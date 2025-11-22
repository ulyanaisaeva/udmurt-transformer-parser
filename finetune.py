from datasets import load_dataset
from typing import List
from clearml import Task
from sklearn.metrics import f1_score
from argparse import ArgumentParser
from datasets import ClassLabel, Sequence
from transformers import BertTokenizerFast, AutoModelForTokenClassification, TrainingArguments, \
    Trainer, DataCollatorForTokenClassification
import numpy as np
import torch
import torch.nn as nn
import yaml
import sys

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/finetune_config.yaml', help='Path to config YAML file')

args = parser.parse_args()

# Load config
try:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if config is None:
        config = {}
except FileNotFoundError:
    print(f"Error: Config file '{args.config}' not found.", file=sys.stderr)
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error: Failed to parse config file '{args.config}': {e}", file=sys.stderr)
    sys.exit(1)

# Get paths and experiment settings from config
paths_config = config.get('paths', {})
experiment_config = config.get('experiment', {})
training_config = config.get('training', {})

# Get paths from config (required)
tokenizer_path = paths_config.get('tok_path', '')
model_path = paths_config.get('model_path', '')
save_path = paths_config.get('save_path', '')
train_file = paths_config.get('train_file', '')
valid_file = paths_config.get('valid_file', '')

# Get experiment settings from config
exp_name = experiment_config.get('exp_name', '')
train_size = experiment_config.get('train_size', 0)

# Validate required paths
if not tokenizer_path:
    print("Error: tokenizer path (tok_path) must be provided in config file", file=sys.stderr)
    sys.exit(1)
if not model_path:
    print("Error: model path (model_path) must be provided in config file", file=sys.stderr)
    sys.exit(1)
if not save_path:
    print("Error: save path (save_path) must be provided in config file", file=sys.stderr)
    sys.exit(1)
if not train_file:
    print("Error: training file (train_file) must be provided in config file", file=sys.stderr)
    sys.exit(1)
if not valid_file:
    print("Error: validation file (valid_file) must be provided in config file", file=sys.stderr)
    sys.exit(1)

Task.init(project_name='udmurt', task_name=exp_name)

data = load_dataset("json", data_files={
    "train": train_file,
    "validation": valid_file,
})

amb = data['validation']['labels_amb']
data['train'].remove_columns('labels_amb')
data['validation'].remove_columns('labels_amb')


def get_label_set(dataset):
    label_set = set()
    for split in dataset:
        for sample in dataset[split]:
            for label in sample["labels"]:
                for l in label:
                    if l != "-1":
                        label_set.add(l)
    return sorted(label_set)


label_set = get_label_set(data)

id2label = {i: label for i, label in enumerate(label_set)}
label2id = {label: i for i, label in enumerate(label_set)}

num_labels = len(label2id)

class_labels = ClassLabel(names=label_set)
class_labels._int2str = {i: label for i, label in enumerate(label_set)}

if train_size > 0:
    data['train'] = data['train'].select(range(train_size))

data = data.cast_column("labels", Sequence(Sequence(class_labels)))

# Note: Use XLMRobertaTokenizerFast for original glot models if needed
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, do_lower_case=False, strip_accents=False,
                                              padding="max_len", is_fast=True)


def tokenize_and_align_labels(examples, tokenizer, multilabel=False, last_token=True):
    tokenized_inputs = tokenizer(examples["words"], truncation=False, is_split_into_words=True)
    labels = []
    if last_token:
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for i, word_idx in enumerate(word_ids[:-1]):
                if word_idx is not None and word_idx != word_ids[i + 1]:
                    curr_label = label[word_idx]
                else:
                    curr_label = [-100]
                label_ids.append(curr_label if multilabel else curr_label[0])
            label_ids.append([-100] if multilabel else -100)
            labels.append(label_ids)
    else:
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append([-100] if multilabel else -100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx] if multilabel else label[word_idx][0])
                else:
                    label_ids.append([-100] if multilabel else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


prepared_data = data.map(lambda x: tokenize_and_align_labels(x, tokenizer, multilabel=False, last_token=False),
                         batched=True, num_proc=16)
label_list = data["train"].features["labels"].feature.feature.names

model = AutoModelForTokenClassification.from_pretrained(
    model_path, num_labels=len(label_list), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)


def get_batch_metrics(pred_labels: List[str], labels: List[str], labels_amb: List[List[str]] = None):
    correct, total = 0, 0
    correct_amb, total_amb = 0, 0
    pred_labels_flat, labels_flat = [], []
    f1_sample_scores = []
    metrics = dict()
    for i, (sample_pred_labels, sample_labels) in enumerate(zip(pred_labels, labels)):
        pred_labels_flat.extend(sample_pred_labels)
        labels_flat.extend(sample_labels)
        sample_labels = [label for label in sample_labels if label not in [-100]]
        assert len(sample_pred_labels) == len(
            sample_labels), f"Predicted~true shape mismatch: {len(sample_pred_labels)} ~ {len(sample_labels)}"
        for j, (x, y) in enumerate(zip(sample_pred_labels, sample_labels)):
            if labels_amb is not None and len(labels_amb[i][j]) > 1:
                total_amb += 1
                if x == y:
                    correct_amb += 1
            total += 1
            if x == y:
                correct += 1
        f1_sample_scores.append(f1_score(sample_labels, sample_pred_labels, average="micro"))

    if labels_amb is not None and total_amb:
        metrics["token_acc_amb"] = correct_amb / total_amb
    else:
        metrics["token_acc_amb"] = None

    metrics["token_acc"] = correct / total
    metrics["f1_macro"] = f1_score(labels_flat, pred_labels_flat, average="macro")
    metrics["f1_instance_level"] = sum(f1_sample_scores) / len(f1_sample_scores)
    return metrics


def compute_metrics(p, true_labels_amb=None):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metrics = get_batch_metrics(true_predictions, true_labels, labels_amb=true_labels_amb)
    return metrics


class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self):
        super(MultiLabelSoftmaxLoss, self).__init__()

    def forward(self, prediction_logits, true_labels):
        global prediction_logits_g, true_labels_g
        prediction_logits_g = prediction_logits
        true_labels_g = true_labels
        loss = torch.logsumexp(prediction_logits, axis=-1) - \
               torch.logsumexp(torch.where(
                   true_labels == 1, prediction_logits,
                   torch.tensor(float("-Inf"), device=prediction_logits.device)
               ), axis=-1)
        return torch.nanmean(torch.where(~loss.isinf(), loss, torch.tensor(np.nan, device=loss.device)))


class MultiLabelSoftmaxTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = MultiLabelSoftmaxLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        try:
            loss = self.loss_fct(outputs.logits, labels)
        except AttributeError:
            loss = self.loss_fct(outputs.logits.view(-1, model.module.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class MultiLabelSigmoidTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fct(outputs.logits[labels != -100], labels[labels != -100].float())

        return (loss, outputs) if return_outputs else loss


# Training hyperparameters loaded from config file
training_args = TrainingArguments(
    output_dir=save_path,
    learning_rate=training_config.get('learning_rate', 5e-5),
    per_device_train_batch_size=training_config.get('per_device_train_batch_size', 1),
    per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 16),
    num_train_epochs=training_config.get('num_train_epochs', 10),
    weight_decay=training_config.get('weight_decay', 0.01),
    evaluation_strategy=training_config.get('evaluation_strategy', 'epoch'),
    save_strategy=training_config.get('save_strategy', 'epoch'),
    logging_strategy=training_config.get('logging_strategy', 'steps'),
    logging_steps=training_config.get('logging_steps', 5),
    report_to=training_config.get('report_to', 'clearml'),
    seed=training_config.get('seed', 42),
    data_seed=training_config.get('data_seed', 42),
    optim=training_config.get('optim', 'adamw_torch'),
    lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
    save_total_limit=training_config.get('save_total_limit', 1)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_data["train"],
    eval_dataset=prepared_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda x: compute_metrics(x, amb),
)

trainer.train()
