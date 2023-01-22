import numpy as np
import torch
import torch.nn as nn
from transformers.optimization import AdamW

from model_utils.ud_dataset_utils import *
from model_utils.multilabel_nll_loss import multilabel_nll_loss


class BasicTransformersTaggingModel(nn.Module):

    def __init__(self, model, labels_number, lr=1e-5, device="cpu", **kwargs):
        super(BasicTransformersTaggingModel, self).__init__()
        self.model = model
        self.labels_number = labels_number
        self.build_network(labels_number)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss(reduction="mean", ignore_index=-100)
        self.device = device
        if self.device is not None:
            self.to(self.device)
        self.optimizer = AdamW(self.parameters(), lr=lr, weight_decay=0.01)

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    def forward(self, input_ids, **kwargs):
        raise NotImplementedError("You should implement forward pass in your derived class.") 

    def train_on_batch(self, x, y, mask=None):
        self.train()
        self.optimizer.zero_grad()
        loss = self._validate(x, y, mask=mask)
        loss["loss"].backward()
        self.optimizer.step()
        return loss

    def validate_on_batch(self, x, y, mask=None):
        self.eval()
        with torch.no_grad():
            return self._validate(x, y, mask=mask)

    def _validate(self, x, y, mask=None):
        if self.device is not None:
            y = y.to(self.device)
        if "mask" not in x:
            x["mask"] = mask
        batch_output = self(**x)
        loss = self.criterion(batch_output["log_probs_tensor"].permute(0, 2, 1), y)
        return {"loss": loss, "labels": batch_output["labels"], "probs": batch_output["probs"]}


class TransformersTaggingModel(BasicTransformersTaggingModel):

    def build_network(self, labels_number):
        self.proj_layer = torch.nn.Linear(self.hidden_size, self.labels_number)
        return self

    def forward(self, input_ids, mask, **kwargs):
        input_ids = input_ids.to(self.device) # shape=(B, W)
        cls_output = self.model(input_ids)["last_hidden_state"] # shape=(B, W, H)
        logits = self.proj_layer(cls_output) # shape=(B, W, K)
        log_probs = self.log_softmax(logits) # shape=(B, W, K)
        _, labels = torch.max(log_probs, dim=-1) # shape=(B, W)
        batch_labels = [None] * len(labels)
        batch_probs = [None] * len(labels)
        for i, elem in enumerate(labels):
            if mask is None:
                curr_mask = [True] * len(elem)
            else:
                curr_mask = mask[i].bool()
            batch_labels[i] = elem[curr_mask].detach().cpu().numpy()
            batch_probs[i] = np.exp(log_probs[i,curr_mask].detach().cpu().numpy())
        return {"log_probs_tensor": log_probs, "labels": batch_labels, "probs": batch_probs}
    
    def predict(self, dataset):
        self.eval()
        dataloader = FieldBatchDataLoader(dataset, device=self.device)
        answer = [None] * len(dataset)
        for batch in dataloader:
            with torch.no_grad():
                batch_answer = self.forward(**batch)
            for i, sent_labels in zip(batch["indexes"], batch_answer["labels"]):
                answer[i] = np.take(dataset.tags_, sent_labels)
        return answer


class MultilabelTransformersTaggingModel(BasicTransformersTaggingModel):

    def build_network(self, labels_number):
        self.proj_layer = torch.nn.Linear(self.hidden_size, self.labels_number)
        # self.criterion = None
        return self

    def forward(self, input_ids, mask, **kwargs):
        input_ids = input_ids.to(self.device) # shape=(B, W)
        cls_output = self.model(input_ids)["last_hidden_state"] # shape=(B, W, H)
        logits = self.proj_layer(cls_output) # shape=(B, W, K)
        log_probs = self.log_softmax(logits) # shape=(B, W, K)
        _, labels = torch.max(logits, dim=-1) # shape=(B, W)
        batch_labels = [None] * len(labels)
        batch_probs = [None] * len(labels)
        for i, elem in enumerate(labels):  # iterate over samples in batch
            if mask is None:
                curr_mask = [True] * len(elem)
            else:
                curr_mask = mask[i].bool()
            batch_labels[i] = elem[curr_mask].detach().cpu().numpy()
            batch_probs[i] = np.exp(log_probs[i,curr_mask].detach().cpu().numpy())
        return {"logits": logits, "log_probs_tensor": log_probs, 
                "labels": batch_labels, "probs": batch_probs}

    def _validate(self, x, y, mask=None):
        if self.device is not None:
            y = y.to(self.device)
        if "mask" not in x:
            x["mask"] = mask
        batch_output = self(**x)
        global test_logits, test_true
        test_logits = batch_output["logits"]
        test_true = y
        loss = multilabel_nll_loss(batch_output["logits"], y)
        return {"loss": loss, "labels": batch_output["labels"], 
                "probs": batch_output["probs"]}
