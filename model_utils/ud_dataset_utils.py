import itertools
import numpy as np
import re
import torch
from collections import Counter
from torch.utils.data.dataset import Dataset
from typing import List


def read_infile(infile, keep_ambiguity=False):

    def split_tag_variants(pos, feats):
        pos_vars = [re.sub("\d+\)", "", s) for s in pos.split("//")]
        feats_vars = [re.sub("\d+\)", "", s) for s in feats.split("//")]
        if len(pos_vars) < len(feats_vars):
            pos_vars = [pos_vars[0]] * len(feats_vars)
        elif len(pos_vars) > len(feats_vars):
            feats_vars = [feats_vars[0]] * len(pos_vars)
        return [f'{p},{f}' if f not in ["", "_"] else p for p, f in zip(pos_vars, feats_vars)]

    answer, sent, labels = [], [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                if len(sent) > 0:
                    answer.append({"words": sent, "labels": labels})
                sent, labels = [], []
                continue
            elif line[:1] == "#":
                continue
            splitted = line.split("\t")
            if not splitted[0].isdigit():
                continue
            sent.append(splitted[1])
            if keep_ambiguity:
                tag_list = split_tag_variants(splitted[3], splitted[5])
                labels.append(tag_list)
            else:
                tag = splitted[3] if splitted[5] in ["", "_"] else f"{splitted[3]},{splitted[5]}"
                labels.append(tag)
    if len(sent) > 0:
        answer.append({"words": sent, "labels": labels})
    return answer


def make_last_subtoken_mask(mask):
    mask = mask[1:-1]
    is_last_word = [False] + list((first != second) for first, second in zip(mask[:-1], mask[1:])) + [True, False]
    return is_last_word


def pad_tensor(vec, length, dim, pad_symbol):
    pad_size = list(vec.shape)
    pad_size[dim] = length - vec.shape[dim]
    answer = torch.cat([vec, torch.ones(*pad_size, dtype=torch.long) * pad_symbol], axis=dim)
    return answer


def pad_tensors(tensors, pad=0):
    if isinstance(tensors[0], (int, np.integer)):
        return torch.LongTensor(tensors)
    elif isinstance(tensors[0], (float, np.float)):
        return torch.Tensor(tensors)
    tensors = [torch.LongTensor(tensor) for tensor in tensors]
    L = max(tensor.shape[0] for tensor in tensors)
    tensors = [pad_tensor(tensor, L, dim=0, pad_symbol=pad) for tensor in tensors]
    return torch.stack(tensors, axis=0)


class TagEncoder:

    def __init__(self, tags=None, unk_index=None, ignore_index=None) -> None:
        self.tags_ = tags
        self.tag_indexes_ = None
        self.unk_index = unk_index
        self.ignore_index = ignore_index
        self.n_unique_indexes = None
        self.n_encoded_indexes = len(self.tags_) if self.tags_ is not None else None
    
    def fit(self, tag_list: List[str], min_count: int = None, additional_tags: List[str] = None) -> None:
        if isinstance(tag_list[0], list):
            tag_list = list(itertools.chain.from_iterable(tag_list))
        if self.tags_ is None:
            tag_counts = Counter(tag_list)
            self.tags_ = additional_tags + [x for x, count in tag_counts.items() if count >= min_count]
        self.tag_indexes_ = {tag: i for i, tag in enumerate(self.tags_)}
        self.n_encoded_indexes = len(self.tags_)
        self.n_unique_indexes = len(tag_counts)
        print(f"Fitted TagEncoder with {self.n_encoded_indexes} tags ({self.n_unique_indexes - self.n_encoded_indexes + len(additional_tags)} ignored according to `min_count = {min_count}`).")
    
    def encode(self, tag: List[str]) -> List[List[int]]:
        return self.tag_indexes_.get(tag, self.unk_index)


class UDDataset(Dataset):

    def __init__(self, data, tokenizer, min_count=3, tags=None):
        self.data = data
        self.tokenizer = tokenizer
        self.tag_encoder = TagEncoder(tags, unk_index=1, ignore_index=-100)
        data_labels = list(itertools.chain.from_iterable([elem["labels"] for elem in data]))
        self.tag_encoder.fit(data_labels,
                             min_count=min_count, additional_tags=["<PAD>", "<UNK>"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokenization = self.tokenizer(item["words"], is_split_into_words=True)
        last_subtoken_mask = make_last_subtoken_mask(tokenization.word_ids())
        answer = {"input_ids": tokenization["input_ids"], "mask": last_subtoken_mask}
        if "labels" in item:
            labels = [self.tag_encoder.encode(tag) for tag in item["labels"]]
            zero_labels = np.array([self.tag_encoder.ignore_index] * len(tokenization["input_ids"]), dtype=int)
            zero_labels[last_subtoken_mask] = labels
            answer["y"] = zero_labels
        return answer


class UDDatasetWithAmbiguity(UDDataset):
    def __init__(self, data, tokenizer, min_count=3, tags=None):
        super().__init__(data, tokenizer, min_count=min_count, tags=tags)

    def __getitem__(self, index):
        item = self.data[index]
        tokenization = self.tokenizer(item["words"], is_split_into_words=True)
        last_subtoken_mask = make_last_subtoken_mask(tokenization.word_ids())
        answer = {"input_ids": tokenization["input_ids"], "mask": last_subtoken_mask}
        if "labels" in item:
            zero_labels = np.full((len(tokenization["input_ids"]), self.tag_encoder.n_encoded_indexes), 
                                       0, dtype=int)#.tolist()  # zeros of shape (W, K)
            assert len(zero_labels) == len(last_subtoken_mask)
            label_idx = 0  # label index from initial tag sequence (to skip unmasked tokens)
            for mask_idx, mask_value in enumerate(last_subtoken_mask):
                if mask_value:
                    # encode tags for selected word
                    label_variants_encoded = [self.tag_encoder.encode(tag) for tag in item["labels"][label_idx]]
                    label_variants_encoded = sorted(list(set(label_variants_encoded)))
                    zero_labels[mask_idx, label_variants_encoded] = 1
                    label_idx += 1
            answer["y"] = zero_labels
        return answer


class FieldBatchDataLoader:

    def __init__(self, X, batch_size=32, sort_by_length=True, 
                 length_field=None, state=115, device="cpu"):
        self.X = X
        self.batch_size = batch_size
        self.sort_by_length = sort_by_length
        self.length_field = length_field  ## добавилось
        self.device = device
        np.random.seed(state)

    def __len__(self):
        return (len(self.X)-1) // self.batch_size + 1 

    def __iter__(self):
        if self.sort_by_length:
            if self.length_field is not None:
                lengths = [len(x[self.length_field]) for x in self.X]
            else:
                lengths = [len(list(x.values())[0]) for x in self.X]
            order = np.argsort(lengths)
            batched_order = np.array([order[start:start+self.batch_size] 
                                      for start in range(0, len(self.X), self.batch_size)])
            np.random.shuffle(batched_order)
            self.order = np.fromiter(itertools.chain.from_iterable(batched_order), dtype=int)
        else:
            self.order = np.arange(len(self.X))
            np.random.shuffle(self.order)
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.X):
            raise StopIteration()
        end = min(self.idx + self.batch_size, len(self.X))
        try:
            indexes = [self.order[i] for i in range(self.idx, end)]
        except IndexError as e:
            print(f"self.idx={self.idx}, end={end}, len(self.order)={len(self.order)}")
            print(e)
        batch = dict()
        for field in self.X[indexes[0]]:
            batch[field] = pad_tensors([self.X[i][field] for i in indexes]).to(self.device)
        batch["indexes"] = indexes
        self.idx = end
        return batch
