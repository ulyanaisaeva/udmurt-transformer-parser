import itertools
import numpy as np
import numpy as np
import torch
from collections import Counter
from torch.utils.data.dataset import Dataset


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


class UDDataset(Dataset):

    def __init__(self, data, tokenizer, min_count=3, tags=None):
        self.data = data
        self.tokenizer = tokenizer
        if tags is None:
            tag_counts = Counter([tag for elem in data for tag in elem["labels"]])
            self.tags_ = ["<PAD>", "<UNK>"] + [x for x, count in tag_counts.items() if count >= min_count]
        else:
            self.tags_ = tags
        self.tag_indexes_ = {tag: i for i, tag in enumerate(self.tags_)}
        self.unk_index = 1
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokenization = self.tokenizer(item["words"], is_split_into_words=True)
        last_subtoken_mask = make_last_subtoken_mask(tokenization.word_ids())
        answer = {"input_ids": tokenization["input_ids"], "mask": last_subtoken_mask}
        if "labels" in item:
            labels = [self.tag_indexes_.get(tag, self.unk_index) for tag in item["labels"]]
            zero_labels = np.array([self.ignore_index] * len(tokenization["input_ids"]), dtype=int)
            zero_labels[last_subtoken_mask] = labels
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
