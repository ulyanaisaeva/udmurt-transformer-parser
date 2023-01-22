import torch
import numpy as np


def multilabel_nll_loss(prediction_logits, true_labels):
    pos_logits = torch.where(true_labels.bool(), prediction_logits, -np.inf)
    mask = torch.where(torch.mean(torch.where(pos_logits == -np.inf, 0, pos_logits), dim=-1) == 0, 0, 1)
    l = torch.logsumexp(prediction_logits, axis=-1)
    r = torch.logsumexp(pos_logits, axis=-1)
    return torch.mean(l - torch.nan_to_num(r * mask))
