import math
import torch
import torch.nn.functional as F


def heuristic_scale(out_features: int):
    return math.sqrt(2) * math.log(out_features - 1)


def one_hot(labels: torch.Tensor, num_classes: int):
    return F.one_hot(labels, num_classes=num_classes).float() if len(labels.shape) <= 1 else labels
