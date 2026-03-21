"""Bloth CrossEntropy nn.Module wrapper."""
import torch.nn as nn
from ..kernels.cross_entropy import (
    fused_cross_entropy as _fce,
    fused_linear_cross_entropy as _flce,
)


class BlothCrossEntropy(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        return _fce(logits, labels, self.ignore_index)


def fused_cross_entropy(logits, labels, ignore_index=-100):
    return _fce(logits, labels, ignore_index)


def fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100):
    return _flce(hidden, weight, labels, ignore_index)
