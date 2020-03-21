import torch.nn as nn
from converter import strLabelConverter
from common import preds_converter
from catalyst.utils.torch import any2device
import torch


class WrapCTCLoss(nn.Module):
    def __init__(self, alphabet, device='cpu'):
        super().__init__()
        self.converter = strLabelConverter(alphabet)
        self.device = device
        self.loss = nn.CTCLoss(reduction='sum', zero_infinity=False)

    def __call__(self, logits, targets):
        text, length = self.converter.encode(targets)
        text, length = text.to(self.device), length.to(self.device)
        sim_preds, preds_size = preds_converter(self.converter, logits, len(targets))
        loss = self.loss(logits, text, preds_size, length)
        return loss


class WrapAccuracy(nn.Module):
    def __init__(self, alphabet, device='cpu'):
        super().__init__()
        self.converter = strLabelConverter(alphabet)
        self.device = device
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def value(self):
        return self.sum / self.count

    def add(self, logits, targets):
        text, length = self.converter.encode(targets)
        text, length = text.to(self.device), length.to(self.device)
        sim_preds, preds_size = preds_converter(self.converter, logits, len(targets))
        self.count += len(targets)
        for cnt in range(len(targets)):
            if sim_preds[cnt] == targets[cnt]:
                self.sum += 1
