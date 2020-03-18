from typing import List  # isort:skip
from collections import defaultdict

import numpy as np

from catalyst.core import Callback, CallbackOrder
from catalyst.utils import get_activation_fn


class CustomCallback(Callback):
    def __init__(
        self,
        metric_names: List[str],
        meter_list: List,
        input_key: str = "targets",
        output_key: str = "logits",
        class_names: List[str] = None,
        num_classes: int = 2,
        activation: str = "Sigmoid",
    ):
        super().__init__(CallbackOrder.Metric)
        self.metric_names = metric_names
        self.meters = meter_list
        self.input_key = input_key
        self.output_key = output_key
        self.class_names = class_names
        self.num_classes = num_classes
        self.activation = activation
        self.activation_fn = get_activation_fn(self.activation)

    def _reset_stats(self):
        for meter in self.meters:
            meter.reset()

    def on_loader_start(self, state):
        self._reset_stats()

    def on_batch_end(self, state):
        logits = state.batch_out[self.output_key].detach().float()
        targets = state.batch_in[self.input_key]
        loader_values = state.loader_metrics

        for i in range(len(self.meters)):
            self.meters[i].add(logits, targets)
            loader_values[f"{self.metric_names[i]}"] = self.meters[i].value()


    def on_loader_end(self, state):
        loader_values = state.loader_metrics

        for i, meter in enumerate(self.meters):
            metric_ = meter.value()
            prefix = self.metric_names[i]
            metric_name = f"{prefix}"
            loader_values[metric_name] = metric_

        self._reset_stats()


__all__ = ["CustomCallback"]
