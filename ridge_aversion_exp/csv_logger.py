################################################################################
# Copyright (c) 2021 KU Leuven                                                 #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-03-2021                                                             #
# Author(s): Matthias De Lange, Eli Verwimp                                    #
# E-mail: matthias.delange@kuleuven.be, eli.verwimp@kuleuven.be                #
################################################################################
import sys
from typing import List

import torch

from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import StrategyLogger
from avalanche.training.plugins import PluggableStrategy


class EvalTextLogger(StrategyLogger):
    def __init__(self, metric_filter, file=sys.stdout):
        """
        Text-based logger that logs metrics in a file.
        By default it prints to the standard output.

        The metrics being logged are determined by the metric_filter.
        This is deprecated in later Avalanche releases where evaluator results
        can be accessed directly.

        :param metric_filter: metric names to log in the csv.
        :param file: destination file (default=sys.stdout).
        """
        super().__init__()
        self.file = file
        self.metric_vals = {}
        self.metrics_filter = [metric_filter]
        self.first_line = True

    def log_metric(self, metric_value: 'MetricValue', callback: str) -> None:
        name = metric_value.name
        x = metric_value.x_plot
        val = metric_value.value
        if name in self.metrics_filter:
            self.metric_vals[name] = (name, x, val)

    def _val_to_str(self, m_val):
        if isinstance(m_val, torch.Tensor):
            return '\n' + str(m_val)
        elif isinstance(m_val, float):
            return f'{m_val:.6f}'  # Adjust accuracy after 0
        else:
            return str(m_val)

    def print_current_metrics(self, split='\t'):
        sorted_vals = sorted(self.metric_vals.values(),
                             key=lambda x: x[0])
        if self.first_line:
            for idx, (name, _, _) in enumerate(sorted_vals):  # Print header
                s = '' if idx == 0 else split
                print(f'{s}{name}{split}step_{name}', file=self.file, flush=True)
            self.first_line = False

        for idx, (name, x, val) in enumerate(sorted_vals):  # Print header
            val = self._val_to_str(val)
            s = '' if idx == 0 else split
            print(f'{s}{val}{split}{x}', file=self.file, flush=True)

    def after_eval(self, strategy: 'PluggableStrategy',
                   metric_values: List['MetricValue'], **kwargs):
        super().after_eval(strategy, metric_values, **kwargs)
        self.print_current_metrics()
        self.metric_vals = {}  # Reset
