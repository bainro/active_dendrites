import math
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DistributedSampler, Sampler

__all__ = [
    "TaskRandomSampler",
]

class TaskRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, task_indices):
        self.task_indices = task_indices
        self.num_classes = len(task_indices)
        self.set_active_tasks(0)

    def set_active_tasks(self, tasks):
        self.active_tasks = tasks
        if not isinstance(self.active_tasks, Iterable):
            self.active_tasks = [tasks]
        self.indices = np.concatenate([self.task_indices[t] for t in self.active_tasks])

    def __iter__(self):
        return (self.indices[i] for i in np.random.permutation(len(self.indices)))

    def __len__(self):
        return len(self.indices)
