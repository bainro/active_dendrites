import os
from collections import defaultdict
from samplers import TaskRandomSampler
import numpy
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader


def make_loader(seed, batch_size, train):
    """
    CIFAR-100 split into 10-way classification tasks. 
    """
    
    # load regular 100 class dataset
    # split into 10 tasks given a seed https://stackoverflow.com/questions/47432168/taking-subsets-of-a-pytorch-dataset
    # return list of dataloaders
    
    num_classes = 100
    num_classes_per_task = 10
    
    dataset = CIFAR100(
        root=os.path.expanduser("~/datasets/CIFAR100"),
        download=False,  # Change to True if running for the first time
        seed=seed,
        train=train,
    )
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        sampler=None,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
    
    return loader
