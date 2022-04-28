import os
import random
from collections import defaultdict
from samplers import TaskRandomSampler
import numpy
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset


def make_loaders(seed, batch_size, train):
    """
    CIFAR-100 split into 10-way classification tasks. Returns 10 loaders, 
    each with 10 random classes. Setting the seed allows the validation 
    data to be of the same classes as training.
    """
    
    whole_dataset = CIFAR100(
        root=os.path.expanduser("~/datasets/CIFAR100"),
        download=False,  # Change to True if running for the first time
        seed=seed,
        train=train,
    )
    
    # load regular 100 class dataset
    whole_loader = DataLoader(whole_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    # deterministically shuffle the tasks' classes
    all_labels = list(range(0,100))
    random.Random(seed).shuffle(all_labels)
    # split shuffled classes into 10 lists
    label_subsets = [all_labels[x:x+10] for x in range(0, 100, 10)]
    
    # list of dataloaders. One for each task.
    loaders = []
    for label_subset in label_subsets:
        subset_idx = []
        for idx, (img, target) in enumerate(whole_loader):
            if target in label_subset:
                subset_idx.append(idx)
        dataset_subset = Subset(whole_dataset, subset_idx)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            sampler=None,
            pin_memory=torch.cuda.is_available(),
            drop_last=train,
        )
        loaders.append(loader)
        
        del whole_loader
    
    return loaders
