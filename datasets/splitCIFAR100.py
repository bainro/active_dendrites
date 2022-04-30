import os
import random
from collections import defaultdict
from samplers import TaskRandomSampler
import numpy
import torch
from torchvision import transforms as trans
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset


def make_loaders(seed, batch_size, train):
    """
    CIFAR-100 split into 10-way classification tasks. Returns 10 loaders, 
    each with 10 random classes. Setting the seed allows the validation 
    data to be of the same classes as training.
    """

    t = trans.Compose([trans.ToTensor(),
                       trans.Normalize((0.4914, 0.4822, 0.4465), 
                                       (0.2023, 0.1994, 0.2010))
                      ])
    
    conf = {"root": os.path.expanduser("~/datasets/CIFAR100"),
            "download": False, "train": train, "transform": t}
    # will throw error if dataset isn't already downloaded
    try:
        whole_dataset = CIFAR100(**conf)  
    except:
        conf.update({'download': True})
        whole_dataset = CIFAR100(**conf)  
    
    # load regular 100 class dataset
    whole_loader = DataLoader(whole_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # deterministically shuffle the tasks' classes
    all_labels = list(range(0,100))
    random.Random(seed).shuffle(all_labels)
    n_c_per_task = 2 # binary classification
    # split shuffled classes into 10 lists
    label_subsets = [all_labels[x:x+n_c_per_task] for x in range(0, 100, n_c_per_task)]
    
    # list of indexes for each task's example subset
    subsets = [[] for _ in range(100//n_c_per_task)]
    # should be parallel lists
    assert len(subsets) == len(label_subsets)
    targets = []
    file_path = os.path.join("./", f".{'train' if train else 'test'}_idx.dat")
    if os.path.exists(file_path):
        targets = numpy.fromfile(file_path, dtype=int)
    else:
        for _imgs, _targets in whole_loader:
            for target in _targets:
                targets.append(target)
        targets = numpy.array(targets)
        targets.tofile(file_path)
    for idx, target in enumerate(targets):
        # find index of label_subset that this class belongs to
        label_sub_idx = None
        for _idx in range(len(label_subsets)):
            if target in label_subsets[_idx]:
                label_sub_idx = _idx
                break
        assert type(label_sub_idx) != type(None)
        subsets[label_sub_idx].append(idx)
    
    # list of dataloaders. One for each task.
    loaders = []
    for i, subset in enumerate(subsets):
        # map the 100 class id's to [0, 9]
        for j, k in enumerate(label_subsets[i]):
            t_copy = numpy.array(whole_dataset.targets)
            t_copy[t_copy == k] = j
            whole_dataset.targets = list(t_copy)
        print(subset)
        dataset_subset = Subset(whole_dataset, subset)
        loader = DataLoader(
            dataset=dataset_subset,
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
