import os
from collections import defaultdict
from samplers import TaskRandomSampler
import numpy
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader


class splitCIFAR100(CIFAR100):
    """
    CIFAR-100 split into 10-way classification tasks. 
    
    """

    def __init__(self, num_tasks, seed, train, root=".", download=False):

        t = [transforms.ToTensor()]
        data_transform = transforms.Compose(t)
        super().__init__(root=root, train=train, transform=data_transform,
                         target_transform=None, download=download)

        # Use a generator object to manually set the seed and generate the same
        # num_tasks splits for both training and validation datasets
        g = torch.manual_seed(seed)

        self.permutations = [
            torch.randperm(784, generator=g) for task_id in range(1, num_tasks)
        ]

    def __getitem__(self, index):        
        img, target = super().__getitem__(index % len(self.data))
        # Determine the which task `index` belongs to
        task_id = self.get_task_id(index)
        return img, target

    def __len__(self):
        return 10 * len(self.data)
    
    @property
    def processed_folder(self):
        return os.path.join(self.root, "CIFAR100", "processed")

    def get_task_id(self, index):
        return index // len(self.data)

def make_loader(seed, batch_size, train):
    num_classes = 100
    num_classes_per_task = 10
    
    dataset = PermutedMNIST(
        root=os.path.expanduser("~/datasets/splitCIFAR100"),
        download=True,  # Change to True if running for the first time
        seed=seed,
        train=train,
    )
    
    # target -> all indices for that target
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        target = int(dataset.targets[idx % len(dataset.data)])
        task_id = dataset.get_task_id(idx)
        target += 10 * task_id
        class_indices[target].append(idx)

    # task -> all indices corresponding to this task
    task_indices = defaultdict(list)
    for i in range(num_tasks):
        for j in range(num_classes_per_task):
            task_indices[i].extend(class_indices[j + (i * num_classes_per_task)])

    sampler = TaskRandomSampler(task_indices)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        num_workers=4,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
    
    return dataset, loader
