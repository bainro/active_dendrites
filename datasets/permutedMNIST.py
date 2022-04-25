import os

import numpy
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class PermutedMNIST(MNIST):
    """
    The permutedMNIST dataset contains MNIST images where the same random permutation
    of pixel values is applied to each image. More specifically, the dataset can be
    broken down into 'tasks', where each such task is the set of all MNIST images, but
    with a unique pixel-wise permutation applied to all images. `num_tasks` gives the
    number of 10-way classification tasks (each corresponding to a unique pixel-wise
    permutation) that a continual learner will try to learn in sequence.
    """

    def __init__(self, num_tasks, seed, train, root=".", target_transform=None,
                 download=False, normalize=True):

        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize((0.13062755,), (0.30810780,)))
        data_transform = transforms.Compose(t)
        super().__init__(root=root, train=train, transform=data_transform,
                         target_transform=target_transform, download=download)

        self.num_tasks = num_tasks

        # Use a generator object to manually set the seed and generate the same
        # num_tasks random permutations for both training and validation datasets; the
        # first one is the identity permutation (i.e., regular MNIST), represented
        # below as `None`
        g = torch.manual_seed(seed)

        self.permutations = [
            torch.randperm(784, generator=g) for task_id in range(1, num_tasks+1) # <- +1 part of debug!
        ]
        # self.permutations.insert(0, None) # part of debug!

    def __getitem__(self, index):
        """
        Returns an (image, target) pair.

        In particular, this method retrieves an MNIST image, and based on the value of
        `index`, determines which pixel-wise permutation to apply. Target values are
        also scaled to be unique to each permutation.
        """
        
        img, target = super().__getitem__(index % len(self.data))
        '''
        ### DEBUG CODE
        visual_test = numpy.transpose(img, (1, 2, 0))
        plt.imshow(visual_test, cmap='gray', vmin=0.4242, vmax=2.8215)
        plt.savefig(f"my_plot_{index}_{target}.png")
        exit()
        '''

        # Determine which task `index` corresponds to
        task_id = self.get_task_id(index)
        
        # Apply permutation to `img`
        permutation = self.permutations[task_id]
        if permutation is not None:
            _, height, width = img.size()
            img = img.view(-1, 1)
            img = img[permutation, :]
            img = img.view(1, height, width)

        # Since target values are not shared between tasks, `target` should be in the
        # range [0 + 10 * task_id, 9 + 10 * task_id]
        # target += 10 * task_id
        return img, target

    def __len__(self):
        return self.num_tasks * len(self.data)
    
    @property
    def processed_folder(self):
        return os.path.join(self.root, "MNIST", "processed")

    def get_task_id(self, index):
        return index // len(self.data)
