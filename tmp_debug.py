'''
Script to train an active dendrite MLP on 10 tasks of PermutedMNIST.
'''

import os
import math
from collections import defaultdict

import dendritic_mlp as D
from mlp import ModifiedInitMLP
from datasets.permutedMNIST import PermutedMNIST
from samplers import TaskRandomSampler

import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


seed = 42
num_epochs = 4
batch_size = 256
test_batch_size = 512
num_tasks = 10
# @TODO evil stare ?
num_classes = 10 * num_tasks
num_classes_per_task = math.floor(num_classes / num_tasks)

conf = dict(
    input_size=784,
    num_classes = 10,
    # output_size=10,  # Single output head shared by all tasks
    hidden_sizes=[2048, 2048],
    # dim_context=10,
    # kw=True,
    # kw_percent_on=0.05,
    # dendrite_weight_sparsity=0.0,
    # weight_sparsity=0.5,
    # context_percent_on=1.0,
    # num_segments=num_tasks
)    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = D.DendriticMLP(**conf)
    model = ModifiedInitMLP(**conf)
    model = model.to(device)
    
    dataset = PermutedMNIST(
        root=os.path.expanduser("~/datasets/permutedMNIST"),
        download=False,  # Change to True if running for the first time
        seed=seed,
        train=True,
        num_tasks=num_tasks,
    )
    
    # target -> all indices for that target
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        target = int(dataset.targets[idx % len(dataset.data)])
        task_id = dataset.get_task_id(idx)
        target += 10 * task_id
        assert task_id < 2
        class_indices[target].append(idx)

    # task -> all indices corresponding to this task
    task_indices = defaultdict(list)
    for i in range(num_tasks):
        for j in range(num_classes_per_task):
            task_indices[i].extend(class_indices[j + (i * num_classes_per_task)])

    sampler = TaskRandomSampler(task_indices)
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        num_workers=4,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    
    test_dataset = PermutedMNIST(
        root=os.path.expanduser("~/datasets/permutedMNIST"),
        download=False,
        seed=seed,
        train=False,
        num_tasks=num_tasks,
    )
    
    test_class_indices = defaultdict(list)
    for idx in range(len(test_dataset)):
        target = int(test_dataset.targets[idx % len(test_dataset.data)])
        task_id = test_dataset.get_task_id(idx)
        target += 10 * task_id
        test_class_indices[target].append(idx)

    test_task_indices = defaultdict(list)
    for i in range(num_tasks):
        for j in range(num_classes_per_task):
            test_task_indices[i].extend(test_class_indices[j + (i * num_classes_per_task)])

    test_sampler = TaskRandomSampler(test_task_indices)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=test_sampler is None,
        num_workers=4,
        sampler=test_sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for curr_task in range(num_tasks):
        train_loader.sampler.set_active_tasks(curr_task)
        for e in tqdm(range(num_epochs)):
            model.train()
            for batch_idx, (imgs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                imgs, targets = imgs.to(device), targets.to(device)
                one_hot_vector = torch.zeros([num_tasks])
                one_hot_vector[curr_task] = 1
                context = torch.FloatTensor(one_hot_vector)
                context = context.to(device)
                context = context.unsqueeze(0)
                context = context.repeat(imgs.shape[0], 1)
                imgs = imgs.flatten(start_dim=1)
                # output = model(imgs, context)
                output = model(imgs)
                pred = output.data.max(1, keepdim=True)[1]
                train_loss = criterion(output, targets)
                train_loss.backward()
                print(f"train_loss: {train_loss.item()}")
                optimizer.step()
                
        train_loader.sampler.set_active_tasks(curr_task+1)
        for e in tqdm(range(num_epochs)):
            model.train()
            for batch_idx, (imgs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                imgs, targets = imgs.to(device), targets.to(device)
                one_hot_vector = torch.zeros([num_tasks])
                one_hot_vector[curr_task+1] = 1
                context = torch.FloatTensor(one_hot_vector)
                context = context.to(device)
                context = context.unsqueeze(0)
                context = context.repeat(imgs.shape[0], 1)
                imgs = imgs.flatten(start_dim=1)
                # output = model(imgs, context)
                output = model(imgs)
                pred = output.data.max(1, keepdim=True)[1]
                train_loss = criterion(output, targets)
                train_loss.backward()
                print(f"[t2] train_loss: {train_loss.item()}")
                optimizer.step()
                
        train_loader.sampler.set_active_tasks(curr_task)
        for e in tqdm(range(num_epochs)):
            model.train()
            for batch_idx, (imgs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                imgs, targets = imgs.to(device), targets.to(device)
                one_hot_vector = torch.zeros([num_tasks])
                one_hot_vector[curr_task] = 1
                context = torch.FloatTensor(one_hot_vector)
                context = context.to(device)
                context = context.unsqueeze(0)
                context = context.repeat(imgs.shape[0], 1)
                imgs = imgs.flatten(start_dim=1)
                # output = model(imgs, context)
                output = model(imgs)
                pred = output.data.max(1, keepdim=True)[1]
                train_loss = criterion(output, targets)
                train_loss.backward()
                print(f"[t1a] train_loss: {train_loss.item()}")
                # optimizer.step()
                
            exit()
                
            model.eval()
            correct = 0
            with torch.no_grad():
                #for t in range(curr_task+1):
                test_loader.sampler.set_active_tasks(curr_task)
                for imgs, targets in test_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    one_hot_vector = torch.zeros([num_tasks])
                    one_hot_vector[curr_task] = 1
                    context = torch.FloatTensor(one_hot_vector)
                    context = context.to(device)
                    context = context.unsqueeze(0)
                    context = context.repeat(imgs.shape[0], 1)
                    imgs = imgs.flatten(start_dim=1)
                    # output = model(imgs, context)
                    output = model(imgs)
                    pred = output.data.max(1, keepdim=True)[1]
                    # print(f"targets: {targets[10, ...]}")
                    # print(f"predictions: {pred[10, ...]}")
                    correct += pred.eq(targets.data.view_as(pred)).sum().item()
                print(f"correct: {correct}")
                # acc = 100. * correct * num_tasks / (curr_task+1) / len(test_loader.dataset)
                acc = 100. * correct / 10000 
                print(f"[t:{curr_task} e:{e}] test acc: {acc}%")

    print("SCRIPT FINISHED!")
