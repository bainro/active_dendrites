'''
Script to train a fully connected, 3 layer MLP on 10 tasks of PermutedMNIST.
'''

import os

from mlp import ModifiedInitMLP
from datasets.permutedMNIST import PermutedMNIST, make_loader

import numpy
import torch
from torch import nn
from tqdm import tqdm


seed = 42
num_epochs = 5
train_bs = 256
test_bs = 512
num_tasks = 10

conf = dict(
    input_size=784,
    num_classes = 10,
    hidden_sizes=[2048, 2048],
)    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedInitMLP(**conf)
    model = model.to(device)
    
    dataset, train_loader = make_loader(num_tasks, seed, train_bs, train=True)
    test_dataset, test_loader = make_loader(num_tasks, seed, test_bs, train=False)
    
    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-6, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    
    for curr_task in range(num_tasks):
        train_loader.sampler.set_active_tasks(curr_task)
        for e in tqdm(range(num_epochs)):
            model.train()
            for batch_idx, (imgs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                imgs, targets = imgs.to(device), targets.to(device)
                imgs = imgs.flatten(start_dim=1)
                output = model(imgs)
                pred = output.data.max(1, keepdim=True)[1]
                train_loss = criterion(output, targets)
                train_loss.backward()
                # print(f"train_loss: {train_loss.item()}")
                optimizer.step()
                
        model.eval()
        correct = 0
        with torch.no_grad():
            for t in range(curr_task+1):
                test_loader.sampler.set_active_tasks(t)
                for imgs, targets in test_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    imgs = imgs.flatten(start_dim=1)
                    output = model(imgs)
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(targets.data.view_as(pred)).sum().item()
            print(f"correct: {correct}")
            acc = 100. * correct * num_tasks / (curr_task+1) / len(test_loader.dataset)
            print(f"[t:{t} e:{e}] test acc: {acc}%")

    print("SCRIPT FINISHED!")
