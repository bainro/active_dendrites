'''
Script to train an active dendrite MLP on 10 tasks of PermutedMNIST.
'''

import os
import dendritic_mlp as D
from datasets.permutedMNIST import make_loader
import numpy
import torch
from torch import nn
from tqdm import tqdm


num_epochs = 3
train_bs = 256
test_bs = 512
num_tasks = 10

conf = dict(
    input_size=784,
    output_size=10,  # Single output head shared by all tasks
    hidden_sizes=[2048, 2048],
    dim_context=10,
    kw=True,
    kw_percent_on=0.05,
    weight_sparsity=0.5,
    context_percent_on=0.1, # used for weight init, but paper reported using dense context...
    num_segments=num_tasks
)    

if __name__ == "__main__":
    for seed in range(10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = D.DendriticMLP(**conf)
        model = model.to(device)    

        train_loader = make_loader(num_tasks, seed, train_bs, train=True)
        test_loader = make_loader(num_tasks, seed, test_bs, train=False)

        # Optimizer and Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
        criterion = nn.CrossEntropyLoss()

        # records latest task's test accuracy
        single_acc = []
        for curr_task in range(num_tasks):
            train_loader.sampler.set_active_tasks(curr_task)
            for e in range(num_epochs):
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
                    output = model(imgs, context)
                    pred = output.data.max(1, keepdim=True)[1]
                    train_loss = criterion(output, targets)
                    train_loss.backward()
                    # print(f"train_loss: {train_loss.item()}")
                    optimizer.step()

            model.eval()
            total_correct = 0
            with torch.no_grad():
                for t in range(curr_task+1):
                    test_loader.sampler.set_active_tasks(t)
                    for imgs, targets in test_loader:
                        imgs, targets = imgs.to(device), targets.to(device)
                        one_hot_vector = torch.zeros([num_tasks])
                        one_hot_vector[t] = 1
                        context = torch.FloatTensor(one_hot_vector)
                        context = context.to(device)
                        context = context.unsqueeze(0)
                        context = context.repeat(imgs.shape[0], 1)
                        imgs = imgs.flatten(start_dim=1)
                        output = model(imgs, context)
                        pred = output.data.max(1, keepdim=True)[1]
                        correct = pred.eq(targets.data.view_as(pred)).sum().item()
                        total_correct += correct
                    # record latest trained task's test acc
                    if t == curr_task:
                        # hardcoded number of test examples per mnist digit/class
                        single_acc.append(100 * correct / 10000)
                # print(f"correct: {total_correct}")
                acc = 100. * total_correct * num_tasks / (curr_task+1) / len(test_loader.dataset)
                print(f"[t:{t} e:{e}] test acc: {acc}%")
                
        print("single accuracies: ", single_acc)

    print("SCRIPT FINISHED!")
