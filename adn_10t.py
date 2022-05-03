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
num_tasks = 100

conf = dict(
    input_size=784,
    output_size=10,
    hidden_sizes=[2048, 2048],
    dim_context=784,
    kw=True,
    kw_percent_on=0.05,
    weight_sparsity=0.5,
    context_percent_on=0.05, # used for weight init, but paper reported using dense context...
    num_segments=10 # num_tasks
)    

if __name__ == "__main__":
    seeds = [33, 34, 35, 36, 37]
    # used for creating avg over all seed runs
    all_single_acc = []
    all_avg_acc = []
    for seed in seeds:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = D.DendriticMLP(**conf)
        model = model.to(device)    

        train_loader = make_loader(num_tasks, seed, train_bs, train=True)
        test_loader = make_loader(num_tasks, seed, test_bs, train=False)

        # Optimizer and Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=7e-5, weight_decay=0)
        criterion = nn.CrossEntropyLoss()

        # @TODO use Euclidian distance to infer which task's input at test time
        # calculate all the context vectors, avg's of each tasks' inputs
        contexts = []
        # hardcoded for mnist train
        assert len(train_loader.dataset) == 50000, f"{len(train_loader.dataset)}"
        for curr_task in range(num_tasks):
            train_loader.sampler.set_active_tasks(curr_task)
            sum = 0
            for batch_idx, (imgs, _) in enumerate(train_loader):
                imgs = imgs.to(device)
                imgs = imgs.flatten(start_dim=1)
                sum += imgs.sum(0)
            avg_task_input = sum / len(train_loader.dataset)
            avg_task_input = avg_task_input.to(device)
            contexts.append(avg_task_input)
        
        # records latest task's test accuracy
        single_acc = []
        avg_acc = []
        for curr_task in range(num_tasks):
            train_loader.sampler.set_active_tasks(curr_task)
            for e in range(num_epochs):
                model.train()
                for batch_idx, (imgs, targets) in enumerate(train_loader):
                    optimizer.zero_grad()
                    imgs, targets = imgs.to(device), targets.to(device)
                    context = contexts[curr_task]
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
                    latest_correct = 0
                    test_loader.sampler.set_active_tasks(t)
                    for imgs, targets in test_loader:
                        imgs, targets = imgs.to(device), targets.to(device)
                        context = contexts[t]
                        imgs = imgs.flatten(start_dim=1)
                        output = model(imgs, context)
                        pred = output.data.max(1, keepdim=True)[1]
                        latest_correct += pred.eq(targets.data.view_as(pred)).sum().item()
                    total_correct += latest_correct
                    # record latest trained task's test acc
                    if t == curr_task:
                        # hardcoded number of test examples per mnist digit/class
                        single_acc.append(100 * latest_correct / 10000)
                # print(f"correct: {total_correct}")
                acc = 100. * total_correct * num_tasks / (curr_task+1) / len(test_loader.dataset)
                avg_acc.append(acc)
                # print(f"[t:{t} e:{e}] test acc: {acc}%")
                
        print("single accuracies: ", single_acc)
        print("running avg accuracies: ", avg_acc)
        all_single_acc.append(single_acc)
        all_avg_acc.append(avg_acc)
        
    # figure out average wrt all seeds
    avg_seed_acc = list(map(lambda x: sum(x)/len(x), zip(*all_avg_acc)))
    avg_single_acc = list(map(lambda x: sum(x)/len(x), zip(*all_single_acc)))
    print("seed avg running avg accuracies: ", avg_seed_acc)
    print("seed avg single accuracies: ", avg_single_acc)
    
    print("SCRIPT FINISHED!")
