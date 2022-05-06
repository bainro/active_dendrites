'''
Script to train fully connected, 3 layer MLPs on consecutive tasks from PermutedMNIST.
Alternatively, pass the --search flag on the cmd line to do a hyperparameter search.
'''

import os
import sys
from datasets.permutedMNIST import make_loader
import numpy
import torch
from torch import nn


class ModifiedInitMLP(nn.Module):  
    def __init__(self, input_size, num_classes, init_dof=1,
                 hidden_sizes=(100, 100)):
        # init_dof: Used in modified Kaiming weight init.
           
        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(int(numpy.prod(input_size)), hidden_sizes[0]),
            nn.ReLU()
        ]
        for idx in range(1, len(hidden_sizes)):
            layers.extend([
                nn.Linear(hidden_sizes[idx - 1], hidden_sizes[idx]),
                nn.ReLU()
            ])
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.classifier = nn.Sequential(*layers)
        
        # Modified Kaiming weight initialization
        # Numenta had this as a variable (ie degree of freedom) even when 
        # context input was dense. Searching over it for a fair comparision
        input_density = init_dof
        weight_density = 1.0
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                _, fan_in = layer.weight.size()
                bound = 1.0 / numpy.sqrt(input_density * weight_density * fan_in)
                nn.init.uniform_(layer.weight, -bound, bound)

    def forward(self, x):
        return self.classifier(x)

    
if __name__ == "__main__":
    test_bs = 512
    num_tasks = 100
    # used for creating avg over all seed runs
    all_single_acc = []
    all_avg_acc = []
    num_args = len(sys.argv)
    # gridsearch flag given on cmd line
    if num_args == 2:
        INIT_DOFs = [0.05, 0.1, 0.5]
        LRs = [5e-7, 1e-6, 5e-6, 1e-5]
        NUM_EPOCHS = [3, 4, 5, 6]
        TRAIN_BSs = [64, 128, 256]
        SEEDs = [1]
    else:
        INIT_DOFs = [1.]
        LRs = [1e-6]
        NUM_EPOCHS = [3]
        TRAIN_BSs = [256]
        SEEDs = range(10) # [33, 34, 35, 36, 37]
    
    for train_bs in TRAIN_BSs:
        for num_epochs in NUM_EPOCHS:
            for lr in LRs:
                for init_dof in INIT_DOFs:    
                    for seed in SEEDs:
                        conf = dict(
                            input_size=784,
                            num_classes = 10,
                            hidden_sizes=[2048, 2048],
                            init_dof=init_dof
                        )
                        
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = ModifiedInitMLP(**conf)
                        model = model.to(device)

                        train_loader = make_loader(num_tasks, seed, train_bs, train=True)
                        test_loader = make_loader(num_tasks, seed, test_bs, train=False)

                        # Optimizer and Loss
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
                        criterion = nn.CrossEntropyLoss()

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
                                    imgs = imgs.flatten(start_dim=1)
                                    output = model(imgs)
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
                                        imgs = imgs.flatten(start_dim=1)
                                        output = model(imgs)
                                        pred = output.data.max(1, keepdim=True)[1]
                                        latest_correct += pred.eq(targets.data.view_as(pred)).sum().item()
                                    total_correct += latest_correct
                                    # record latest trained task's test acc
                                    if t == curr_task:
                                        # hardcoded number of test examples per mnist digit/class
                                        single_acc.append(100 * latest_correct / 10000)
                                # print(f"correct: {correct}")
                                acc = 100. * total_correct * num_tasks / (curr_task+1) / len(test_loader.dataset)
                                avg_acc.append(acc)
                                # print(f"[t:{t} e:{e}] test acc: {acc}%")

                        # print("single accuracies: ", single_acc)
                        # print("running avg accuracies: ", avg_acc)
                        all_single_acc.append(single_acc)
                        all_avg_acc.append(avg_acc)

                    # figure out average wrt all seeds
                    avg_seed_acc = list(map(lambda x: sum(x)/len(x), zip(*all_avg_acc)))
                    avg_single_acc = list(map(lambda x: sum(x)/len(x), zip(*all_single_acc)))
                    print("\n", f"init_dof: {init_dof}, lr: {lr}, num_tasks: {num_tasks}, num_epochs: {num_epochs}, train_bs: {train_bs}")
                    print("seed avg running avg accuracies: ", avg_seed_acc)
                    print("seed avg single accuracies: ", avg_single_acc)

    print("SCRIPT FINISHED!")
