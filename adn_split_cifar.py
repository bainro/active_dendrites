''' 
Trains lenet 5 with active dendrite FC layers on CIFAR100 split into 10-way classification tasks.
'''

import os
from k_winners import KWinners, KWinners2d
from datasets.splitCIFAR100 import make_loaders
from dendritic_mlp import AbsoluteMaxGatingDendriticLayer as dends1D
from dendritic_mlp import AbsoluteMaxGatingDendriticLayer2d as dends2D
import numpy
import torch
from torch import nn


num_epochs = 100
test_bs = 512
test_freq = 1
num_tasks = 10
tolerance = test_freq * 30

class LeNet5(nn.Module):
    def __init__(self, device, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.ModuleList()
        layers = [
            dends2D(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
                    num_segments=10, # Testing! Should change back to num_tasks!
                    dim_context=num_tasks,
                    module_sparsity=0,
                    dendrite_sparsity=0),
            KWinners2d(percent_on=0.2,
                       channels=64,
                       k_inference_factor=1.125,
                       boost_strength=0.75,
                       boost_strength_factor=0.4),
            nn.MaxPool2d(kernel_size=2),
            dends2D(nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1),
                    num_segments=10, # Testing! Should change back to num_tasks!
                    dim_context=num_tasks,
                    module_sparsity=0,
                    dendrite_sparsity=0),
            KWinners2d(percent_on=0.2,
                       channels=32,
                       k_inference_factor=1.125,
                       boost_strength=0.75,
                       boost_strength_factor=0.4),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(1)
        ]
        for l in layers:
            self.features.append(l)
        self.dends = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.final_l = nn.ModuleList()
        self.dends.append(dends1D(nn.Linear(32*8*8, 256),
                          num_segments=10, # Testing! Should change back to num_tasks!
                          dim_context=num_tasks,
                          module_sparsity=0.5,
                          dendrite_sparsity=0))
        self.activations.append(KWinners(256, percent_on=0.05,
                                         k_inference_factor=1.0,
                                         boost_strength=0.0,
                                         boost_strength_factor=0.0))
        self.dends.append(dends1D(nn.Linear(256, 128),
                          num_segments=10, # Testing! Should change back to num_tasks!
                          dim_context=num_tasks,
                          module_sparsity=0.5,
                          dendrite_sparsity=0))
        self.activations.append(KWinners(128, percent_on=0.05,
                                         k_inference_factor=1.0,
                                         boost_strength=0.0,
                                         boost_strength_factor=0.0))
        self.final_l.append(nn.Linear(128, num_classes))

    def forward(self, x, context):
        for i, l in enumerate(self.features):
            # @TODO will want to change for long term use
            # conv layers that need dendritic context
            if i == 0 or i == 3:
                x = l(x, context)
            else:
                x = l(x)
        x = self.dends[0](x, context)
        x = self.activations[0](x)
        x = self.dends[1](x, context)
        x = self.activations[1](x)
        x = self.final_l[0](x)
        return x
    
def train(seed, train_bs, lr,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5(device, num_classes=2)
    model = model.to(device)
    backup = LeNet5(device, num_classes=2)
    backup = backup.to(device)
    
    train_loaders = make_loaders(seed, train_bs, train=True)
    test_loaders  = make_loaders(seed, test_bs, train=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    final_e, final_acc = [], []
    
    for curr_t in range(num_tasks):        
        best_acc = 0.   # best task test acc so far
        best_e = 0      # epoch of best_acc
        for e in range(num_epochs):
            model.train()
            for batch_idx, (imgs, targets) in enumerate(train_loaders[curr_t]):
                # need to visualize images & labels for sanity check (& paper?)
                optimizer.zero_grad()
                imgs, targets = imgs.to(device), targets.to(device)
                one_hot_vector = torch.zeros([num_tasks])
                one_hot_vector[curr_t] = 1
                context = torch.FloatTensor(one_hot_vector)
                context = context.to(device)
                context = context.unsqueeze(0)
                context = context.repeat(imgs.shape[0], 1)
                output = model(imgs, context)
                pred = output.data.max(1, keepdim=True)[1]
                train_loss = criterion(output, targets)
                train_loss.backward()
                optimizer.step()

            if e % test_freq == 0:
                print(f"train_loss: {train_loss.item()}")    
                model.eval()
                correct = 0
                with torch.no_grad():
                    for imgs, targets in test_loaders[curr_t]:
                        imgs, targets = imgs.to(device), targets.to(device)
                        one_hot_vector = torch.zeros([num_tasks])
                        one_hot_vector[curr_t] = 1
                        context = torch.FloatTensor(one_hot_vector)
                        context = context.to(device)
                        context = context.unsqueeze(0)
                        context = context.repeat(imgs.shape[0], 1)
                        output = model(imgs, context)
                        pred = output.data.max(1, keepdim=True)[1]
                        correct += pred.eq(targets.data.view_as(pred)).sum().item()
                    # print(f"correct: {correct}")
                    acc = 100. * correct / len(test_loaders[curr_t].dataset)
                    print(f"[t:{curr_t} e:{e}] test acc: {acc}%")
                    if acc > best_acc:
                        best_acc = acc
                        best_e = e
                        backup.load_state_dict(model.state_dict())
                    elif best_e + tolerance <= e:
                        # haven't improved test acc recently
                        # reload best checkpoint & stop early
                        model.load_state_dict(backup.state_dict())
                        final_e.append(best_e)
                        final_acc.append(best_acc)
                        break
                        
        model.eval()
        correct = 0
        with torch.no_grad():
            for t in range(curr_t+1):
                for imgs, targets in test_loaders[t]:
                    imgs, targets = imgs.to(device), targets.to(device)
                    one_hot_vector = torch.zeros([num_tasks])
                    one_hot_vector[t] = 1
                    context = torch.FloatTensor(one_hot_vector)
                    context = context.to(device)
                    context = context.unsqueeze(0)
                    context = context.repeat(imgs.shape[0], 1)
                    output = model(imgs, context)
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(targets.data.view_as(pred)).sum().item()
            print(f"correct: {correct}")
            acc = 100. * correct / (curr_t+1) / len(test_loaders[t].dataset)
            print(f"\n\n[t:{t} e:{e}] test acc: {acc}%\n\n")

    # final task-avg accuracy
    # epochs that best test acc occurred 
    # best test acc for each task
    return acc, final_e, final_acc 

if __name__ == "__main__":
    _ = train(seed=15, train_bs=256, lr=1e-3)
    print(_)
    print("SCRIPT FINISHED!")
