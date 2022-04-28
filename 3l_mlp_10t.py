'''
Script to train a fully connected, 3 layer MLP on 10 tasks of PermutedMNIST.
'''

import os
from datasets.permutedMNIST import make_loader
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

class ModifiedInitMLP(nn.Module):  
    def __init__(self, input_size, num_classes,
                 hidden_sizes=(100, 100)):
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
        
        # Modified Kaiming weight initialization which considers 1) the density of
        # the input vector and 2) the weight density in addition to the fan-in

        weight_density = 1.0
        input_flag = False
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):

                # Assume input is fully dense, but hidden layer activations are only
                # 50% dense due to ReLU
                input_density = 1.0 if not input_flag else 0.5
                input_flag = True

                _, fan_in = layer.weight.size()
                bound = 1.0 / numpy.sqrt(input_density * weight_density * fan_in)
                nn.init.uniform_(layer.weight, -bound, bound)

    def forward(self, x):
        return self.classifier(x)

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedInitMLP(**conf)
    model = model.to(device)
    
    train_loader = make_loader(num_tasks, seed, train_bs, train=True)
    test_loader = make_loader(num_tasks, seed, test_bs, train=False)
    
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
