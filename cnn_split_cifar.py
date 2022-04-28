'''
Train a lenet 5 CNN on cifar-100 split into 10-way classification tasks.
'''

import os
from datasets.splitCIFAR100 import make_loader
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

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*7*7, 512),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    
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
