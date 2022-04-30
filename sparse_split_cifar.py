'''
Trains sparse lenet 5 on CIFAR100 split into 10-way classification tasks.
'''

import os
from sparse_weights import SparseWeights
from k_winners import KWinners, KWinners2d
from datasets.splitCIFAR100 import make_loaders
import numpy
import torch
from torch import nn
from tqdm import tqdm

num_epochs = 1000
test_bs = 512
test_freq = 5
num_tasks = 1
tolerance = test_freq * 6

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            KWinners2d(percent_on=0.2,
                       channels=64,
                       k_inference_factor=1.,
                       boost_strength=0.,
                       boost_strength_factor=0.),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1),
            KWinners2d(percent_on=0.2,
                       channels=32,
                       k_inference_factor=1.,
                       boost_strength=0.,
                       boost_strength_factor=0.),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            SparseWeights(module=nn.Linear(32*8*8, 256),
                          sparsity=0.5, allow_extremes=True),
            KWinners(n=256,
                     percent_on=0.2,
                     k_inference_factor=1.0,
                     boost_strength=0.,
                     boost_strength_factor=0.),
            SparseWeights(module=nn.Linear(256, 128),
                          sparsity=0.5, allow_extremes=True),
            KWinners(n=128,
                     percent_on=0.2,
                     k_inference_factor=1.0,
                     boost_strength=0.,
                     boost_strength_factor=0.),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def train(seed, train_bs, lr, w_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5(num_classes=10)
    model = model.to(device)
    backup = LeNet5(num_classes=10)
    backup = backup.to(device)
    
    train_loaders = make_loaders(seed, train_bs, train=True)
    test_loaders  = make_loaders(seed, test_bs, train=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    criterion = nn.CrossEntropyLoss()

    final_e, final_acc = [], []
    
    for curr_t in range(num_tasks):        
        best_acc = 0.   # best task test acc so far
        best_e = 0      # epoch of best_acc
        # for e in tqdm(range(num_epochs)):
        for e in range(num_epochs):
            model.train()
            for batch_idx, (imgs, targets) in enumerate(train_loaders[curr_t]):
                optimizer.zero_grad()
                imgs, targets = imgs.to(device), targets.to(device)
                output = model(imgs)
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
                        output = model(imgs)
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
                    output = model(imgs)
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
    _ = train(seed=43, train_bs=32, lr=1e-3, w_decay=0)
    print(_)
    print("SCRIPT FINISHED!")