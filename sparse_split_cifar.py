'''
Trains sparse lenet 5 on CIFAR100 split into 10-way classification tasks.
'''

import os
from sparse_weights import SparseWeights, rezero_weights
from k_winners import KWinners, KWinners2d
from datasets.splitCIFAR100 import make_loaders
from dendritic_mlp import DendriticMLP as D
import numpy
import torch
from torch import nn

num_epochs = 1000
test_bs = 512
test_freq = 1
num_tasks = 10
tolerance = test_freq * 30

class SparseLeNet5(nn.Module):
    def __init__(self, c_a_s, f_a_s, f_w_s, boost_set, num_classes=10):
        super(SparseLeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            KWinners2d(percent_on=c_a_s,
                       channels=64,
                       k_inference_factor=boost_set[0],
                       boost_strength=boost_set[1],
                       boost_strength_factor=boost_set[2]),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1),
            KWinners2d(percent_on=c_a_s,
                       channels=32,
                       k_inference_factor=boost_set[0],
                       boost_strength=boost_set[1],
                       boost_strength_factor=boost_set[2]),
            nn.MaxPool2d(kernel_size=2),
        )
        sparse_w_1 = SparseWeights(module=nn.Linear(32*8*8, 256),
                                   sparsity=f_w_s, allow_extremes=True)
        D._init_sparse_weights(sparse_w_1, 1 - f_w_s)
        sparse_w_2 = SparseWeights(module=nn.Linear(256, 128),
                                   sparsity=f_w_s, allow_extremes=True)
        D._init_sparse_weights(sparse_w_2, 1 - f_w_s)
        self.classifier = nn.Sequential(
            sparse_w_1,
            KWinners(n=256,
                     percent_on=f_a_s,
                     k_inference_factor=boost_set[0],
                     boost_strength=boost_set[1],
                     boost_strength_factor=boost_set[2]),
            sparse_w_2,
            KWinners(n=128,
                     percent_on=f_a_s,
                     k_inference_factor=boost_set[0],
                     boost_strength=boost_set[1],
                     boost_strength_factor=boost_set[2]),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def train(seed, train_bs, lr, c_a_s=.2, f_a_s=.2, f_w_s=0.5, boost_set=(1.,0.,0.)):
    """
    c_a_s: convolutional layer's 2D WTA activation sparsity
    f_a_s: fully connected layer's 1D WTA activation sparsity
    f_w_s: fully connected layer's weight sparsity
    boost_set: tuple of 3 items.
        1) k_inference_factore 
        2) boost_strength
        3) boost_strength_factor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseLeNet5(c_a_s, f_a_s, f_w_s, boost_set, num_classes=10)
    model = model.to(device)
    backup = SparseLeNet5(c_a_s, f_a_s, f_w_s, boost_set, num_classes=10)
    backup = backup.to(device)
    
    train_loaders = make_loaders(seed, train_bs, train=True)
    test_loaders  = make_loaders(seed, test_bs, train=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    running_acc, single_acc = [], []

    break_early = False
    for curr_t in range(num_tasks):        
        best_acc = 0.   # best task test acc so far
        best_e = 0
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
                model.apply(rezero_weights)
            
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
                        single_acc.append(best_acc)
                        # if best_acc < 60:
                            # break_early = True
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
            running_acc.append(acc)
            print(f"\n\n[t:{t} e:{e}] test acc: {acc}%\n\n")
            # let's speed this grid search up!
            # if acc < 20 or break_early:
                # break
        
    # running avg task test acc
    # best test acc for each task
    return running_acc, single_acc 

if __name__ == "__main__":
    all_running = []
    all_latest = []
    for s in range(5):
        running, latest = train(seed=s, train_bs=32, lr=1e-3, c_a_s=.2, f_a_s=.1, f_w_s=0.5)
        all_running.append(running)
        all_latest.append(latest)
    
    # figure out average wrt all seeds
    avg_running = list(map(lambda x: sum(x)/len(x), zip(*all_running)))
    avg_latest = list(map(lambda x: sum(x)/len(x), zip(*all_latest)))
    print("avg running: ", avg_running)
    print("avg latest: ", avg_latest)
    print("SCRIPT FINISHED!")
