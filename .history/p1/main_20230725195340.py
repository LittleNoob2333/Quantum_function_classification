
import sys
import pickle
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from deepquantum import *

from model import Net, NetDJ

# global settings
n = 4
batch_size = 8
num_epochs = 30


# load datatset
with open('data/p1_X.pickle', 'rb') as handle:
    p1_X = pickle.load(handle)
with open('./data/p1_Y.pickle', 'rb') as handle:
    p1_Y = pickle.load(handle)



p1_train_X, p1_train_Y = p1_X[:100], p1_Y[:100]
p1_val_X, p1_val_Y = p1_X[100:], p1_Y[100:]



class MyDataSet(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.Y != None:
            return self.X[index], self.Y[index]
        else:
            return self.X[index], -1
    


trainset = MyDataSet(p1_train_X, p1_train_Y)
valset = MyDataSet(p1_val_X, p1_val_Y)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)



def cross_entropy_loss(out, labels):
    loss = 0
    for o, l in zip(out, labels):
        loss += -torch.log(o[l]+1e-10)
    return loss


def val(net):
    net.eval()
    criterion =  cross_entropy_loss # nn.CrossEntropyLoss()
    loss_tot = 0
    count = 0
    for x, labels in valloader:
        out, cir = net(x)
        loss = criterion(out, labels)
        loss_tot += loss.item()
        count += 1
    print('val loss', loss_tot/count)
    return loss_tot/count, cir


def train_epoch(epoch, net, optimizer, criterion, log):
    net.train()
    loss_epoch = []
    for x, labels in trainloader:
        out, _ = net(x)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch.append(loss.detach().item())
    
    log.update(epoch, np.array(loss_epoch).mean(), val(net)[0])



class OptimizerLog:
    """Log to store optimizer's intermediate results"""
    def __init__(self):
        self.steps = []
        self.costs = []
        self.val_costs = []
    
    def update(self, step, cost, val_cost):
        #write_log = evaluation%9 == 0
        write_log = True 
        if write_log:
            self.steps.append(step)
            self.costs.append(cost)
            self.val_costs.append(val_cost)



#if sys.argv[1] == 'dj':
net = NetDJ(n=n) # DJ algorithm
#elif sys.argv[1] == 'pqc':
#    net = Net(n=n)   # PQC algorithm
  

_, cir = val(net)
cir.draw(filename='data/pqc.png')

# training
if len(list(net.parameters())) != 0:
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = cross_entropy_loss # nn.CrossEntropyLoss()
    log = OptimizerLog()
    for epoch in range(num_epochs): 
        train_epoch(epoch, net=net, optimizer=optimizer, criterion=criterion, log=log)
    
    # plot the learning curve
    fig, ax = plt.subplots()
    ax.set_title("Cost function value against iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost function value")
    ax.plot(log.steps, log.costs, label='train')       
    ax.plot(log.steps, log.val_costs, label='val')     
    #plt.yscale('log')
    ax.legend()
    plt.show()

val(net)


PATH = 'data/net.pt'
torch.save(net.state_dict(), PATH)








