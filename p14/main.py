
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
num_epochs = 130


# load datatset
with open('data/p1_X.pickle', 'rb') as handle:
    p1_X = pickle.load(handle)
with open('data/p1_Y.pickle', 'rb') as handle:
    p1_Y = pickle.load(handle)



p1_train_X, p1_train_Y = p1_X[:100], p1_Y[:100]
p1_val_X, p1_val_Y = p1_X[100:], p1_Y[100:]

c1 = 0  # 标签为1的样本数量
c2 = 0  # 标签为0的样本数量
list_0 = []
list_1 = []
for idx,i in enumerate(p1_train_Y):
    if i == 1:
        c1+=1
        list_1.append(idx)
    else:
        list_0.append(idx)
        c2+=1

print(f"train_0:{c2}")
print(f"train_1:{c1}")
print(len(list_0))

def add_samples(l:list[int],p1_train_X,p1_train_Y):
    for i in l:
        a = torch.unsqueeze(p1_train_X[i], 0)
        p1_train_X = torch.cat((p1_train_X, a), 0)
        b = torch.unsqueeze(p1_train_Y[i], 0)
        p1_train_Y = torch.cat([p1_train_Y, b], 0)

# 对0样本进行过采样
l = random.sample(list_0,c1-c2)
add_samples(l,p1_train_X,p1_train_Y)

# 对0 1 样本同时进行采样
num = 0
l_0 = random.sample(list_0, num)
l_1 = random.sample(list_1, num)

add_samples(l_0,p1_train_X,p1_train_Y)
add_samples(l_1,p1_train_X,p1_train_Y)

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



def cal_acc(result,label):
    result = result.detach().numpy()
    label = label.detach().numpy()
    #print(np.argmax(result,1),label)
    return sum(np.argmax(result, 1) == label)
w = [c2/(c1+c2),c1/(c1+c2)]
print(w)
def effecient(pt,gamma):
    return (1-pt)**gamma
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
    acc = 0
    for x, labels in valloader:
        out, cir = net(x)
        loss = criterion(out, labels)
        acc+=cal_acc(out,labels)
        loss_tot += loss.item()
        count += 1
    print('val loss', loss_tot/count,' val acc',acc/50)
    return loss_tot/count, cir


def train_epoch(epoch, net, optimizer, criterion, log):
    net.train()
    loss_epoch = []
    for x, labels in trainloader:
        out, _ = net(x)
        #print(out)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch.append(loss.detach().item())
    print(f'epoch:{epoch}', end=' ')
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



if sys.argv[1] == 'dj':
    net = NetDJ(n=n) # DJ algorithm
elif sys.argv[1] == 'pqc':
    net = Net(n=n)   # PQC algorithm    
  

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
    plt.savefig('data/savefig_example.png')
    plt.show()

val(net)

PATH = 'data/net.pt'
torch.save(net.state_dict(), PATH)








