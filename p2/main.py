import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from deepquantum import *
from model import Net

# global settings
n = 4
batch_size = 16
num_classes = 3
num_epochs = 1


# load datatset
with open('data/p2_X.pickle', 'rb') as handle:
    p2_X = pickle.load(handle)
with open('data/p2_Y.pickle', 'rb') as handle:
    p2_Y = pickle.load(handle)



TRAIN_DATA, VAL_DATA, TRAIN_LABELS, VAL_LABELS = train_test_split(p2_X, p2_Y, test_size=0.2, random_state=21)



# basis encoding
def basis_encoding(func, n, debug=False):
    """Given n bit string as input, there are n+1 qubits
    
    Args:
        func (dict): bitstring function
    """
    
    if debug:
        print('Given a black box fucntion:')
        print(func)
        print('we encoding its all input-output relations as:')
        print('|psi> = sum_x |x>|f(x)> = 1/sqrt(2**n) * (')
    desired_state = np.zeros(2**(n+1))
    i = 0
    for k, v in func.items():
        b = format(k, '0'+str(n)+'b')+str(v)
        if debug: 
            if i == 0:
                print(f'|{b}>', end=' ')
            else:
                print(f'+ |{b}>', end=' ')
        b = int(b, 2)
        desired_state[b] = (2**n)**-0.5
        i += 1
    if debug: print(')')

    return desired_state

print('=', basis_encoding(VAL_DATA[1], n=n, debug=True))

def get_tensor(data, labels=None):
    x_lst = []
    y_lst = []
    for i in range(len(data)):
        x = torch.tensor(basis_encoding(data[i], n=n))
        if labels == None:
            y = torch.tensor(-1)
        else:
            y = torch.tensor(labels[i])
        x_lst.append(x)
        y_lst.append(y)
    return torch.stack(x_lst), torch.stack(y_lst)



x_train, y_train = get_tensor(TRAIN_DATA, TRAIN_LABELS)
x_val, y_val = get_tensor(VAL_DATA, VAL_LABELS)


trainset = TensorDataset(x_train,y_train)
valset = TensorDataset(x_val,y_val)


trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

print('max_steps:', len(trainloader)*num_epochs)

    


def train(net, log):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        net.train()
        loss_epoch = []
        for step, data in enumerate(trainloader):
            x, labels = data[0], data[1]
            out = net(x)   
            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.detach().item())  
            
            if step % 100 == 0:
                gstep = epoch*len(trainloader)+step
                log.update(gstep, np.array(loss_epoch).mean(), val(net).detach().item())


def val(net):
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    loss_tot = 0
    count = 0
    net.eval()
    for data in valloader:
        x, labels = data[0], data[1]
        out = net(x)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        loss = criterion(out, labels)
        loss_tot += loss
        _, predicted = torch.max(out.data, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        count += 1
    print('val acc', correct/total)
    return loss_tot/count


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

net = Net(n=n)
log = OptimizerLog()

net.cir.draw(filename='data/pqc.png')

val(net)

print('-'*10)
train(net, log)
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
print('-'*10)

val(net)

PATH = 'data/net.pt'
torch.save(net.state_dict(), PATH)

