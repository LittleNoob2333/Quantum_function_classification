import torch
import sys
 #print(sys.path)
from deepquantum import *
import torch.nn as nn
import torch.nn.functional as F

class NetDJ(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.cir1 = QubitCircuit(self.n+1)
        self.cir2 = QubitCircuit(self.n+1)
        self.circuit1()
        self.circuit2()

    def circuit1(self):
        self.cir1.x(self.n)
        self.cir1.hlayer()
        
    def circuit2(self):        
        self.cir2.hlayer(wires=list(range(self.n)))

    
    def forward(self, oracles):
        out = []
        for oracle in oracles:
            cir_o = QubitCircuit(self.n+1)
            cir_o.any(oracle, name='dj_oracle')
            cir = self.cir1 + cir_o + self.cir2
            state = cir()
            state = state.squeeze(0)
            x = torch.abs(state[0,0]) ** 2 + torch.abs(state[1,0]) ** 2
            logits = torch.stack([x , 1 - x], dim=-1) # p(00...0), 1-p(00...0) 
            out.append(logits)
            
        return torch.stack(out), cir
    
    def get_results(self, oracles):
        """
        Args:
            oracles (tensor): a batch of oracles, shape=(batch_size, 2**(n+1), 2**(n+1))
        Return:
            states (tensor): a batch of final states,  shape=(batch_size, 2**(n+1), 1)
        """
        states = []
        for oracle in oracles:
            cir_o = QubitCircuit(self.n+1)
            cir_o.any(oracle)
            cir = self.cir1 + cir_o + self.cir2
            state = cir()
            state = state.squeeze(0)
            states.append(state)
            
        return torch.stack(states)    

    


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

        self.cir1 = QubitCircuit(self.n+1)
        self.cir2 = QubitCircuit(self.n+1)

        self.circuit1()
        self.circuit2()
        self.l1 = nn.Linear(2, 4)
        self.l2 = nn.Linear(4,2)

    def strong_entangle(self,n=1):
        self.cir1.cnot(0, 2)
        self.cir1.cnot(1, 3)
        self.cir1.cnot(2, 4)
        self.cir1.cnot(2, 0)
        self.cir1.cnot(3, 1)
        self.cir1.cnot(4, 2)

    def strong_entangle1(self,n=1):
        self.cir2.cnot(0, 2)
        self.cir2.cnot(1, 3)
        self.cir2.cnot(2, 0)
        self.cir2.cnot(3, 1)
    def entangle(self,n):
        self.cir1.cnot(0, 1)
        self.cir1.cnot(1, 2)
        self.cir1.cnot(2, 3)
        self.cir1.cnot(3, 4)
    def entangle1(self,n):
        self.cir2.cnot(0, 1)
        self.cir2.cnot(1, 2)
        self.cir2.cnot(2, 3)
        #self.cir2.cnot(3, 4)
    def circuit1(self):
        #self.cir1.cry()

        self.cir1.rylayer()
        self.cir1.rzlayer()

        self.cir1.cx(1,0)
        self.cir1.cx(2,1)
        self.cir1.cx(3,2)
        self.cir1.cx(4, 3)
        self.cir1.cx(0, 4)
        self.cir1.barrier()

        self.cir1.rylayer()
        self.cir1.rzlayer()
        self.cir1.cx(3, 0)
        self.cir1.cx(1, 3)
        self.cir1.cx(4, 1)
        self.cir1.cx(2, 4)
        self.cir1.cx(0, 2)
        self.cir1.barrier()


    def circuit2(self):

        self.cir2.rylayer()
        self.cir2.rzlayer()
        self.cir2.cx(1, 0)
        self.cir2.cx(2, 1)
        self.cir2.cx(3, 2)
        self.cir2.cx(4, 3)
        self.cir2.cx(0, 4)

        self.cir2.barrier()
        self.cir2.rylayer()
        self.cir2.rzlayer()
        self.cir2.cx(3, 0)
        self.cir2.cx(1, 3)
        self.cir2.cx(4, 1)
        self.cir2.cx(2, 4)
        self.cir2.cx(0, 2)
        self.cir2.barrier()

    
    def forward(self, oracles):
        out = []
        for oracle in oracles:
            cir_o = QubitCircuit(self.n+1)
            cir_o.any(oracle, name='dj_oracle')
            cir = self.cir1 + cir_o + self.cir2
            state = cir()
            state = state.squeeze(0)
            x = torch.abs(state[0,0]) ** 2 + torch.abs(state[1,0]) ** 2
            logits = torch.stack([x , 1 - x], dim=-1) # p(00...0), 1-p(00...0)
            #o = self.l1(logits)
            #o = F.relu(o)
            #o = self.l2(o)
            #logits = F.softmax(o)
            out.append(logits)
        return torch.stack(out), cir
    
    def get_results(self, oracles):
        """
        Args:
            oracles (tensor): a batch of oracles, shape=(batch_size, 2**(n+1), 2**(n+1))
        Return:
            states (tensor): a batch of final states,  shape=(batch_size, 2**(n+1), 1)
        """
        states = []
        for oracle in oracles:
            cir_o = QubitCircuit(self.n+1)
            cir_o.any(oracle)
            cir = self.cir1 + cir_o + self.cir2
            state = cir()
            state = state.squeeze(0)
            print()
            states.append(state)
            
        return torch.stack(states)        


    







