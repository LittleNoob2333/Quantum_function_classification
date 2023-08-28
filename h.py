import torch
import torch.nn as nn
import torch.nn.functional as F
import deepquantum as dq
from deepquantum import *
class MyCircuit(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.params = nn.Parameter(torch.ones(n))
        self.cir = self.circuit(n)
    def circuit(self,n):
        cir = dq.QubitCircuit(n)
        cir.hlayer()
        cir.rylayer(encode=True)
        cir.cnot_ring()
        for i in range(n):
            cir.observable(i)
        return cir
    def forward(self):
        self.cir(self.params)
        return self.cir.expectation().mean()

print("dasdas")
n1 = 4
cir1 = MyCircuit(n1)
optimizer = torch.optim.SGD(cir1.parameters(),lr=0.01)
for i in range(10):
    for j in cir1.parameters():
        print(j)
    optimizer.zero_grad()
    loss = cir1()
    loss.backward()
    optimizer.step()
    print(f"epoch: {i} loss: {loss}")
cir1.cir.draw(filename="./1.png")