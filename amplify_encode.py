import deepquantum as dq
import torch
nqubit = 4
batch = 2
data = torch.randn(batch, 2 ** nqubit)
print(data.shape)