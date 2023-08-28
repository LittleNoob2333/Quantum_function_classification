import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import math

print(type(os.path.join('home','lib','g++')))

dtype = torch.float

# 这里使用了三目运算符
# expression1 if condition1 is true else expression2
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

x = torch.linspace(-math.pi,math.pi,2000,dtype=dtype)
y = torch.sin(x)
import random
list = [0,1,2,3,4]
rs = random.sample(list,4)
print(rs)
