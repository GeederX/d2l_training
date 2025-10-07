import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
from torch import nn


X = torch.arange(30).reshape([10,3])
w = torch.arange(15).reshape([5,3])

print(torch.matmul(X,w.T).shape)
