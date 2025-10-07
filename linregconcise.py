import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
from torch import nn


batch_size = 10
num_epochs = 3


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def sample_iter(data_arrays, batch_size, is_train):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
num_examples = 1000
features, labels = synthetic_data(true_w, true_b, num_examples)


net = nn.Sequential(nn.Linear(2, 1))


net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


for epoch in range(num_epochs):
    for X, y in sample_iter([features,labels],batch_size,True):
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')