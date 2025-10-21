import torch
import myrepo as mr
from torch import nn


# hyper parameters
batch_size = 256
num_epochs = 10
lr = 0.05
num_inputs = 784
num_middle = 256
num_outputs = 10


# parameters init
W1 = nn.Parameter(torch.randn((num_inputs, num_middle), requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_middle, requires_grad=True))
W2 = nn.Parameter(torch.randn((num_middle, num_outputs), requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]


# loss
loss = nn.CrossEntropyLoss()


# updater
updater = torch.optim.SGD(params, lr)


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X: torch.Tensor):
    X = X.reshape(-1, num_inputs)
    O1 = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(O1, W2) + b2


def main():
    train_iter, test_iter = mr.load_data_fashion_mnist(batch_size, num_threads=4)
    mr.train(net, train_iter, test_iter, loss, updater, num_epochs)


if __name__ == "__main__":
    main()
