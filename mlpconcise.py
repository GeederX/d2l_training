import torch
import myrepo as mr
from torch import nn


# hyper params
batch_size = 256
lr = 0.05
num_epochs = 10

num_inputs = 784
num_middle = 256
num_outputs = 10


def main():
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_middle),
        nn.ReLU(),
        nn.Linear(num_middle, num_outputs),
    )
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    trian_iter, test_iter = mr.load_data_fashion_mnist(batch_size, num_threads=4)
    mr.train(net, trian_iter, test_iter, loss, updater, num_epochs)


if __name__ == "__main__":
    main()
