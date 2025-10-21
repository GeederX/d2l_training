import torch
import torchvision
import os
from torch.utils import data
from torchvision import transforms


batch_size = 256
lr = 0.1
num_epochs = 10

num_inputs = 784
num_outputs = 10


W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros((num_outputs), requires_grad=True)


def load_data_fashin_mnist(batch_size, resize=None, num_threads=0):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=os.path.join(".", "data"), train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=os.path.join(".", "data"), train=False, transform=trans, download=True
    )
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_threads),
        data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_threads),
    )


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=-1, keepdim=True)
    return X_exp / partition


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y]).mean()


def accuracy(y, y_hat):
    y_hat = y_hat.argmax(axis=-1)
    cmp = y_hat.type(y.dtype) == y
    return float((cmp.type(y.dtype).sum()) / len(y))


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(y, net(X)), 1)
    return metric[0] / metric[1]


def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)


def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= param.grad * lr
            param.grad.zero_()


def train_single_epoch(net, train_iter, loss, updater, lr):
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        sgd([W, b], lr)
        metric.add(float(l.detach().item()), accuracy(y, y_hat), 1)
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, updater, lr, num_epochs):
    for epoch in range(num_epochs):
        train_metrics = train_single_epoch(net, train_iter, loss, updater, lr)
        test_acc = evaluate_accuracy(net, test_iter)
        train_loss, train_acc = train_metrics
        print(f"Loss in {epoch+1} round is {train_loss}")
        print(f"Train Accuracy in {epoch+1} round is {train_acc}")
        print(f"Test Accuracy in {epoch+1} round is {test_acc}")
