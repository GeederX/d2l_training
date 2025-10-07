import torch
import torchvision
import os
from torch.utils import data
from torchvision import transforms


batch_size = 256
num_inputs = 784
num_outputs = 10


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
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def cross_entropy(y,y_hat,batch_size):
    return -torch.log(y_hat[range(len(y_hat)),y])


