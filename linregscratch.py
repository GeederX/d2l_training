import torch
import random
from d2l import torch as d2l


batch_size = 10
lr = 0.001
num_epochs = 1000


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def sample_selector(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        num_sample = min(batch_size, num_examples - i)
        yield features[batch_indices], labels[batch_indices], num_sample


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y, batch_size):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2).sum() / 2 / batch_size


def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


true_w = torch.tensor([2, -3.4])
true_b = 4.2
num_examples = 1000
features, labels = synthetic_data(true_w, true_b, num_examples)


w = torch.normal(0, 0.01, [2, 1], requires_grad=True)
b = torch.normal(0, 0.01, [1], requires_grad=True)


for epoch in range(num_epochs):
    for X, y, num_samples in sample_selector(batch_size, features, labels):
        loss = squared_loss(linreg(X, w, b), y, num_samples)
        loss.backward()
        sgd([w, b], lr)
    with torch.no_grad():
        loss_after_train = squared_loss(linreg(features, w, b), labels, num_examples)
        print(f"epoch {epoch + 1}, loss {float(loss_after_train):f}")
