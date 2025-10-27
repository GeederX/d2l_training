import hashlib
import os
import tarfile
import zipfile
import requests


import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


DATA_HUB = dict()
DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"


def download(name, cache_dir=os.path.join(".", "data")):
    assert name in DATA_HUB, f"{name} is not existed in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f"Downloading {fname} from {url}")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname


DATA_HUB["kaggle_house_train"] = (
    DATA_URL + "kaggle_house_pred_train.csv",
    "585e9cc93e70b39160e7921475f9bcd7d31219ce",
)

DATA_HUB["kaggle_house_test"] = (
    DATA_URL + "kaggle_house_pred_test.csv",
    "fa19780a7b011d9b009e8bff8e99922a8ee2eb90",
)


train_data = pd.read_csv(download("kaggle_house_train"))
test_data = pd.read_csv(download("kaggle_house_test"))


all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))


numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: ((x - x.mean()) / x.std())
)
all_features[numeric_features] = all_features[numeric_features].fillna(0)


all_features = pd.get_dummies(all_features, dummy_na=True)


n_train = train_data.shape[0]
train_features = torch.tensor(
    all_features[:n_train].astype("float32").values, dtype=torch.float32
)
test_features = torch.tensor(
    all_features[n_train:].astype("float32").values, dtype=torch.float32
)
train_labels = torch.tensor(
    train_data["SalePrice"].astype("float32").values, dtype=torch.float32
)


num_inputs = train_features.shape[-1]


net = nn.Sequential(nn.Linear(num_inputs, 1))


loss = nn.MSELoss()
updater = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)


def train(
    net,
    train_features,
    train_labels,
    test_features,
    test_labels,
    num_epochs,
    batch_size,
):
    pass


d2l.load_array
