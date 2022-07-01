import os
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# import pdb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from model.CDAE import add_noise, CAE, loss_function

batch_size = 32

# load data and add noise
original_data = pd.read_csv('data/transformed_data/ms.csv', index_col=0)
original_data = add_noise(original_data)
original_data = torch.from_numpy(np.array(original_data)).float()
train_dataset = TensorDataset(original_data)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

model = CAE()
mse_loss = nn.BCELoss(reduction=False)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

original_data = np.array(original_data)


def loss_function(W, x, recons_x, h, lam):
    mse = mse_loss(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h)
    # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W) ** 2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1)  # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh ** 2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


model = CAE()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train(epoch):
    log_interval = 10
    lam = 1e-4
    model.train()
    train_loss = 0

    for idx, data in enumerate(trainloader):
        data = Variable(data)

        optimizer.zero_grad()

        hidden_representation, recons_x = model(data)

        # Get the weights
        # model.state_dict().keys()
        # change the key by seeing the keys manually.
        # (In future I will try to make it automatic)
        W = model.state_dict()['fc1.weight']
        loss = loss_function(W, data.view(-1, 784), recons_x,
                             hidden_representation, lam)

        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if idx % log_interval == 0:
            print('Train epoch: {} [{}/{}({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, idx * len(data), len(trainloader.dataset),
                       100 * idx / len(trainloader),
                       loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(trainloader.dataset)))
    model.samples_write(data, epoch)
