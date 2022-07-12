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
import pickle
from torch.utils.data import TensorDataset, DataLoader

from model.CDAE import add_noise, CAE

# load data and add noise
# original_data = pd.read_csv('data/transformed_data/ms.csv', index_col=0)
# original_data = add_noise(original_data)
# original_data = torch.from_numpy(np.array(original_data)).float()
#
# train_dataset = TensorDataset(original_data)
# trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#
model = CAE()
mse_loss = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# original_data = np.array(original_data)


# def loss_function(W, x, recons_x, h, lam):
#     mse = mse_loss(recons_x, x)
#     # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
#     # opposed to #1
#     dh = h * (1 - h)
#     # Hadamard product produces size N_batch x N_hidden
#     # Sum through the input dimension to improve efficiency, as suggested in #1
#     w_sum = torch.sum(Variable(W) ** 2, dim=1)
#     # unsqueeze to avoid issues with torch.mv
#     w_sum = w_sum.unsqueeze(1)  # shape N_hidden x 1
#     contractive_loss = torch.sum(torch.mm(dh ** 2, w_sum), 0)
#     return mse + contractive_loss.mul_(lam)

def loss_function(outputs_e, outputs, imgs, labels, lamda=1e-4, device=torch.device('cuda:1')):
    criterion = nn.MSELoss()
    assert outputs.shape == imgs.shape, f'outputs.shape : {outputs.shape} != imgs.shape : {imgs.shape}'
    loss1 = criterion(outputs, labels)

    outputs_e.backward(torch.ones(outputs_e.size()).to(device), retain_graph=True)

    # Frobenious norm, the square root of sum of all elements (square value)
    # in a jacobian matrix
    loss2 = torch.sqrt(torch.sum(torch.pow(imgs.grad, 2)))
    imgs.grad.data.zero_()
    loss = loss1 + (lamda * loss2)
    return loss


if __name__ == "__main__":
    batch_size = 32
    epochs = 300
    lam = 1e-4
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    f = open('./data/transformed_data/data_list_1_1.pickle', 'rb')
    result = pickle.load(f)
    noise_result = []
    for i in range(len(result)):
        noise_result.append(add_noise(result[i]))
    with open('./data/transformed_data/noise_data_list_1_1.pickle', "wb") as f:
        pickle.dump(noise_result, f)

    original_list = []
    for i in range(len(result)):
        original_data = torch.from_numpy(np.array(result[i])).float()
        original_list.append(original_data)
    train_tensor = torch.cat(original_list)
    noise_list = []
    for i in range(len(noise_result)):
        noise_data = torch.from_numpy(np.array(noise_result[i])).float()
        noise_list.append(noise_data)
    noise_tensor = torch.cat(noise_list)
    train_dataset = TensorDataset(train_tensor, noise_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # train
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for idx, [label, feature] in enumerate(train_loader):
            feature = feature.to(device)
            label = label.to(device)

            feature.requires_grad_(True)
            feature.retain_grad()
            # label.requires_grad_(True)
            # label.retain_grad()
            hidden_representation, recons_x = model(feature)

            loss = loss_function(hidden_representation, recons_x, feature, label, lam, device)

            feature.requires_grad_(False)
            # label.requires_grad_(False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.mean()
        print(f'train loss for epoch{epoch}: {train_loss / len(train_loader)}')
    torch.save(model, './out/model.pth')
