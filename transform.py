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

model = torch.load('./out/model.pth', map_location=torch.device('cpu'))
model.eval()

f = open('./data/transformed_data/data_list_1_1.pickle', 'rb')
result = pickle.load(f)
noise_result = []
result_list = []
for i in range(len(result)):
    noise_result.append(torch.from_numpy(add_noise(result[i])).float())
for i in range(len(noise_result)):
    input = noise_result[i]
    hidden_representation, recons_x = model(input)
    result_list.append(hidden_representation.detach().numpy())
with open('./data/transformed_data/transformed_data_list_1_1.pickle', "wb") as f:
    pickle.dump(result_list, f)


