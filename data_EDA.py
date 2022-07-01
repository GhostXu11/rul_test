import os
import re
import seaborn as sns
from numpy.fft import *
import tftb.processing
import pandas as pd
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt

idx = ''
folder = 'Bearing' + idx
name = ['h', 'm', 's', '0.000001s', 'hori', 'verti']
folder_path = './data/IEEE_phm_2022/Learning_set'
path = folder_path + '/' + str(folder)

# file_dir = []
# for file in os.listdir(path):
#     if re.match('acc.*', file) != None:
#         file_dir.append(file)

df = pd.read_csv('./data/IEEE_phm_2022/Learning_set/Bearing1_1/acc_00001.csv', names=name)
hori = signal.savgol_filter(df.hori,49,3)
plt.plot(hori)
plt.show()