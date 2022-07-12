import os, sys
import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.fftpack import fft, rfft, fftfreq
from scipy.signal import hilbert, stft, periodogram, spectrogram, find_peaks, medfilt
import scipy.signal as signal
from scipy.signal.windows import blackman, hann, hamming, flattop
from scipy.signal import hilbert
from pyhht.emd import EMD
import pickle


# params = {'fs': 51200, 'win': 'blackman', 'startFreq':1000, 'endFreq': 3000}
def load_data():
    name = ['h', 'm', 's', '0.000001s', 'hori', 'verti']
    df = pd.DataFrame()
    path = './data/IEEE_phm_2022/Learning_set/Bearing1_1'
    dirs = os.listdir(path)
    for file in dirs:
        file = os.path.join(path, file)
        df = pd.concat([df, pd.read_csv(file, names=name)])

    # df = pd.read_csv('./data/IEEE_phm_2022/Learning_set/Bearing1_1/acc_00001.csv', names=name)

    hori = signal.savgol_filter(df.hori, 49, 3)
    with open('./data/transformed_data/hori_bearing_1_1.pickle', "wb") as f:
        pickle.dump(hori, f)


# load_data()


def create_marginal_spectrum(data):
    fs = 25600

    N = len(data)

    decomposer = EMD(data)
    imf = decomposer.decompose()

    imf_data = {}
    if_data = {}
    Marginal_Spectrum_data = {}
    n = 1
    for i in imf:
        analytic_signal = hilbert(i)
        amplitude_envelope = np.abs(analytic_signal)  # 幅值
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                   (2.0 * np.pi) * fs)
        imf_data["imf_%d" % n] = list(i)
        if_data["instantaneous_frequency_%d" % n] = list(instantaneous_frequency)

        # 计算边际谱
        instantaneous_frequency_int = instantaneous_frequency.astype(int)
        ssp = np.zeros(int(fs / 2))
        for j in range(ssp.shape[0]):
            ssp[j] = np.sum(amplitude_envelope[1:][instantaneous_frequency_int == j])
        Marginal_Spectrum_data["Marginal_Spectrum_%d" % n] = list(ssp)
        n += 1
    output_data_imf = pd.DataFrame(imf_data)
    output_data_if = pd.DataFrame(if_data)
    output_data_ms = pd.DataFrame(Marginal_Spectrum_data)
    df_new = output_data_ms.apply(lambda x: x.sum())
    df_new = pd.DataFrame(df_new).T
    return df_new


name = ['h', 'm', 's', '0.000001s', 'hori', 'verti']
path = './data/IEEE_phm_2022/train/Bearing1_1'
dirs = os.listdir(path)
data_list = []
for file in dirs:
    print(f'start transform {file}!')
    file_1 = os.path.join(path, file)
    df = pd.read_csv(file_1, names=name)
    hori = signal.savgol_filter(df.hori, 49, 3)
    hori_margin = create_marginal_spectrum(hori)
    count = 0
    ans = pd.DataFrame()
    for i in range(len(hori_margin)):
        count += 1
        i += 1
        if count % 10 == 0:
            data = hori_margin.iloc[i - 10:i, :]
            column = data.columns.values
            columns = []
            for j in column:
                columns.append(j)
            data_mean = data[columns].mean()
            ans = ans.append(data_mean, ignore_index=True)
        else:
            continue
    hori_margin = ans.T.to_numpy()
    data_list.append(hori_margin)
    # print(data_list)

with open('./data/transformed_data/data_list_1_1.pickle', "wb") as f:
    pickle.dump(data_list, f)