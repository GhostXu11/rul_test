
import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.fftpack import fft, rfft, fftfreq
from scipy.signal import hilbert, stft, periodogram, spectrogram, find_peaks, medfilt
import scipy.signal as signal
from scipy.signal.windows import blackman, hann, hamming, flattop
from scipy.signal import hilbert
from pyhht.emd import EMD

# params = {'fs': 51200, 'win': 'blackman', 'startFreq':1000, 'endFreq': 3000}

name = ['h', 'm', 's', '0.000001s', 'hori', 'verti']

df = pd.read_csv('./data/IEEE_phm_2022/Learning_set/Bearing1_1/acc_00001.csv', names=name)
hori = signal.savgol_filter(df.hori, 49, 3)
fs = 51200

N = len(hori)

decomposer = EMD(hori)
imf = decomposer.decompose()

n = 1
imf_data = {}
if_data = {}
Marginal_Spectrum_data = {}
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
    ssp = np.zeros(int(51200 / 2))
    for j in range(ssp.shape[0]):
        ssp[j] = np.sum(amplitude_envelope[1:][instantaneous_frequency_int == j])
    Marginal_Spectrum_data["Marginal_Spectrum_%d" % n] = list(ssp)
    n += 1
output_data_imf = pd.DataFrame(imf_data)
output_data_if = pd.DataFrame(if_data)
output_data_ms = pd.DataFrame(Marginal_Spectrum_data)
print(output_data_imf, output_data_if, output_data_ms)
output_data_ms.to_csv('ms.csv')
