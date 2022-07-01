import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Json, Int, Csv, Float

import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.fftpack import fft, rfft, fftfreq
from scipy.signal import hilbert, stft, periodogram, spectrogram, find_peaks, medfilt
import scipy.signal as signal
from scipy.signal import blackman, hann, hamming, flattop, hilbert
from pyhht.emd import EMD

# params = {'fs': 51200, 'win': 'blackman', 'startFreq':1000, 'endFreq': 3000}

@app.input(Csv(key="inputData1", alias="data"))
@app.param(Int(key="param1", alias="fs", default=51200)) # 最小距离
@app.output(Csv(key="outputData1"))
@app.output(Csv(key="outputData2"))
@app.output(Csv(key="outputData3"))
def HelloWorld(context):
    global params
    args = context.args
    data = args.data
    fs = args.fs
    print(data)
    print(fs)
    x = data['value']
    N = len(x)

    # fs = params

    decomposer = EMD(x)
    imf = decomposer.decompose()

    n=1
    imf_data = {}
    if_data = {}
    Marginal_Spectrum_data = {}
    for i in imf:
        analytic_signal = hilbert(i)
        amplitude_envelope = np.abs(analytic_signal) # 幅值
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                (2.0*np.pi) * fs)

        imf_data["imf_%d" % n] = list(i)
        if_data["instantaneous_frequency_%d" % n] = list(instantaneous_frequency)


        # 计算边际谱
        instantaneous_frequency_int = instantaneous_frequency.astype(int)
        ssp = np.zeros(int(51200/2))
        for i in range(ssp.shape[0]):
            ssp[i] = np.sum(amplitude_envelope[1:][instantaneous_frequency_int==i])
        Marginal_Spectrum_data["Marginal_Spectrum_%d" % n] = list(ssp)

        n+=1
    output_data_imf = pd.DataFrame(imf_data)
    output_data_if = pd.DataFrame(if_data)
    output_data_ms = pd.DataFrame(Marginal_Spectrum_data)
    print(output_data_imf, output_data_if, output_data_ms)

    return output_data_imf, output_data_if, output_data_ms

if __name__ == "__main__":
    suanpan.run(app)
