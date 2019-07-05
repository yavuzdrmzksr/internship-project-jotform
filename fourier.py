import scipy as sc
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

###############################################################################

def fit_model(y, n_predict=1):
    n = y.size
    n_harm = 5                         # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, y, 1)            # find linear trend in x
    y_notrend = y - p[0] * t           # detrended x
    y_freqdom = np.fft.fft(y_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))

    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(y_freqdom[i]) / n   # amplitude
        phase = np.angle(y_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

###############################################################################

def last_years_rmse(y):
    y_forecasted = fit_model(y[:-12],n_predict=12)[-12:]

    y_truth = y[-12:]

    mse = ((y_forecasted - y_truth) ** 2).mean()**0.5
    return mse

###############################################################################

def next_month(pred):
    return pred[-1]

###############################################################################

def plot_last_year(y,title,filename,x_label="Months",y_label="Value"):
    t = np.arange(12)
    plt.plot(t, fit_model(y[:-12],n_predict=12)[-12:], label='Predictions')
    plt.plot(t, y[-12:], label='Observations')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)

    plt.legend()

    plt.savefig(filename)

###############################################################################

def plot_next_month(y,pred,title,filename,x_label="Months",y_label="Value"):
    t1 = np.arange(12)
    t2 = np.arange(13)
    plt.plot(t2, pred[-13:], label='Predictions')
    plt.plot(t1, y[-12:], label='Observations')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)

    plt.legend()

    plt.savefig(filename)
