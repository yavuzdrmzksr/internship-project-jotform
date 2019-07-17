import scipy as sc
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
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

def last_years_rmse(y,results):
    #results=fit_model(y[:-12],n_predict=12)
    y_forecasted = results[-12:]

    y_truth = y[-12:]

    mse = ((y_forecasted - y_truth) ** 2).mean()**0.5
    x=sorted(abs((y_forecasted - y_truth)))
    mean = abs((y_forecasted - y_truth)).mean()
    std = abs((y_forecasted - y_truth)).std()
    return (mse,mean,std,(x[-3])*2.5-(x[2])*1.5)

###############################################################################

def next_month(pred):
    #pred=fit_model(y[:-1])
    return pred[-1]

###############################################################################

def plot_last_year(y,results,title,filename,t,x_label="Months",y_label="Value"):
    #results=fit_model(y[:-12],n_predict=12)
    plt.figure(figsize=(12.8,4.8))
    plt.plot(t, results[-12:], label='Predictions')
    plt.plot(t, y[-12:], label='Observations')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)

    plt.legend()

    plt.savefig(filename)

###############################################################################

def next_6_months(pred):
    return pred[-6:]

###############################################################################

def plot_next_6_months(y,pred,title,filename,t1,t2,x_label="Months",y_label="Value"):
    #pred=fit_model(y,6)
    plt.figure(figsize=(16.0,4.8))
    plt.plot(t2, pred[-18:], label='Predictions')
    plt.plot(t1, y[-12:], label='Observations')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)

    plt.legend()

    plt.savefig(filename)
