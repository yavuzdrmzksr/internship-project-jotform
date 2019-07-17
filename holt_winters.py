from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

###############################################################################

def fit_model(y):
    pos_y=y
    for j in range(len(pos_y)):
        if(pos_y[j]!=0):
            pos_y=pos_y[j:]
            break

    ts=("add","mul")
    best_rmse=float("INF")
    best_results=None
    for tre in ts:
        for sea in ts:
            try:
                if(tre!="mul" and sea!="mul"):
                    model = ExponentialSmoothing(y.astype(np.float),trend=tre,seasonal=sea,seasonal_periods=12)
                    results = model.fit()
                else:
                    model = ExponentialSmoothing(pos_y.astype(np.float),trend=tre,seasonal=sea,seasonal_periods=12)
                    results = model.fit()
                y_forecasted = results.predict(len(y)-11, len(y))
                y_truth = y[-12:]
                last_rmse = ((y_forecasted - y_truth) ** 2).mean()**0.5

                if(last_rmse<best_rmse):
                    best_rmse=last_rmse
                    best_results=results
            except:
                continue
    return best_results

###############################################################################

def last_years_rmse(y,results):
    #results = fit_model(y[:-12])
    y_forecasted = results.forecast(12)
    y_truth = y[-12:]

    mse = ((y_forecasted - y_truth) ** 2).mean()**0.5
    x=sorted(abs((y_forecasted - y_truth)))
    mean = abs((y_forecasted - y_truth)).mean()
    std = abs((y_forecasted - y_truth)).std()
    return (mse,mean,std,(x[-3])*2.5-(x[2])*1.5)

###############################################################################

def next_month(results):
    return results.forecast(1)[0]

###############################################################################

def plot_last_year(y,results,title,filename,t,x_label="Months",y_label="Value"):
    #results = fit_model(y[:-12])
    plt.figure(figsize=(12.8,4.8))
    plt.plot(t, results.forecast(12), label='Predictions')
    plt.plot(t, y[-12:], label='Observations')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)

    plt.legend()

    plt.savefig(filename)

###############################################################################

def next_6_months(results):
    return results.forecast(6)

###############################################################################

def plot_next_6_months(y,results,title,filename,t1,t2,x_label="Months",y_label="Value"):
    plt.figure(figsize=(16.0,4.8))
    plt.plot(t2, results.predict(len(y)-12, len(y)+5), label='Predictions')
    plt.plot(t1, y[-12:], label='Observations')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)

    plt.legend()

    plt.savefig(filename)
