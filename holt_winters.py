from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

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
                last_rmse=last_years_rmse(y,results)
                if(last_rmse<best_rmse):
                    best_rmse=last_rmse
                    best_results=results
            except:
                continue
    return best_results

###############################################################################

def last_years_rmse(y,results):
    y_forecasted = results.predict(len(y)-11, len(y))
    y_truth = y[-12:]

    mse = ((y_forecasted - y_truth) ** 2).mean()**0.5
    return mse

###############################################################################

def next_month(results):
    return results.forecast(1)[0]
