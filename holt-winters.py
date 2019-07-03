from statsmodels.tsa.holtwinters import ExponentialSmoothing

###############################################################################

def fit_model(y):
    pos_y=y
    for j in range(len(pos_y)):
        if(pos_y[j]!=0):
            pos_y=pos_y[j:]
            break

    ts=("add","mul",None)
    best_mse=float("INF")
    best_results=None
    for tre in ts:
        for sea in ts:
            try:
                if(tre=="mul" and sea==None):
                    model = ExponentialSmoothing(pos_y.astype(np.float),trend=tre,seasonal=sea)
                    results = model.fit()
                elif(sea==None):
                    model = ExponentialSmoothing(y.astype(np.float),trend=tre,seasonal=sea)
                    results = model.fit()

                elif(tre!="mul" and sea!="mul"):
                    model = ExponentialSmoothing(y.astype(np.float),trend=tre,seasonal=sea,seasonal_periods=12)
                    results = model.fit()
                else:
                    model = ExponentialSmoothing(pos_y.astype(np.float),trend=tre,seasonal=sea,seasonal_periods=12)
                    results = model.fit()
                last_mse=last_years_mse(y,results)
                if(last_mse<best_mse):
                    best_mse=last_mse
                    best_results=results
            except:
                continue
    return best_results

###############################################################################

def last_years_mse(y,results):
    y_forecasted = results.predict(len(y)-12, len(y))
    y_truth = y[-12:]

    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()
    return mse

###############################################################################

def next_month(results):
    return results.forecast(1)
