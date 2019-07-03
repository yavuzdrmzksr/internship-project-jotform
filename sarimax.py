import itertools
import statsmodels.api as sm

###############################################################################

def fit_model(y,pdq_limit=2):
    # Define the p, d and q parameters to take any value between 0 and pdq_limit
    p = d = q = range(0, pdq_limit)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    # Find best model using AIC value
    best_aic=float("INF")
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit(disp=0)

                if(results.aic<best_aic):
                    best_aic=results.aic
                    best_param=param
                    best_param_seasonal=param_seasonal
            except:
                continue

    mod = sm.tsa.statespace.SARIMAX(y,
                                order=best_param,
                                seasonal_order=best_param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit(disp=0)
    return results

###############################################################################

def last_years_mse(y,results):
    pred=results.get_prediction(start=y.size-12,dynamic=False)

    y_forecasted = pred.predicted_mean
    y_truth = y[-12:]

    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()
    return mse

###############################################################################

def next_month(results):
    pred=results.get_forecast(steps=1)
    return pred.predicted_mean[0]
