from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import warnings

warnings.filterwarnings('ignore')

###############################################################################

# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

###############################################################################

def fit_model(raw_seq,n_steps=12):
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    return model

###############################################################################

def last_years_rmse(raw_seq,model,n_steps=12):
	# model=fit_model(raw_seq[:-12],n_steps)
	n_features = 1
	y_forecasted = []
	for i in range(-24,-12):
		x_input = raw_seq[i:i+n_steps]
		x_input = x_input.reshape((1, n_steps, n_features))
		y_forecasted.append(model.predict(x_input, verbose=0)[0][0])

	y_truth=raw_seq[-12:]

	mse = ((y_forecasted - y_truth) ** 2).mean()**0.5
	x=sorted(abs((y_forecasted - y_truth)))
	mean = abs((y_forecasted - y_truth)).mean()
	std = abs((y_forecasted - y_truth)).std()
	return (mse,mean,std,(x[-3])*2.5-(x[2])*1.5)

###############################################################################

def next_month(raw_seq,model,n_steps=12):
    n_features = 1
    x_input = raw_seq[-n_steps:]
    x_input = x_input.reshape((1, n_steps, n_features))
    return model.predict(x_input, verbose=0)[0][0]

###############################################################################

def plot_last_year(raw_seq,model,title,filename,t,x_label="Months",y_label="Value",n_steps=12):
	# model=fit_model(raw_seq[:-12],n_steps)
	n_features = 1

	n_features = 1
	y_forecasted = []
	for i in range(-24,-12):
		x_input = raw_seq[i:i+n_steps]
		x_input = x_input.reshape((1, n_steps, n_features))
		y_forecasted.append(model.predict(x_input, verbose=0)[0][0])
	plt.figure(figsize=(12.8,4.8))
	plt.plot(t, y_forecasted, label='Predictions')
	plt.plot(t, raw_seq[-12:], label='Observations')

	plt.xlabel(x_label)
	plt.ylabel(y_label)

	plt.title(title)

	plt.legend()

	plt.savefig(filename)

###############################################################################

def next_6_months(raw_seq,model,n_steps=12):
	n_features = 1
	x_input = raw_seq[-n_steps:]
	x_input = x_input.reshape((1, n_steps, n_features))
	result = model.predict(x_input, verbose=0)[0]
	for i in range(1,6):
		x_input = np.append(raw_seq[-n_steps+i:],result)
		x_input = x_input.reshape((1, n_steps, n_features))
		result = np.append(result,model.predict(x_input, verbose=0)[0])
	return result

###############################################################################

def plot_next_6_months(raw_seq,model,title,filename,t1,t2,x_label="Months",y_label="Value",n_steps=12):
	n_features = 1

	y_forecasted = []
	for i in range(-24,-12):
		x_input = raw_seq[i:i+n_steps]
		x_input = x_input.reshape((1, n_steps, n_features))
		y_forecasted.append(model.predict(x_input, verbose=0)[0][0])

	y_forecasted.extend(next_6_months(raw_seq,model))
	plt.figure(figsize=(16.0,4.8))
	plt.plot(t2, y_forecasted, label='Predictions')
	plt.plot(t1, raw_seq[-12:], label='Observations')

	plt.xlabel(x_label)
	plt.ylabel(y_label)

	plt.title(title)

	plt.legend()

	plt.savefig(filename)
