# Time Series Predictions of Stock Prices with Recurrent Neural Network (RNN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training dataset
dataset_train = pd.read_csv('stock-price-data-train.csv')

# Importing the test dataset
dataset_test = pd.read_csv('stock-price-data-test.csv')

training_set = dataset_train.iloc[:, 1].values
training_set = training_set.reshape(-1, 1)

test_set_real = dataset_test.iloc[:, 1].values
test_set_real = test_set_real.reshape(-1, 1)


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 120 timesteps and 1 output
X_train = []
y_train = []
for i in range(120, 11954):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# calculate root mean squared error of the training set
import math
trainPredict = regressor.predict(X_train)
y_train = y_train.reshape(-1, 1)
trainScore = math.sqrt(np.mean((y_train - trainPredict)**2))
print('RMSE of training set:', trainScore)

# Making the predictions for testset and visualising the results

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(120, 240):
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# calculate root mean squared error of the test set
import math
testScore = math.sqrt(np.mean((test_set_real - predicted_stock_price)**2))
print('RMSE of test set:', testScore)


# Visualising the results

plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Stock Price for Test set')
plt.plot(test_set_real, color = 'blue', label = 'Real Stock Price for Test set')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

trainPredict = sc.inverse_transform(trainPredict)
plt.plot(trainPredict, color = 'red', label = 'Predicted Stock Price for Training set')
plt.plot(training_set, color = 'blue', label = 'Real Stock Price for Training set')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

