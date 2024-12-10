from sklearn.discriminant_analysis import StandardScaler
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Fetch data for the past 3 months
data = yf.Ticker("NDAQ").history(period="6mo", interval="1d")
df = pd.DataFrame(data)

# Scale data
features = df[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

def create_sequences_with_predict_length(data, sequence_length, predict_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - predict_length + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + predict_length])
    return np.array(X), np.array(y)

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, 30)

# Split Train and Test
train_size = int(len(X) * 0.8)
predict_size = int(len(X) * 0.5)
X_train, X_test, X_prediction  = X[:train_size], X[train_size:], X[predict_size:]
y_train, y_test, y_prediction = y[:train_size], y[train_size:], y[predict_size:]

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
X_prediction = X_prediction.reshape((X_prediction.shape[0], X_prediction.shape[1], 1))

# print(f"Total: {len(features)}: X: {len(X)}, y: {len(y)}, X_train: {len(X_train)}, y_train: {len(y_train)}, X_test: {len(X_test)}, y_test: {len(y_test)}")
# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

model = Sequential()
model.add(LSTM(units=256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1)) 
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse'])
# model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=0)

test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

# plot history
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['accuracy'], label='train')
# plt.legend()
# plt.show()

# Forecast
predicted = model.predict(X_prediction)
predicted = scaler.inverse_transform(predicted)
y_train_actual = scaler.inverse_transform(y_train)
y_prediction_actual = scaler.inverse_transform(y_prediction)

last_sequence = X_prediction[-1].flatten()
future_predictions = []
input_sequence = last_sequence.copy()

for _ in range(5):  # Predict 5 days ahead
    input_sequence_reshaped = input_sequence.reshape((1, input_sequence.shape[0], 1))  # Reshape input
    next_prediction = model.predict(input_sequence_reshaped, verbose=0)  # Predict next value
    future_predictions.append(next_prediction[0, 0])  # Store prediction
    input_sequence = np.append(input_sequence[1:], next_prediction)  # Update sequence

# Inverse transform the 5-day forecast
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Combine predictions for plotting
full_predictions = np.concatenate((predicted, future_predictions), axis=0)

# Plot the results
plt.figure(figsize=(14, 7))

train_range = range(0, train_size)
predict_range = range(predict_size, len(y))
forecast_range = range(len(y) - 1, len(y) - 1 + len(future_predictions))

plt.plot(train_range, y_train_actual, label='Actual Train Data',linewidth=2, color='dimgray', alpha=0.6)
plt.plot(predict_range, y_prediction_actual, label='Actual Predict Data', linewidth=1.8, color='sienna', alpha=0.6)
plt.plot(predict_range, predicted, label='Predictions for X_prediction', color='red')
plt.plot(forecast_range, future_predictions, label='Next 5-Day Forecast', color='steelblue')

plt.title('Predictions and 5-Day Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()
