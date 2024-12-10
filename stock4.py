from sklearn.discriminant_analysis import StandardScaler
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

data = pd.read_csv('./stock/features_sentiment.csv', parse_dates=['Date'])
data = data.sort_values(by='Date')

# scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[['Close', 'score']])
scaled_data = pd.DataFrame(scaled_features, columns=['Close_scaled', 'score_scaled'])
scaled_data['label'] = data['label'].values
scaled_data['Date'] = data['Date'].values 


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

data_array = scaled_data[['Close_scaled', 'score_scaled', 'label']].values
X, y = create_sequences(data_array, 30)

# Split Train and Test
train_size = int(len(X) * 0.8)
predict_size = int(len(X) * 0.5)
X_train, X_test, X_prediction  = X[:train_size], X[train_size:], X[predict_size:]
y_train, y_test, y_prediction = y[:train_size], y[train_size:], y[predict_size:]

# # Reshape input to be [samples, time steps, features]
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
# X_prediction = X_prediction.reshape((X_prediction.shape[0], X_prediction.shape[1], 1))

# print(f"Total: {len(data)}: X: {len(X)}, y: {len(y)}, X_train: {len(X_train)}, y_train: {len(y_train)}, X_test: {len(X_test)}, y_test: {len(y_test)}")
# print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

model = Sequential()
model.add(LSTM(units=256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3)) 
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse'])
# model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=0)

test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

predicted_features = model.predict(X_prediction)
predicted_df = pd.DataFrame(predicted_features, columns=['Close_scaled', 'score_scaled', 'label'])
predicted_df[['Close', 'score']] = scaler.inverse_transform(predicted_df[['Close_scaled', 'score_scaled']])
predicted_df['label'] = y_prediction[:, 2] 
predicted_df['Date'] = data['Date'].values[-len(predicted_df):]

y_prediction_actual = scaler.inverse_transform(y_prediction[:, :2]) 
y_prediction_df = pd.DataFrame(y_prediction_actual, columns=['Close', 'score'])
y_prediction_df['label'] = y_prediction[:, 2] 
y_prediction_df['Date'] = data['Date'].values[-len(y_prediction_df):] 

# Number of days to predict
num_days = 5

# Initialize the list to store future predictions
future_predictions = []

# Use the last sequence from X_prediction
last_sequence = X_prediction[-1]

for i in range(num_days):
    input_sequence_reshaped = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    next_prediction = model.predict(input_sequence_reshaped, verbose=0)
    next_prediction_df = pd.DataFrame(next_prediction, columns=['Close_scaled', 'score_scaled', 'label'])
    next_prediction_df[['Close', 'score']] = scaler.inverse_transform(next_prediction_df[['Close_scaled', 'score_scaled']])
    if i == 0:
        last_date = pd.to_datetime(predicted_df['Date'].values[-1])
    else:
        last_date = future_predictions[-1]['Date']
    next_date = last_date + pd.Timedelta(days=1)
    next_prediction_df['Date'] = next_date
    future_predictions.append(next_prediction_df[['Date', 'Close', 'score', 'label']].iloc[0])
    last_sequence = np.append(last_sequence[1:], next_prediction, axis=0)
future_predictions_df = pd.DataFrame(future_predictions)

plt.figure(figsize=(14, 7))
plt.plot(predicted_df['Date'], predicted_df['Close'], label='Predicted Close Prices', color='blue')
plt.plot(y_prediction_df['Date'], y_prediction_df['Close'], label='Actual Close Prices (Test)', color='red')
plt.plot(future_predictions_df['Date'], future_predictions_df['Close'], label='Next Day Predicted Close Price', color='orange',  marker='o')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()