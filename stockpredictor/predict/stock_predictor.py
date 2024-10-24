# predictor/stock_predictor.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Step 1: Fetch Stock Data
def get_stock_data(stock_symbol, period='1y'):
    stock_data = yf.download(stock_symbol, period=period)
    stock_data.reset_index(inplace=True)  # Reset index to make the date a column
    return stock_data

# Step 2: Preprocess Data
def preprocess_data(stock_data):
    data = stock_data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Step 3: Create LSTM Model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(scaled_data, train_size=0.8):
    train_len = int(len(scaled_data) * train_size)
    train_data = scaled_data[:train_len]
    
    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    return model

# Step 4: Predict Future Stock Prices
def predict_future(stock_data, model, scaler, days_to_predict=30):
    last_60_days = stock_data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)

    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = []
    for _ in range(days_to_predict):
        predicted_price = model.predict(X_test)
        predictions.append(predicted_price[0, 0])
        last_60_days_scaled = np.append(last_60_days_scaled[1:], predicted_price)
        X_test = np.reshape(last_60_days_scaled, (1, last_60_days_scaled.shape[0], 1))

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_prices

# Step 5: Plot the Predictions
def plot_predictions(stock_data, predicted_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'], label='Actual Stock Price')

    future_dates = pd.date_range(stock_data['Date'].iloc[-1], periods=len(predicted_prices) + 1, freq='B')[1:]
    plt.plot(future_dates, predicted_prices, label='Predicted Stock Price', linestyle='--')

    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
