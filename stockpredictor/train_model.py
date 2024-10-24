import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Function to fetch and preprocess stock data
def fetch_and_preprocess_data(stock_symbol):
    # Fetch historical data
    stock_data = yf.download(stock_symbol, start="2020-01-01", end="2024-09-18", interval="1d")
    
    # Convert columns to lowercase to avoid case sensitivity issues
    stock_data.columns = stock_data.columns.str.lower()

    # Feature columns
    X = stock_data[['open', 'high', 'low', 'volume']].dropna()

    # Target column: Try 'close' or 'adj close'
    if 'close' in stock_data.columns:
        y = stock_data['close'].dropna()  # Predicting the 'close' price
    elif 'adj close' in stock_data.columns:
        y = stock_data['adj close'].dropna()  # Fallback to 'adj close' if 'close' is unavailable
    else:
        raise ValueError("Neither 'close' nor 'adj close' columns found in the data")

    return X, y

# Function to train the model
def train_model():
    stock_symbol = 'AAPL'  # Example stock symbol
    X, y = fetch_and_preprocess_data(stock_symbol)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Save the trained model to a file
    joblib.dump(model, 'predict/models/stock_model.pkl')
    print("Model saved to 'predict/models/stock_model.pkl'")

    return model, X

# Function to predict the next 5 days' stock prices
def predict_next_days(model, X):
    # Get the last available data point
    latest_features = X.iloc[-1].values.reshape(1, -1)

    # Predict for the next 5 days (assuming similar patterns for simplicity)
    future_predictions = []
    for _ in range(5):
        next_prediction = model.predict(latest_features)[0]
        future_predictions.append(next_prediction)

        # Update features for the next prediction (e.g., shifting the predicted 'close' as 'open')
        # You can refine this by predicting new features, but for simplicity, we update 'open' with the predicted close
        latest_features[0][0] = next_prediction  # Update the 'open' value

    return future_predictions

if __name__ == "__main__":
    # Train the model
    model, X = train_model()

    # Predict the next 5 days' prices
    future_prices = predict_next_days(model, X)
    print("Predicted stock prices for the next 5 days:", future_prices)

    # Predict and display the latest day's price
    latest_price = model.predict(X.iloc[[-1]])
    print(f"Predicted stock price for the latest day: {latest_price[0]}")
