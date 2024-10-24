import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import time

# Initialize Alpha Vantage API
api_key = 'RNZPXZ6Q9FEFMEHM'
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch intraday data for Microsoft (MSFT)
data, meta_data = ts.get_intraday(symbol='MSFT', interval='1min', outputsize='full')

# Display the data (optional)
print(data)

# Extract closing prices and calculate percentage change
close_data = data['4. close']
percentage_change = close_data.pct_change()

# Print the percentage change
print(percentage_change)

# Check for significant last change
last_change = percentage_change[-1]
if abs(last_change) > 0.0004:
    print("MSFT Alert:" + str(last_change))

# Plotting the closing prices
plt.figure(figsize=(12, 6))
plt.plot(close_data.index, close_data, label='MSFT Closing Prices', color='blue')
plt.title('MSFT Intraday Closing Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()
