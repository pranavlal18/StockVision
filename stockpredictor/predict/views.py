
from django.contrib.auth.decorators import login_required
import requests
import pandas as pd
import pickle
from django.shortcuts import render, redirect
import os,io 
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import urllib, base64

from django.http import HttpResponse
from django.contrib.auth import login



def index(request):
    return render(request, 'predict/index.html')
def about(request):
    return render(request,'predict/about.html')
def service(request):
    return render(request,'predict/service.html')
def login(request):
    return render(request,'predict/login.html')
@login_required
def dashboard(request):
    return render(request, 'predict/dashboard.html')
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib import messages

def custom_login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        # Check if the user exists first
        try:
            user = User.objects.get(username=username)

            # Try authenticating the user with the provided password
            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)
                return redirect('dashboard')  # Redirect to the dashboard on successful login
            else:
                # Username exists but password is wrong
                messages.error(request, "Password not correct")
        except User.DoesNotExist:
            # If the username does not exist
            messages.error(request, "Username not found")
    
    return render(request, 'predict/login.html')




import yfinance as yf

def fetch_stock_data(stock_symbol):
    # Fetch the stock data from Yahoo Finance
    stock_data = yf.download(stock_symbol, period="1d", interval="1m")

    # Check if the data contains the necessary columns
    if all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Volume']):
        # Extract the latest row of data and only required columns
        latest_data = stock_data[['Open', 'High', 'Low', 'Volume']].tail(1)
        return latest_data
    else:
        print(f"Missing required columns in the stock data for {stock_symbol}")
        return None

# Example usage
stock_symbol = 'AAPL'  # You can change this to any stock symbol
stock_data = fetch_stock_data(stock_symbol)

if stock_data is not None:
    print(stock_data)
else:
    print("Stock data could not be fetched.")


def predict(request):
    if request.method == 'POST':
        stock_symbol = request.POST.get('stock_symbol').upper().strip()  # Ensure symbol is uppercase and stripped
        
        # Fetch stock data
        stock_data = fetch_stock_data(stock_symbol)
        
        if stock_data is None:
            return render(request, 'predict/predict.html', {'error': 'Stock data could not be retrieved. Please try again.'})
        
        # Print the actual column names for debugging purposes
        print("Stock Data Columns:", stock_data.columns)
        
        # Clean up column names: strip spaces, remove special characters, and lowercase
        stock_data.columns = stock_data.columns.str.strip().str.lower()
        
        # Print cleaned column names for debugging
        print("Cleaned Stock Data Columns:", stock_data.columns)
        
        # Check if the expected columns are present
        expected_columns = ['open', 'high', 'low', 'volume']
        missing_columns = [col for col in expected_columns if col not in stock_data.columns]
        
        if missing_columns:
            return render(request, 'predict/predict.html', {'error': f'Missing columns in data: {missing_columns}'})
        
        # Load trained model
        model_path = os.path.join('predict', 'models', 'stock_model.pkl')
        if not os.path.exists(model_path):
            return render(request, 'predict/predict.html', {'error': 'Prediction model not found.'})
        
        with open(model_path, 'rb') as file:
            stock_prediction_model = pickle.load(file)
        
        try:
            # Prepare data for prediction
            latest_data = stock_data[['open', 'high', 'low', 'volume']].tail(1)
            print(f"Data used for prediction: {latest_data}")
            
            # Make the prediction
            predicted_price = stock_prediction_model.predict(latest_data)[0]
        
        except KeyError as e:
            return render(request, 'predict/predict.html', {'error': f'Column error: {str(e)}'})
        except Exception as e:
            return render(request, 'predict/predict.html', {'error': 'Error in prediction: ' + str(e)})
        
        return render(request, 'predict/result.html', {'predicted_price': predicted_price, 'stock_symbol': stock_symbol})
    
    return render(request, 'predict/predict.html')


import io
import base64
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from datetime import timedelta

# Function to predict stock prices for the next 5 days
def predict_next_days(model, latest_features):
    future_predictions = []
    
    # Predict for the next 5 days
    for _ in range(5):
        next_prediction = model.predict(latest_features)[0]
        future_predictions.append(next_prediction)

        # Update the latest features for the next prediction (e.g., shift the 'open' value)
        latest_features[0][0] = next_prediction  # Use predicted 'close' price for 'open'
    
    return future_predictions

import numpy as np
import yfinance as yf
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import joblib
import io
import base64
from django.http import HttpResponse
from django.shortcuts import render

def result(request):
    if request.method == 'POST':
        stock_symbol = request.POST.get('stock_symbol')

        if not stock_symbol:
            return HttpResponse("Stock symbol is required.", status=400)

        try:
            # Load the pre-trained model
            model = joblib.load('predict/models/stock_model.pkl')

            # Fetch historical stock data from yfinance
            stock_data = yf.download(stock_symbol, start="2024-01-01", end="2024-10-20", interval="1d")

            if stock_data.empty:
                return HttpResponse("No data available for the given stock symbol.", status=404)

            # If stock_data has a MultiIndex, reset the index
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)  # Use first level if it’s MultiIndex
            
            # Now you can use .str accessor without issues
            stock_data.columns = stock_data.columns.str.strip().str.lower()

            # Preprocess stock data for prediction
            latest_data = stock_data[['open', 'high', 'low', 'volume']].tail(1)

            if latest_data.empty:
                return HttpResponse("No recent data available for prediction.", status=404)

            # Make the initial prediction
            initial_prediction = model.predict(latest_data)[0]

            # Prepare for next 5 business days prediction
            future_dates = []
            future_predictions = []

            prediction_start_date = datetime(2024, 10, 23)
            next_open = latest_data['open'].values[0]
            next_high = latest_data['high'].values[0]
            next_low = latest_data['low'].values[0]
            next_volume = latest_data['volume'].values[0]

            # Predict for the next 5 business days
            i = 0
            while len(future_dates) < 5:
                future_date = prediction_start_date + timedelta(days=i)
                if future_date.weekday() >= 5:
                    i += 1
                    continue

                future_dates.append(future_date.strftime('%Y-%m-%d'))

                # Prepare the input data for prediction
                next_data = np.array([[next_open, next_high, next_low, next_volume]])

                # Predict the stock price
                future_prediction = model.predict(next_data)[0]
                future_predictions.append(future_prediction)

                # Simulate the next day's open, high, low, and volume
                next_open = future_prediction * np.random.uniform(0.98, 1.02)
                next_high = next_open * np.random.uniform(1.00, 1.02)
                next_low = next_open * np.random.uniform(0.98, 1.00)
                next_volume = next_volume * np.random.uniform(0.95, 1.05)

                i += 1

            # Plotting the predictions
            plt.figure(figsize=(12, 6))
            plt.style.use('seaborn-darkgrid')
            plt.plot(future_dates, future_predictions, label='Predicted Price', color='red', linestyle='--', marker='o', markersize=6)
            plt.title(f'{stock_symbol} Stock Prices: Predicted Business Days from 23-10-2024', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(loc='upper left', fontsize=12)
            plt.grid(color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_facecolor('#f9f9f9')

            # Save plot to BytesIO object and encode as base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            # Pass predictions and plot image to the template
            zipped_predictions = zip(future_dates, future_predictions)
            context = {
                'stock_symbol': stock_symbol,
                'zipped_predictions': zipped_predictions,
                'plot_url': f"data:image/png;base64,{image_base64}"
            }
            return render(request, 'predict/result.html', context)

        except Exception as e:
            # Print the full exception to the logs for debugging
            print(f"An error occurred: {str(e)}")
            return HttpResponse(f"An error occurred: {str(e)}", status=500)


from django.shortcuts import render, redirect
from django.contrib.auth import login
from .forms import CustomUserCreationForm

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Automatically log the user in after registration
            return redirect('dashboard')  # Redirect to the dashboard after successful registration
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'predict/register.html', {'form': form})

import yfinance as yf
from django.shortcuts import render
from datetime import datetime

def get_stock_trend(ticker):
    stock = yf.Ticker(ticker)
    
    # Get historical data
    hist = stock.history(period='2d')  # Fetching data for the last 2 days

    # Ensure that there are at least 2 data points
    if hist.empty or len(hist['Close']) < 2:
        # Try fetching data for the last 5 days as a fallback
        hist = stock.history(period='5d')
        if hist.empty or len(hist['Close']) < 2:
            return None, 'no_data'  # Still no data, return None

    # Get the current price (latest day) and previous closing price
    current_price = hist['Close'][-1]  # Last available close price
    previous_close = hist['Close'][-2]  # Second last available close price

    if current_price > previous_close:
        trend = 'up'
    elif current_price < previous_close:
        trend = 'down'
    else:
        trend = 'neutral'
    
    return current_price, trend



from django.shortcuts import render
import yfinance as yf
from datetime import datetime
import random

def trending_news(request):
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Default tickers
    all_news = []

    # Check if a ticker has been submitted
    user_ticker = request.GET.get('ticker', '').upper()
    if user_ticker:
        tickers.append(user_ticker)  # Add user-provided ticker to the list

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        news = stock.news

        for item in news:
            publish_time = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
            stock_price, trend = get_stock_trend(ticker)

            if stock_price is None:
                stock_price = 'N/A'
                trend = 'no_data'

            all_news.append({
                'ticker': ticker,
                'title': item['title'],
                'description': item.get('summary', 'No description available'),
                'link': item['link'],
                'publisher': item['publisher'],
                'publish_time': publish_time,
                'stock_price': stock_price,
                'trend': trend,
            })

    random.shuffle(all_news)

    context = {
        'news': all_news,
    }
    return render(request, 'predict/trending_news.html', context)




# views.py
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

def stock_alert_page(request):
    """Render the stock alert page."""
    return render(request, 'predict/stock_alert.html')


# views.py
import requests
from django.http import JsonResponse
from django.conf import settings

def check_stock_alerts(request):
    """Check stock alerts for predefined tickers and send messages via Telegram."""
    
    tickers_to_monitor = ["TSLA", "AAPL", "AMZN", "NVDA", "GOOGL"]  # Add your tickers here
    significant_change_threshold = 1.0  # Change this to your desired threshold

    for STOCK_NAME in tickers_to_monitor:
        STOCK_ENDPOINT = "https://www.alphavantage.co/query"

        # Get yesterday's and day-before-yesterday's closing prices
        stock_params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": STOCK_NAME,
            "apikey": settings.STOCK_API_KEY,
        }

        response = requests.get(STOCK_ENDPOINT, params=stock_params)
        data = response.json().get("Time Series (Daily)")

        if not data:
            continue  # Skip to the next ticker if data is not available

        data_list = [value for (key, value) in data.items()]
        yesterday_data = data_list[0]
        yesterday_closing_price = float(yesterday_data["4. close"])

        day_before_yesterday_data = data_list[1]
        day_before_yesterday_closing_price = float(day_before_yesterday_data["4. close"])

        # Calculate the price difference and percentage
        difference = yesterday_closing_price - day_before_yesterday_closing_price
        diff_percent = round((difference / day_before_yesterday_closing_price) * 100)

        # Check for significant changes
        if diff_percent > significant_change_threshold:  # Only alert for increases
            telegram_url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": f"{STOCK_NAME}: 🔺{diff_percent}%",
                "parse_mode": "Markdown"
            }
            requests.post(telegram_url, data=payload)

    return JsonResponse({"message": "Stock alerts checked"}, status=200)
# views.py
from django.shortcuts import render, redirect
from .models import StockTicker

def stock_alert_page(request):
    tickers = StockTicker.objects.all()
    return render(request, 'predict/stock_alert.html', {'tickers': tickers})

# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import StockTicker

@csrf_exempt
def add_ticker(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        ticker = data.get('ticker')
        if ticker:
            StockTicker.objects.get_or_create(ticker=ticker)
            return JsonResponse({'message': f'{ticker} added to the list.'}, status=201)
    return JsonResponse({'error': 'Invalid request'}, status=400)





from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import UserUpdateForm, ProfileUpdateForm
from .models import Profile

@login_required
def user_profile(request):
    # Ensure the profile exists, create it if it doesn't
    profile, created = Profile.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        user_form = UserUpdateForm(request.POST, instance=request.user)
        profile_form = ProfileUpdateForm(request.POST, request.FILES, instance=profile)
        
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            return redirect('user_profile')
    else:
        user_form = UserUpdateForm(instance=request.user)
        profile_form = ProfileUpdateForm(instance=profile)

    # Pass necessary context to the template
    context = {
        'user_form': user_form,
        'profile_form': profile_form,
    }
    return render(request, 'predict/user_profile.html', context)





import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from django.http import JsonResponse
from django.views import View

api_key = 'RNZPXZ6Q9FEFMEHM'

class StockDataView(View):
    def get(self, request, symbol):
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, _ = ts.get_intraday(symbol=symbol, interval='1min', outputsize='full')
        close_data = data['4. close']
        return JsonResponse(close_data.to_dict())


from django.shortcuts import render
import yfinance as yf
import json
from django.http import JsonResponse



import yfinance as yf
from datetime import datetime, timedelta
from django.http import JsonResponse

def get_stock_data(request):
    if request.method == 'POST':
        ticker = request.POST.get('ticker')
        time_range = request.POST.get('range', '1m')  # Default to 1 month
        
        if ticker:
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            
            # Determine start_date based on the time range
            if time_range == '1w':
                start_date = end_date - timedelta(weeks=1)
            elif time_range == '1m':
                start_date = end_date - timedelta(days=30)
            elif time_range == '6m':
                start_date = end_date - timedelta(days=180)
            elif time_range == '1y':
                start_date = end_date - timedelta(days=365)
            else:
                return JsonResponse({'error': 'Invalid time range'}, status=400)

            history = stock.history(start=start_date, end=end_date)

            # Prepare data for response
            history_data = {
                'dates': history.index.strftime('%Y-%m-%d').tolist(),
                'closingPrices': history['Close'].tolist(),
            }

            return JsonResponse(history_data)

    return JsonResponse({'error': 'Invalid request method'}, status=405)




from django.http import JsonResponse
import yfinance as yf

def get_stock_data(request):
    if request.method == 'POST':
        ticker = request.POST.get('ticker')
        if ticker:
            try:
                # Fetch historical data for the last month
                data = yf.Ticker(ticker).history(period="1mo")
                
                if data.empty:
                    return JsonResponse({'error': 'No data found for the ticker'}, status=404)

                # Prepare data for the response
                dates = data.index.strftime('%Y-%m-%d').tolist()
                opening_prices = data['Open'].tolist()  # Fetch opening prices
                closing_prices = data['Close'].tolist()  # Fetch closing prices

                return JsonResponse({
                    'dates': dates,
                    'openingPrices': opening_prices,
                    'closingPrices': closing_prices
                })
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=400)
        else:
            return JsonResponse({'error': 'Ticker is required'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)



from django.http import JsonResponse
import yfinance as yf
from datetime import datetime

def get_stock_news(request):
    ticker = request.GET.get('ticker')
    if ticker:
        stock = yf.Ticker(ticker)
        news = stock.news[:5]  # Get the latest 5 news articles
        news_data = []
        
        for item in news:
            publish_time = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
            news_data.append({
                'title': item['title'],
                'link': item['link'],
                'publish_time': publish_time
            })
        
        return JsonResponse(news_data, safe=False)
    
    return JsonResponse([], safe=False)


from django.http import JsonResponse
import yfinance as yf
from datetime import datetime, timedelta



def get_financials(request):
    ticker = request.GET.get('ticker')
    if ticker:
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet
        financials_data = {
            'assets': balance_sheet.loc['Total Assets'].tolist(),
            'liabilities': balance_sheet.loc['Total Liabilities Net Minority Interest'].tolist(),
            # Add more as needed
        }

        return JsonResponse(financials_data)

    return JsonResponse({}, safe=False)



import yfinance as yf
from django.shortcuts import render, redirect
from .models import Portfolio

def portfolio_view(request):
    if request.method == "POST":
        stock_ticker = request.POST.get('stock_ticker').upper()
        shares = float(request.POST.get('shares'))
        purchase_price = float(request.POST.get('purchase_price'))

        # Save the portfolio entry
        Portfolio.objects.create(
            user=request.user,
            stock_ticker=stock_ticker,
            shares=shares,
            purchase_price=purchase_price
        )

        return redirect('portfolio')

    # Fetch user portfolio and stock prices
    portfolio = Portfolio.objects.filter(user=request.user)
    stocks_data = []
    for stock in portfolio:
        ticker = yf.Ticker(stock.stock_ticker)
        try:
            current_price = ticker.history(period='1d')['Close'][0]
        except IndexError:
            current_price = "N/A"  # Handle case where no data is available

        profit_loss = (current_price - stock.purchase_price) * stock.shares if current_price != "N/A" else "N/A"
        stocks_data.append({
            'ticker': stock.stock_ticker,
            'shares': stock.shares,
            'purchase_price': stock.purchase_price,
            'current_price': current_price,
            'profit_loss': profit_loss,
        })

    context = {'portfolio': stocks_data}
    return render(request, 'predict/portfolio.html', context)














# views.py
import yfinance as yf
from django.shortcuts import render

def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    stock_info = stock.info

    return {
        'symbol': stock_info.get('symbol', symbol),
        'current_price': stock_info.get('regularMarketPrice', 'N/A'),
        'open_price': stock_info.get('regularMarketOpen', 'N/A'),
        'high_price': stock_info.get('regularMarketDayHigh', 'N/A'),
        'low_price': stock_info.get('regularMarketDayLow', 'N/A'),
        'market_cap': stock_info.get('marketCap', 'N/A'),
        'pe_ratio': stock_info.get('trailingPE', 'N/A')
    }

def suggest_stock(stock1_data, stock2_data):
    # Initialize suggestions
    suggestions = []

    # Compare P/E ratio if available
    if stock1_data['pe_ratio'] != 'N/A' and stock2_data['pe_ratio'] != 'N/A':
        if stock1_data['pe_ratio'] < stock2_data['pe_ratio']:
            suggestions.append(f"{stock1_data['symbol']} has a lower P/E ratio, indicating it may be undervalued compared to {stock2_data['symbol']}.")
        elif stock2_data['pe_ratio'] < stock1_data['pe_ratio']:
            suggestions.append(f"{stock2_data['symbol']} has a lower P/E ratio, suggesting it may offer better value compared to {stock1_data['symbol']}.")

    # Compare Market Capitalization
    if stock1_data['market_cap'] != 'N/A' and stock2_data['market_cap'] != 'N/A':
        if stock1_data['market_cap'] > stock2_data['market_cap']:
            suggestions.append(f"{stock1_data['symbol']} has a larger market cap, indicating it is a bigger company compared to {stock2_data['symbol']}.")
        else:
            suggestions.append(f"{stock2_data['symbol']} has a larger market cap, showing it is bigger than {stock1_data['symbol']}.")

    # Compare current prices
    if stock1_data['current_price'] > stock2_data['current_price']:
        suggestions.append(f"{stock2_data['symbol']} is currently trading at a lower price than {stock1_data['symbol']}.")
    else:
        suggestions.append(f"{stock1_data['symbol']} is currently trading at a lower price than {stock2_data['symbol']}.")

    return suggestions

def compare_stocks(request):
    stock1_symbol = request.GET.get('stock1')
    stock2_symbol = request.GET.get('stock2')

    if stock1_symbol and stock2_symbol:
        stock1_data = fetch_stock_data(stock1_symbol)
        stock2_data = fetch_stock_data(stock2_symbol)

        # Generate suggestions based on comparison
        suggestions = suggest_stock(stock1_data, stock2_data)

        context = {
            'stock1_symbol': stock1_symbol,
            'stock2_symbol': stock2_symbol,
            'stock1_data': stock1_data,
            'stock2_data': stock2_data,
            'suggestions': suggestions,
        }
        return render(request, 'predict/compare_stocks.html', context)
    
    return render(request, 'predict/compare_stocks.html')



from django.shortcuts import render, redirect
from .models import Stock, AnalystRating
from .forms import StockForm




def get_trending_stocks():
    # Dummy implementation for trending stocks
    return [
        {'ticker': 'AAPL'},
        {'ticker': 'GOOGL'},
        {'ticker': 'AMZN'},
    ]



import yfinance as yf

def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    try:
        return stock.history(period="1d")['Close'][0]
    except IndexError:
        return "N/A"  # If no data available



from django.http import JsonResponse

def get_stock_data(request):
    if request.method == 'POST':
        ticker = request.POST.get('ticker')
        # Perform operations to get stock data here
        stock_data = {'dates': [], 'openingPrices': [], 'closingPrices': []}
        
        # Return the JSON response
        return JsonResponse(stock_data)

def get_stock_news(request):
    ticker = request.GET.get('ticker')
    # Perform operations to get news here
    news_data = [{'title': 'Example News', 'link': '#', 'publish_time': '2024-10-24'}]

    # Return the JSON response
    return JsonResponse(news_data, safe=False)














