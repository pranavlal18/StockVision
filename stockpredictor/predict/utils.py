
import yfinance as yf

def fetch_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.info

        current_price = stock_info.get('regularMarketPrice', 'N/A')
        today_open_price = stock_info.get('regularMarketOpen', 'N/A')

        if current_price != 'N/A' and today_open_price != 'N/A':
            if current_price > today_open_price:
                trend = 'up'
            elif current_price < today_open_price:
                trend = 'down'
            else:
                trend = 'neutral'
        else:
            trend = 'N/A'

        return {
            'real_time_price': current_price,
            'today_open_price': today_open_price,
            'trend': trend
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {
            'real_time_price': 'N/A',
            'today_open_price': 'N/A',
            'trend': 'N/A'
        }
