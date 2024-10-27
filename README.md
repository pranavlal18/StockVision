# Stock Vision

Stock Vision is a web application designed to provide users with a comprehensive dashboard for stock analysis, prediction, and real-time monitoring. It allows users to track historical and real-time stock data, perform predictions, and compare different stocks to make informed investment decisions.



## Features

- **User Authentication**: Register, login, and manage profiles with additional customization options.
- **Real-Time Stock Tracking**: View and monitor real-time stock prices.
- **Stock Prediction**: Predict future stock prices using historical data.
- **Stock Comparison**: Compare stocks based on P/E ratio, market capitalization, and current prices.
- **Watchlist**: Track favorite stocks and monitor their performance.
- **Theming Options**: Switch between light and dark mode for a more personalized experience.

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Backend**: Django (Python), Django Rest Framework (DRF)
- **APIs**: Alpha Vantage, Yahoo Finance (`yfinance`)
- **Database**: SQLite (development), Postgres (production-ready setup recommended)


## Installation

### Prerequisites

- Python 3.8+
- Django 4.0+
- Virtual Environment setup (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/stockvision.git
cd stockvision


python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

python manage.py migrate

python manage.py collectstatic

python manage.py runserver

The application will be accessible at http://127.0.0.1:8000/.

Usage
Register/Login: Sign up or log in to access personalized features.
Dashboard: View your customized dashboard with options for real-time stock tracking, stock comparison, and predictions.
Prediction: Enter stock tickers to predict prices using historical data.
Watchlist: Add your favorite stocks to your watchlist and monitor them in real-time.

Future Enhancements
Enhanced Prediction Algorithms: Implementing machine learning for more accurate predictions.
Portfolio Management: Users can create portfolios and track their investments.
Advanced Theming Options: Additional customization features.
Social Sharing: Share insights and stock predictions with friends or social media.

License
This project is licensed under the MIT License. See the LICENSE file for details.

