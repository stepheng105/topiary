import pandas as pd
import datetime
import yfinance as yf
from consts.constants import TICKER_SET, DESIRED_STOCKS, BANNED_STOCKS
from curl_cffi import requests


# A function for filtering out all stock data other than selected stocks
# ------------------------------------------------------------------------------------------------------
# Inputs:
#   data (pd dataframe): A pandas dataframe of stock data
#   desired_stocks (list): List of tickers for desired stocks
# Output:
#   new_data (pd dataframe): data of only desired stocks
def use_only_desired_stocks(tickers, desired_stocks):
    if desired_stocks == 'all':
        return tickers
    return [t for t in tickers if t in desired_stocks]


# A function for filtering out all undesired stock data
# ------------------------------------------------------------------------------------------------------
# Inputs:
#   data (pd dataframe): A pandas dataframe of stock data
#   banned_stocks (list): List of tickers for undesired stocks
# Output:
#   data (pd dataframe): The data without the stocks in banned_stocks
def remove_banned_stocks(tickers, banned_stocks):
    if banned_stocks == 'none':
        return tickers
    return [t for t in tickers if t not in banned_stocks]


# A function that inputs a start time and an end time and outputs the data restricted to that timeframe
# ------------------------------------------------------------------------------------------------------
# Inputs:
#   start_date (tuple): A tuple of integers representing the beginning date of your data formatted like (month, day, year)
#   end_date (tuple): A tuple of integers representing the end date of your data formatted like (month, day, year)
def get_data(start_date, end_date, inteval='1d'):
    tickers_df = pd.read_csv('data/ticker_names.csv')
    tickers_list = tickers_df.columns  # Column 0 is all nasdaq stocks, Column 1 is SMP500 stocks, column 2 is stocks from James' google sheet
    tickers = list(set([T[1].upper() for T in tickers_df[tickers_list[TICKER_SET]].items() if isinstance(T[1], str) and T[1].isalnum()]))
    tickers = use_only_desired_stocks(tickers, DESIRED_STOCKS)
    tickers = remove_banned_stocks(tickers, BANNED_STOCKS)

    # GRAH YAHOO FINANCE WHY MUST YOU RATE LIMIT ME SO NOW I MUST LIE TO YOU AND PRETEND I AM A CHROME BROWSER
    session = requests.Session(impersonate="chrome")    # Beep Boop I am a robot
    tickers = yf.Tickers(tickers, session=session)

    return tickers.history(start=start_date, end=end_date, interval=inteval).loc[:, 'Close'].dropna(axis=1)

