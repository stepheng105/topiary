import pandas as pd
import datetime
from consts.constants import JSON_FILE_NAME, DESIRED_STOCKS, BANNED_STOCKS


# A function for filtering out all stock data other than selected stocks
# ------------------------------------------------------------------------------------------------------
# Inputs:
#   data (pd dataframe): A pandas dataframe of stock data
#   desired_stocks (list): List of tickers for desired stocks
# Output:
#   new_data (pd dataframe): data of only desired stocks
def use_only_desired_stocks(data, desired_stocks):
    stocks = data.columns.to_list()
    keep = []
    for ticker in desired_stocks:
        if ticker in stocks:
            keep.append(ticker)
    new_data = data[keep]
    return new_data


# A function for filtering out all undesired stock data
# ------------------------------------------------------------------------------------------------------
# Inputs:
#   data (pd dataframe): A pandas dataframe of stock data
#   banned_stocks (list): List of tickers for undesired stocks
# Output:
#   data (pd dataframe): The data without the stocks in banned_stocks
def remove_banned_stocks(data, banned_stocks):
    stocks = data.columns.to_list()
    banned = []
    for ticker in banned_stocks:
        if ticker in stocks:
            banned.append(ticker)
    data.drop(columns=banned, inplace=True)
    return data


# A function to remove stocks from data with too many NaN values (i.e. stock wasn't public at the time)
# ------------------------------------------------------------------------------------------------------
# Inputs:
#   data (pd dataframe): A pandas dataframe of stock data
#   i1 (integer): The index corresponding to the initial day of the day range we wish to check
#   i2 (integer): The index corresponding to the final day of the day range we wish to check
# Output:
#   data (pd dataframe): The data without the stocks with too many NaN values
def remove_stocks_without_enough_data(data, i1, i2):
    data_temp = data.iloc[i1:i2]
    num_nans = data_temp.isnull().sum()
    max_nan = num_nans.min()
    good_tickers = []
    for i, ticker in enumerate(data.columns):
        if num_nans[ticker] == max_nan:
            good_tickers.append(ticker)
    return data[good_tickers]


# A function that inputs the start time and end times that we wish to base our historical data on
# and outputs the corresponding indeces in our data
# ------------------------------------------------------------------------------------------------------
# Inputs:
#   start_time (integer): The unix timestamp (in seconds) of the day we wish to start our historical data
#   end_time (integer): The unix timestamp (in seconds) of the day we wish to end our historical data
# Output:
#   i1 (integer): The index in our data corresponding to the day specified by start_time
#   i2 (integer): The index in our data corresponding to the day specified by end_time
def get_partition_timeseries_indices(start_time, end_time):
    i1, i2 = int((start_time - (start_time % 86400)) / 86400), int((end_time - (end_time % 86400)) / 86400)
    return i1, i2


# A function that inputs a date then returns the unix timestamp (in seconds) for the corresponding date
# ------------------------------------------------------------------------------------------------------
# Inputs:
#   month (integer): A number between 1 and 12 corresponding to the month of year
#   day (integer): A number between 1 and 31 corresponding to the day of the month
#   year (integer): A number greater than 1970 corresponding to the year
# Output:
#   s (integer): The number of seconds since January 1, 1970
def get_unix_time_from_date(month, day, year):
    s = int(datetime.datetime(year, month, day).timestamp())
    return s


# A function that inputs a start time and an end time and outputs the data restricted to that timeframe
# ------------------------------------------------------------------------------------------------------
# Inputs:
#   start_date (tuple): A tuple of integers representing the beginning date of your data formatted like (month, day, year)
#   end_date (tuple): A tuple of integers representing the end date of your data formatted like (month, day, year)
def get_data(start_date, end_date):
    start_time_historical = get_unix_time_from_date(start_date[0], start_date[1], start_date[2])
    end_time_historical = get_unix_time_from_date(end_date[0], end_date[1], end_date[2])
    i1, i2 = get_partition_timeseries_indices(start_time_historical, end_time_historical)
    json_data = pd.read_json('data/' + JSON_FILE_NAME)
    if DESIRED_STOCKS == "all":
        data = json_data
    else:
        data = use_only_desired_stocks(json_data, DESIRED_STOCKS)
    if not (BANNED_STOCKS is None):
        data = remove_banned_stocks(data, BANNED_STOCKS)
    data = remove_stocks_without_enough_data(data, i1, i2)
    data = data.iloc[i1:i2]
    return data.dropna()
