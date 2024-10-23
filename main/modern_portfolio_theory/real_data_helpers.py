import pandas as pd
import os
import datetime
from consts.constants import JSON_FILE_NAME, DESIRED_STOCKS, BANNED_STOCKS


def use_only_desired_stocks(data, desired_stocks):
    stocks = data.columns.to_list()
    keep = []
    for ticker in desired_stocks:
        if ticker in stocks:
            keep.append(ticker)
    data = data[keep]
    return data


def remove_banned_stocks(data, banned_stocks):
    stocks = data.columns.to_list()
    banned = []
    for ticker in banned_stocks:
        if ticker in stocks:
            banned.append(ticker)
    data.drop(columns=banned, inplace=True)
    return data


def remove_stocks_without_enough_data(data, i1, i2):
    data_temp = data.iloc[i1:i2]
    num_nans = data_temp.isnull().sum()
    max_nan = num_nans.min()
    good_tickers = []
    for i, ticker in enumerate(data.columns):
        if num_nans[ticker] == max_nan:
            good_tickers.append(ticker)
    return data[good_tickers]


def get_partition_timeseries_indices(start_time, end_time):
    i1, i2 = int((start_time - (start_time % 86400)) / 86400), int((end_time - (end_time % 86400)) / 86400)
    return i1, i2


def get_unix_time_from_date(day, month, year):
    return int(datetime.datetime(year, month, day).timestamp())


def get_data(start_date, end_date):
    start_time_historical = get_unix_time_from_date(start_date[0], start_date[1], start_date[2])
    end_time_historical = get_unix_time_from_date(end_date[0], end_date[1], end_date[2])
    i1, i2 = get_partition_timeseries_indices(start_time_historical, end_time_historical)
    json_data = pd.read_json('/Users/rattzombie/Desktop/School/CPE/rtd_research_project/main/data/' + JSON_FILE_NAME)
    if DESIRED_STOCKS == "all":
        data = json_data
    else:
        data = use_only_desired_stocks(json_data, DESIRED_STOCKS)
    if not (BANNED_STOCKS is None):
        data = remove_banned_stocks(data, BANNED_STOCKS)
    data = remove_stocks_without_enough_data(data, i1, i2)
    data = data.iloc[i1:i2]
    return data.dropna()
