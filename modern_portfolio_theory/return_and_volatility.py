import numpy as np
import math
from consts.constants import RISK_FREE_INTEREST_RATE


# A function that takes in portfolio weights and calculates the expected return of the portfolio
# ------------------------------------------------------------------------------------------------------
# Input:
#   weights (numpy array): An array (that sums to 1) corresponding to the weight assigned to each stock
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   T (int): The number of periods measured per year
# Output:
#   returns (float): the expected return of the portfolio after time T
def portfolio_expected_return(weights, mean_returns, T):
    returns = np.sum(mean_returns * weights) * T
    return returns


# A function that takes in a portfolio weights and calculates the volatility of the portfolio
# ------------------------------------------------------------------------------------------------------
# Input:
#   weights (numpy array): An array (that sums to 1) corresponding to the weight assigned to each stock
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
#   T_root (float): The square root of T
def portfolio_volatility(weights, cov_matrix, T_root):
    val = np.dot(weights, np.dot(cov_matrix, weights))
    if val < 0:  # the cov_matrix matrix is positive semidefinite, so theoretically this value should never be negative. If it is, then we have run into a precision error that occurs when our numbers get too small (on the order of 10e-20)
        std = 0
    else:
        std = math.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * T_root
    return std


# A function that takes in a portfolio weights and calculates the Sharpe ratio of the portfolio
# ------------------------------------------------------------------------------------------------------
# Input:
#   weights (numpy array): An array (that sums to 1) corresponding to the weight assigned to each stock
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
#   T (int): The number of periods measured per year
#   T_root (float): The square root of T
# Output:
#   s_ratio (float): The sharpe ratio of the portfolio
def sharpe_ratio(weights, mean_returns, cov_matrix, T, T_root):
    p_var = portfolio_volatility(weights, cov_matrix, T_root)
    p_ret = portfolio_expected_return(weights, mean_returns, T)
    s_ratio = (p_ret - RISK_FREE_INTEREST_RATE) / p_var
    return s_ratio


# A function that quickly computes the Sharpe ratio if we are already given the expected return and
# volatility of a portfolio
# ------------------------------------------------------------------------------------------------------
# Input:
#   expected_return (float): The expected return of a portfolio
#   volatility (float): The standard deviation of a portfolio
# Output:
#   sharpe (float): The Sharpe ratio of a portfolio
def sharpe_ratio_given_ret_and_vol(expected_return, volatility):
    sharpe = (expected_return - RISK_FREE_INTEREST_RATE) / volatility
    return sharpe


# A helper function for computing the rate of returns vector and covariance matrix from our data
# ------------------------------------------------------------------------------------------------------
# Input:
#   data (pd dataframe): A pandas dataframe of our historical data
# Output:
#   mean_returns (np vector): The vector of means for the rates of return of our historical data
#   cov_matrix (np matrix): The covariance matrix for the rates of return of our historical data
def get_mean_returns_and_covariance_from_data(data):
    returns = data.pct_change()  # Rate of returns for the stocks
    mean_returns = np.array(returns.mean())  # Mean rate of returns
    cov_matrix = np.array(returns.cov())  # Covariance matrix
    return mean_returns, cov_matrix
