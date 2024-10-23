import numpy as np
import math
import sys
sys.path.append('../consts')
from consts.constants import RISK_FREE_INTEREST_RATE


# The following code has been STOLEN from https://github.com/tthustla/efficient_frontier/blob/master/Efficient%20_Frontier_implementation.ipynb
# Criminal, I know. For whom would stoop so low as to plagerize code from github :'(
# Comments are my own though. I may also go in and try to optimize the code, as it is a bit slow at the moment

# Given the portfolio weights and mean returns, calculate the expected return of the portfolio
# ------------------------------------------------------------------------------------------------------
# Input:
#   weights (numpy array): An array (that sums to 1) corresponding to the weight assigned to each stock
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   T (int): The number of periods measured per year
# Output:
#   std (float): The standard deviation of the portfolio
def portfolio_expected_return(weights, mean_returns, T):
    returns = np.sum(mean_returns * weights) * T
    return returns

def portfolio_volatility(weights, cov_matrix, T_root):
    val = np.dot(weights, np.dot(cov_matrix, weights))
    if val < 0:  # the cov_matrix matrix is positive semidefinite, so theoretically this value should never be negative. If it is, then we have run into a precision error that occurs when our numbers get too small (on the order of 10e-20)
        std = 0
    else:
        std = math.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * T_root
    return std


def sharpe_ratio(weights, mean_returns, cov_matrix, T, T_root):
    p_var = portfolio_volatility(weights, cov_matrix, T_root)
    p_ret = portfolio_expected_return(weights, mean_returns, T)
    s_ratio = (p_ret - RISK_FREE_INTEREST_RATE) / p_var
    return s_ratio


def sharpe_ratio_given_ret_and_vol(expected_return, volatility):
    return (expected_return - RISK_FREE_INTEREST_RATE) / volatility


def get_mean_returns_and_covariance_from_data(data):
    returns = data.pct_change()  # Rate of returns for the stocks
    mean_returns = np.array(returns.mean())  # Mean rate of returns
    cov_matrix = np.array(returns.cov())  # Covariance matrix
    return mean_returns, cov_matrix
