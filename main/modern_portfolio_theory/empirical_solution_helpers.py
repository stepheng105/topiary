import numpy as np
import math
from modern_portfolio_theory.return_and_volatility import portfolio_expected_return, portfolio_volatility, sharpe_ratio
import scipy as sci


# Given the mean returns, covariance matrix, and risk-free-interest rate, randomly select different portfolios
# and calculate their expected return, standard deviation, and sharpe ratios.
# ------------------------------------------------------------------------------------------------------
# Input:
#   num_portfolios (int): The number of random portfolios to observe (i.e. number of data points desired)
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
#   risk_free_rate (float): The risk-free interest rate
#   T (int): The number of periods measured per year
# Output:
#   results (numpy matrix): An (3)x(num_portfolios) matrix, where the first column corresponds to the standard deviations,
#                           the second column corresponds to the expected returns, and the third column corresponds to the
#                           Sharpe ratios for each random portfolio
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, T):
    results = np.zeros((3, num_portfolios))
    T_root = math.sqrt(T)
    for i in range(num_portfolios):
        weights = np.random.random(size=len(mean_returns))
        weights /= np.sum(weights)
        portfolio_std_dev = portfolio_volatility(weights, cov_matrix, T_root)
        portfolio_return = portfolio_expected_return(weights, mean_returns, T)
        results[0, i] = portfolio_std_dev  # Standard deviation
        results[1, i] = portfolio_return  # expected return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev  # Sharpe ratio
    return results


# Given the expected returns and covariance matrix, find the portfolio that minimizes volatility
# ------------------------------------------------------------------------------------------------------
# Input:
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
#   T (int): The number of periods measured per year
#   tolerance (float): The tolerance of sci.optimize.minimize for finding the solution
# Ouptut:
#   result (numpy array): The weights corresponding to the portfolio that minimizes volatility
def min_variance(mean_returns, cov_matrix, T, tolerance):
    num_assets = len(mean_returns)
    args = (cov_matrix, math.sqrt(T))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Making sure sum of weights must be equal to 1
                   {'type': 'ineq', 'fun': lambda x: np.min(x)})  # Making sure all elements are positive
    bound = (0.0, 1.0)
    bounds = tuple(
        bound for asset in range(num_assets))  # Each asset can range from 0% of the weight to 100% of the weight

    result = sci.optimize.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args,
                                   # Might be good to test how fast this algorithm converges/test different methods
                                   method='SLSQP', bounds=bounds, constraints=constraints, tol=tolerance)['x']
    return result


# Helper function used simply as input into sci.optimize.minimize in max_sharpe_ratio
# ------------------------------------------------------------------------------------------------------
# Input:
#   weights (numpy array): An array (that sums to 1) corresponding to the weight assigned to each stock
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
#   risk_free_rate (float): The risk-free-interest rate
#   T (int): The number of periods measured per year
#   T_root (float): The square root of T (for efficiency)
# Output:
#   neg_sharpe_ratio: The NEGATIVE of the Sharpe ratio of the given portfolio
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, T, T_root):
    negative_sharpe_ratio = -sharpe_ratio(weights, mean_returns, cov_matrix, T, T_root)
    return negative_sharpe_ratio


# Given the expected returns, covariance matrix, and risk-free-interest rate, find the portfolio that maximizes Sharpe ratio
# ------------------------------------------------------------------------------------------------------
# Input:
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
#   risk_free_rate (float): The risk-free-interest rate
#   T (int): The number of periods measured per year
#   tolerance (float): The tolerance of sci.optimize.minimize for finding the solution
# Ouptut:
#   result (numpy array): The weights corresponding to the portfolio that minimizes volatility
def max_sharpe_ratio(mean_returns, cov_matrix, T, tolerance):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, T, math.sqrt(T))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Making sure sum of weights must be equal to 1
                   {'type': 'ineq', 'fun': lambda x: np.min(x)})  # Making sure all elements are positive
    bound = (0.0, 1.0)
    bounds = tuple(
        bound for asset in range(num_assets))  # Each asset can range from 0% of the weight to 100% of the weight
    result = sci.optimize.minimize(neg_sharpe_ratio, num_assets * [1. / num_assets, ], args=args,
                                   method='SLSQP', bounds=bounds, constraints=constraints, tol=tolerance)['x']
    return result


def negative_eta(mu, psi, M, const_for_eta):
    return .5 * np.dot(mu, np.dot(M, mu)) - const_for_eta * np.dot(psi, mu)


def max_eta(psi, M, const_for_eta, tolerance):
    num_assets = len(psi)
    args = (psi, M, const_for_eta)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Making sure sum of weights must be equal to 1
                   {'type': 'ineq', 'fun': lambda x: np.min(x)})  # Making sure all elements are positive
    bound = (0.0, 1.0)
    bounds = tuple(
        bound for asset in range(num_assets))  # Each asset can range from 0% of the weight to 100% of the weight

    result = sci.optimize.minimize(negative_eta, num_assets * [1. / num_assets, ], args=args,
                                   # Might be good to test how fast this algorithm converges/test different methods
                                   method='SLSQP', bounds=bounds, constraints=constraints, tol=tolerance)['x']
    return result


# Given a expected returns, covariance matrix, target expected return, find the portfolio that has an expected return equal to
# the target value that minimizes the variance
# ------------------------------------------------------------------------------------------------------
# Input:
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
#   target (float): Target expected return
#   T (int): The number of periods measured per year
#   tolerance (float): The tolerance of sci.optimize.minimize for finding the solution
# Ouptut:
#   result (numpy array): The weights corresponding to the portfolio that minimizes volatility

def efficient_return(mean_returns, cov_matrix, target_mean, T, tolerance):
    num_assets = len(mean_returns)
    args = (cov_matrix, math.sqrt(T))

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_expected_return(x, mean_returns, T) - target_mean},
                   # Want expected return to be equal to target
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'ineq', 'fun': lambda x: np.min(x)})  # and weights to sum to 1
    bounds = tuple(
        (0, 1) for asset in range(num_assets))  # Each asset can range from 0% of the weight to 100% of the weight
    result = sci.optimize.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args,
                                   method='SLSQP', bounds=bounds, constraints=constraints, tol=tolerance)['fun']
    return result


# Function that traces out the efficient frontier given the range of expected returns
# ------------------------------------------------------------------------------------------------------
# Input:
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
#   returns_range (numpy array): An array of floats that you wish to measure the efficient frontier on
#   T (int): The number of periods measured per year
# Output:
#   efficients (list): A list of the portfolio weights that lie on the efficient frontier
def efficient_frontier(mean_returns, cov_matrix, target_means, T, tolerance):
    length = len(target_means)
    efficients = [None] * length
    for i in range(length):
        efficients[i] = efficient_return(mean_returns, cov_matrix, target_means[i], T, tolerance)
    return efficients
