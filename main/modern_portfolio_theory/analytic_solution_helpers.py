import numpy as np
import scipy as sci
import math
from modern_portfolio_theory.return_and_volatility import portfolio_expected_return, portfolio_volatility
import sys
sys.path.append('../consts')
from consts.constants import RISK_FREE_INTEREST_RATE


# Function for solving the matrix equation Ax = b for x. It will use sci.linalg.solve to get the exact answer so long as A
# is not too close to being singular. If it is, it will switch to using least squares to solve the matrix equation
# ------------------------------------------------------------------------------------------------------
# Input:
#   A (numpy matrix): An n x n square matrix
#   b (numpy vector): A length n vector
#   assume_a (string): A constant for sci.linalg.solve to assume things about the matrix A (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html#scipy.linalg.solve)
# Output:
#   x (numpy vector): The vector that solves the equation Ax = b. If A is singular or close to singular, it returns None.
def solve_matrix_equation(A, b, assume_a='gen'):
    if np.linalg.cond(A) < 10 ** 15:
        x = sci.linalg.solve(A, b, assume_a=assume_a)
    else:
        x = None
    return x

# Function that uses the analytic equations described above to get the minimum variance portfolio
# ------------------------------------------------------------------------------------------------------
# Input:
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
# Output:
#   w_min_var (numpy vector): The portfolio that has minimum variance, or None if the covariance matrix is singular
def get_minimum_variance_portfolio_analytically(mean_returns, cov_matrix):
    e = np.ones(len(mean_returns))
    sigma_inv_e = solve_matrix_equation(cov_matrix, e, assume_a="sym")
    if not (sigma_inv_e is None):
        w_min_var = sigma_inv_e / np.dot(e, sigma_inv_e)
        w_min_var /= np.sum(w_min_var)
    else:
        w_min_var = None
    return w_min_var


# Function that uses the analytic equations described above to get the market portfolio
# ------------------------------------------------------------------------------------------------------
# Input:
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
# Output:
#   w_mk (numpy vector): The portfolio that has minimum variance, or None if the covariance matrix is singular
def get_market_portfolio_analytically(mean_returns, cov_matrix):
    e = np.ones(len(mean_returns))
    sigma_inv_m = solve_matrix_equation(cov_matrix, mean_returns, assume_a="sym")
    if not (sigma_inv_m is None):
        w_mk = sigma_inv_m / np.dot(e, sigma_inv_m)
        w_mk /= np.sum(w_mk)
    else:
        w_mk = None
    return w_mk


def convert_mean_range_to_lambda_range(mean_range, mean_returns, cov_matrix, mk_exp_ret, min_var_exp_ret):
    c = np.dot(mean_returns, solve_matrix_equation(cov_matrix, np.ones(len(mean_returns)), assume_a="sym"))
    a = (mean_range[0] - min_var_exp_ret)/(c*(mk_exp_ret - min_var_exp_ret))
    b = (mean_range[1] - min_var_exp_ret)/(c*(mk_exp_ret - min_var_exp_ret))
    lambda_range = min(a, b), max(a, b)
    return lambda_range


# Given the range of expected returns, analytically compute the efficient frontier for these values.
# ------------------------------------------------------------------------------------------------------
# Input:
#   w_mk (numpy array): The market portfolio
#   w_min (numpy array): The minimum variance portfolio
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
#   T (int): The number of periods measured per year
#   returns_range (numpy array): An array of floats that you wish to measure the efficient frontier on
# Output:
#   analytic_efficient_frontier_volatilities (numpy array): A list of the portfolio weights that lie on the efficient frontier
def get_analytic_efficient_frontier(w_mk, w_min, mean_returns, cov_matrix, T, lambda_range, efficient_frontier_granularity):
    c = np.dot(mean_returns, solve_matrix_equation(cov_matrix, np.ones(len(mean_returns)), assume_a="sym"))
    targets = np.linspace(lambda_range[0], lambda_range[1], efficient_frontier_granularity)
    dx = targets[1] - targets[0]
    targets = np.append(targets, targets[len(targets) - 1] + dx)
    targets = np.append(targets[0] - dx, targets)
    efficient_means = [None] * len(targets)
    efficient_vols = [None] * len(targets)
    T_root = math.sqrt(T)
    for i in range(len(targets)):
        w = (1 - targets[i]*c)*w_min + targets[i]*c*w_mk
        efficient_means[i] = portfolio_expected_return(w, mean_returns, T)
        efficient_vols[i] = portfolio_volatility(w, cov_matrix, T_root)
    return np.array(efficient_means), np.array(efficient_vols)


# Given the range of expected returns, compute the capitcal market line for these values.
# ------------------------------------------------------------------------------------------------------
# Input:
#   w_mk (numpy array): The market portfolio
#   mean_returns (numpy array): An array with the mean returns of n different stocks
#   cov_matrix (numpy matrix): The (n)x(n) covariance matrix for the stocks
#   T (int): The number of periods measured per year
#   risk_free_interest_rate (float): The risk-free interest rate
#   returns_range (numpy array): An array of floats that you wish to measure the efficient frontier on
# Output:
#   analytic_capital_market_line_volatilities (numpy array): A list of the portfolio weights that lie on the capital market line
def get_analytic_capital_market_line(w_mk, mean_returns, cov_matrix, T, means_range, granularity):
    market_volatility, market_return = portfolio_volatility(w_mk, cov_matrix, math.sqrt(T)), portfolio_expected_return(
        w_mk, mean_returns, T)
    CML_means = np.linspace(means_range[0], means_range[1], granularity)
    CML_vols = [None] * len(CML_means)
    c = market_volatility/(market_return - RISK_FREE_INTEREST_RATE)
    for i in range(len(CML_vols)):
        CML_vols[i] = c*(CML_means[i] - RISK_FREE_INTEREST_RATE)
    return np.array(CML_means), np.array(CML_vols)
