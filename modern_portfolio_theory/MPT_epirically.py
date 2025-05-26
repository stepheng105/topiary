import pandas as pd
import numpy as np
import math
import time
from modern_portfolio_theory.return_and_volatility import portfolio_volatility, portfolio_expected_return, sharpe_ratio
from modern_portfolio_theory.empirical_solution_helpers import max_sharpe_ratio, min_variance, efficient_frontier, negative_eta
from consts.constants import RISK_FREE_INTEREST_RATE


def mpt_empirically(data, mean_returns, cov_matrix, T, const_for_eta, number_of_decimals, tolerance, effecient_frontier_means_range,
                    efficient_frontier_granularity, efficient_frontier_polynomial_approximation_degree):
    print("Finding portfolio with maximum Sharpe ratio...")
    t = time.time()
    T_root = math.sqrt(T)
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, T,
                                  tolerance)  # Get the portfolio with the max Sharpe ratio
    max_sharpe_vol = portfolio_volatility(max_sharpe, cov_matrix, T_root)
    max_sharpe_ret = portfolio_expected_return(max_sharpe, mean_returns, T)  # Calculate expected value and standard deviation
    max_sharpe_sharpe = sharpe_ratio(max_sharpe, mean_returns, cov_matrix, T, T_root)
    max_sharpe_eta = -negative_eta(max_sharpe, mean_returns, cov_matrix, const_for_eta)
    max_sharpe_allocation = pd.DataFrame(max_sharpe, index=data.columns,
                                         columns=['allocation'])  # Convert to pandas dataframe for better printing
    max_sharpe_allocation.allocation = [round(i * 100, number_of_decimals - 2) for i in max_sharpe]
    max_sharpe_allocation = max_sharpe_allocation.T
    print("Portfolio found! This took", time.time() - t, "seconds.")

    print("Finding portfolio with minimum volatility...")
    t = time.time()
    min_vol = min_variance(mean_returns, cov_matrix, T, tolerance)  # Get the portfolio with the min volatility
    min_volatility_vol = portfolio_volatility(min_vol, cov_matrix, T_root)
    min_volatility_ret = portfolio_expected_return(min_vol, mean_returns, T)  # Calculate expected value and standard deviation
    min_volatility_sharpe = sharpe_ratio(min_vol, mean_returns, cov_matrix, T, T_root)
    min_volatility_eta = -negative_eta(min_vol, mean_returns, cov_matrix, const_for_eta)
    min_vol_allocation = pd.DataFrame(min_vol, index=data.columns,
                                      columns=['allocation'])  # Convert to pandas dataframe for better printing
    min_vol_allocation.allocation = [round(i * 100, number_of_decimals - 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    print("Portfolio found! This took", time.time() - t, "seconds.")

    print("Finding efficient frontier...")
    t = time.time()
    target_means = np.linspace(effecient_frontier_means_range[0], effecient_frontier_means_range[1], efficient_frontier_granularity)
    efficient_portfolio_vols = efficient_frontier(target_means, cov_matrix, mean_returns, T, tolerance)
    boundary_poly_coefs = np.polynomial.polynomial.polyfit(target_means, efficient_portfolio_vols,
                                                           efficient_frontier_polynomial_approximation_degree)
    print("Efficient frontier found! This took", time.time() - t, "seconds.")

    return ((max_sharpe_allocation, max_sharpe_ret, max_sharpe_vol, max_sharpe_sharpe, max_sharpe_eta),
            (min_vol_allocation, min_volatility_ret, min_volatility_vol, min_volatility_sharpe, min_volatility_eta),
            (target_means, efficient_portfolio_vols, boundary_poly_coefs))

