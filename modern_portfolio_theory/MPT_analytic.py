import numpy as np
import time
import pandas as pd
import math
from modern_portfolio_theory.empirical_solution_helpers import negative_eta
from modern_portfolio_theory.return_and_volatility import portfolio_volatility, portfolio_expected_return, sharpe_ratio
from modern_portfolio_theory.analytic_solution_helpers import (get_minimum_variance_portfolio_analytically,
                                                               get_market_portfolio_analytically,
                                                               get_analytic_efficient_frontier,
                                                               get_analytic_capital_market_line,
                                                               convert_mean_range_to_lambda_range)


def mpt_analytic(data, mean_returns, cov_matrix, T, const_for_eta, number_of_decimals, effecient_frontier_means_range,
                 efficient_frontier_granularity, efficient_frontier_polynomial_approximation_degree):
    print("Calculating Analytic Solution...")
    if np.linalg.cond(cov_matrix) < 10 ** 15:
        t = time.time()
        T_root = math.sqrt(T)
        w_min_vol = get_minimum_variance_portfolio_analytically(mean_returns, cov_matrix)
        min_volatility_vol = portfolio_volatility(w_min_vol, cov_matrix, T_root)
        min_volatility_ret = portfolio_expected_return(w_min_vol, mean_returns, T)  # Calculate expected value and standard deviation
        min_volatility_sharpe = sharpe_ratio(w_min_vol, mean_returns, cov_matrix, T, T_root)
        min_volatility_eta = -negative_eta(w_min_vol, mean_returns, cov_matrix, const_for_eta)

        w_mk = get_market_portfolio_analytically(mean_returns, cov_matrix)
        mk_vol = portfolio_volatility(w_mk, cov_matrix, T_root)
        mk_ret = portfolio_expected_return(w_mk, mean_returns, T)  # Calculate expected value and standard deviation
        mk_sharpe = sharpe_ratio(w_mk, mean_returns, cov_matrix, T, T_root)
        mk_eta = -negative_eta(w_mk, mean_returns, cov_matrix, const_for_eta)

        lambda_range = convert_mean_range_to_lambda_range(effecient_frontier_means_range, mean_returns,
                                                          cov_matrix, mk_ret, min_volatility_ret)

        efficient_frontier_means, efficient_frontier_vols = get_analytic_efficient_frontier(w_mk, w_min_vol,
                                                                                            mean_returns, cov_matrix,
                                                                                            T, lambda_range,
                                                                                            efficient_frontier_granularity)
        boundary_poly_coefs = np.polynomial.polynomial.polyfit(efficient_frontier_means, efficient_frontier_vols,
                                                               efficient_frontier_polynomial_approximation_degree)

        # CML_volatility_range = (0, max(efficient_frontier_vols))
        CML_means, CMl_vars = get_analytic_capital_market_line(w_mk.copy(), mean_returns, cov_matrix, T,
                                                               effecient_frontier_means_range, efficient_frontier_granularity)

        market_portfolio_allocation = pd.DataFrame(w_mk, index=data.columns,
                                             columns=['allocation'])  # Convert to pandas dataframe for better printing
        market_portfolio_allocation.allocation = [round(i * 100, number_of_decimals - 2) for i in w_mk]
        market_portfolio_allocation = market_portfolio_allocation.T
        min_vol_allocation = pd.DataFrame(w_min_vol, index=data.columns,
                                             columns=['allocation'])  # Convert to pandas dataframe for better printing
        min_vol_allocation.allocation = [round(i * 100, number_of_decimals - 2) for i in w_min_vol]
        min_vol_allocation = min_vol_allocation.T
        singular = False
        print("Analytic solution computed! This took", time.time() - t, "seconds.")
    else:
        print("Warning: Covariance matrix is too close to being singular. Analytic solution cannot be computed.")
        (market_portfolio_allocation, mk_ret, mk_vol, mk_sharpe, mk_eta, min_vol_allocation,
         min_volatility_ret, min_volatility_vol, min_volatility_sharpe, min_volatility_eta, efficient_frontier_vols,
         efficient_frontier_means, boundary_poly_coefs, CMl_vars, CML_means) = (None, None, None, None, None, None,
                                                                                None, None, None, None, None, None,
                                                                                None, None, None)
        singular = True

    return ((market_portfolio_allocation, mk_ret, mk_vol, mk_sharpe, mk_eta),
            (min_vol_allocation, min_volatility_ret, min_volatility_vol, min_volatility_sharpe, min_volatility_eta),
            (efficient_frontier_vols, efficient_frontier_means, boundary_poly_coefs),
            (CMl_vars, CML_means), singular)
