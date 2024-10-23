import numpy as np
import math
import time
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from consts.constants import RISK_FREE_INTEREST_RATE
from modern_portfolio_theory.empirical_solution_helpers import random_portfolios
from modern_portfolio_theory.return_and_volatility import portfolio_expected_return, portfolio_volatility, sharpe_ratio
from modern_portfolio_theory.MPT_epirically import mpt_empirically
from modern_portfolio_theory.MPT_analytic import mpt_analytic
from modern_portfolio_theory.topiary import topiary


def print_portfolio(name, allocation, expected_return, volatility, sharpe_ratio, num_decimals, show_0_allocation=False):
    print("-" * 80)  # Printing info about maximal Sharpe ratio portfolio
    print(name, "Portfolio Allocation\n")
    print("Annualised Return:", round(expected_return, num_decimals))
    print("Annualised Volatility:", round(volatility, num_decimals))
    print("Sharpe Ratio:", round(sharpe_ratio, num_decimals))
    print("\n")
    if show_0_allocation:
        print(allocation)
    else:
        print(allocation.drop([t for _, t in enumerate(allocation.columns) if allocation[t].iloc[0] <= 0], axis=1))


def print_polynomial_approximation(analytic_or_empirical, boundary_poly_coefs, number_of_decimals):
    print("-" * 80)
    print("Polynomial Approximation for the", analytic_or_empirical, "Efficient Frontier\n")
    s = str(round(boundary_poly_coefs[0], number_of_decimals))
    if boundary_poly_coefs[1] >= 0:
        s += " + "
    else:
        s += " - "
        boundary_poly_coefs[1] = (-1) * boundary_poly_coefs[1]
    s += str(round(boundary_poly_coefs[1], number_of_decimals)) + "y"
    for i in range(2, len(boundary_poly_coefs)):
        if boundary_poly_coefs[i] >= 0:
            s += " + "
        else:
            s += " - "
            boundary_poly_coefs[i] = (-1) * boundary_poly_coefs[i]
        s += str(round(boundary_poly_coefs[i], number_of_decimals)) + "y^" + str(i)
    print(s)


def plot_efficient_frontier(data, T, num_portfolios_to_simulate, empirical_MPT=True, analytic_MPT=False, num_decimals=2,
                            tolerance=10e-10, efficient_frontier_granularity=50, const_for_topiary=None,
                            positive_weights_only_for_topiary=True, too_high_rate=None,
                            starting_weight_for_topiary=None,
                            effecient_frontier_polynomial_approximation_degree=4):
    T = round(T * 252)
    returns = data.pct_change()  # Rate of returns for the stocks
    mean_returns = np.array(returns.mean())  # Mean rate of returns
    cov_matrix = np.array(returns.cov())  # Covariance matrix
    T_root = math.sqrt(T)

    if too_high_rate is not None:
        for i, stock in enumerate(data.columns):
            if (mean_returns[i] - cov_matrix[i, i]) > RISK_FREE_INTEREST_RATE:
                diff = (mean_returns[i] - cov_matrix[i, i] - RISK_FREE_INTEREST_RATE) / 2
                mean_returns[i] -= diff
                cov_matrix[i, i] += diff

            if mean_returns[i] > too_high_rate:
                cov_matrix[i, i] += (mean_returns[i] - too_high_rate) / 2
                mean_returns[i] -= (mean_returns[i] - too_high_rate) / 2

    print("Simulating portfolios...")
    t = time.time()
    results = random_portfolios(num_portfolios_to_simulate, mean_returns.copy(), cov_matrix.copy(),
                                RISK_FREE_INTEREST_RATE,
                                T)  # Calculate standard deviation, expected return, and Sharpe ratio of different portfolios
    w, an_vol_individual_stocks, an_rt_individual_stocks = np.zeros(len(mean_returns)), np.zeros(
        len(mean_returns)), np.zeros(len(mean_returns))
    for i in range(len(mean_returns)):
        w[i - 1] = 0
        w[i] = 1
        an_vol_individual_stocks[i] = portfolio_volatility(w, cov_matrix, T_root)
        an_rt_individual_stocks[i] = portfolio_expected_return(w, mean_returns, T)

    print("Finished simulating portfolios! This took", time.time() - t, "seconds.")

    if const_for_topiary is not None:
        ((topiary_allocation, topiary_expected_return, topiary_volatility, topiary_sharpe_ratio),
         (min_eta_allocation, min_eta_expected_return, min_eta_volatility, min_eta_sharpe_ratio),
         path_taken_info) = topiary(data, mean_returns.copy(), cov_matrix.copy(), T, tolerance,
                                    positive_weights_only=positive_weights_only_for_topiary,
                                    starting_weight=starting_weight_for_topiary, const_for_eta=const_for_topiary,
                                    number_of_decimals=num_decimals)

    if empirical_MPT:
        print("Calculating empircal solution (Note: This is only for positive portfolios)...")
        t = time.time()
        efficient_frontier_range = (min(min(results[1]), min(an_rt_individual_stocks)),
                                    max(max(results[1]), max(an_rt_individual_stocks)))
        ((max_sharpe_allocation_emp, max_sharpe_ret_emp, max_sharpe_vol_emp, max_sharpe_sharpe_emp),
         (min_vol_allocation_emp, min_volatility_ret_emp, min_volatility_vol_emp, min_volatility_sharpe_emp),
         (target_emp, efficient_portfolio_means_emp, boundary_poly_coefs_emp)) = mpt_empirically(data,
                                                                                                 mean_returns.copy(),
                                                                                                 cov_matrix.copy(),
                                                                                                 T, num_decimals,
                                                                                                 tolerance,
                                                                                                 efficient_frontier_range,
                                                                                                 efficient_frontier_granularity,
                                                                                                 effecient_frontier_polynomial_approximation_degree)
        print("Finished calculating empirical solution! This took", time.time() - t, "")

    if analytic_MPT:
        print("Calculating analytic solution...")
        t = time.time()
        lambda_range = (min(min(results[1]), min(an_rt_individual_stocks)),
                        max(max(results[1]), max(an_rt_individual_stocks)))
        ((market_portfolio_allocation_anal, mk_ret_anal, mk_vol_anal, mk_sharpe_anal),
         (min_vol_allocation_anal, min_volatility_ret_anal, min_volatility_vol_anal, min_volatility_sharpe_anal),
         (efficient_frontier_vols_anal, efficient_frontier_means_anal, boundary_poly_coefs_anal),
         (CMl_vars_anal, CML_means_anal), singular) = mpt_analytic(data, mean_returns.copy(), cov_matrix.copy(), T,
                                                                   num_decimals,
                                                                   lambda_range,
                                                                   efficient_frontier_granularity,
                                                                   effecient_frontier_polynomial_approximation_degree)
        if singular:
            print("Error: The covariance is too close to singular and the analytic solution was not able to be "
                  "calculated.")
        else:
            print("Finished calculating analytic solution! This took", time.time() - t, "")

    print("Displaying results")
    print("\n\n\n")

    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    color_bar = plt.colorbar()
    color_bar.set_label('Sharpe Ratio')
    plt.scatter(an_vol_individual_stocks, an_rt_individual_stocks, color='red', marker='o', s=20, zorder=1)
    for i, txt in enumerate(data.columns):
        plt.annotate(txt, (an_vol_individual_stocks[i], an_rt_individual_stocks[i]), xytext=(10, 0),
                     textcoords='offset points')

    if const_for_topiary is not None:
        print_portfolio("Topiaric", topiary_allocation, topiary_expected_return, topiary_volatility,
                        topiary_sharpe_ratio, num_decimals)
        print_portfolio("SciPy-Topiaric", min_eta_allocation, min_eta_expected_return, min_eta_volatility,
                        min_eta_sharpe_ratio, num_decimals)
        plt.scatter(path_taken_info[0, :], path_taken_info[1, :], c=path_taken_info[2, :], cmap='YlGnBu', marker='o',
                    s=10)
        for i in range(len(path_taken_info[0, :]) - 1):
            plt.annotate('', xy=(path_taken_info[0, i + 1], path_taken_info[1, i + 1]),
                         xytext=(path_taken_info[0, i], path_taken_info[1, i]),
                         arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.1))
        plt.scatter(topiary_volatility, topiary_expected_return, marker='*', color='cyan', s=75,
                    label='Topiaric portfolio', zorder=4)
        plt.scatter(min_eta_volatility, min_eta_expected_return, marker='*', color='gray', s=50,
                    label='SciPy-Topiaric portfolio', zorder=4)

    if empirical_MPT:
        print_portfolio("Maximum Sharpe Ratio", max_sharpe_allocation_emp, max_sharpe_ret_emp, max_sharpe_vol_emp,
                        max_sharpe_sharpe_emp, num_decimals)
        print_portfolio("Minimum Volatility", min_vol_allocation_emp, min_volatility_ret_emp, min_volatility_vol_emp,
                        min_volatility_sharpe_emp, num_decimals)
        print_polynomial_approximation("Empirical", boundary_poly_coefs_emp.copy(), num_decimals)
        print('\n')
        plt.plot(efficient_portfolio_means_emp, target_emp, linestyle='-.', color='black',
                 label='Positive Efficient Frontier', zorder=1)
        plt.plot(np.polyval(boundary_poly_coefs_emp[::-1], target_emp), target_emp, color='green', alpha=0.5,
                 label='polynomial approximation', zorder=1)
        plt.scatter(max_sharpe_vol_emp, max_sharpe_ret_emp, marker='*', color='y', s=150, label='Maximum Sharpe ratio',
                    zorder=2)
        plt.scatter(min_volatility_vol_emp, min_volatility_ret_emp, marker='*', color='orange', s=150,
                    label='Minimum volatility', zorder=2)

    if analytic_MPT and not singular:
        print_portfolio("Analytic Market", market_portfolio_allocation_anal, mk_ret_anal, mk_vol_anal, mk_sharpe_anal,
                        num_decimals)
        print_portfolio("Analytic Minimum Volatility", min_vol_allocation_anal, min_volatility_ret_anal,
                        min_volatility_vol_anal, min_volatility_sharpe_anal, num_decimals)
        print_polynomial_approximation("Analytic", boundary_poly_coefs_anal.copy(), num_decimals)
        print('\n')

        plt.plot(efficient_frontier_vols_anal, efficient_frontier_means_anal, linestyle='-.', color='red',
                 label='Analytic Efficient Frontier', zorder=1)
        plt.plot(CMl_vars_anal, CML_means_anal, linestyle='-.', color='y',
                 label='Capital Market Line', zorder=1)
        plt.scatter(mk_vol_anal, mk_ret_anal, marker='*', color='purple', s=100, label='Market Portflio', zorder=3)
        plt.scatter(min_volatility_vol_anal, min_volatility_ret_anal, marker='*', color='b', s=100,
                    label='Analytic Minimum Volatility Portflio', zorder=3)

    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    plt.savefig('temp.png', dpi=300)
