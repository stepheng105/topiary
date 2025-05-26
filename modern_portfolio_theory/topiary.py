import numpy as np
import math
import time
import pandas as pd
import sys
from modern_portfolio_theory.return_and_volatility import portfolio_expected_return, portfolio_volatility, sharpe_ratio
from modern_portfolio_theory.empirical_solution_helpers import max_eta, negative_eta
from consts.constants import RISK_FREE_INTEREST_RATE


def topiary_main_algorithm(mean_returns, cov_matrix, starting_weight, precision, positive_weights_only,
                           const_for_eta, keep_track=True, traditional_alg=True):

    psi = const_for_eta * mean_returns.copy()
    if starting_weight is None:
        mu = np.zeros(len(psi))
        mu[0] = 1
    elif starting_weight == 'max':
        mu = np.zeros(len(psi))
        mu[np.argmax(psi-.5*np.diag(cov_matrix))] = 1
    elif starting_weight == 'random':
        mu = np.random.random(size=len(psi))
        mu /= np.sum(mu)
    else:
        mu = starting_weight
    C = np.dot(cov_matrix, mu)
    precision *= 2
    previous_mus = [np.copy(mu)]
    memory_usage_of_previous_mus = sys.getsizeof(previous_mus)
    cov_diag = np.diagonal(cov_matrix)


    if traditional_alg:
        while True:
            norm_mu_squared = np.dot(mu, C)  # Calculating <mu, mu>
            phi_mu = np.dot(mu, psi)  # Calculating <psi, mu>
            delta_z = np.argmax(psi - C)
            iota = (psi[delta_z] - C[delta_z]) - (phi_mu - norm_mu_squared)
            if iota <= 0:
                break
            norm_delta_minus_mu_squared = cov_matrix[delta_z, delta_z] - 2 * C[delta_z] + norm_mu_squared
            if math.isclose(0, norm_delta_minus_mu_squared):
                raise ZeroDivisionError("norm_delta_minus_mu_squared is too close to 0\n mu:", mu, "\n z=", delta_z, "\nnorm_delta_minus_mu_squared=",norm_delta_minus_mu_squared)
            t = iota / norm_delta_minus_mu_squared

            if positive_weights_only:
                if t > 1:
                    t = 1
                elif t < 0:
                    t = 0

            mu = (1 - t) * mu
            mu[delta_z] += t
            if keep_track:
                if memory_usage_of_previous_mus < 2e+10:
                    previous_mus.append(np.copy(mu))
                    memory_usage_of_previous_mus = sys.getsizeof(previous_mus)
                else:
                    print("Used too much memory (" + str(memory_usage_of_previous_mus) + "). Dumping memory.")  # Sure...
                    keep_track = False
                    previous_mus = None


            if t * iota <= precision:
                break
            C = (1 - t) * C + t * cov_matrix[delta_z]
    else:
        while True:
            norm_mu_squared = np.dot(mu, C)  # Calculating <mu, mu>
            phi_mu = np.dot(mu, psi)  # Calculating <psi, mu>
            iota = (psi - C) - (phi_mu - norm_mu_squared)
            norm_delta_minus_mu_squared = cov_diag - 2 * C + norm_mu_squared
            with np.errstate(divide='ignore', invalid='ignore'):
                max_out = np.sign(iota)*np.true_divide((np.square(iota)), 2*norm_delta_minus_mu_squared)
                max_out[max_out == np.inf] = 0
                max_out = np.nan_to_num(max_out)
            delta_z = np.argmax(max_out)
            iota = iota[delta_z]
            if iota <= 0:
                break
            if math.isclose(0, norm_delta_minus_mu_squared[delta_z]):
                raise ZeroDivisionError("norm_delta_minus_mu_squared is too close to 0\n mu:", mu, "\n z=", delta_z, "\nnorm_delta_minus_mu_squared=",norm_delta_minus_mu_squared[delta_z])

            t = iota / norm_delta_minus_mu_squared[delta_z]

            if positive_weights_only:
                if t > 1:
                    t = 1
                elif t < 0:
                    t = 0

            mu = (1 - t) * mu
            mu[delta_z] += t
            if keep_track:
                if memory_usage_of_previous_mus < 2e+10:
                    previous_mus.append(np.copy(mu))
                    memory_usage_of_previous_mus = sys.getsizeof(previous_mus)
                else:
                    print(memory_usage_of_previous_mus)

            if t * iota <= precision:
                break
            C = (1 - t) * C + t * cov_matrix[delta_z]
    if not keep_track:
        previous_mus = None
    return mu, previous_mus

def topiary_effecient_frontier(mean_returns, cov_matrix, T, lambda_range, precision, starting_weight='max',
                               positive_weights_only=True, topiaric_efficient_frontier_granularity=1000):
    efficient_means = []
    efficient_vols = []
    T_root = math.sqrt(T)

    start_mu, _ = topiary_main_algorithm(mean_returns=mean_returns, cov_matrix=cov_matrix, starting_weight=starting_weight,
                                   precision=precision, positive_weights_only=positive_weights_only,
                                   const_for_eta=lambda_range[1], keep_track=False)
    mu = start_mu.copy()
    step = 0
    while (start_mu == mu).all():
        step += .1
        mu, _ = topiary_main_algorithm(mean_returns=mean_returns, cov_matrix=cov_matrix, starting_weight=starting_weight,
                                   precision=precision, positive_weights_only=positive_weights_only,
                                   const_for_eta=lambda_range[1]-step, keep_track=False)
        if lambda_range[1] - step < lambda_range[0]:
            raise Exception("ERROR")
    largest_lambda = lambda_range[1]-step + .1

    start_mu, _ = topiary_main_algorithm(mean_returns=mean_returns, cov_matrix=cov_matrix,
                                         starting_weight=starting_weight,
                                         precision=precision, positive_weights_only=positive_weights_only,
                                         const_for_eta=lambda_range[0], keep_track=False)
    mu = start_mu.copy()
    step = 0
    while (start_mu == mu).all():
        step += .1
        mu, _ = topiary_main_algorithm(mean_returns=mean_returns, cov_matrix=cov_matrix, starting_weight=starting_weight,
                                    precision=precision, positive_weights_only=positive_weights_only,
                                    const_for_eta=lambda_range[0] + step, keep_track=False)
        if lambda_range[0] + step > lambda_range[1]:
            raise Exception("ERROR")
    smallest_lambda = lambda_range[0] + step - .1


    consts_for_etas = np.linspace(smallest_lambda, largest_lambda, topiaric_efficient_frontier_granularity+1)
    for i in range(len(consts_for_etas)):
        mu, _ = topiary_main_algorithm(mean_returns=mean_returns, cov_matrix=cov_matrix, starting_weight=starting_weight,
                                       precision=precision, positive_weights_only=positive_weights_only,
                                       const_for_eta=consts_for_etas[i], keep_track=False)
        efficient_means.append(portfolio_expected_return(mu, mean_returns, T))
        efficient_vols.append(portfolio_volatility(mu, cov_matrix, T_root))
    return efficient_means, efficient_vols

def topiary(data, mean_returns, cov_matrix, T, precision, positive_weights_only=False, starting_weight=None,
            const_for_eta=1, number_of_decimals=2, silence_prints=False, topiaric_efficient_frontier_granularity=1000,
            efficient_frontier_polynomial_approximation_degree=2):
    if not silence_prints:
        print("Getting scipy-topiaric portfolio..." )
    tm = time.time()
    T_root = math.sqrt(T)
    max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta, precision)
    max_eta_volatility = portfolio_volatility(max_eta_scipy, cov_matrix, T_root)
    max_eta_expected_return = portfolio_expected_return(max_eta_scipy, mean_returns, T)
    max_eta_sharpe_ratio = sharpe_ratio(max_eta_scipy, mean_returns, cov_matrix, T, T_root)
    max_eta_eta = -negative_eta(max_eta_scipy, mean_returns, cov_matrix, const_for_eta)
    max_eta_allocation = pd.DataFrame(max_eta_scipy, index=data.columns,
                                      columns=['allocation'])  # Convert to pandas dataframe for better printing
    max_eta_allocation.allocation = [round(i * 100, number_of_decimals - 2) for i in max_eta_scipy]
    max_eta_allocation = max_eta_allocation.T
    if not silence_prints:
        print("Algorithm done! This took", time.time() - tm, "seconds.")
        print("Running Topiary Algorithm...")

    tm = time.time()
    mu, previous_mus = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight, precision, positive_weights_only, const_for_eta)
    topiary_volatility = portfolio_volatility(mu, cov_matrix, T_root)
    topiary_expected_return = portfolio_expected_return(mu, mean_returns, T)
    topiary_sharpe_ratio = sharpe_ratio(mu, mean_returns, cov_matrix, T, T_root)
    topiary_eta = -negative_eta(mu, mean_returns, cov_matrix, const_for_eta)
    path_taken_info = np.zeros((3, len(previous_mus)))
    for i in range(len(previous_mus)):
        weights = previous_mus[i]
        portfolio_std_dev = portfolio_volatility(weights, cov_matrix, T_root)
        portfolio_return = portfolio_expected_return(weights, mean_returns, T)
        path_taken_info[0, i] = portfolio_std_dev  # Standard deviation
        path_taken_info[1, i] = portfolio_return  # expected return
        path_taken_info[2, i] = (portfolio_return - RISK_FREE_INTEREST_RATE) / portfolio_std_dev
    topiary_allocation = pd.DataFrame(mu, index=data.columns,
                           columns=['allocation'])  # Convert to pandas dataframe for better printing
    topiary_allocation.allocation = [round(i * 100, number_of_decimals - 2) for i in mu]
    topiary_allocation = topiary_allocation.T

    if not silence_prints:
        print("Algorithm done! This took", time.time() - tm, "seconds.")
        print("Getting Topiaric efficient frontier...")

    tm = time.time()
    topiary_efficient_means, topiary_efficient_vols = topiary_effecient_frontier(mean_returns, cov_matrix, T, (-1, 1),
                                                                                 precision, starting_weight, positive_weights_only,
                                                                                 topiaric_efficient_frontier_granularity)
    boundary_poly_coefs = np.polynomial.polynomial.polyfit(topiary_efficient_means, topiary_efficient_vols,
                                                           efficient_frontier_polynomial_approximation_degree)
    if not silence_prints:
        print("Efficients found! This took", time.time() - tm, "seconds.")


    return ((topiary_allocation, topiary_expected_return, topiary_volatility, topiary_sharpe_ratio, topiary_eta),
            (max_eta_allocation, max_eta_expected_return, max_eta_volatility, max_eta_sharpe_ratio, max_eta_eta),
            (topiary_efficient_means, topiary_efficient_vols), boundary_poly_coefs, path_taken_info)
