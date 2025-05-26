import numpy as np
import math
import time
from stock_simulation.simulate_stocks import simulate_stocks
from consts.constants import TIME_PERIOD_IN_YEARS
from modern_portfolio_theory.topiary import topiary_main_algorithm
from modern_portfolio_theory.empirical_solution_helpers import max_eta, negative_eta
from modern_portfolio_theory.real_data_helpers import get_data
import random

def compare_times(lambdas, precisions):
    data = get_data('2020-01-01', '2021-01-01')
    T = round(252 * TIME_PERIOD_IN_YEARS)
    data = np.log(data).dropna()
    returns = data.pct_change()  # Rate of returns for the stocks
    mean_returns = np.array(returns.mean())  # Mean rate of returns
    Cov_matrix = np.array(returns.cov())  # Covariance matrix
    for const_for_topiary in lambdas:
        for p in precisions:
            times_topiary = [0]*10
            times_scipy = [0]*10
            for i in range(10):
                t = time.time()
                mu, _ = topiary_main_algorithm(mean_returns, Cov_matrix, 'max', p, True,
                                    const_for_topiary, keep_track=False, traditional_alg=True)
                times_topiary[i] = time.time() - t
                t = time.time()
                nu = max_eta(mean_returns, Cov_matrix, const_for_topiary, p)
                times_scipy[i] = time.time() - t
            print("For " + str(len(mean_returns)) + " assets, lambda = " + str(const_for_topiary) + ", and precision = " + str(p) + ", here are the results:")
            print()
            print("Topiary took on average: " + str(np.mean(times_topiary)))
            print("Topiary eta: " + str(-negative_eta(mu, mean_returns, Cov_matrix, const_for_topiary)))
            print("Topiary number of assets: " + str(len(np.nonzero(mu)[0])))
            print()
            print("Scipy took on average: " + str(np.mean(times_scipy)))
            print("Scipy eta: " + str(-negative_eta(nu, mean_returns, Cov_matrix, const_for_topiary)))
            print("Scipy number of assets: " + str(len(np.nonzero(~np.isclose(nu, np.zeros_like(nu)))[0])))
            print("-"*30)
    for const_for_topiary in lambdas:
        for p in precisions:
            times_topiary = [0]*100
            times_scipy = [0]*100
            for i in range(100):
                sample = random.sample(range(len(mean_returns)), 100)
                psi = mean_returns[sample]  # Mean rate of returns
                C = Cov_matrix[np.ix_(sample, sample)]
                t = time.time()
                mu, _ = topiary_main_algorithm(psi, C, 'max', p, True,
                                    const_for_topiary, keep_track=False, traditional_alg=True)
                times_topiary[i] = time.time() - t
                t = time.time()
                nu = max_eta(psi, C, const_for_topiary, p)
                times_scipy[i] = time.time() - t
            print("For " + str(len(psi)) + " assets, lambda = " + str(const_for_topiary) + ", and precision = " + str(p) + ", here are the results:")
            print()
            print("Topiary took on average: " + str(np.mean(times_topiary)))
            print("Topiary eta: " + str(-negative_eta(mu, psi, C, const_for_topiary)))
            print("Topiary number of assets: " + str(len(np.nonzero(mu)[0])))
            print()
            print("Scipy took on average: " + str(np.mean(times_scipy)))
            print("Scipy eta: " + str(-negative_eta(nu, psi, C, const_for_topiary)))
            print("Scipy number of assets: " + str(len(np.nonzero(~np.isclose(nu, np.zeros_like(nu)))[0])))
            print("-"*30)

    for const_for_topiary in lambdas:
        for p in precisions:
            times_topiary = [0] * 100
            times_scipy = [0] * 100
            for i in range(100):
                sample = random.sample(range(len(mean_returns)), 10)
                psi = mean_returns[sample]  # Mean rate of returns
                C = Cov_matrix[np.ix_(sample, sample)]
                t = time.time()
                mu, _ = topiary_main_algorithm(psi, C, 'max', p, True,
                                               const_for_topiary, keep_track=False, traditional_alg=True)
                times_topiary[i] = time.time() - t
                t = time.time()
                nu = max_eta(psi, C, const_for_topiary, p)
                times_scipy[i] = time.time() - t
            print("For " + str(len(psi)) + " assets, lambda = " + str(const_for_topiary) + ", and precision = " + str(
                p) + ", here are the results:")
            print()
            print("Topiary took on average: " + str(np.mean(times_topiary)))
            print("Topiary eta: " + str(-negative_eta(mu, psi, C, const_for_topiary)))
            print("Topiary number of assets: " + str(len(np.nonzero(mu)[0])))
            print()
            print("Scipy took on average: " + str(np.mean(times_scipy)))
            print("Scipy eta: " + str(-negative_eta(nu, psi, C, const_for_topiary)))
            print("Scipy number of assets: " + str(len(np.nonzero(~np.isclose(nu, np.zeros_like(nu)))[0])))
            print("-" * 30)

if __name__ == "__main__":
    compare_times([1, .4, .1, .03, 0], [1e-15])