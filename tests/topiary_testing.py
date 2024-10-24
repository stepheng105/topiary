import unittest
import time
import numpy as np
import math
from modern_portfolio_theory.topiary import topiary_main_algorithm
from stock_simulation.simulate_stocks import simulate_stocks
from modern_portfolio_theory.empirical_solution_helpers import negative_eta
from modern_portfolio_theory.empirical_solution_helpers import max_eta
from modern_portfolio_theory.return_and_volatility import (portfolio_expected_return, portfolio_volatility,
                                                           sharpe_ratio_given_ret_and_vol)


# WARNING: These tests take a while. I ran them over a week. I also only ran them on my own personal computer, so
# if they fail on your machine it may be a hardware difference. If you wish to run these tests yourself, I recommend
# running them individually and/or reducing the number of stocks/iterations.
class TestTopiary(unittest.TestCase):

    def test_convergence(self):
        num_stocks = [20, 1, 100, 500, 2000]
        timeouts = [2, .1, 5, 10, 30]
        test_number = 1
        iteration_number = 0

        for i in range(len(num_stocks)):
            iteration_number += 1
            T = 1
            data, _, _, _, _ = simulate_stocks(num_stocks[i], T, mu_range=(-.1, 2), sigma_range=(.1, 1),
                                               S0_range=(5, 100))
            returns = data.pct_change()  # Rate of returns for the stocks
            mean_returns = np.array(returns.mean())  # Mean rate of returns
            cov_matrix = np.array(returns.cov())  # Covariance matrix
            precision = 10e-5
            timeout_time = timeouts[i]
            target = np.linspace(-2, 2, 501)

            print("test", str(test_number) + "." + str(iteration_number) + ".1")
            for l in target:
                t = time.time()
                topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                       positive_weights_only=False, const_for_eta=l)
                self.assertTrue(time.time() - t < timeout_time)
            print("test", str(test_number) + "." + str(iteration_number) + ".2")
            for l in target:
                t = time.time()
                topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                       positive_weights_only=True, const_for_eta=l)
                t_final = time.time() - t
                self.assertTrue(t_final < timeout_time)
            print("test", str(test_number) + "." + str(iteration_number) + ".3")
            for l in target:
                t = time.time()
                topiary_main_algorithm(mean_returns, cov_matrix, starting_weight='random', precision=precision,
                                       positive_weights_only=False, const_for_eta=l)
                t_final = time.time() - t
                self.assertTrue(t_final < timeout_time)
            print("test", str(test_number) + "." + str(iteration_number) + ".4")
            for l in target:
                t = time.time()
                topiary_main_algorithm(mean_returns, cov_matrix, starting_weight='max', precision=precision,
                                       positive_weights_only=False, const_for_eta=l)
                t_final = time.time() - t
                self.assertTrue(t_final < timeout_time)
            print("test", str(test_number) + "." + str(iteration_number) + ".5")
            precision = 10e-15
            for l in target:
                t = time.time()
                topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                       positive_weights_only=False, const_for_eta=l)
                t_final = time.time() - t
                self.assertTrue(t_final < timeout_time)

    def test_topiary_against_scipy(self):
        num_stocks = [20, 1, 100]
        test_number = 2
        iteration_number = 0
        for i in range(len(num_stocks)):
            iteration_number += 1
            T = 1
            data, _, _, _, _ = simulate_stocks(num_stocks[i], T, mu_range=(-.1, 2), sigma_range=(.1, 1),
                                               S0_range=(5, 100))
            returns = data.pct_change()  # Rate of returns for the stocks
            mean_returns = np.array(returns.mean())  # Mean rate of returns
            cov_matrix = np.array(returns.cov())  # Covariance matrix
            precision = 10e-10
            target = np.linspace(-3, 5, 501)
            n = 3
            print("test", str(test_number) + "." + str(iteration_number) + ".1")
            for l in target:
                mu, _ = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                               positive_weights_only=True, const_for_eta=l)
                max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)

                eta_top = -negative_eta(mu, mean_returns, cov_matrix, l)
                eta_sci = -negative_eta(max_eta_scipy, mean_returns, cov_matrix, l)
                if eta_top < eta_sci:
                    self.assertAlmostEqual(eta_top, eta_sci, n)
            print("test", str(test_number) + "." + str(iteration_number) + ".2")
            for l in target:
                mu, _ = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                               positive_weights_only=False, const_for_eta=l)
                max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)

                eta_top = -negative_eta(mu, mean_returns, cov_matrix, l)
                eta_sci = -negative_eta(max_eta_scipy, mean_returns, cov_matrix, l)
                if eta_top < eta_sci:
                    self.assertAlmostEqual(eta_top, eta_sci, n)
            print("test", str(test_number) + "." + str(iteration_number) + ".3")
            for l in target:
                mu, _ = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight='max', precision=precision,
                                               positive_weights_only=True, const_for_eta=l)
                max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)

                eta_top = -negative_eta(mu, mean_returns, cov_matrix, l)
                eta_sci = -negative_eta(max_eta_scipy, mean_returns, cov_matrix, l)
                if eta_top < eta_sci:
                    self.assertAlmostEqual(eta_top, eta_sci, n)
            print("test", str(test_number) + "." + str(iteration_number) + ".4")
            for l in target:
                mu, _ = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight='random', precision=precision,
                                               positive_weights_only=True, const_for_eta=l)
                max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)

                eta_top = -negative_eta(mu, mean_returns, cov_matrix, l)
                eta_sci = -negative_eta(max_eta_scipy, mean_returns, cov_matrix, l)
                if eta_top < eta_sci:
                    self.assertAlmostEqual(eta_top, eta_sci, n)
            print("test", str(test_number) + "." + str(iteration_number) + ".5")
            precision = 10e-20
            n = 4
            for l in target:
                mu, _ = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                               positive_weights_only=True, const_for_eta=l)
                max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)

                eta_top = -negative_eta(mu, mean_returns, cov_matrix, l)
                eta_sci = -negative_eta(max_eta_scipy, mean_returns, cov_matrix, l)
                if eta_top < eta_sci:
                    self.assertAlmostEqual(eta_top, eta_sci, n)

    def test_topiary_outputs_against_scipy_outputs(self):
        num_stocks = [20, 1, 100, 500, 1000]
        test_number = 3
        iteration_number = 0
        for i in range(len(num_stocks)):
            iteration_number += 1
            T = 1
            T_root = math.sqrt(T)
            data, _, _, _, _ = simulate_stocks(num_stocks[i], T, mu_range=(0, 2), sigma_range=(0, 1),
                                               S0_range=(0, 1000))
            returns = data.pct_change()  # Rate of returns for the stocks
            mean_returns = np.array(returns.mean())  # Mean rate of returns
            cov_matrix = np.array(returns.cov())  # Covariance matrix
            precision = 10e-15
            target = np.linspace(0, 5, 501)
            n = 3
            print("test", str(test_number) + "." + str(iteration_number) + ".1")
            for l in target:
                mu, _ = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                               positive_weights_only=True, const_for_eta=l)
                max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)
                topiary_expected_return = portfolio_expected_return(mu, mean_returns, T)
                max_eta_expected_return = portfolio_expected_return(max_eta_scipy, mean_returns, T)
                topiary_volatility = portfolio_volatility(mu, cov_matrix, T_root)
                max_eta_volatility = portfolio_volatility(max_eta_scipy, cov_matrix, T_root)
                topiary_sharpe_ratio = sharpe_ratio_given_ret_and_vol(topiary_expected_return, topiary_volatility)
                max_eta_sharpe_ratio = sharpe_ratio_given_ret_and_vol(max_eta_expected_return, max_eta_volatility)

                if topiary_expected_return < max_eta_expected_return:
                    self.assertAlmostEqual(topiary_expected_return, max_eta_expected_return, n)
                if topiary_volatility > max_eta_volatility:
                    self.assertAlmostEqual(topiary_volatility, max_eta_volatility, n)
                if topiary_sharpe_ratio < max_eta_sharpe_ratio:
                    self.assertAlmostEqual(topiary_sharpe_ratio, max_eta_sharpe_ratio, n)
            print("test", str(test_number) + "." + str(iteration_number) + ".2")
            for l in target:
                mu, _ = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                               positive_weights_only=False, const_for_eta=l)
                max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)
                topiary_expected_return = portfolio_expected_return(mu, mean_returns, T)
                max_eta_expected_return = portfolio_expected_return(max_eta_scipy, mean_returns, T)
                topiary_volatility = portfolio_volatility(mu, cov_matrix, T_root)
                max_eta_volatility = portfolio_volatility(max_eta_scipy, cov_matrix, T_root)
                topiary_sharpe_ratio = sharpe_ratio_given_ret_and_vol(topiary_expected_return, topiary_volatility)
                max_eta_sharpe_ratio = sharpe_ratio_given_ret_and_vol(max_eta_expected_return, max_eta_volatility)
                if topiary_expected_return < max_eta_expected_return:
                    self.assertAlmostEqual(topiary_expected_return, max_eta_expected_return, n)
                if topiary_volatility > max_eta_volatility:
                    self.assertAlmostEqual(topiary_volatility, max_eta_volatility, n)
                if topiary_sharpe_ratio < max_eta_sharpe_ratio:
                    self.assertAlmostEqual(topiary_sharpe_ratio, max_eta_sharpe_ratio, n)
            print("test", str(test_number) + "." + str(iteration_number) + ".3")
            for l in target:
                mu, _ = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight='max', precision=precision,
                                               positive_weights_only=True, const_for_eta=l)
                max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)
                topiary_expected_return = portfolio_expected_return(mu, mean_returns, T)
                max_eta_expected_return = portfolio_expected_return(max_eta_scipy, mean_returns, T)
                topiary_volatility = portfolio_volatility(mu, cov_matrix, T_root)
                max_eta_volatility = portfolio_volatility(max_eta_scipy, cov_matrix, T_root)
                topiary_sharpe_ratio = sharpe_ratio_given_ret_and_vol(topiary_expected_return, topiary_volatility)
                max_eta_sharpe_ratio = sharpe_ratio_given_ret_and_vol(max_eta_expected_return, max_eta_volatility)

                if topiary_expected_return < max_eta_expected_return:
                    self.assertAlmostEqual(topiary_expected_return, max_eta_expected_return, n)
                if topiary_volatility > max_eta_volatility:
                    self.assertAlmostEqual(topiary_volatility, max_eta_volatility, n)
                if topiary_sharpe_ratio < max_eta_sharpe_ratio:
                    self.assertAlmostEqual(topiary_sharpe_ratio, max_eta_sharpe_ratio, n)
            print("test", str(test_number) + "." + str(iteration_number) + ".4")
            for l in target:
                mu, _ = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight='random', precision=precision,
                                               positive_weights_only=True, const_for_eta=l)
                max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)
                topiary_expected_return = portfolio_expected_return(mu, mean_returns, T)
                max_eta_expected_return = portfolio_expected_return(max_eta_scipy, mean_returns, T)
                topiary_volatility = portfolio_volatility(mu, cov_matrix, T_root)
                max_eta_volatility = portfolio_volatility(max_eta_scipy, cov_matrix, T_root)
                topiary_sharpe_ratio = sharpe_ratio_given_ret_and_vol(topiary_expected_return, topiary_volatility)
                max_eta_sharpe_ratio = sharpe_ratio_given_ret_and_vol(max_eta_expected_return, max_eta_volatility)

                if topiary_expected_return < max_eta_expected_return:
                    self.assertAlmostEqual(topiary_expected_return, max_eta_expected_return, n)
                if topiary_volatility > max_eta_volatility:
                    self.assertAlmostEqual(topiary_volatility, max_eta_volatility, n)
                if topiary_sharpe_ratio < max_eta_sharpe_ratio:
                    self.assertAlmostEqual(topiary_sharpe_ratio, max_eta_sharpe_ratio, n)
            precision = 10e-30
            n = 2
            print("test", str(test_number) + "." + str(iteration_number) + ".5")
            for l in target:
                mu, _ = topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                               positive_weights_only=True, const_for_eta=l)
                max_eta_scipy = max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)
                topiary_expected_return = portfolio_expected_return(mu, mean_returns, T)
                max_eta_expected_return = portfolio_expected_return(max_eta_scipy, mean_returns, T)
                topiary_volatility = portfolio_volatility(mu, cov_matrix, T_root)
                max_eta_volatility = portfolio_volatility(max_eta_scipy, cov_matrix, T_root)
                topiary_sharpe_ratio = sharpe_ratio_given_ret_and_vol(topiary_expected_return, topiary_volatility)
                max_eta_sharpe_ratio = sharpe_ratio_given_ret_and_vol(max_eta_expected_return, max_eta_volatility)

                if topiary_expected_return < max_eta_expected_return:
                    self.assertAlmostEqual(topiary_expected_return, max_eta_expected_return, n)
                if topiary_volatility > max_eta_volatility:
                    self.assertAlmostEqual(topiary_volatility, max_eta_volatility, n)
                if topiary_sharpe_ratio < max_eta_sharpe_ratio:
                    self.assertAlmostEqual(topiary_sharpe_ratio, max_eta_sharpe_ratio, n)

    def test_topiary_time_against_scipy_time(self):
        num_stocks = [20, 1, 100, 500, 1000]
        test_number = 4
        iteration_number = 0
        for i in range(len(num_stocks)):
            iteration_number += 1
            T = 1
            data, _, _, _, _ = simulate_stocks(num_stocks[i], T, mu_range=(0, 2), sigma_range=(0, 1),
                                               S0_range=(0, 1000))
            returns = data.pct_change()  # Rate of returns for the stocks
            mean_returns = np.array(returns.mean())  # Mean rate of returns
            cov_matrix = np.array(returns.cov())  # Covariance matrix
            precision = 10e-15
            target = np.linspace(-2, 5, 501)
            print("test", str(test_number) + "." + str(iteration_number) + ".1")
            faster = [0, 0]
            sums = [0, 0]
            faster_scaler = .5
            for l in target:
                t1 = time.time()
                topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                               positive_weights_only=True, const_for_eta=l)
                t1_diff = time.time() - t1
                t2 = time.time()
                max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)
                t2_diff = time.time() - t2
                if t1_diff < t2_diff:
                    faster[0] += 1
                else:
                    faster[1] += 1
                sums[0] += t1_diff
                sums[1] += t2_diff

            self.assertLess(faster[1], faster_scaler*faster[0])
            # self.assertLess(sums[0], sums[1])

            print("test", str(test_number) + "." + str(iteration_number) + ".2")
            for l in target:
                t1 = time.time()
                topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                       positive_weights_only=False, const_for_eta=l)
                t1_diff = time.time() - t1
                t2 = time.time()
                max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)
                t2_diff = time.time() - t2

                if t1_diff < t2_diff:
                    faster[0] += 1
                else:
                    faster[1] += 1
                sums[0] += t1_diff
                sums[1] += t2_diff

            self.assertLess(faster[1], faster_scaler*faster[0])
            # self.assertLess(sums[1], sums[0])

            print("test", str(test_number) + "." + str(iteration_number) + ".3")
            for l in target:
                t1 = time.time()
                topiary_main_algorithm(mean_returns, cov_matrix, starting_weight='max', precision=precision,
                                       positive_weights_only=True, const_for_eta=l)
                t1_diff = time.time() - t1
                t2 = time.time()
                max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)
                t2_diff = time.time() - t2

                if t1_diff < t2_diff:
                    faster[0] += 1
                else:
                    faster[1] += 1
                sums[0] += t1_diff
                sums[1] += t2_diff

            self.assertLess(faster[1], faster_scaler*faster[0])
            # self.assertLess(sums[1], sums[0])

            print("test", str(test_number) + "." + str(iteration_number) + ".4")
            for l in target:
                t1 = time.time()
                topiary_main_algorithm(mean_returns, cov_matrix, starting_weight='random', precision=precision,
                                       positive_weights_only=True, const_for_eta=l)
                t1_diff = time.time() - t1
                t2 = time.time()
                max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)
                t2_diff = time.time() - t2

                if t1_diff < t2_diff:
                    faster[0] += 1
                else:
                    faster[1] += 1
                sums[0] += t1_diff
                sums[1] += t2_diff

            self.assertLess(faster[1], faster_scaler*faster[0])
            # self.assertLess(sums[1], sums[0])

            precision = 10e-30
            print("test", str(test_number) + "." + str(iteration_number) + ".5")
            for l in target:
                t1 = time.time()
                topiary_main_algorithm(mean_returns, cov_matrix, starting_weight=None, precision=precision,
                                       positive_weights_only=True, const_for_eta=l)
                t1_diff = time.time() - t1
                t2 = time.time()
                max_eta(mean_returns, cov_matrix, const_for_eta=l, tolerance=precision)
                t2_diff = time.time() - t2

                if t1_diff < t2_diff:
                    faster[0] += 1
                else:
                    faster[1] += 1
                sums[0] += t1_diff
                sums[1] += t2_diff

            self.assertLess(faster[1], faster_scaler*faster[0])
            # self.assertLess(sums[1], sums[0])


if __name__ == '__main__':
    unittest.main()
