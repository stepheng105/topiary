from stock_simulation.simulate_stocks import simulate_stocks
from plotting.plot_efficient_frontier import plot_efficient_frontier
from modern_portfolio_theory.real_data_helpers import get_data
from consts.constants import RISK_FREE_INTEREST_RATE, TIME_PERIOD_IN_YEARS


if __name__ == '__main__':
    T = TIME_PERIOD_IN_YEARS
    data, _, _, _, _ = simulate_stocks(10, T, mu_range=(-.1, 2), sigma_range=(.1, 1), S0_range=(5, 100)) # Simulated data
    # data = get_data((1, 1, 2001), (1, 1, 2002))   # Real world data
    plot_efficient_frontier(data, T, 100000, empirical_MPT=True, analytic_MPT=True, num_decimals=4,
                            tolerance=10e-16, efficient_frontier_granularity=100, const_for_topiary=1,
                            positive_weights_only_for_topiary=True, too_high_rate=RISK_FREE_INTEREST_RATE * 4,
                            starting_weight_for_topiary='max',
                            effecient_frontier_polynomial_approximation_degree=5)
