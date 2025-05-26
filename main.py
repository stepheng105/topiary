from stock_simulation.simulate_stocks import simulate_stocks
from plotting.plot_efficient_frontier import plot_efficient_frontier
from plotting.plot_stock_data import plot_stock_data
from modern_portfolio_theory.real_data_helpers import get_data
from consts.constants import RISK_FREE_INTEREST_RATE, TIME_PERIOD_IN_YEARS
from Real_Hardy_Space.compute_real_hardy_space import compute_real_hardy_space
from modern_portfolio_theory.CAPM import display_CAPM


if __name__ == '__main__':
    T = 10
    # data, lognormal_data, drift_mus, std_dev_sigmas, S0s = simulate_stocks(1, T, mu_range=(.6, .6), sigma_range=(.5, .5), S0_range=(5, 100)) # Simulated data
    # plot_stock_data(lognormal_data, T, 1/252, S0s, drift_mus, std_dev_sigmas, lognormal=True)
    # plot_stock_data(data, T, 1 / 252, S0s, drift_mus, std_dev_sigmas, lognormal=False)
    # data = get_data('2021-01-01', '2022-01-01')   # Real world data
    # I = [0, 1, 3]

    # display_CAPM(data, I, 1e-15, .04)
    # plot_efficient_frontier(data, T, 100000, empirical_MPT=False, analytic_MPT=True, num_decimals=4,
    #                          tolerance=1e-16, efficient_frontier_granularity=100, topiaric_efficient_frontier_granularity=10000,
    #                         const_for_topiary=.025,
    #                         positive_weights_only_for_topiary=True, too_high_rate=None,
    #                         starting_weight_for_topiary='max',
    #                         efficient_frontier_polynomial_approximation_degree=3)
    compute_real_hardy_space(num_points=35000, num_points_fill=150000, precision=1e-30, region_type='shifted_donut', compute_scipy=False, create_gif=False)
