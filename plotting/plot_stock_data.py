import numpy as np
import matplotlib.pyplot as plt
from consts.constants import RISK_FREE_INTEREST_RATE


def plot_stock_data(data, T, dt, S0s, drift_mus, std_dev_sigmas, lognormal=False):

    for i, txt in enumerate(data.columns):
        S0, drift_mu, std_dev_sigma = S0s[i], drift_mus[i], std_dev_sigmas[i]
        t = np.arange(0, T, dt)
        y1 = data[i]
        if lognormal:
            y2 = RISK_FREE_INTEREST_RATE * t    # Risk-free interest rate
            y3 = (drift_mu - .5 * (std_dev_sigma * std_dev_sigma)) * t  # Drift
        else:
            y2 = S0 * np.exp(RISK_FREE_INTEREST_RATE * t)  # Risk-free interest rate
            y3 = S0 * np.exp(drift_mu * t)  # Drift

        plt.plot(t, y1)
        plt.plot(t, y2, color='red')
        plt.plot(t, y3, color='purple')
        plt.show()
        print("stock =", txt)
        print("mu =", drift_mus[i])
        print("sigma =", std_dev_sigmas[i])
        print("S0 = ", S0s[i])
