import pandas as pd
import numpy as np
import math


# Construct simulated data
# Using geometric brownian motion to model
# Brownian motion is the limit of random walk (Weiner process) with mean 0 and variance 1
# dS = mu*S*dt + sigma*S*dB
# See https://en.wikipedia.org/wiki/Geometric_Brownian_motion#Solving_the_SDE for information on solution to diffeq
def simulate_stocks(num_stocks, T, mu_range=(-.1, 2), sigma_range=(.1, 1), S0_range=(5, 100)):
    drift_mus = np.random.uniform(mu_range[0], mu_range[1],
                                  size=num_stocks)  # Randomly assigning different stocks different drift values (mu),
    std_dev_sigmas = np.random.uniform(sigma_range[0], sigma_range[1],
                                       size=num_stocks)  # volatilities/standard deviations (sigma),
    S0s = np.random.randint(S0_range[0], S0_range[1], size=num_stocks)  # and starting prices (S0).
    num_steps = round(252 * T)
    dt = T / num_steps
    inv_sqrt = math.sqrt(dt)

    df = pd.DataFrame(
        np.random.randint(0, 2,
                          size=(num_steps, num_stocks))) * 2 - 1  # Random numbers, either -1 or 1 (mean 0 varience 1)
    data = pd.DataFrame(np.zeros((num_steps, num_stocks)))
    lognormal_data = pd.DataFrame(np.zeros((num_steps, num_stocks)))
    for i in range(df.shape[1]):
        for j in range(df.shape[0]):
            # Function S(t) for price of an asset
            # Sum and multiply by 1/sqrt(n) to get weiner process
            data[i][j] = S0s[i] * np.exp(
                (drift_mus[i] - .5 * (std_dev_sigmas[i] * std_dev_sigmas[i])) * j * dt + inv_sqrt * std_dev_sigmas[i] *
                df[i][:j].sum())
            lognormal_data[i][j] = (drift_mus[i] - .5 * (std_dev_sigmas[i] * std_dev_sigmas[i])) * j * dt + inv_sqrt * \
                                   std_dev_sigmas[i] * df[i][:j].sum()

    return data, lognormal_data, drift_mus, std_dev_sigmas, S0s
