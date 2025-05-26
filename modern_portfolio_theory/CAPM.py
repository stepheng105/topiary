import math
import numpy as np
from modern_portfolio_theory.topiary import topiary_main_algorithm
import matplotlib.pyplot as plt

def display_CAPM(data, wanted_indices, precision, const_for_eta):
    data = np.log(data).dropna()
    returns = data.pct_change()  # Rate of returns for the stocks
    mean_returns = const_for_eta*np.array(returns.mean())  # Mean rate of returns
    cov_matrix = np.array(returns.cov())  # Covariance matrix

    mean_returns_wanted = mean_returns[wanted_indices]
    cov_matrix_wanted = cov_matrix[np.ix_(wanted_indices, wanted_indices)]
    temp_mu, _ = topiary_main_algorithm(mean_returns_wanted, cov_matrix_wanted, 'max', precision, True,
                                   1, False)
    mu = np.zeros_like(mean_returns)
    mu[wanted_indices] = temp_mu
    r_k = np.dot(mean_returns, mu) - np.dot(mu, np.dot(cov_matrix, mu))

    psi_of_x_topiaric_index = []
    topiary_of_x_topiaric_index = []
    nonzero_indices = mu.nonzero()

    for i in range(len(nonzero_indices)):
        psi_of_x_topiaric_index.append(mean_returns[nonzero_indices[i]])
        topiary_of_x_topiaric_index.append(np.dot(mu, cov_matrix[:, nonzero_indices[i]]))

    psi_of_x_not_topiaric_index = mean_returns_wanted.copy()
    topiary_of_x_not_topiaric_index = []

    for i in range(len(mean_returns_wanted)):
        topiary_of_x_not_topiaric_index.append(np.dot(temp_mu, cov_matrix_wanted[:, i]))

    psi_of_x_not_wanted = []
    topiary_of_x_not_wanted = []
    for i in range(len(mean_returns)):
        psi_of_x_not_wanted.append(mean_returns[i])
        topiary_of_x_not_wanted.append(np.dot(mu, cov_matrix[:, i]))

    fig, ax = plt.subplots()

    ax.scatter(psi_of_x_not_wanted, topiary_of_x_not_wanted, color='purple', s=20)
    ax.scatter(psi_of_x_not_topiaric_index, topiary_of_x_not_topiaric_index, color='brown', s=20)
    ax.scatter(psi_of_x_topiaric_index, topiary_of_x_topiaric_index, color='red', s=20)
    for i, txt in enumerate(data.columns):
        plt.annotate(txt, (psi_of_x_not_wanted[i], topiary_of_x_not_wanted[i]), xytext=(10, 0),
                     textcoords='offset points')

    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = x_vals - r_k
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.plot(x_vals, y_vals, '--')
    plt.scatter(r_k, 0, color='black', s=20)
    plt.annotate(r'$r_K$', (r_k, 0), xytext=(5, -8), textcoords='offset points')
    plt.title('Payout vs Covariance Against Topiaric Portfolio')
    plt.xlabel(r'$ \lambda \psi(x)$ (Payout)')
    plt.ylabel(r'$\mu(x)$ (Covariance against $\mu$)')

    plt.savefig('CAPM.png', dpi=300)
