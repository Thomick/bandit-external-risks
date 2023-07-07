# Illustrative plot for the concept of risk in rl

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


value_range = -5, 5
nb_points = 1000
x = np.linspace(value_range[0], value_range[1], nb_points)


def compute_asymetric_left_density(x, mu, alpha):
    return 0 if x > mu else alpha * np.exp((x - mu) / alpha)


def compute_asymetric_right_density(x, mu, alpha):
    return 0 if x < mu else alpha * np.exp(-(x - mu) / alpha)


def compute_gaussian_density(x, mu, sigma):
    return (
        1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )


test1 = np.array([compute_asymetric_left_density(xx, 1.13, 1) for xx in x])
test2 = np.array([compute_gaussian_density(xx, 0.12, 0.25) for xx in x])
test3 = np.array([compute_asymetric_right_density(xx, 0, 0.1) for xx in x])
test1 = test1 / np.sum(test1) * nb_points / (value_range[1] - value_range[0])
test2 = test2 / np.sum(test2) * nb_points / (value_range[1] - value_range[0])
test3 = test3 / np.sum(test3) * nb_points / (value_range[1] - value_range[0])


sns.lineplot(x=x, y=test1, label="Reward 1")
sns.lineplot(x=x, y=test2, label="Reward 2")
sns.lineplot(x=x, y=test3, label="Reward 3")
plt.fill_between(x, test1, alpha=0.2)
plt.fill_between(x, test2, alpha=0.2)
plt.fill_between(x, test3, alpha=0.2)


plt.gca().set_prop_cycle(None)
plt.plot([np.average(x, weights=test1)] * 2, [0, 3], linestyle="--")
plt.plot([np.average(x, weights=test2)] * 2, [0, 3], linestyle="--")
plt.plot([np.average(x, weights=test3)] * 2, [0, 3], linestyle="--")
plt.ylim(0, 2.5)
plt.xlim(-0.9, 1.3)
plt.title("Reward probability density function for 3 strategies")
plt.legend()
plt.tight_layout()
plt.show()
