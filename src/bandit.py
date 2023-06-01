import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed


# Simple CLI progressbar that do not need a particular library
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()


# Parent class for bandits
# Arms always return the same values
#
# If the means are unspecified, they are chosen uniformly in the given range
class Bandit:
    def __init__(self, nb_arm, means=None):
        self.nb_arm = nb_arm
        self.means = means

    def pull(self, arm):
        return self.means[arm]

    def get_means(self):
        return self.means


# Bandit with Gaussian distributed rewards
# If unspecified, the standard deviations are set to 1 by default
class GaussianBandit(Bandit):
    def __init__(self, nb_arm, means=None, sigmas=None):
        super().__init__(nb_arm, means=means)
        if sigmas is not None:
            self.sigmas = sigmas
        else:
            self.sigmas = np.ones(nb_arm)

    def pull(self, arm):
        return np.random.randn() * self.sigmas[arm] + self.means[arm]


# Bandit with Bernoulli distributed rewards
class BernoulliBandit(Bandit):
    def __init__(self, nb_arm, means=None):
        super().__init__(nb_arm, means=means)

    def pull(self, arm):
        if np.random.rand() < self.means[arm]:
            return 1.0
        else:
            return 0.0


# Parent class for bandit algorithms
class BanditAlgorithm:
    name = "BanditAlgorithm"

    def __init__(self, bandit):
        self.bandit = bandit
        self.nb_arm = bandit.nb_arm
        self.t = 0
        self.total_rewards = np.zeros(self.nb_arm)
        self.pull_count = np.zeros(self.nb_arm)
        self.regrets = []

    def _update(self, arm, reward):
        self.t += 1
        self.total_rewards[arm] += reward
        self.pull_count[arm] += 1

    def run(self, T):
        for t in range(T):
            arm = self.select()
            reward = self.bandit.pull(arm)
            self._update(arm, reward)
            self.regrets.append(self._compute_regret(arm))
        return self.regrets

    def _compute_regret(self, arm):
        means = self.bandit.get_means()
        return means.max() - means[arm]

    def _compute_empirical_means(self):
        return self.total_rewards / np.maximum(self.pull_count, 1)

    def select(self):
        pass


# Epsilon-greedy algorithm
class EpsilonGreedy(BanditAlgorithm):
    name = "Epsilon Greedy"

    def __init__(self, bandit, epsilon=0.1):
        super().__init__(bandit)
        self.epsilon = epsilon

    def select(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nb_arm)
        else:
            return np.argmax(self._compute_empirical_means())


# UCB algorithm
class UCB1(BanditAlgorithm):
    name = "UCB1"

    def __init__(self, bandit):
        super().__init__(bandit)

    def select(self):
        if self.t < self.nb_arm:
            return self.t
        else:
            return np.argmax(
                self._compute_empirical_means()
                + np.sqrt(2 * np.log(self.t) / self.pull_count)
            )


# Thompson sampling algorithm for Bernoulli bandits
class BernoulliThompsonSampling(BanditAlgorithm):
    name = "Thompson Sampling"

    def __init__(self, bandit):
        super().__init__(bandit)
        self.alpha = np.ones(self.nb_arm)
        self.beta = np.ones(self.nb_arm)

    def _update(self, arm, reward):
        super()._update(arm, reward)
        if reward == 1.0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def select(self):
        return np.argmax(np.random.beta(self.alpha, self.beta))


# Thompson sampling algorithm for Gaussian bandits
class GaussianThompsonSampling(BanditAlgorithm):
    name = "Thompson Sampling"

    def __init__(self, bandit):
        super().__init__(bandit)
        self.mu = np.zeros(self.nb_arm)
        self.lambda_ = np.ones(self.nb_arm)

    def _update(self, arm, reward):
        super()._update(arm, reward)
        self.lambda_[arm] += 1
        self.mu[arm] = (self.mu[arm] * (self.lambda_[arm] - 1) + reward) / self.lambda_[
            arm
        ]

    def select(self):
        return np.argmax(np.random.randn(self.nb_arm) / np.sqrt(self.lambda_) + self.mu)


# Experiment class
class Experiment:
    def __init__(self):
        self.learners = []

    def add_learner(self, algorithm_class, algorithm_args={}, name=None):
        self.learners.append((algorithm_class, algorithm_args, name))

    def one_run(self, bandit_class, bandit_args, T, algorithm_class, algorithm_args):
        bandit = bandit_class(**bandit_args)
        algorithm = algorithm_class(bandit, **algorithm_args)
        regrets = algorithm.run(T)
        return regrets

    def run(self, bandit_class, bandit_args, T, nb_simu):
        regrets = []
        for algorithm_class, algorithm_args, name in progressbar(self.learners):
            regrets_algo = np.array(
                Parallel(n_jobs=12)(
                    delayed(self.one_run)(
                        bandit_class, bandit_args, T, algorithm_class, algorithm_args
                    )
                    for _ in range(nb_simu)
                )
            )
            regrets.append(regrets_algo)
        self.regrets = regrets
        return regrets

    def run_and_plot(
        self, bandit_class, bandit_args, T, nb_simu, plot_instant_regret=False
    ):
        regrets = self.run(bandit_class, bandit_args, T, nb_simu)
        for regrets_algo, (algorithm_class, algorithm_args, name) in zip(
            regrets, self.learners
        ):
            self.add_cumulative_plot(regrets_algo, T, name)
        self.show_regret_plots(show=False)

        if plot_instant_regret:
            plt.figure()
            for regrets_algo, (algorithm_class, algorithm_args, name) in zip(
                regrets, self.learners
            ):
                self.add_instant_plot(regrets_algo, T, name)
            self.show_regret_plots(show=False)

        plt.show()

    def add_cumulative_plot(self, regrets, T, name=None):
        plt.plot(np.arange(T), np.cumsum(regrets.mean(axis=0)), label=name)

    def add_instant_plot(self, regrets, T, name=None):
        plt.plot(np.arange(T), regrets.mean(axis=0), label=name)

    def show_regret_plots(self, show=True):
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Regret")
        if show:
            plt.show()
