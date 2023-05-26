import numpy as np
import sys
from scipy.stats import norm
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns


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


def ExternalRiskBandit(bandit_class):
    class ExternalRiskBandit(Bandit):
        def __init__(self, nb_arm, mu=None, lmbd=None):
            self.normal_bandit = bandit_class(nb_arm, mu)
            self.hazard_bandit = bandit_class(nb_arm, lmbd)
            self.mu = mu
            self.lmbd = lmbd

        def pull(self, arm, p):
            if np.random.rand() < p:
                return self.normal_bandit.pull(arm), False
            else:
                return self.hazard_bandit.pull(arm), True

    return ExternalRiskBandit


# Parent class for bandit algorithms
class BanditAlgorithm:
    name = "BanditAlgorithm"
    def __init__(self, bandit):
        self.bandit = bandit
        self.nb_arm = bandit.nb_arm
        self.means = bandit.means
        self.t = 0
        self.total_rewards = np.zeros(self.nb_arm)
        self.pull_count = np.zeros(self.nb_arm)
        self.regrets = []

    def update(self, arm, reward):
        self.t += 1
        self.total_rewards[arm] += reward
        self.pull_count[arm] += 1
        self.regrets.append(self.means.max() - self.means[arm])

    def run(self, T):
        for t in range(T):
            arm = self.select()
            reward = self.bandit.pull(arm)
            self.update(arm, reward)
        return self.regrets

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
            return np.argmax(self.total_rewards / self.pull_count)


# UCB algorithm
class UCB(BanditAlgorithm):
    name = "UCB1"
    def __init__(self, bandit):
        super().__init__(bandit)

    def select(self):
        if self.t < self.nb_arm:
            return self.t
        else:
            return np.argmax(
                self.total_rewards / self.pull_count
                + np.sqrt(2 * np.log(self.t) / self.pull_count)
            )


# Thompson sampling algorithm for Bernoulli bandits
class BernoulliThompsonSampling(BanditAlgorithm):
    name = "Thompson Sampling"
    def __init__(self, bandit):
        super().__init__(bandit)
        self.alpha = np.ones(self.nb_arm)
        self.beta = np.ones(self.nb_arm)

    def update(self, arm, reward):
        super().update(arm, reward)
        if reward == 1.0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def select(self):
        return np.argmax(np.random.beta(self.alpha, self.beta))


# Experiment class
class Experiment:
    def run(
        self, bandit_class, bandit_args, algorithm_class, algorithm_args, T, nb_simu
    ):
        regrets = np.zeros((nb_simu, T))
        for i in progressbar(range(nb_simu), "Simulation: ", 40):
            bandit = bandit_class(**bandit_args)
            algorithm = algorithm_class(bandit, **algorithm_args)
            regrets[i] = algorithm.run(T)
        return regrets

    def plot_cumulated_regret(self, regrets, T, names=None):
        plt.plot(np.arange(T), np.cumsum(regrets.mean(axis=0)), label=names)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Regret")


# Experiments
if __name__ == "__main__":
    # Bandit parameters
    nb_arm = 10
    range_means = [0, 1]
    means = np.random.rand(nb_arm) * (range_means[1] - range_means[0]) + range_means[0]
    sigmas = np.ones(nb_arm)

    # Algorithm parameters
    epsilon = 0.1

    # Experiment parameters
    T = 1000
    nb_simu = 100

    # Set plot style
    plt.style.use("ggplot")

    # Run experiments
    experiment = Experiment()
    regrets = experiment.run(
        BernoulliBandit,
        {"nb_arm": nb_arm, "means": means},
        EpsilonGreedy,
        {"epsilon": epsilon},
        T,
        nb_simu,
    )
    experiment.plot_cumulated_regret(regrets, T, "Epsilon Greedy")

    # Run experiments
    experiment = Experiment()
    regrets = experiment.run(
        BernoulliBandit,
        {"nb_arm": nb_arm, "means": means},
        UCB,
        {},
        T,
        nb_simu,
    )
    experiment.plot_cumulated_regret(regrets, T, "UCB1")

    # Run experiments
    experiment = Experiment()
    regrets = experiment.run(
        BernoulliBandit,
        {"nb_arm": nb_arm, "means": means},
        BernoulliThompsonSampling,
        {},
        T,
        nb_simu,
    )
    experiment.plot_cumulated_regret(regrets, T, "Thompson Sampling")

    plt.show()

    # Run experiments
    experiment = Experiment()
    regrets = experiment.run(
        GaussianBandit,
        {
            "nb_arm": nb_arm,
            "means": means,
            "sigmas": sigmas,
        },
        EpsilonGreedy,
        {"epsilon": epsilon},
        T,
        nb_simu,
    )
    experiment.plot_cumulated_regret(regrets, T, "Epsilon Greedy")

    # Run experiments
    experiment = Experiment()
    regrets = experiment.run(
        GaussianBandit,
        {
            "nb_arm": nb_arm,
            "means": means,
            "sigmas": sigmas,
        },
        UCB,
        {},
        T,
        nb_simu,
    )
    experiment.plot_cumulated_regret(regrets, T, "UCB1")

    # Run experiments
    experiment = Experiment()
    regrets = experiment.run(
        GaussianBandit,
        {
            "nb_arm": nb_arm,
            "means": means,
            "sigmas": sigmas,
        },
        BernoulliThompsonSampling,
        # TODO : change to GaussianThompsonSampling
        {},
        T,
        nb_simu,
    )
    experiment.plot_cumulated_regret(regrets, T, "Thompson Sampling")

    plt.show()