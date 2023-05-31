from bandit import *


def ForecastBandit(bandit_class):
    class ForecastBandit(Bandit):
        def __init__(self, nb_arm, mu=None, lmbd=None, bandit_args={}):
            self.normal_bandit = bandit_class(nb_arm, mu, **bandit_args)
            self.hazard_bandit = bandit_class(nb_arm, lmbd, **bandit_args)
            self.nb_arm = nb_arm
            self.mu = mu
            self.lmbd = lmbd
            self.p = np.random.rand()

        def pull(self, arm):
            current_p = self.p
            self.p = self.get_next_p()
            if np.random.rand() < current_p:
                return self.normal_bandit.pull(arm), False
            else:
                return self.hazard_bandit.pull(arm), True

        def get_next_p(self):
            return np.random.rand() * 0.3

        def get_means(self, p):
            return (1 - p) * self.normal_bandit.means + p * self.hazard_bandit.means

        def get_smallest_gap(self, p):
            means = self.get_means(p)
            means_sorted = np.sort(means)
            return means_sorted[-1] - means_sorted[-2]

    return ForecastBandit


# Parent class for bandit algorithms with forecasts
class ForecastBanditAlgorithm(BanditAlgorithm):
    name = "ForecastBanditAlgorithm"

    def __init__(self, bandit):
        super().__init__(bandit)
        self.normal_pull_count = np.zeros(self.nb_arm)
        self.total_rewards_normal = np.zeros(self.nb_arm)
        self.observed_p = []
        self.smallest_gap = []

    def _update(self, arm, reward, event):
        super()._update(arm, reward)
        if not event:
            self.normal_pull_count[arm] += 1
            self.total_rewards_normal[arm] += reward

    def run(self, T):
        for t in range(T):
            current_p = self.bandit.p
            arm = self.select(current_p)
            self.observed_p.append(current_p)
            self.regrets.append(self._compute_regret(arm, current_p))
            self.smallest_gap.append(self.bandit.get_smallest_gap(current_p))
            reward, event = self.bandit.pull(arm)
            self._update(arm, reward, event)
        return self.regrets

    def select(self, p):
        pass

    def _compute_regret(self, arm, p):
        means = self.bandit.get_means(p)
        return means.max() - means[arm]

    def _compute_empirical_means(self, p):
        return (1 - p) * self.total_rewards_normal / np.maximum(
            self.normal_pull_count, 1
        ) + p * (self.total_rewards - self.total_rewards_normal) / np.maximum(
            self.pull_count - self.normal_pull_count, 1
        )


# UCB1 algorithm with forecasts
class ForecastUCB1(ForecastBanditAlgorithm):
    name = "ForecastUCB1"

    def __init__(self, bandit):
        super().__init__(bandit)

    def select(self, p):
        return np.argmax(
            self._compute_empirical_means(p)
            + 0.3
            * np.sqrt(
                2
                * np.log(1 + self.t * np.log(self.t + 1) ** 2)
                * (
                    (1 - p) ** 2 / np.maximum(self.normal_pull_count, 1)
                    + p**2 / np.maximum(self.pull_count - self.normal_pull_count, 1)
                )
            )
        )


class ForecastEpsilonGreedy(ForecastBanditAlgorithm):
    name = "ForecastEpsilonGreedy"

    def __init__(self, bandit, epsilon):
        super().__init__(bandit)
        self.epsilon = epsilon

    def select(self, p):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nb_arm)
        else:
            return np.argmax(self._compute_empirical_means(p))


# Thompson sampling algorithm for Bernoulli bandits
# WIP
class ForecastBernoulliThompsonSampling(ForecastBanditAlgorithm):
    name = "Thompson Sampling"

    def __init__(self, bandit):
        super().__init__(bandit)
        self.alpha_normal = np.ones(self.nb_arm)
        self.beta_normal = np.ones(self.nb_arm)
        self.alpha_hazard = np.ones(self.nb_arm)
        self.beta_hazard = np.ones(self.nb_arm)

    def _update(self, arm, reward, event):
        super()._update(arm, reward, event)
        if event:
            if reward == 1.0:
                self.alpha_hazard[arm] += 1
            else:
                self.beta_hazard[arm] += 1
        else:
            if reward == 1.0:
                self.alpha_normal[arm] += 1
            else:
                self.beta_normal[arm] += 1

    def select(self, p):
        normal_samples = np.random.beta(self.alpha_hazard, self.beta_hazard)
        hazard_samples = np.random.beta(self.alpha_normal, self.beta_normal)
        return np.argmax((1 - p) * normal_samples + p * hazard_samples)


class ForecastBanditExperiment(Experiment):
    def plot_ratio_pull(self, bandit_class, bandit_args, T):
        for algorithm_class, algorithm_args, name in self.learners:
            bandit = bandit_class(**bandit_args)
            algorithm = algorithm_class(bandit, **algorithm_args)
            _ = algorithm.run(T)
            print(name, algorithm.normal_pull_count / algorithm.pull_count)

    def plot_smallest_gap(self, bandit_class, bandit_args, T):
        bandit = bandit_class(**bandit_args)
        print()
        plt.plot([bandit.get_smallest_gap(p) for p in np.linspace(0, 1)])
        plt.title("Smallest gap as a function of p")
        plt.set_xlabel("p")
        plt.set_ylabel("Smallest optimality gap")
        plt.figure()
        plt.errorbar(
            range(bandit.nb_arm),
            (bandit.mu + bandit.lmbd) / 2,
            (bandit.mu - bandit.lmbd) / 2,
            linestyle="None",
        )
        plt.title("Arm intervals")
        plt.show()
