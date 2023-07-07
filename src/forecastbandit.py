from bandit import *
import numpy.random as ra
import scipy.linalg as sla
import numpy.linalg as la
import pdb


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
            if np.random.rand() < 1 - current_p:
                return self.normal_bandit.pull(arm), False
            else:
                return self.hazard_bandit.pull(arm), True

        def get_next_p(self):
            return np.random.rand()

        def get_means(self, p):
            return (1 - p) * self.normal_bandit.means + p * self.hazard_bandit.means

        def get_smallest_gap(self, p):
            means = self.get_means(p)
            means_sorted = np.sort(means)
            return means_sorted[-1] - means_sorted[-2]

    return ForecastBandit


def FBFixedPSequence(bandit_class, p_sequence=None):
    if p_sequence is None:
        return ForecastBandit(bandit_class)

    class FBFixedSequence(ForecastBandit(bandit_class)):
        def __init__(self, nb_arm, mu=None, lmbd=None, bandit_args={}):
            super().__init__(nb_arm, mu, lmbd, bandit_args)
            self.p_sequence = p_sequence
            self.t = 0

        def get_next_p(self):
            next_p = self.p_sequence[self.t % len(self.p_sequence)]
            self.t = (self.t + 1) % len(self.p_sequence)
            return next_p

    return FBFixedSequence


def FBWithSubPullOutput(alg_class):
    class FBWithSubPullOutput(alg_class):
        def _compute_regret(self, arm, p):
            means = self.bandit.get_means(p)
            return means.max() - means[arm] > 0

    return FBWithSubPullOutput


# Parent class for bandit algorithms with forecasts
class ForecastBanditAlgorithm(BanditAlgorithm):
    name = "ForecastBanditAlgorithm"

    def __init__(self, bandit):
        super().__init__(bandit)
        self.normal_pull_count = np.zeros(self.nb_arm)
        self.total_rewards_normal = np.zeros(self.nb_arm)
        self.observed_p = []
        self.smallest_gap = []
        self.estimation_error = []

    def _update(self, arm, reward, event, p):
        super()._update(arm, reward)
        if not event:
            self.normal_pull_count[arm] += 1
            self.total_rewards_normal[arm] += reward

    def run(self, T):
        for t in range(T):
            current_p = self.bandit.p
            arm, empirical_means = self.select(current_p)
            self.estimation_error.append(
                np.sum(np.abs(empirical_means - self.bandit.get_means(current_p)))
            )
            self.observed_p.append(current_p)
            self.regrets.append(self._compute_regret(arm, current_p))
            self.smallest_gap.append(self.bandit.get_smallest_gap(current_p))
            reward, event = self.bandit.pull(arm)
            self._update(arm, reward, event, current_p)
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

    def _compute_global_empirical_means(self):
        return self.total_rewards / np.maximum(self.pull_count, 1)


# UCB1 algorithm with forecasts
class ForecastUCB1(ForecastBanditAlgorithm):
    name = "ForecastUCB1"

    def __init__(self, bandit):
        super().__init__(bandit)

    def select(self, p):
        empirical_means = self._compute_empirical_means(p)
        if self.normal_pull_count.min() == 0:
            return np.argmin(self.normal_pull_count), empirical_means
        elif (self.pull_count - self.normal_pull_count).min() == 0:
            return np.argmin(self.pull_count - self.normal_pull_count), empirical_means
        return (
            np.argmax(
                empirical_means
                + np.sqrt(
                    2
                    * np.log(1 + self.t * np.log(self.t + 1) ** 2)
                    * (
                        (1 - p) ** 2 / self.normal_pull_count
                        + p**2 / (self.pull_count - self.normal_pull_count)
                    )
                )
            ),
            empirical_means,
        )


class ForecastEpsilonGreedy(ForecastBanditAlgorithm):
    name = "ForecastEpsilonGreedy"

    def __init__(self, bandit, epsilon):
        super().__init__(bandit)
        self.epsilon = epsilon

    def select(self, p):
        empirical_means = self._compute_empirical_means(p)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nb_arm), empirical_means
        else:
            return np.argmax(empirical_means), empirical_means


class UCB1(ForecastBanditAlgorithm):
    name = "UCB1"

    def __init__(self, bandit):
        super().__init__(bandit)

    def select(self, p):
        empirical_means = self._compute_global_empirical_means()
        for arm in range(self.nb_arm):
            if self.pull_count[arm] == 0:
                return arm, empirical_means
        return (
            np.argmax(
                empirical_means
                + np.sqrt(
                    2
                    * np.log(1 + self.t * np.log(self.t + 1) ** 2)
                    / np.maximum(self.pull_count, 1)
                )
            ),
            empirical_means,
        )


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

    def _update(self, arm, reward, event, p):
        super()._update(arm, reward, event, p)
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


class ForecastUCB1Approx(ForecastUCB1):
    name = "ForecastUCB1"

    def __init__(self, bandit):
        super().__init__(bandit)
        self.event_count = 0

    def select(self, p):
        p_estimated = self.event_count / (self.t + 1)
        return super().select(p_estimated)

    def _update(self, arm, reward, event, p):
        super()._update(arm, reward, event, p)
        if event:
            self.event_count += 1


class OptimalAgent(ForecastBanditAlgorithm):
    name = "Optimal"

    def __init__(self, bandit):
        super().__init__(bandit)

    def select(self, p):
        return np.argmax(self.bandit.get_means(p)), self.bandit.get_means(p)


class OFUL(ForecastBanditAlgorithm):
    # Adapted from https://github.com/zackbh/bandit_algorithms
    name = "OFUL"

    ridge = 0.1
    delta = 0.1
    S_hat = 1
    R = 1
    my_c = 1  # "agressiveness"

    def __init__(self, bandit):
        super().__init__(bandit)
        self.d = self.nb_arm * 2

        self.XTy = np.zeros(self.d)
        self.invVt = np.eye(self.d) / self.ridge
        self.logdetV = self.d * np.log(self.ridge)
        self.sqrt_beta = self._calc_sqrt_beta_det2()
        self.theta_hat = np.zeros(self.d)
        self.Vt = self.ridge * np.eye(self.d)

    def select(self, p):
        x = np.array([self._make_context(arm, p) for arm in range(self.nb_arm)])
        X_invVt_norm_sq = np.sum(np.dot(x, self.invVt) * x, 1)
        obj_func = np.dot(x, self.theta_hat) + self.my_c * self.sqrt_beta * np.sqrt(
            X_invVt_norm_sq
        )
        pulled_idx = np.argmax(obj_func)

        return pulled_idx, np.dot(x, self.theta_hat)

    def _update(self, arm, reward, event, p):
        super()._update(arm, reward, event, p)

        xt = self._make_context(arm, p)
        self.XTy += reward * xt
        self.Vt += np.outer(xt, xt)

        tempval1 = np.dot(self.invVt, xt)
        tempval2 = np.dot(tempval1, xt)
        self.logdetV += np.log(1 + tempval2)

        if self.t % 20 == 0:
            self.invVt = la.inv(self.Vt)
        else:
            self.invVt -= np.outer(tempval1, tempval1) / (1 + tempval2)

        self.theta_hat = np.dot(self.invVt, self.XTy)
        self.sqrt_beta = self._calc_sqrt_beta_det2()

    def _calc_sqrt_beta_det2(self):
        return (
            self.R
            * np.sqrt(
                self.logdetV
                - self.d * np.log(self.ridge)
                + np.log(1 / (self.delta**2))
            )
            + np.sqrt(self.ridge) * self.S_hat
        )

    def _make_context(self, arm, p):
        return np.array(
            [
                p if i == 2 * arm else 1 - p if i == 2 * arm + 1 else 0
                for i in range(self.d)
            ]
        )


class ForecastBanditExperiment(Experiment):
    def plot_ratio_pull(self, bandit_class, bandit_args, T):
        for algorithm_class, algorithm_args, name in self.learners:
            bandit = bandit_class(**bandit_args)
            algorithm = algorithm_class(bandit, **algorithm_args)
            _ = algorithm.run(T)
            print(name, algorithm.normal_pull_count / algorithm.pull_count)

    def plot_smallest_gap(self, bandit_class, bandit_args, T):
        bandit = bandit_class(**bandit_args)
        plt.plot(
            np.linspace(0, 1, 200),
            [bandit.get_smallest_gap(p) for p in np.linspace(0, 1, 200)],
        )
        print(np.argmax(bandit.get_means(0.0)))
        print(np.argmax(bandit.get_means(0.5)))
        print(np.argmax(bandit.get_means(1.0)))
        plt.title("Smallest gap as a function of $p$")
        plt.xlabel("$p$")
        plt.ylabel("Smallest gap")
        plt.figure()
        plt.errorbar(
            range(bandit.nb_arm),
            (bandit.mu + bandit.lmbd) / 2,
            (bandit.mu - bandit.lmbd) / 2,
            linestyle="None",
        )
        plt.title("Arm intervals")
        plt.xticks(range(bandit.nb_arm), [str(i) for i in range(1, bandit.nb_arm + 1)])
        plt.xlabel("Arm")
        plt.ylabel("Mean reward interval")
        plt.show()

    def plot_estimation_error(self, bandit_class, bandit_args, T):
        plt.figure()
        for algorithm_class, algorithm_args, name in self.learners:
            bandit = bandit_class(**bandit_args)
            algorithm = algorithm_class(bandit, **algorithm_args)
            _ = algorithm.run(T)
            plt.plot(np.cumsum(algorithm.estimation_error), label=name)
        plt.legend()
        plt.title("Estimation error")
        plt.xlabel("Time")
        plt.ylabel("Estimation error")
        plt.show()
