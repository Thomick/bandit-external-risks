from forecastbandit import *

# Bandit parameters
nb_arm = 10
range_means = [0, 1]
mu = np.sort(
    np.random.rand(nb_arm) * (range_means[1] - range_means[0]) + range_means[0]
)
lmbd = np.random.rand(nb_arm) * mu
sigmas = np.ones(nb_arm)

# Algorithm parameters
epsilon = 0.1

# Experiment parameters
T = 10000
nb_simu = 10

# Toggle experiments
plot_bernoulli = True
plot_gaussian = True
print_ratio_pull = True
plot_smallest_gap = True

# Set plot style
plt.style.use("ggplot")


# Run experiments
experiment = ForecastBanditExperiment()
experiment.add_learner(
    ForecastEpsilonGreedy, {"epsilon": epsilon}, name="Forecast Epsilon Greedy"
)
experiment.add_learner(ForecastUCB1, name="Forecast UCB1")
"""experiment.add_learner(
    ForecastBernoulliThompsonSampling, name="Forecast Thompson Sampling"
)"""

# plot smallest gaps
if plot_smallest_gap:
    print("Smallest gaps")
    experiment.plot_smallest_gap(
        ForecastBandit(BernoulliBandit), {"nb_arm": nb_arm, "mu": mu, "lmbd": lmbd}, T
    )

if plot_bernoulli:
    print("Bernoulli bandit")
    experiment.run_and_plot(
        ForecastBandit(BernoulliBandit),
        {"nb_arm": nb_arm, "mu": mu, "lmbd": lmbd},
        T,
        nb_simu,
        plot_instant_regret=True,
    )

if plot_gaussian:
    print("Gaussian bandit")
    experiment.run_and_plot(
        ForecastBandit(GaussianBandit),
        {"nb_arm": nb_arm, "mu": mu, "lmbd": lmbd, "bandit_args": {"sigmas": sigmas}},
        T,
        nb_simu,
        plot_instant_regret=True,
    )

if print_ratio_pull:
    print("Ratio pull without event")
    experiment.plot_ratio_pull(
        ForecastBandit(BernoulliBandit), {"nb_arm": nb_arm, "mu": mu, "lmbd": lmbd}, T
    )
