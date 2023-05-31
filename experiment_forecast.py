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
T = 100000
nb_simu = 10

# Toggle experiments
plot_bernoulli = False
plot_gaussian = False
print_ratio_pull = False
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

if plot_bernoulli:
    experiment.run_and_plot(
        ForecastBandit(BernoulliBandit),
        {"nb_arm": nb_arm, "mu": mu, "lmbd": lmbd},
        T,
        nb_simu,
        plot_instant_regret=True,
    )

if plot_gaussian:
    experiment.run_and_plot(
        ForecastBandit(GaussianBandit),
        {"nb_arm": nb_arm, "mu": mu, "lmbd": lmbd, "bandit_args": {"sigmas": sigmas}},
        T,
        nb_simu,
        plot_instant_regret=True,
    )

if print_ratio_pull:
    experiment.plot_ratio_pull(
        ForecastBandit(BernoulliBandit), {"nb_arm": nb_arm, "mu": mu, "lmbd": lmbd}, T
    )

# plot smallest gaps
if plot_smallest_gap:
    experiment.plot_smallest_gap(
        ForecastBandit(BernoulliBandit), {"nb_arm": nb_arm, "mu": mu, "lmbd": lmbd}, T
    )
