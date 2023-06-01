from bandit import *

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
experiment.add_learner(EpsilonGreedy, {"epsilon": epsilon}, name="Epsilon Greedy")
experiment.add_learner(UCB1, name="UCB1")
experiment.add_learner(BernoulliThompsonSampling, name="Thompson Sampling")
experiment.run_and_plot(BernoulliBandit, {"nb_arm": nb_arm, "means": means}, T, nb_simu)

# Experiment with gaussian bandits
experiment = Experiment()
experiment.add_learner(EpsilonGreedy, {"epsilon": epsilon}, name="Epsilon Greedy")
experiment.add_learner(UCB1, name="UCB1")
experiment.add_learner(GaussianThompsonSampling, name="Thompson Sampling")
experiment.run_and_plot(
    GaussianBandit, {"nb_arm": nb_arm, "means": means, "sigmas": sigmas}, T, nb_simu
)
