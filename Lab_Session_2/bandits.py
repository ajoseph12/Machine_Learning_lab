import numpy as np
import matplotlib.pyplot as plt


class Arm(object):

    def __init__(self, distribution="normal", p=0.6, mu=0, sigma=1, low=0, high=1):
        self.distribution = distribution
        self.p = p
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high
        self.rewards = []
        self.times_pulled = 0
        self._set_best_reward()

    def _set_best_reward(self):

        if self.distribution == "normal":
            self.best_reward = self.mu
        elif self.distribution == "uniform":
            self.best_reward = self.high
        elif self.distribution == "bernoulli":
            self.best_reward = self.p

    def pull(self):
        distributions = {"normal": self._get_normal,
                         "uniform": self._get_uniform,
                         "bernoulli": self._get_bernoulli}
        self.times_pulled += 1
        return distributions[self.distribution]()

    def _get_normal(self):

        reward = np.random.normal(self.mu, self.sigma)
        self.rewards.append(reward)
        return reward

    def _get_uniform(self):
        reward = np.random.uniform(self.low, self.high)
        self.rewards.append(reward)
        return reward

    def _get_bernoulli(self):
        reward = np.random.binomial(1, self.p)
        self.rewards.append(reward)
        return reward

    @property
    def regret(self):
        return self.best_reward - sum(self.rewards)*(1/self.times_pulled)

    @property
    def mean_revard(self):
        return np.mean(self.rewards)

    def get_ucb(self, t):
        return self.mean_revard + np.sqrt(2*np.log(t)/self.times_pulled)


class Bandit(object):

    def __init__(self, distribution="bernoulli", n_arms=8):
        self.n_arms = n_arms
        self.best_arm = None
        self.arms = self._get_arms(distribution)
        self.trials_performed = 0
        self.regrets = []

    @property
    def regret(self):
        observed_reward = 0
        best_reward = 0
        for arm in self.arms:
            observed_reward += sum(arm.rewards)
            if best_reward < arm.best_reward:
                best_reward = arm.best_reward
        print(best_reward)
        print(observed_reward*(1/self.trials_performed))
        return best_reward - observed_reward*(1/self.trials_performed)

    def _get_arms(self, distribution):
        arms = []
        if distribution == "normal":
            # means = np.random.uniform(0, 0.5, size=self.n_arms)
            # stds = np.random.uniform(0, 0.5, size=self.n_arms)
            means = [0.47 for _ in range(self.n_arms)]
            means[0] = 0.5
            stds = [0.5 for _ in range(self.n_arms)]
            stds[0] = 0.5
            self.best_arm = np.argmax(means)
            for mean, std in zip(means, stds):
                arms.append(Arm(distribution="normal", mu=mean, sigma=std))

        elif distribution == "bernoulli":
            # probas = np.random.uniform(0, 1, size=self.n_arms)
            probas = [0.5 for i in range(self.n_arms)]
            probas[0] = 0.51
            self.best_arm = np.argmax(probas)
            for p in probas:
                arms.append(Arm(distribution="bernoulli", p=p))

        elif distribution == "uniform":
            # probas = np.random.uniform(0, 1, size=self.n_arms)
            probas = [0.5 for i in range(self.n_arms)]
            probas[0] = 0.8
            self.best_arm = np.argmax(probas)
            for high in probas:
                arms.append(Arm(distribution="uniform", high=high))
        return arms

    def use_incremental_uniform(self, n_trials=100):
        for _ in range(int(n_trials)):

            for arm in self.arms:
                arm.pull()
                self.trials_performed += 1
            self.regrets.append(self.regret)

        rewards = []
        for arm in self.arms:
            rewards.append(arm.mean_revard)

        best_rewards = max(rewards)
        best_arm = np.argmax(rewards)

        print("After {} iterations. Best arm {}. Mean reward {}".format(
            n_trials, best_arm, best_rewards))
        return self

    def use_ucb(self, n_trials=100):
        for trial in range(n_trials):

            if trial == 0:
                for arm in self.arms:
                    arm.pull()
                    self.trials_performed += 1
            else:
                ucbs = []
                for arm in self.arms:
                    ucbs.append(arm.get_ucb(trial))

                best_arm = np.argmax(ucbs)
                self.arms[best_arm].pull()
                self.trials_performed += 1
            self.regrets.append(self.regret)

        means = []
        ucbs = []
        for arm in self.arms:
            means.append(arm.mean_revard)
            ucbs.append(arm.get_ucb(n_trials))
        best_mean = max(means)
        best_arm_means = np.argmax(means)

        best_ucb = max(ucbs)
        best_arm_ucb = np.argmax(ucbs)
        best_mean_ucb = self.arms[best_arm_ucb].mean_revard

        print("Best arm {}".format(best_arm))
        print("After {} iterations. Best arm by means {}. Mean {}.".format(
            n_trials, best_arm_means, best_mean))
        print("After {} iterations. Best arm by ucb {}. UCB {}. Mean {}".format(
            n_trials, best_arm_ucb, best_ucb, best_mean_ucb))
        return self

    def use_greedy(self, n_trials, epsilon):
        for trial in range(n_trials):
            self.trials_performed += 1
            c = np.random.rand()
            if c < epsilon:
                best_arm = self.pick_best_arm()
                self.arms[best_arm].pull()
            else:
                random_arm = np.random.randint(0, self.n_arms)
                self.arms[random_arm].pull()
            self.regrets.append(self.regret)

        best_arm = self.pick_best_arm()
        best_mean = self.arms[best_arm].mean_revard
        print("Best arm: {}".format(self.best_arm))
        print("After {} iterations. Best arm {}. Best mean {}".format(
            n_trials, best_arm, best_mean))
        return self

    def pick_best_arm(self):
        means = []
        for arm in self.arms:
            means.append(arm.mean_revard)
        return np.argmax(means)


if __name__ == "__main__":
    np.random.seed(42)
    regrets_uniform = Bandit(
        distribution="bernoulli").use_incremental_uniform(500).regrets
    regrets_ucb = Bandit(distribution="uniform").use_ucb(500).regrets
    regrets_epsilon_0_5 = Bandit(
        distribution="bernoulli").use_greedy(500, 0.5).regrets
    regrets_epsilon_0_8 = Bandit(
        distribution="bernoulli").use_greedy(500, 0.5).regrets
    regrets_epsilon_0_1 = Bandit(
        distribution="bernoulli").use_greedy(500, 0.1).regrets

    plt.plot(list(range(len(regrets_uniform))), regrets_uniform,
             c="c", label="Incremental Uniform")
    plt.plot(list(range(len(regrets_ucb))), regrets_ucb, c="g", label="UCB")
    plt.plot(list(range(len(regrets_epsilon_0_5))),
             regrets_epsilon_0_5, c="r", label="Greedy, epsilon=0.5")
    plt.plot(list(range(len(regrets_epsilon_0_8))),
             regrets_epsilon_0_8, c="b", label="Greedy, epsilon=0.8")

    plt.legend()
    plt.show()
