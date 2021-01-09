from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

import utils
import problem
import algorithms


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        self.epochs: int = 0
        self.time_steps: int = 0
        self.non_stationary = False
        self.problem_center = 0.0
        self.algorithms: List[algorithms.Algorithm] = []
        self.algorithms_i: List[int] = []
        self.timer = utils.Timer()

        self.time_steps_x: np.ndarray = np.ndarray(shape=(0,), dtype=int)
        # self.learning_curves()

        self.powers: np.ndarray = np.ndarray(shape=(0,), dtype=float)
        self.hyperparameters: np.ndarray = np.ndarray(shape=(0, ), dtype=float)

        # self.stationary_parameter_study()
        self.non_stationary_parameter_study()

    def learning_curves(self):
        self.e_greedy_comparison()
        # self.sample_vs_alpha()
        # self.optimistic_vs_realistic()
        # self.optimistic_biased_vs_unbiased()
        # self.e_greedy_vs_ucb()
        # self.gradient_bandit_comparison()

        self.run()
        self.timer.stop()
        self.learning_curve_graph()

    def e_greedy_comparison(self):
        self.epochs = 200
        self.time_steps = 1000

        alg1 = algorithms.EGreedy(name="greedy", time_steps=self.time_steps, epsilon=0.0)
        alg2 = algorithms.EGreedy(name="ε=0.01", time_steps=self.time_steps, epsilon=0.01)
        alg3 = algorithms.EGreedy(name="ε=0.1", time_steps=self.time_steps, epsilon=0.1)
        self.algorithms = [alg1, alg2, alg3]

    def sample_vs_alpha(self):
        self.epochs = 2000
        self.time_steps = 10000
        self.non_stationary = True

        alg1 = algorithms.EGreedy(name="sample averages", time_steps=self.time_steps,
                                  epsilon=0.1)
        alg2 = algorithms.EGreedyAlpha(name="constant step-size", time_steps=self.time_steps,
                                       epsilon=0.1, alpha=0.1)
        self.algorithms = [alg1, alg2]

    def optimistic_vs_realistic(self):
        self.epochs = 2000
        self.time_steps = 1000

        alg1 = algorithms.EGreedyAlpha(name="optimistic greedy", time_steps=self.time_steps,
                                       epsilon=0.0, alpha=0.1, q1=5.0)
        alg2 = algorithms.EGreedyAlpha(name="realistic non-greedy", time_steps=self.time_steps,
                                       epsilon=0.1, alpha=0.1)
        self.algorithms = [alg1, alg2]

    def optimistic_biased_vs_unbiased(self):
        self.epochs = 2000
        self.time_steps = 1000

        alg1 = algorithms.EGreedyAlpha(name="optimistic non-greedy biased", time_steps=self.time_steps,
                                       epsilon=0.1, alpha=0.1, q1=5.0)
        alg2 = algorithms.EGreedyAlpha(name="optimistic non-greedy unbiased", time_steps=self.time_steps,
                                       epsilon=0.1, alpha=0.1, q1=5.0, biased=False)
        self.algorithms = [alg1, alg2]

    def e_greedy_vs_ucb(self):
        self.epochs = 2000
        self.time_steps = 1000
        alg1 = algorithms.EGreedy(name="e-greedy", time_steps=self.time_steps, epsilon=0.1)
        alg2 = algorithms.UCB(name="UCB", time_steps=self.time_steps, c=2.0)
        self.algorithms = [alg1, alg2]

    def gradient_bandit_comparison(self):
        self.epochs = 2000
        self.time_steps = 1000
        self.problem_center = 4.0

        gb = algorithms.GradientBandit
        alg1 = gb(name="alpha=0.1", time_steps=self.time_steps, alpha=0.1)
        alg2 = gb(name="alpha=0.1 no baseline", time_steps=self.time_steps, alpha=0.1, baseline_enabled=False)
        alg3 = gb(name="alpha=0.4", time_steps=self.time_steps, alpha=0.4)
        alg4 = gb(name="alpha=0.4 no baseline", time_steps=self.time_steps, alpha=0.4, baseline_enabled=False)
        self.algorithms = [alg1, alg2, alg3, alg4]

    def run(self, reporting_frequency: int = 100):
        for epoch in range(self.epochs):
            if self.verbose and epoch % reporting_frequency == 0:
                self.timer.lap(f"epoch = {epoch}", show=True)

            problem_ = problem.Problem(center=self.problem_center, non_stationary=self.non_stationary)

            for alg in self.algorithms:
                alg.set_problem(problem_, epoch)

            for t in range(1, self.time_steps):
                problem_.do_time_step(t)
                for alg in self.algorithms:
                    alg.do_time_step_and_record(t)

    def learning_curve_graph(self):
        self.time_steps_x = np.arange(self.time_steps)
        self.learning_curve_av_return()
        self.learning_curve_av_percent()

    def learning_curve_av_return(self):
        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()

        for alg in self.algorithms:
            ax.plot(self.time_steps_x, alg.av_return, label=alg.name)
            ax.legend()
        plt.show()

    def learning_curve_av_percent(self):
        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()

        for alg in self.algorithms:
            ax.plot(self.time_steps_x, alg.av_percent, label=alg.name)
            ax.legend()

        plt.show()

    def stationary_parameter_study(self):
        self.epochs = 2000
        self.time_steps = 1000

        self.powers = np.arange(-7, 2+1)
        self.hyperparameters = 2.0**self.powers

        e_greedy = np.empty(shape=self.powers.shape, dtype=float)
        e_greedy[:] = np.nan
        self.e_greedy_parameter_study()

        gradient_bandit = np.empty(shape=self.powers.shape, dtype=float)
        gradient_bandit[:] = np.nan
        self.gradient_bandit_parameter_study()

        ucb = np.empty(shape=self.powers.shape, dtype=float)
        ucb[:] = np.nan
        self.ucb_parameter_study()

        optimistic = np.empty(shape=self.powers.shape, dtype=float)
        optimistic[:] = np.nan
        self.optimistic_parameter_study()

        self.run(reporting_frequency=10)
        self.timer.stop()
        for i, alg in zip(self.algorithms_i, self.algorithms):
            av_reward = alg.get_av_reward()
            # print(f"{alg.name}\t{av_reward}")
            alg_type = type(alg)
            if alg_type == algorithms.EGreedy:
                e_greedy[i] = av_reward
            elif alg_type == algorithms.GradientBandit:
                gradient_bandit[i] = av_reward
            elif alg_type == algorithms.UCB:
                ucb[i] = av_reward
            elif alg_type == algorithms.EGreedyAlpha:
                assert isinstance(alg, algorithms.EGreedyAlpha)
                if alg.q1 > 0:
                    optimistic[i] = av_reward

        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()
        ax.set_xlim(xmin=np.min(self.powers), xmax=np.max(self.powers))
        ax.set_xlabel("hyperparameter power of 2")
        # ax.set_xscale("log")
        ax.plot(self.powers, e_greedy, label="e-greedy")
        ax.plot(self.powers, gradient_bandit, label="gradient bandit")
        ax.plot(self.powers, ucb, label="ucb")
        ax.plot(self.powers, optimistic, label="greedy with optimistic initialization")
        ax.legend()
        plt.show()

    def non_stationary_parameter_study(self):
        self.non_stationary = True
        self.epochs = 100
        self.time_steps = 200_000

        self.powers = np.arange(-9, 2+1)
        self.hyperparameters = 2.0**self.powers

        e_greedy = np.empty(shape=self.powers.shape, dtype=float)
        e_greedy[:] = np.nan
        self.e_greedy_parameter_study()

        gradient_bandit = np.empty(shape=self.powers.shape, dtype=float)
        gradient_bandit[:] = np.nan
        self.gradient_bandit_parameter_study()

        ucb = np.empty(shape=self.powers.shape, dtype=float)
        ucb[:] = np.nan
        self.ucb_parameter_study()

        constant_step = np.empty(shape=self.powers.shape, dtype=float)
        constant_step[:] = np.nan
        self.constant_step_parameter_study()

        self.run(reporting_frequency=1)
        self.timer.stop()
        for i, alg in zip(self.algorithms_i, self.algorithms):
            av_reward = alg.get_av_reward(final_steps=100_000)
            # print(f"{alg.name}\t{av_reward}")
            alg_type = type(alg)
            if alg_type == algorithms.EGreedy:
                e_greedy[i] = av_reward
            elif alg_type == algorithms.GradientBandit:
                gradient_bandit[i] = av_reward
            elif alg_type == algorithms.UCB:
                ucb[i] = av_reward
            elif alg_type == algorithms.EGreedyAlpha:
                constant_step[i] = av_reward

        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()
        ax.set_xlim(xmin=np.min(self.powers), xmax=np.max(self.powers))
        ax.set_xlabel("hyperparameter power of 2")
        # ax.set_xscale("log")
        ax.plot(self.powers, e_greedy, label="e-greedy")
        ax.plot(self.powers, gradient_bandit, label="gradient bandit")
        ax.plot(self.powers, ucb, label="ucb")
        ax.plot(self.powers, constant_step, label="e-greedy with constant step-size")
        ax.legend()
        plt.show()

    def e_greedy_parameter_study(self):
        for i, power in enumerate(self.powers):
            if -7 <= power <= -2:
                epsilon = self.hyperparameters[i]
                alg = algorithms.EGreedy(name=f"e-greedy epsilon={epsilon}", time_steps=self.time_steps,
                                         epsilon=epsilon)
                self.algorithms_i.append(i)
                self.algorithms.append(alg)

    def gradient_bandit_parameter_study(self):
        for i, power in enumerate(self.powers):
            if -5 <= power <= 0:
                alpha = self.hyperparameters[i]
                alg = algorithms.GradientBandit(name=f"gradient alpha={alpha}", time_steps=self.time_steps,
                                                alpha=alpha)
                self.algorithms_i.append(i)
                self.algorithms.append(alg)

    def ucb_parameter_study(self):
        for i, power in enumerate(self.powers):
            if -3 <= power <= 2:
                c = self.hyperparameters[i]
                alg = algorithms.UCB(name=f"UCB c={c}", time_steps=self.time_steps,
                                     c=c)
                self.algorithms_i.append(i)
                self.algorithms.append(alg)

    def optimistic_parameter_study(self):
        for i, power in enumerate(self.powers):
            if -2 <= power <= 2:
                q1 = self.hyperparameters[i]
                alg = algorithms.EGreedyAlpha(name=f"optimistic q1={q1}", time_steps=self.time_steps,
                                              epsilon=0.0, alpha=0.1, q1=q1)
                self.algorithms_i.append(i)
                self.algorithms.append(alg)

    def constant_step_parameter_study(self):
        for i, power in enumerate(self.powers):
            if -9 <= power <= -2:
                epsilon = self.hyperparameters[i]
                alg = algorithms.EGreedyAlpha(name=f"constant_step epsilon={epsilon}", time_steps=self.time_steps,
                                              epsilon=epsilon, alpha=0.1)
                self.algorithms_i.append(i)
                self.algorithms.append(alg)
