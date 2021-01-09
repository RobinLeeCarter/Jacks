from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure, cm

# import utils
import state
import location


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose

        # hyperparameters
        self.theta = 0.1    # accuracy

        self.max_cars: int = 20
        self.max_transfer: int = 5
        self.l1 = location.Location(rental_rate=3, return_rate=3)
        self.l2 = location.Location(rental_rate=4, return_rate=2)
        self.cost_per_transfer: float = 2.0
        self.gamma = 0.9  # discount factor

        self.states_shape: tuple = (self.max_cars + 1, self.max_cars + 1)
        self.states: List[state.State] = [state.State(c1, c2)
                                          for c1 in range(self.states_shape[0])
                                          for c2 in range(self.states_shape[1])]
        self.actions: List[int] = [a for a in range(-self.max_transfer, self.max_transfer+1)]

        self.V: np.ndarray = np.zeros(shape=self.states_shape, dtype=float)
        self.policy: np.ndarray = np.zeros(shape=self.states_shape, dtype=int)
        self.policies: List[np.ndarray] = []

    def run(self):
        i: int = 0
        policy_stable: bool = False

        while not policy_stable:
            self.policies.append(self.policy.copy())
            if self.verbose:
                print(f"Iteration {i}")
                print(self.policy)
            # self.graph_policy(self.policy, i)
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            i += 1

        if self.verbose:
            print(self.V)
        for i, policy in enumerate(self.policies):
            self.graph_policy(policy, i)
        self.graph_v()

    def policy_evaluation(self):
        cont: bool = True
        i: int = 0

        if self.verbose:
            print(f"Start policy evaluation")
        while cont:
            delta: float = 0.0
            for s in self.states:
                v = self.get_v(s)
                action = self.get_action(s)
                return_ = self.get_expected_return(s, action)
                self.set_v(s, return_)
                diff = abs(v - return_)
                delta = max(delta, diff)
            if self.verbose:
                print(f"policy_evaluation iteration = {i}\tdelta={delta:.2f}")
            cont = not (delta < self.theta)
            i += 1

    def policy_improvement(self) -> bool:
        policy_stable: bool = True
        q: np.ndarray = np.zeros(shape=len(self.actions), dtype=float)

        if self.verbose:
            print(f"Start policy improvement")
        for s in self.states:
            old_action = self.get_policy(s)
            for i, a in enumerate(self.actions):
                q[i] = self.get_expected_return(s, a)
            max_a: float = np.max(q)
            max_i_bool = (q == max_a)
            arg_max_i = np.flatnonzero(max_i_bool)
            arg_max_a = [self.actions[i] for i in arg_max_i]
            if old_action not in arg_max_a:
                policy_stable = False
                new_a = min(arg_max_a, key=abs)
                self.set_policy(s, new_a)
        return policy_stable

    def get_expected_return(self, s: state.State, action: int) -> float:
        if action >= 0:
            # min of action, availability, spare capacity
            actual_l1_to_l2: int = min(action, s.cars_l1, self.l2.max_cars - s.cars_l2)
        else:
            actual_l1_to_l2: int = max(action, -s.cars_l2, s.cars_l1 - self.l1.max_cars)

        transfer_cost: float = self.get_transfer_cost(actual_l1_to_l2)
        start_l1: int = s.cars_l1 - actual_l1_to_l2
        start_l2: int = s.cars_l2 + actual_l1_to_l2
        expected_revenue_l1: float = self.l1.get_expected_revenue(start_cars=start_l1)
        expected_revenue_l2: float = self.l2.get_expected_revenue(start_cars=start_l2)
        expected_immediate_reward: float = expected_revenue_l1 + expected_revenue_l2 - transfer_cost
        return_: float = expected_immediate_reward
        for s_dash in self.states:
            probability_l1 = self.l1.probability_transition(start_cars=start_l1, end_cars=s_dash.cars_l1)
            probability_l2 = self.l2.probability_transition(start_cars=start_l2, end_cars=s_dash.cars_l2)
            s_dash_probability = probability_l1 * probability_l2
            expected_future_reward = s_dash_probability * self.gamma * self.get_v(s_dash)
            return_ += expected_future_reward
        return return_

    def get_v(self, state_: state.State) -> float:
        return self.V[state_.cars_l1, state_.cars_l2]

    def set_v(self, state_: state.State, value: float):
        self.V[state_.cars_l1, state_.cars_l2] = value

    def get_policy(self, state_: state.State) -> int:
        return self.policy[state_.cars_l1, state_.cars_l2]

    def set_policy(self, state_: state.State, value: int):
        self.policy[state_.cars_l1, state_.cars_l2] = value

    def get_action(self, state_: state.State) -> int:
        return self.policy[state_.cars_l1, state_.cars_l2]

    def get_transfer_cost(self, action: int) -> float:
        return abs(action) * self.cost_per_transfer

    def graph_policy(self, policy: np.ndarray, iteration):
        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.subplots()
        # x = np.arange(self.states_shape[0])
        # y = np.arange(self.states_shape[1])
        # x_grid, y_grid = np.meshgrid(x, y)

        # rows, cols = policy.shape
        ax.imshow(policy, interpolation='nearest', origin='lower',
                  vmin=-5, vmax=5, cmap='bwr')
        # extent=[0.5, cols-0.5, 0.5, rows-0.5]
        ax.set_xlim(xmin=0, xmax=20)
        ax.set_ylim(ymin=0, ymax=20)

        # contour_set = ax.contour(x_grid, y_grid, policy, levels=self.actions)
        # ax.clabel(contour_set, inline=True, fontsize=10)
        ax.set_title(f'Policy {iteration}')
        plt.show()

    def graph_v(self):
        fig: figure.Figure = plt.figure()
        ax: figure.Axes = fig.add_subplot(111, projection='3d')
        # ax: figure.Axes = fig.subplots(1, 1, 1, projection='3d')
        x = np.arange(self.states_shape[0])
        y = np.arange(self.states_shape[1])
        x_grid, y_grid = np.meshgrid(x, y)
        # noinspection PyUnresolvedReferences
        ax.plot_surface(x_grid, y_grid, self.V, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax.set_zlim(400.0, 700.0)
        plt.show()
