import numpy as np
from scipy import stats


class Location:
    def __init__(self, rental_rate: float, return_rate: float):
        self._rental_rate: float = rental_rate
        self._return_rate: float = return_rate

        self.max_cars: int = 20
        self._revenue_per_car: float = 10.0
        self._park_penalty: float = 4.0

        self._demand_prob: np.ndarray = np.zeros(self.max_cars + 1, float)
        self._return_prob: np.ndarray = np.zeros(self.max_cars + 1, float)

        # given starting_cars as input, value is expected revenue
        # E[r[l] | s, a]
        self._expected_revenue: np.ndarray = np.zeros(self.max_cars + 1, float)

        # given starting_cars as first value, value is probability of ending_cars
        # Pr(s'[l] | s, a)
        self._prob_ending_cars: np.ndarray = np.zeros(shape=(self.max_cars + 1, self.max_cars + 1), dtype=float)

        self._build()

    def _build(self):
        self._rental_return_prob()
        self._daily_outcome_tables()

    def _rental_return_prob(self):
        car_count = [c for c in range(self.max_cars + 1)]
        self._demand_prob = np.array([self._poisson(self._rental_rate, c) for c in car_count])
        self._return_prob = np.array([self._poisson(self._return_rate, c) for c in car_count])
        self._demand_prob[-1] += 1.0 - np.sum(self._demand_prob)
        self._return_prob[-1] += 1.0 - np.sum(self._return_prob)

    def _daily_outcome_tables(self):
        self._prob_ending_cars = np.zeros(shape=(self.max_cars + 1, self.max_cars + 1), dtype=float)

        for starting_cars in range(self.max_cars + 1):
            expected_revenue: float = 0.0
            for car_demand, demand_probability in enumerate(self._demand_prob):
                cars_rented = min(starting_cars, car_demand)
                revenue = cars_rented * self._revenue_per_car
                expected_revenue += demand_probability * revenue
                for cars_returned, return_probability in enumerate(self._return_prob):
                    joint_probability = demand_probability * return_probability
                    ending_cars = starting_cars - cars_rented + cars_returned
                    if ending_cars > 20:
                        ending_cars = 20
                    self._prob_ending_cars[starting_cars, ending_cars] += joint_probability
            self._expected_revenue[starting_cars] = expected_revenue

    def _poisson(self, lambda_: float, n: int):
        return stats.poisson.pmf(k=n, mu=lambda_)

    def get_expected_revenue(self, start_cars: int) -> float:
        return self._expected_revenue[start_cars]

    def probability_transition(self, start_cars: int, end_cars: int) -> float:
        return self._prob_ending_cars[start_cars, end_cars]

    def get_parking_penalty(self, end_cars: int) -> float:
        if end_cars > 10:
            return -self._park_penalty
        else:
            return 0.0
