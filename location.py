import numpy as np
from scipy import stats


class Location:
    def __init__(self, rental_rate: float, return_rate: float):
        self.rental_rate: float = rental_rate
        self.return_rate: float = return_rate

        self.max_cars: int = 20
        self.revenue_per_car: float = 10.0

        self.demand_prob: np.ndarray = np.zeros(self.max_cars+1, float)
        self.return_prob: np.ndarray = np.zeros(self.max_cars+1, float)

        # given starting_cars as input, value is expected revenue
        self.expected_revenue: np.ndarray = np.zeros(self.max_cars+1, float)
        # given starting_cars as first value, value is probability of ending_cars
        self.prob_ending_cars: np.ndarray = np.zeros(shape=(self.max_cars+1, self.max_cars+1), dtype=float)

        self.build()

    def build(self):
        self.rental_return_prob()
        self.daily_outcome_tables()

    def rental_return_prob(self):
        car_count = [c for c in range(self.max_cars+1)]
        self.demand_prob = np.array([self.poisson(self.rental_rate, c) for c in car_count])
        self.return_prob = np.array([self.poisson(self.return_rate, c) for c in car_count])
        self.demand_prob[-1] += 1.0 - np.sum(self.demand_prob)
        self.return_prob[-1] += 1.0 - np.sum(self.return_prob)

    def daily_outcome_tables(self):
        self.prob_ending_cars = np.zeros(shape=(self.max_cars + 1, self.max_cars + 1), dtype=float)

        for starting_cars in range(self.max_cars+1):
            expected_revenue: float = 0.0
            for car_demand, demand_probability in enumerate(self.demand_prob):
                cars_rented = min(starting_cars, car_demand)
                revenue = cars_rented * self.revenue_per_car
                expected_revenue += demand_probability * revenue
                for cars_returned, return_probability in enumerate(self.return_prob):
                    joint_probability = demand_probability * return_probability
                    ending_cars = starting_cars - cars_rented + cars_returned
                    if ending_cars > 20:
                        ending_cars = 20
                    self.prob_ending_cars[starting_cars, ending_cars] += joint_probability
            self.expected_revenue[starting_cars] = expected_revenue

    def poisson(self, lambda_: float, n: int):
        return stats.poisson.pmf(k=n, mu=lambda_)


