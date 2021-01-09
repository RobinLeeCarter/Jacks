import numpy as np
from scipy import stats


class Location:
    def __init__(self, rental_rate: float, return_rate: float):
        self.rental_rate: float = rental_rate
        self.return_rate: float = return_rate

        self.max_cars: int = 20
        self.rent_revenue: float = 10.0

        self.rental_prob: np.ndarray = np.zeros(self.max_cars, float)
        self.return_prob: np.ndarray = np.zeros(self.max_cars, float)

        self.build()

    def build(self):
        car_count = [c for c in range(self.max_cars+1)]
        self.rental_prob = np.array([self.poisson(self.rental_rate, c) for c in car_count])
        self.return_prob = np.array([self.poisson(self.return_rate, c) for c in car_count])
        self.rental_prob[-1] += 1.0 - np.sum(self.rental_prob)
        self.return_prob[-1] += 1.0 - np.sum(self.return_prob)

    def poisson(self, lambda_: float, n: int):
        return stats.poisson.pmf(k=n, mu=lambda_)


