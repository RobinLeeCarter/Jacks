import numpy as np

import location

loc1 = location.Location(5, 3)

# noinspection PyProtectedMember
print(np.sum(loc1._prob_ending_cars[2, :]))

# should be 1s
# noinspection PyProtectedMember
print(np.sum(loc1._prob_ending_cars, axis=1))
