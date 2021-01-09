import numpy as np

import location

loc1 = location.Location(5, 3)

print(np.sum(loc1.prob_ending_cars[2, :]))

# should be 1s
print(np.sum(loc1.prob_ending_cars, axis=1))
