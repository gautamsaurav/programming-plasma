from itertools import islice
import numpy as np

with open('liness') as lines:
    array = np.genfromtxt(islice(lines, 3, 6))

print array
