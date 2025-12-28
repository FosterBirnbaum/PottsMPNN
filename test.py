import numpy as np

t1 = np.NaN
print(t1)

t2 = ['a', 'b']
t3 = ['c', 'd']
tsum = []
tsum2 = [tsum + t for t in [t2, t2, t3]]
print(tsum2)

t0 = ['A'] * 0
print(t0)