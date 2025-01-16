# quick demonstration of how the burger sim class can be used to easily perform
# numerical experiments

import Burger_Sim as b
import numpy as np

def u0(x):
    t1 = np.exp(-(x-1)**2 / 2)
    t2 = np.exp(-(x+1)**2 / 2)
    return t1 - t2


test = b.Burger_Sim(-6, 6, 200, 0.002, 0)

for i in range(10):
    test.add_sim(lambda x, i=i: 1/(0.15*i*x**2 + 1), 5, u0)

test.play(1000)