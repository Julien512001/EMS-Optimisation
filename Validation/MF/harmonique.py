import numpy as np
import matplotlib.pyplot as plt

ham = np.array([5, 10, 15, 20, 25, 30, 35])
err = np.array([52.61, 8.701072145332034, 3.8628651665000007, 3.4530144371360194, 3.0772280397047354, 2.92481007669781, 2.803807809186438])



plt.figure()
plt.scatter(ham, err)
plt.grid()
plt.show()