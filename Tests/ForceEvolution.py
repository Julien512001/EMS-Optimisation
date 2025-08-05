import numpy as np
import matplotlib.pyplot as plt

air = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

Force = np.array([222.28, 224.20436116144438, 224.063092071888, 223.72, 223.36, 223.02, 221.6978, 222.4, 221.69, 221.8655])











F_ref = 55.04921744193346

F = 56.07444338178229


print(f"err = {100-np.abs(F_ref/F)*100}")









plt.figure()
plt.grid()
plt.plot(air, Force)
plt.scatter(air, Force)


# plt.show()