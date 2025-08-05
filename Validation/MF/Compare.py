import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data from Maxwell Fourier, only Bz
Fz = np.array([
    2.67400705e-05, 2.67311633e-05, 2.67047602e-05, 2.66617700e-05,
    2.66035796e-05, 2.65319052e-05, 2.64486422e-05, 2.63557355e-05
])

Fy = np.array([
    1.10185483e-12, -2.94725479e-07, -5.85827120e-07, -8.69958957e-07,
    -1.14427368e-06, -1.40654493e-06, -1.65520050e-06, -1.88928904e-06
])


df = pd.read_excel("Validation/MF/MaxwellFourier - validation.xlsx", engine='openpyxl')


v = df['v'].to_numpy()
Fz_global = df["Fz_global"].to_numpy()
Fy_global = df["Fy_global"].to_numpy()
Fz_simp = df["Fz_simp"].to_numpy()
Fy_simp = df["Fy_simp"].to_numpy()

plt.figure()
plt.title("Global")
plt.scatter(v, -Fz_global,c='r')
plt.scatter(v, Fz,c='b')

plt.scatter(v, -Fy_global,c='r',label='Comsol')
plt.scatter(v, -Fy,c='b', label="MF")
plt.grid()
plt.legend()


plt.figure()
plt.title("Simple")
plt.plot(v, Fz_simp,c='r')
plt.plot(v, Fz,c='b')

plt.plot(v, Fy_simp,c='r',label='Comsol')
plt.plot(v, -Fy,c='b', label="MF")
plt.grid()
plt.legend()

plt.show()



