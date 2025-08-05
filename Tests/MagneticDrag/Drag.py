import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Tests/MagneticDrag/Drag.csv",sep=';')


v = df.iloc[:, 0].to_numpy()
Fz = df.iloc[:, 1].to_numpy()
Fy = df.iloc[:, 2].to_numpy()
Fz0 = df.iloc[:, 3].to_numpy()[0]


print(Fz0)


plt.figure()
plt.title("Fz en fonction de v")
plt.plot(v, Fz)
plt.axhline(Fz0)
plt.grid()

plt.figure()
plt.title("-Fy en fonction de v")
plt.plot(v, -Fy)
plt.grid()

plt.figure()
plt.title("Taux décroissance de Fz")
plt.plot(v, Fz/Fz0*100)
plt.grid()


plt.show()