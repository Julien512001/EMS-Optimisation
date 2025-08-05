import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(
    "Validation/Longitudinal/Bz_0.csv",
    comment='%',  
    header=None,
    names=['y', 'Bz_comsol']
)
y0_comsol = df['y'].to_numpy()*1e-3
Bz0_comsol = df['Bz_comsol'].to_numpy()

df = pd.read_csv(
    "Validation/Longitudinal/Bz_1.csv",
    comment='%',  
    header=None,
    names=['y', 'Bz_comsol']
)
y1_comsol = df['y'].to_numpy()*1e-3
Bz1_comsol = df['Bz_comsol'].to_numpy()

df = pd.read_csv(
    "Validation/Longitudinal/Bz_2.csv",
    comment='%',  
    header=None,
    names=['y', 'Bz_comsol']
)
y2_comsol = df['y'].to_numpy()*1e-3
Bz2_comsol = df['Bz_comsol'].to_numpy()



df = pd.read_csv("Validation/Longitudinal/Bz_long.csv")

y = df["y"].to_numpy()
Bz0 = df["Bz0"].to_numpy()
Bz1 = df["Bz1"].to_numpy()
Bz2 = df["Bz2"].to_numpy()
B0 = df["B0"].to_numpy()
B1 = df["B1"].to_numpy()
B2 = df["B2"].to_numpy()



plt.figure()
plt.title("0")
plt.plot(y0_comsol,Bz0_comsol)
plt.plot(y, Bz0,label="Bz")
# plt.plot(y, B0,label="B")
plt.legend()
plt.grid()

plt.figure()
plt.title("1")
plt.plot(y1_comsol,Bz1_comsol)
plt.plot(y, Bz1,label="Bz")
# plt.plot(y, B1,label="B")
plt.legend()
plt.grid()

plt.figure()
plt.title("2")
plt.plot(y2_comsol,Bz2_comsol)
plt.plot(y, Bz2,label="Bz")
# plt.plot(y, B2,label="B")
plt.legend()
plt.grid()




plt.show()