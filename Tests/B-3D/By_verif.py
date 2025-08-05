import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv("dataR/magneticField - 3D/B_fit.csv")
x = df["x"].to_numpy()
By_ag = df["Bx_ag"].to_numpy()
Bz_ag = df["Bz_ag"].to_numpy()
B_ag = df["B_ag"].to_numpy()


index = ((x >= 0.0) & (x <= 0.25 + 0.05))
x = x[index]
By_ag = By_ag[index]
Bz_ag = Bz_ag[index]
B_ag = B_ag[index]

B_ag_inv = B_ag[::-1]
By_ag_inv = By_ag[::-1]
Bz_ag_inv = Bz_ag[::-1]

# B_normalized = B_ag_inv[0]

# B_ag_inv = B_ag_inv/B_normalized
# By_ag_inv = By_ag_inv/B_normalized
# Bz_ag_inv = Bz_ag_inv/B_normalized

df = pd.read_csv("dataR/Comsol/airgapField/By.txt", delim_whitespace=True, comment='%', 
                    names=['Arc length (mm)', 'Magnetic flux density norm'])
y = df['Arc length (mm)'].to_numpy()*1e-3
By = df['Magnetic flux density norm'].to_numpy()

df1 = pd.read_csv("dataR/Comsol/airgapField/Bz.txt", delim_whitespace=True, comment='%', 
                    names=['Arc length (mm)', 'Magnetic flux density norm'])
y1 = df1['Arc length (mm)'].to_numpy()*1e-3
Bz = df1['Magnetic flux density norm'].to_numpy()

df2 = pd.read_csv("dataR/Comsol/airgapField/B.txt", delim_whitespace=True, comment='%', 
                    names=['Arc length (mm)', 'Magnetic flux density norm'])
y2 = df2['Arc length (mm)'].to_numpy()*1e-3
B = df2['Magnetic flux density norm'].to_numpy()


# B1_normalized = np.max(B)

# B = B/B1_normalized
# By = By/B1_normalized
# Bz = Bz/B1_normalized

# plt.figure()
# plt.plot(y, By)
# plt.plot(x, -By_ag_inv)

offset = np.abs(B_ag_inv[-1] - B[-1])

plt.figure()
plt.plot(y1, Bz)
plt.plot(x, Bz_ag_inv-offset)
plt.title("Bz")

plt.figure()
plt.plot(y, By)
plt.plot(x, -By_ag_inv)
plt.title("By")

plt.figure()
plt.plot(y2, B)
plt.plot(x, B_ag_inv-offset)
plt.title("B")








plt.show()