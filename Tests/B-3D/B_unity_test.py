import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Tests/B-3D/B_fit_test.csv")
x = df["x"].to_numpy()
By_ag = df["Bx_ag"].to_numpy()
Bz_ag = df["Bz_ag"].to_numpy()
B_ag = df["B_ag"].to_numpy()


ref = (2010e-3+2038e-3)/2
index = ((x >= 0.0) & (x <= ref))
x = x[index]
By_ag = By_ag[index]
Bz_ag = Bz_ag[index]
B_ag = B_ag[index]

B_ag_inv = B_ag[::-1]
By_ag_inv = By_ag[::-1]
Bz_ag_inv = Bz_ag[::-1]

B_normalized = B_ag_inv[0]

B_ag_inv = B_ag_inv/B_normalized
By_ag_inv = By_ag_inv/B_normalized
Bz_ag_inv = Bz_ag_inv/B_normalized

plt.figure()
plt.title("before")
plt.plot(x, By_ag, label="Bx")
plt.plot(x, Bz_ag, label="Bz")
plt.plot(x, B_ag, label="B")
plt.legend()
plt.grid()


plt.figure()
plt.title("normalized")
plt.plot(x, By_ag_inv, label="Bx")
plt.plot(x, Bz_ag_inv, label="Bz")
plt.plot(x, B_ag_inv, label="B")
plt.legend()
plt.grid()

By_ag_new = By_ag_inv*B_normalized
Bz_ag_new = Bz_ag_inv*B_normalized
B_ag_new = B_ag_inv*B_normalized

plt.figure()
plt.title("reconstructed")
plt.plot(x, By_ag_new, label="Bx")
plt.plot(x, Bz_ag_new, label="Bz")
plt.plot(x, B_ag_new, label="B")
plt.legend()
plt.grid()


# df = pd.DataFrame({"x":x, "By":By_ag_inv, "Bz":Bz_ag_inv, "B":B_ag_inv})
# df.to_csv("myBunit.csv", index=False)



# df = pd.read_csv("myBunit.csv")
# x_test = df["x"].to_numpy()
# By_ag_test = df["By"].to_numpy()
# Bz_ag_test = df["Bz"].to_numpy()
# B_ag_test = df["B"].to_numpy()



# plt.figure()
# plt.plot(x_test, By_ag_test*B_normalized)
# plt.plot(x_test, Bz_ag_test*B_normalized)
# plt.plot(x_test, B_ag_test*B_normalized)
# plt.grid()



plt.show()