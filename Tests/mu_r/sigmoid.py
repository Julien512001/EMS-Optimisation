import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd


# 1020 Steel

B = np.array([
    0.000000, 0.166252, 0.208725, 0.261635, 0.327146, 0.407495, 0.504604,
    0.619366, 0.750502, 0.893195, 1.038258, 1.173276, 1.286731, 1.373533,
    1.437658, 1.489189, 1.538344, 1.591321, 1.649723, 1.711927, 1.774712,
    1.834976, 1.891442, 1.945262, 1.998705, 2.052852, 2.106184, 2.155149,
    2.196375, 2.229008, 2.255450, 2.280053, 2.307136, 2.339975, 2.381044,
    2.432708, 2.497748, 2.579627
])

H = np.array([
    0.000000, 79.577472, 100.182101, 126.121793, 158.777930, 199.889571,
    251.646061, 316.803620, 398.832128, 502.099901, 632.106325, 795.774715,
    1001.821011, 1261.217929, 1587.779301, 1998.895710, 2516.460605,
    3168.036204, 3988.321282, 5020.999013, 6321.063250, 7957.747155,
    10018.210114, 12612.179293, 15877.793010, 19988.957103, 25164.606052,
    31680.362037, 39883.212823, 50209.990127, 63210.632497, 79577.471546,
    100182.101136, 126121.792926, 158777.930096, 199889.571030,
    251646.060522, 316803.620370
])

# df = pd.DataFrame({"B":B, "H":H})
# df.to_csv("BH.csv")

mu_0 = 4*np.pi*1e-7


mu_r_app = B/(H*mu_0)
mu_r_app = mu_r_app[1:]
B = B[1:]
print(mu_r_app[0])



plt.figure()
plt.scatter(B,mu_r_app)
plt.grid()

def sigmoid(B):
    L = 1638.3025031133686
    k = -4.503479204801756
    B0 = 1.3692966370327697
    
    exp_term = np.exp(-k * (B - B0))
    sig = L / (1 + exp_term)
    d_sig = (L * k * exp_term) / ((1 + exp_term)**2)
    return sig, d_sig



B_sig = np.linspace(0,3)
mu_r = sigmoid(B_sig)[0]
mu_rd = mu_r/(1 - B_sig/mu_r*sigmoid(B_sig)[1])


plt.figure()
plt.plot(B_sig, mu_r, label="$\mu_r$")
plt.plot(B_sig, mu_rd, label="$\mu_{r,d}$")
plt.scatter(B, mu_r_app, label="FEMM")
plt.legend()
plt.grid()
plt.title("Perméabilité magnétique relative")
plt.ylabel("$\mu$[-]")
plt.xlabel("$B$ [$T$]")
plt.savefig("Tests/mu_r/mu.pdf")








# def sigmoid_fit(B, L, k, x0):
#     return L / (1 + np.exp(-k * (B - x0)))

# initial_guess = [np.max(mu_r_app), 0.01, np.median(B)]

# popt, _ = curve_fit(sigmoid_fit, B, mu_r_app, p0=initial_guess)
# L_fit, k_fit, x0_fit = popt



# print(f"L = {L_fit}\n",
#       f"k = {k_fit}\n",
#       f"x0 = {x0_fit}\n")



# # b_1 = sigmoid(B[-1], L_fit, k_fit, x0_fit)
# # b_2 = mu_r_app[-1]
# # diff = b_2-b_1
# # print(f"b_2 = {b_2}")
# # print(f"b_1 = {b_1}")

# # print(f"diff = {diff}")


# b = np.linspace(B[0], B[-1], 100)
# sigmo = sigmoid_fit(b, L_fit, k_fit, x0_fit)





# plt.figure()
# plt.plot(b, sigmo)
# plt.scatter(B, mu_r_app)
# plt.grid()

# B_sat = 6
# print(sigmoid_fit(B_sat, L_fit, k_fit, x0_fit))

plt.show()