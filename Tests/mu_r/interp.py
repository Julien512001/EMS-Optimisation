import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator



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

mu_0 = 4*np.pi*1e-7

interp_HB = PchipInterpolator(B, H, extrapolate=True)

b_vec = np.linspace(B[0], B[-1], 1000)
h_vec = interp_HB(b_vec)


dy_dx = np.gradient(b_vec, h_vec)/mu_0




index=27


a = (B[27]-B[26])/(H[27]-H[26])
b = B[27] - a*H[27]



plt.figure()
plt.plot(h_vec, b_vec, label="B(H)")
# plt.scatter(H, B, alpha=0.5)
plt.scatter(H[index], B[index])
plt.tick_params(labelbottom=False, labelleft=False)
plt.grid()

x = [0,H[index]]
y = [0, B[index]]
plt.plot(x,y,c='b',label="$\mu_r$")

x=np.linspace(0,200000,100)
plt.plot(x, a*x+b, c='r',label="$\mu_{r,d}$")

plt.legend()

plt.ylim([0,B[-1]+0.2])
plt.xlabel("$H$ $[A/m]$")
plt.ylabel("$B$ $[T]$")
# plt.title("Caractéristique B-H du \"1020 steel\"")
plt.title("Caractéristique B-H d'un matériau quelconque")
plt.savefig("mu_r.pdf")

plt.show()  