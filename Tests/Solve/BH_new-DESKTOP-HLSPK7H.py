import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

B = np.array([
    0.000000, 0.227065, 0.454130, 0.681195, 0.908260, 1.135330, 1.362390,
    1.589350, 1.812360, 2.010040, 2.133160, 2.199990, 2.254790, 2.299930,
    2.342510, 2.378760, 2.415010, 2.451260, 2.487500, 2.523750, 2.560000
])

H = np.array([
    0.000000, 13.898400, 27.796700, 42.397400, 61.415700, 82.382400,
    144.669000, 897.760000, 4581.740000, 17736.200000, 41339.300000,
    68321.800000, 95685.500000, 123355.000000, 151083.000000, 178954.000000,
    206825.000000, 234696.000000, 262568.000000, 290439.000000, 318310.000000
])



mu_0 = 4*np.pi*1e-7


B_sat = 1.7
H_sat = np.interp(B_sat, B, H)

mu_r = B_sat/(mu_0*H_sat)




b_vec = np.linspace(B[0], B[-1], len(B))
interp_HB = PchipInterpolator(B, H, extrapolate=True)
h_vec = interp_HB(b_vec)

a = (B[-2] - B[-1])/(H[-2] - H[-1])
b = B[-1] - a*H[-1]

h_vec_bis = np.linspace(H[-1], 3000000, 100)
b_vec_bis = a*h_vec_bis + b

h_final = np.hstack([h_vec, h_vec_bis])
b_final = np.hstack([b_vec, b_vec_bis])


h_mu = np.interp(B_sat, b_final, h_final)

mu_r = B_sat/(mu_0*h_mu)




mu_r_fit = b_final/(mu_0*h_final)
mu_r_fit[0] = mu_r_fit[1]


print(f"B_sat = {B_sat}\n",
      f"H_sat = {h_mu}\n",
      f"mu_r = {mu_r}\n")

plt.figure()
plt.plot(h_vec, b_vec)
plt.scatter(H,B)
plt.grid()

plt.figure()
plt.scatter(h_final, b_final)
plt.plot(h_final, b_final)
plt.grid()

plt.figure()
plt.plot(b_final, mu_r_fit)
plt.grid()

# Perméabilité différentielle méthode 1
# On prend la défition direct de la perméabilité différentielle
b_vec = np.linspace(B[0], B[-1], len(B))
interp_HB = PchipInterpolator(B, H, extrapolate=True)
h_vec = interp_HB(b_vec)

a = (B[-2] - B[-1])/(H[-2] - H[-1])
b = B[-1] - a*H[-1]

h_vec_bis = np.linspace(H[-1], 3000000, 100)
b_vec_bis = a*h_vec_bis + b

h_final = np.hstack([h_vec, h_vec_bis])
b_final = np.hstack([b_vec, b_vec_bis])


B_sat = 6
if (B_sat < 1e-6):
    B_sat = 1e-6
if (B_sat > 6):
    B_sat = 6
B_down = B_sat - 1e-6   
B_up   = B_sat + 1e-6
H_down = np.interp(B_down, b_final, h_final)
H_up = np.interp(B_up, b_final, h_final)

mu_r_diff = 1/mu_0 * (B_up-B_down)/(H_up - H_down)

print(f"mu_r_diff = {mu_r_diff}")

a = mu_r_diff*mu_0
b = B_down - a*H_down
plt.figure()
plt.plot(h_final, b_final)
plt.plot(h_final, a*h_final + b)
plt.ylim(0,b_final[-1])
plt.grid()




# Perméabilité différentielle méthode 2
# mu_r_diff = mu_r + dmu_r/dB * B
# Fitting mu_r with a sigmoïde



B_sat = 10
def sigmoid(B, L, k, x0):
    return L / (1 + np.exp(-k * (B - x0)))

initial_guess = [np.max(mu_r_fit), 0.01, np.median(b_final)]

popt, _ = curve_fit(sigmoid, b_final, mu_r_fit, p0=initial_guess)
L_fit, k_fit, x0_fit = popt


print(f"L = {L_fit}\n",
      f"k = {k_fit}\n",
      f"x0 = {x0_fit}\n")
b_fit = np.linspace(min(b_final), max(b_final), 500)
mu_r_sig = sigmoid(b_fit, L_fit, k_fit, x0_fit)

def d_sigmoid(B, L, k, x0):
    exp_term = np.exp(-k * (B - x0))
    return (L * k * exp_term) / ((1 + exp_term)**2)

d_mu_r_sig = d_sigmoid(b_fit, L_fit, k_fit, x0_fit)
plt.figure()
plt.plot(b_fit, d_mu_r_sig)
plt.grid()

dmu_dB = d_sigmoid(B_sat, L_fit, k_fit, x0_fit)
print(dmu_dB)
mu_r = sigmoid(B_sat, L_fit, k_fit, x0_fit)
print(mu_r)


# mu_r_diff_bis = mu_r + dmu_dB*B_sat
mu_r_diff_bis = mu_r/(1 - B_sat/mu_r * dmu_dB)


print(f"mu_r_diff_bis = {mu_r_diff_bis}")


plt.figure(figsize=(8, 5))
plt.plot(b_final, mu_r_fit, 'o', label='Données')
plt.plot(b_fit, mu_r_sig, '-', label='Sigmoïde ajustée')
plt.xlabel('B (T)')
plt.ylabel('mu_r')
plt.legend()
plt.grid()
plt.title('Ajustement d\'une sigmoïde à B(H)')



plt.show()