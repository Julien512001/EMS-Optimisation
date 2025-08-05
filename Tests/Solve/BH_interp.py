import numpy as np
import matplotlib.pyplot as plt
# Données BH
B_vals = np.array([
    0.000000, 0.227065, 0.454130, 0.681195, 0.908260, 1.135330, 1.362390,
    1.589350, 1.812360, 2.010040, 2.133160, 2.199990, 2.254790, 2.299930,
    2.342510, 2.378760, 2.415010, 2.451260, 2.487500, 2.523750, 2.560000
])

H_vals = np.array([
    0.000000, 13.898400, 27.796700, 42.397400, 61.415700, 82.382400,
    144.669000, 897.760000, 4581.740000, 17736.200000, 41339.300000,
    68321.800000, 95685.500000, 123355.000000, 151083.000000, 178954.000000,
    206825.000000, 234696.000000, 262568.000000, 290439.000000, 318310.000000
])

# Fonction d'interpolation H(B)
def get_H_from_B(B_query):
    """
    Retourne H(B) interpolé linéairement à partir du tableau BH fourni.
    B_query peut être un scalaire ou un tableau numpy.
    """
    B_query = np.asarray(B_query)
    return np.interp(B_query, B_vals, H_vals)
    
B_test = 2
H_test = np.interp(B_test, B_vals, H_vals)
print(H_test)

B_up = B_test + 1e-3
B_down = B_test - 1e-3

H_up = get_H_from_B(B_up)
H_down = get_H_from_B(B_down)


mu_0 = 4*np.pi*1e-7

dB = B_up - B_down
dH = H_up - H_down

mu_r = (dB)/(mu_0*dH)



print("B =", B_test)
print("H =", H_test)
print(f"B = {B_test} : mu_r = {mu_r}, mu = {mu_r*mu_0}")


a = (B_vals[-1] - B_vals[-2])/(mu_0*(H_vals[-1] - H_vals[-2]))
print(f"mu_r for B > 2.56 T : {a}")
a = (B_vals[1] - B_vals[0])/(mu_0*(H_vals[1] - H_vals[0]))
print(f"mu_r for B = 0 T : {a}")






plt.figure()
plt.scatter(H_vals,B_vals)
plt.plot(H_vals, B_vals)
plt.grid()

# plt.show()


# Calcul de mu = dB/dH
dB = np.gradient(B_vals)
dH = np.gradient(H_vals)
mu = dB / dH
mu0 = 4 * np.pi * 1e-7
mu_r = mu / mu0

# Afficher mu_r en fonction de B
plt.figure()
plt.plot(B_vals, mu_r)
plt.xlabel("B [T]")
plt.ylabel("μ_r")
plt.title("Perméabilité relative différentielle")
plt.grid(True)
plt.show()



import numpy as np
from scipy.interpolate import interp1d

def get_mu_r_from_B(B_sat):
    B_vals = np.array([
        0.000000, 0.227065, 0.454130, 0.681195, 0.908260, 1.135330, 1.362390,
        1.589350, 1.812360, 2.010040, 2.133160, 2.199990, 2.254790, 2.299930,
        2.342510, 2.378760, 2.415010, 2.451260, 2.487500, 2.523750, 2.560000
    ])

    H_vals = np.array([
        0.000000, 13.898400, 27.796700, 42.397400, 61.415700, 82.382400,
        144.669000, 897.760000, 4581.740000, 17736.200000, 41339.300000,
        68321.800000, 95685.500000, 123355.000000, 151083.000000, 178954.000000,
        206825.000000, 234696.000000, 262568.000000, 290439.000000, 318310.000000
    ])


    mu0 = 4 * np.pi * 1e-7  # perméabilité du vide en H/m

    # Calcul de la dérivée dB/dH (différentielle)
    dB_dH = np.gradient(B_vals, H_vals)

    # Calcul de mu_r = dB / (mu0 * dH)
    mu_r_array = dB_dH / mu0

    # Interpolation de mu_r en fonction de B
    mu_r_interp = interp1d(B_vals, mu_r_array, kind='linear',
                           bounds_error=False, fill_value="extrapolate")

    return float(mu_r_interp(B_sat))

print(get_mu_r_from_B(0.883067444213307))


B = np.linspace(0, 5, 1000)

my_mu = get_mu_r_from_B(B)

plt.figure()
plt.plot(B, my_mu)
plt.show()