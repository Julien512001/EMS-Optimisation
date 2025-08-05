import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# Lecture des données
# df = pd.read_excel("Tests/Solve/fer.xlsx")
# H = df["Grade:"][2:].astype(float).to_numpy()
# B = df["430-FR"][2:].astype(float).to_numpy()


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

mu_0 = 4e-7 * np.pi

# Calcul de la dérivée différentielle mu_r = dB/dH / mu_0
dB = np.diff(B)
dH = np.diff(H)
mu_r_diff = dB / (mu_0 * dH)
# mu_r_diff = B/(mu_0*H)
print(mu_r_diff)
# Pour avoir un vecteur mu_r de même taille que B, on rajoute le dernier élément
mu_r_full = np.append(mu_r_diff, mu_r_diff[-1])
# mu_r_full = mu_r_diff
print(mu_r_full)
# Interpolateur mu_r en fonction de B
interp_mu_r = interp1d(B, mu_r_full, kind='linear', fill_value='extrapolate')

# Exemple d'utilisation
B_test = 1.6
mu_r_test = interp_mu_r(B_test)
print(f"mu_r pour B = {B_test} T : {mu_r_test:.1f}")

# Tracé
plt.figure(figsize=(8, 4))
plt.plot(B, mu_r_full, label='mu_r(B)')
plt.xlabel("B (T)")
plt.ylabel("mu_r")
plt.title("Perméabilité relative différentielle")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8, 4))
plt.plot(H, B, label='mu_r(B)')
plt.xlabel("H (H/m)")
plt.ylabel("B (T)")
plt.scatter(H, B)
plt.title("Perméabilité relative différentielle")
plt.grid(True)
plt.legend()
plt.tight_layout()




plt.show()
