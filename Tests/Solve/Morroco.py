import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0
from scipy.optimize import minimize

# Données BH
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

# Calcul de la perméabilité relative effective (évite H = 0)
H[H == 0] = 1e-9
mu_eff = B / (mu_0 * H)

# Fonction de Morroco définie proprement
def mu_r_morroco(B, k1, k2, k3, k4):
    B2 = B**2
    num = (k4 - k1) * B2 * k1
    denom_internal = (B2 * k1 + k3)
    denom = k2 + (B2 * k1 + k3 * (2 * k1 + 1)) * (num / denom_internal**2)
    return 1.0 / denom

# Fonction de coût à minimiser : erreur quadratique entre mu_r données et modèle
def cost(params):
    k1, k2, k3, k4 = params
    mu_model = mu_r_morroco(B, k1, k2, k3, k4)
    return np.mean((mu_model - mu_eff)**2)

# Bornes et initialisation
bounds = [(1e-3, 1e4), (1e-6, 1), (1e-6, 10), (1e-3, 1e4)]
initial_guess = [10, 0.01, 0.1, 100]

result = minimize(cost, initial_guess, bounds=bounds, method='L-BFGS-B')
k1, k2, k3, k4 = result.x

# Affichage
print("Paramètres optimisés :")
print(f"k1 = {k1:.4e}")
print(f"k2 = {k2:.4e}")
print(f"k3 = {k3:.4e}")
print(f"k4 = {k4:.4e}")

# Tracé comparatif
B_plot = np.linspace(min(B), max(B), 300)
mu_fit = mu_r_morroco(B_plot, k1, k2, k3, k4)

plt.figure(figsize=(8, 5))
plt.plot(B, mu_eff, 'o', label='μᵣ (données BH)')
plt.plot(B_plot, mu_fit, '-', label='μᵣ (fit Morroco)')
plt.xlabel("B [T]")
plt.ylabel("μᵣ [-]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title("Ajustement robuste avec la fonction de Morroco")
plt.show()
