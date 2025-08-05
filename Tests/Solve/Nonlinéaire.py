import numpy as np
import time

def fixPointSystem(x0, tol, nmax, g):
    n = 0
    x_old = np.array(x0, dtype=float)
    delta = np.inf

    while delta > tol and n < nmax:
        x_new = g(x_old)
        delta = np.linalg.norm(x_new - x_old) / np.linalg.norm(x_old)
        x_old = x_new
        n += 1
        print(f"Iter {n} : x = {x_new}, erreur = {delta:.3e}")
        time.sleep(0.2)

    if delta <= tol:
        print("Convergence atteinte.")
    else:
        print("Convergence non atteinte.")
    return x_new

# Système de fonctions g(x) = [g1(x), g2(x), g3(x)]
def g(x):
    x1, x2, x3 = x
    return np.array([
        (x1**3 + 1) / 3,         # g1
        (x1**5 + x2**2),         # g2
        (x3**2 + 1) / 6          # g3
    ])

# Condition initiale
x0 = [0.5, 0.5, 0.5]
tol = 1e-6
nmax = 100

# Appel
x_sol = fixPointSystem(x0, tol, nmax, g)
