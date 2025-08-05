import numpy as np
from scipy import linalg


def f(x1,x2):
    f1 = x1**2 - 2*x1 - x2 + 0.5
    f2 = x1**2 + 4*x2**2 - 4
    return np.array([f1, f2])


def J(X, h):
    J = np.zeros((2,2))
    J[0,0] = (f(X[0]+h, X[1])[0] - f(X[0]-h, X[1])[0])/(2*h)
    J[0,1] = (f(X[0], X[1]+h)[0] - f(X[0], X[1])[0]-h)/(2*h)
    J[1,0] = (f(X[0]+h, X[1])[1] - f(X[0]-h, X[1])[1])/(2*h)
    J[1,1] = (f(X[0], X[1]+h)[1] - f(X[0], X[1]-h)[1])/(2*h)

    return J   



x1 = 2
x2 = 0.25

X = np.array([x1,x2])
h = 1e-3


delta_X = np.inf
tol = 1e-3
n = 0
while np.linalg.norm(delta_X) > tol: 
    delta_X = -linalg.solve(J(X,h), f(X[0],X[1]))
    X = X + delta_X
    print(n)
    n+=1


print(X)

print("===================")

import numpy as np
from scipy.optimize import newton_krylov

def F(x):
    x1, x2 = x
    f1 = x1**2 - 2*x1 - x2 + 0.5
    f2 = x1**2 + 4*x2**2 - 4
    return np.array([f1, f2])


# Point de départ
x0 = np.array([2.0, 0.25])

# Appel de Newton-Krylov
sol = newton_krylov(F, x0, f_tol=1e-8)

# Affichage des résultats
print("Solution trouvée :", sol)
print("Résidu :", F(sol))
