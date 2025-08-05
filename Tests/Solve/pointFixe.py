import numpy as np
import matplotlib.pyplot as plt


def fixPoint(x, tol, nmax, g):
    n = 0
    delta = np.inf
    xold = x
    while (abs(delta) > tol and n < nmax):
        n += 1
        x = g(x)
        delta = xold - x
        xold = x
        print(f"x[{n}] = {x}")
    return x


g = lambda x : (x**3 + 1)/3
# g = lambda x : (1-0.5)*x + 0.2



tol = 1e-3
nmax = 100


x_sol = fixPoint(0, tol, nmax, g)


x = np.linspace(0, 1, 100)
plt.figure()
plt.plot(x, g(x))
plt.plot(x, x)

plt.grid()




print(x_sol)

plt.show()