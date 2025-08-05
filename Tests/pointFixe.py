import numpy as np



def fixPoint(x, tol, nmax, g):
    n = 0
    delta = np.inf
    while (abs(delta) > tol and n < nmax):
        n += 1
        xold = x
        x = g(x)
        delta = xold - x
        print(f"x[{n}] = {x}")
    return x


# g = lambda x : (x**3 + 1)/3
g = lambda x : (1-0.5)*x + 0.2



tol = 1e-3
nmax = 100


x_sol = fixPoint(0, tol, nmax, g)


print(x_sol)