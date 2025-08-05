import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simpson

sigma = 1.12e7 # Track conductivity
mu0 = 4*np.pi*1e-7
air_gap = 9e-5
mur_plate = 10


# Pq on en prend pas en compte tc ????
tc = 28e-5 # Track thicness 

wc = 11.0e-3 # Track width
lc = 0.00356 # Track length
tm = 0.0 # PM thickness
tag = 4.5e-5


def compute_forces(bx_grid, by_grid, bz_grid, X, Y, speed, back_iron=True, N=30, M=30, mur_plate=10):
    # A - phi formulation with a back-iron
    bx = 0j
    bz = 0j
    by = 0j
    t = tm + air_gap
    dx = abs(X[0, 1] - X[0, 0])
    dy = abs(Y[1, 0] - Y[0, 0])
    for n in tqdm(range(-N, N + 1)):
        kn = 2 * n * np.pi / wc
        for m in range(-M, M + 1):
            km = 2 * m * np.pi / lc
            Sz = integ_cube(X, Y, bz_grid, 2 * n * np.pi / wc, 2 * m * np.pi / lc)
            # Sx = integ_cube(X, Y, bx_grid, 2 * n * np.pi / wc, 2 * m * np.pi / lc)
            # Sy = integ_cube(X, Y, by_grid, 2 * n * np.pi / wc, 2 * m * np.pi / lc)
            Sx = 0.0
            Sy = 0.0
            anm = np.sqrt(kn ** 2 + km ** 2)
            gamma = np.sqrt(anm ** 2 + 1j * mu0 * mur_plate * sigma * km * speed)
            if m == 0 and n == 0:
                B = 0
            else:
                A = np.array([[np.exp(anm * t), np.exp(-anm * t), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, np.exp(anm * t), np.exp(-anm * t), 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, anm * np.exp(anm * tag), -anm * np.exp(anm * -tag), 0, 0, -anm * np.exp(anm * tag), anm * np.exp(anm * -tag), 0, 0, 0, 0, 0],
                              [-anm * np.exp(anm * tag), anm * np.exp(anm * -tag), 0, 0, anm * np.exp(anm * tag), -anm * np.exp(anm * -tag), 0, 0, 0, 0, 0, 0, 0],
                              [1j * km * np.exp(anm * tag), 1j * km * np.exp(-anm * tag), -1j * kn * np.exp(anm * tag), -1j * kn * np.exp(-anm * tag), -1j * km* np.exp(anm * tag), -1j * km* np.exp(-anm * tag), 1j * kn* np.exp(anm * tag), 1j * kn* np.exp(-anm * tag), 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, anm, -anm, 0, 0, -gamma/mur_plate, gamma/mur_plate, 0],
                              [0, 0, 0, 0, -anm, anm , 0, 0, gamma/mur_plate, -gamma/mur_plate, 0, 0, 0],
                              [0, 0, 0, 0, 1j * km, 1j * km, -1j * kn, -1j * kn, -1j * km, -1j * km, 1j * kn, 1j * kn, 0],
                              [0, 0, 0, 0, 1j * kn, 1j * kn, 1j * km, 1j * km, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1j * kn, 1j * kn, 1j * km, 1j * km, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gamma/mur_plate/mu0 * np.exp(-gamma * tc), -gamma/mur_plate/mu0 * np.exp(gamma * tc), -1j * kn * np.exp(-anm * tc)],
                              [0, 0, 0, 0, 0, 0, 0, 0, -gamma/mur_plate/mu0 * np.exp(-gamma * tc), gamma/mur_plate/mu0 * np.exp(gamma * tc), 0, 0, -1j * km * np.exp(-anm * tc)],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1j * km * np.exp(-gamma * tc), 1j * km * np.exp(gamma * tc), -1j * kn * np.exp(-gamma * tc), -1j * kn * np.exp(gamma * tc), -anm * mu0 * np.exp(-anm * tc)]])
                if not back_iron:
                    A[0, :] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    A[1, :] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                b_vec = np.array([[0, 0, Sx, Sy, Sz, 0, 0, 0, 0, 0, 0, 0, 0]]).T
                coefficient = np.linalg.solve(A, b_vec)[:, 0]
                bx += (-anm * (coefficient[6] * np.exp(anm * tm) - coefficient[7] * np.exp(-anm * tm))) * np.exp(1j * km * Y) * np.exp(1j * kn * X)
                by += (anm * (coefficient[4] * np.exp(anm * tm) - coefficient[5] * np.exp(-anm * tm))) * np.exp(1j * km * Y) * np.exp(1j * kn * X)
                bz += (1j * kn * (coefficient[6] * np.exp(anm * tm) + coefficient[7] * np.exp(-anm * tm)) - 1j * km * (coefficient[4] * np.exp(anm * tm) + coefficient[5] * np.exp(-anm * tm))) * np.exp(1j * km * Y) * np.exp(1j * kn * X)
    forces = np.zeros(3)
    forces[0] = np.real(np.sum(bx * np.conjugate(bz) * dx * dy) / mu0)
    forces[1] = np.real(np.sum(by * np.conjugate(bz) * dx * dy) / mu0)
    forces[2] = np.real(np.sum(bz * np.conjugate(bz) - bx * np.conjugate(bx) - by * np.conjugate(by)) * dx * dy / mu0 / 2)
    return forces

def integ_cube(X, Y, b, hn, hm):
    return simpson(simpson(b * np.exp(-1j * (hn * X + hm * Y)),X[0, :]),Y[:, 0])/ (wc * lc)

    return np.trapz(np.trapz(b * np.exp(-1j * (hn * X + hm * Y)), x=Y[:, 0], axis=0), x=X[0, :]) / (wc * lc)



# Chargement des données
dfx = pd.read_csv("Validation/MF/Bx_MF.csv", comment='%', header=None, names=['x', 'y', 'Bx'])
dfy = pd.read_csv("Validation/MF/By_MF.csv", comment='%', header=None, names=['x', 'y', 'By'])
dfz = pd.read_csv("Validation/MF/Bz_MF.csv", comment='%', header=None, names=['x', 'y', 'Bz'])

# Extraction
x = dfx["x"].to_numpy()*1e-3
y = dfx["y"].to_numpy()*1e-3
Bx = dfx["Bx"].to_numpy()
By = dfy["By"].to_numpy()
Bz = dfz["Bz"].to_numpy()


xi = np.linspace(x.min(), x.max(), 101)
yi = np.linspace(y.min(), y.max(), 101)
Xi, Yi = np.meshgrid(xi, yi)


print(np.min(x)*2, 2*np.min(y))

Bx_grid = -griddata((x, y), Bx, (Xi, Yi), method='nearest')
By_grid = griddata((x, y), By, (Xi, Yi), method='nearest')
Bz_grid = -griddata((x, y), Bz, (Xi, Yi), method='nearest')


# xi = np.linspace(-wc/2, wc/2, 200)
# yi = np.linspace(-lc/2, lc/2, 200)
# Xi, Yi = np.meshgrid(xi, yi)


plt.figure()
plt.grid()
plt.axis("equal")
plt.scatter(x, y, c=Bx, cmap='viridis', s=5)
plt.colorbar(label='Bx')
plt.title("Scatter Bx")

plt.figure()
plt.grid()
plt.axis("equal")
plt.scatter(x, y, c=By, cmap='viridis', s=5)
plt.colorbar(label='By')
plt.title("Scatter By")

plt.figure()
plt.grid()
plt.axis("equal")
plt.scatter(x, y, c=Bz, cmap='viridis', s=5)
plt.colorbar(label='Bz')
plt.title("Scatter Bz")



fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xi, Yi, Bx_grid.T, cmap='viridis')
ax.set_title("Surface Bx")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xi, Yi, By_grid.T, cmap='viridis')
ax.set_title("Surface By")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xi, Yi, Bz_grid.T, cmap='viridis')
ax.set_title("Surface Bz")
# plt.show()

speed = 100

Fz = np.trapz(np.trapz(Bz_grid**2 - 0.5*(Bx_grid**2 + By_grid**2 + Bz_grid**2), xi, axis=0), yi, axis=0)
Fz = Fz/mu0

Fy = np.trapz(np.trapz(By_grid*Bz_grid, xi, axis=0), yi, axis=0)
Fy = Fy/mu0

print(f"Fz = {Fz}\n"
      f"Fy = {Fy}\n")



Fz = simpson(simpson(Bz_grid**2 - 0.5*(Bx_grid**2 + By_grid**2 + Bz_grid**2), yi), xi)
Fz = Fz/mu0

Fz = 1/2*np.trapz(np.trapz(Bz_grid**2, xi, axis=0), yi, axis=0)
Fz = Fz/mu0


# forces = compute_forces(Bx_grid, By_grid, Bz_grid, Xi, Yi, speed, mur_plate=mur_plate)
# print(forces)

speed = np.array([0.0,20.0,40.0,60.0,80.0,100.0,120.0,140.0])
Fz = np.zeros_like(speed)
Fy = np.zeros_like(speed)

for i in range(len(speed)):
    print(speed[i])
    forces = compute_forces(Bx_grid, By_grid, Bz_grid, Xi,Yi, speed[i], mur_plate=mur_plate)
    print(forces)
    Fy[i] = forces[1]
    Fz[i] = forces[2]

print(Fy)
print(Fz)
