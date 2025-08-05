import numpy as np
import matplotlib.pyplot as plt


sigma = 3.5e7 # Track conductivity
mu0 = 4*np.pi*1e-7
air_gap = 15e-3



# Pq on en prend pas en compte tc ????
tc = 1e-2 # Track thicness 

wc = 0.1 # Track width
lc = 0.08 # Track length
tm = 0.02 # PM thickness
Br = 1.4 # Remanence PM
tau = 0.04 # PM pole pitch
wm = 0.02 # PM width

def compute_forces(bx_grid, by_grid, bz_grid, X, Y, speed, back_iron=False, N=30, M=30, mur_plate=1):
    # A - phi formulation with a back-iron
    fx = 0j
    fy = 0j
    fz = 0j
    t = air_gap
    for n in range(-N, N + 1):
        # Nombre d'onde spatial dans le sens d'avancement (selon y)
        kn = 2 * n * np.pi / lc
        for m in range(-M, M + 1):
            # Nombre d'onde spatial transversale (selon x)
            km = 2 * m * np.pi / wc

            # Calcul des coefficients de Fourier
            Sx = integ_cube(X, Y, bx_grid, kn, km)
            Sy = integ_cube(X, Y, by_grid, kn, km)
            Sz = integ_cube(X, Y, bz_grid, kn, km)

            # Nombre d'onde spatial total
            anm = np.sqrt(kn ** 2 + km ** 2)
            # Nombre d'onde dans la direction verticale modélisant la formation de courants induits dans la plaque de réaction
            # kn*speed = omega ---> fréquence angulaire
            gamma = np.sqrt(anm ** 2 + 1j * mu0 * mur_plate * sigma * kn * speed)

            # Composante DC donc aucun impact
            if m == 0 and n == 0:
                B = 0
            else:
                if back_iron:
                    A = np.array([[1j * kn * (1 - np.exp(-2 * anm * t)), 0, -gamma/mur_plate],
                                  [1j * km * (1 - np.exp(-2 * anm * t)), gamma/mur_plate, 0],
                                  [-anm * (1 + np.exp(-2 * anm * t)), -1j * km, 1j * kn]])
                else:
                    A = np.array([[1j * kn, 0, -gamma/mur_plate],
                                  [1j * km, gamma/mur_plate,  0,],
                                  [-anm, -1j * km, 1j * kn]])
                b_vec = np.array([[Sx, Sy, Sz]]).T

                coefficient = np.linalg.solve(A, b_vec)[:, 0]
                B = coefficient[0]
            if back_iron:
                fx += wc * lc / mu0 * (-1j * kn * B * (1 - np.exp(-2 * anm * (tm + air_gap))) + Sx) * np.conjugate(anm * B * (1 + np.exp(-2 * anm * (tm + air_gap))) + Sz)
                # fy += wc * lc / mu0 * (-1j * km * B * (1 - np.exp(-2 * anm * (tm + air_gap))) + Sy) * np.conjugate(anm * B * (1 + np.exp(-2 * anm * (tm + air_gap))) + Sz)
                fz += wc * lc / mu0 * ((anm * B * (1 + np.exp(-2 * anm * (tm + air_gap))) + Sz) * np.conjugate(anm * B * (1 + np.exp(-2 * anm * (tm + air_gap))) + Sz) - (-1j * kn * B * (1 - np.exp(-2 * anm * (tm + air_gap))) + Sx) * np.conjugate(-B * (1 - np.exp(-2 * anm * (tm + air_gap))) * 1j * kn + Sx) - (
                            -B * (1 - np.exp(-2 * anm * (tm + air_gap))) * 1j * km + Sy) * np.conjugate(-B * (1 - np.exp(-2 * anm * (tm + air_gap))) * 1j * km + Sy)) / 2
            else:
                fx += wc * lc / mu0 * (-1j * kn * B + Sx) * np.conjugate(anm * B + Sz)
                # fy += wc * lc / mu0 * (-1j * km * B + Sy) * np.conjugate(anm * B + Sz)
                fz += wc * lc / mu0 * ((anm * B + Sz) * np.conjugate(anm * B + Sz) - (-1j * kn * B + Sx) * np.conjugate(-B * 1j * kn + Sx) - (
                        -B * 1j * km + Sy) * np.conjugate(-B * 1j * km + Sy)) / 2

    forces = np.zeros(3)
    forces[1] = np.real(fx)
    forces[0] = np.real(fy)
    forces[2] = np.real(fz)
    return forces


def B_znm(n, m):
    ff = 0.5 #ratio vertical to horizontal PM HA
    beta = wm / wc
    return Br / (n * m * np.pi**2) * np.sin(n * np.pi * ff / 2) * np.sin(m * np.pi * beta / 2) * (1 - np.cos(np.pi * n)) * (1 - np.cos(np.pi * m))



def B_xnm(n, m):
    ff = 0.5 #ratio vertical to horizontal PM HA
    beta = wm / wc
    return 1 * Br / (1j * n * m * np.pi**2) * np.sin(m * np.pi * beta / 2) * np.cos(ff * np.pi * n / 2.) * (1 - np.cos(np.pi * n)) * (1 - np.cos(np.pi * m))


def compute_b_field_HA(X, Y, thickness, width, tau, N=30, M=30):
    mur = 1
    tm = thickness
    tag = air_gap
    Bx = np.zeros(X.shape, dtype=complex)
    By = np.zeros(X.shape, dtype=complex)
    Bz = np.zeros(X.shape, dtype=complex)
    for n in range(-N, N+1):
        for m in range(-M, M+1):
            if n != 0 and m != 0:
                kn = n * np.pi / tau
                km = m * np.pi / wc
                anm = np.sqrt(kn ** 2 + km ** 2)
                coeff = -B_xnm(n, m) * 1j * kn / anm ** 2 * (1 / 2. - 1 / 2. * (1 - mur) / (1 + mur) * np.exp(-2 * anm * tm) - mur / (1 + mur) * np.exp(-anm * tm))
                coeff -= B_znm(n, m) / anm * (1 / 2. + 1 / 2. * (1 - mur) / (1 + mur) * np.exp(-2 * anm * tm) - 1 / (1 + mur) * np.exp(-anm * tm))

                Bx += coeff * 1j * kn * np.exp(anm * (-tag)) * np.exp(1j * kn * X) * np.exp(1j * km * Y)
                By += coeff * 1j * km * np.exp(anm * (-tag)) * np.exp(1j * kn * X) * np.exp(1j * km * Y)
                Bz += coeff * anm * np.exp(anm * (-tag)) * np.exp(1j * kn * X) * np.exp(1j * km * Y)
    return Bx, By, Bz

# def integ_cube(X, Y, b, hn, hm):
#     dx = abs(X[0, 1] - X[0, 0])
#     dy = abs(Y[1, 0] - Y[0, 0])
#     integ = np.sum(b * np.exp(-1j * X * hn) * np.exp(-1j * hm * Y) * dx * dy)
#     return integ / wc / lc

def integ_cube(X, Y, b, hn, hm):
    return np.trapz(np.trapz(b * np.exp(-1j * (hn * X + hm * Y)), x=Y[:, 0], axis=0), x=X[0, :]) / (wc * lc)




# ACTION

x = np.linspace(-lc/2, lc/2, 101)
y = np.linspace(-wc/2, wc/2, 101)
X, Y = np.meshgrid(x, y)


bx, by, bz = compute_b_field_HA(X, Y, tm, wm, tau)

print(bx.shape, by.shape, bz.shape)


# fig = plt.figure()

speed = 700



N = 5
M = 5
forces = compute_forces(bx, by, bz, X, Y, speed=speed, back_iron=True, mur_plate=4000, N=N, M=M)
print(forces)

# plt.show()



P_drag = speed*np.abs(forces[1])


print(P_drag)