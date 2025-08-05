import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Utility import *
from tqdm import tqdm
from scipy.integrate import simpson
from scipy.interpolate import griddata

from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.interpolate import RectBivariateSpline

class MaxwellFourier:

    def __init__(self, myMEC, myMagneticField, vy, Bz):
        self.myMEC = myMEC

        self.x, self.y, _ = myMagneticField.get_coordinate()
        self.Bz = Bz
        # print(self.Bz.shape)
        self.tag = self.myMEC.h7 - myMagneticField.tag

        self.vy = vy
        self.translateVariables()
        self.Fx, self.Fy, self.Fz = self.Fourier()

    def Fourier(self, back_iron=True, N=20, M=20):
        bx = 0j
        by = 0j
        bz = 0j
        t = self.air_gap

        mu0 = self.mu0
        mur = self.mur_plate
        sigma = self.sigma
        speed = self.speed
        tag = self.tag
        tc = self.tc
        tm = self.tm
        wc = self.wc
        lc = self.lc
        X = self.X
        Y = self.Y
        Bz = self.Bz

        for n in (range(-N, N + 1)):
            kn = 2 * n * np.pi / wc
            for m in range(-M, M + 1):
                km = 2 * m * np.pi / lc
                anm2 = kn ** 2 + km ** 2
                if anm2 == 0:
                    continue  # évite division par zéro inutile
                Sz = self.integ_cube(X, Y, Bz, kn, km)
                if Sz == 0:
                    continue  # évite toute la résolution du système pour rien
                anm = np.sqrt(anm2)
                gamma = np.sqrt(anm2 + 1j * mu0 * mur * sigma * km * speed)

                exp = np.exp
                exp_anm_t = exp(anm * t)
                exp_anm_tag = exp(anm * tag)
                exp_minus_anm_tag = exp(-anm * tag)
                exp_gamma_tc = exp(gamma * tc)
                exp_minus_gamma_tc = exp(-gamma * tc)
                exp_minus_anm_tc = exp(-anm * tc)
                exp_anm_tm = exp(anm * tm)
                exp_minus_anm_tm = exp(-anm * tm)

                A = np.array([
                    [exp_anm_t, 1/exp_anm_t, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, exp_anm_t, 1/exp_anm_t, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, anm * exp_anm_tag, -anm * exp_minus_anm_tag, 0, 0, -anm * exp_anm_tag, anm * exp_minus_anm_tag, 0, 0, 0, 0, 0],
                    [-anm * exp_anm_tag, anm * exp_minus_anm_tag, 0, 0, anm * exp_anm_tag, -anm * exp_minus_anm_tag, 0, 0, 0, 0, 0, 0, 0],
                    [1j * km * exp_anm_tag, 1j * km * exp_minus_anm_tag, -1j * kn * exp_anm_tag, -1j * kn * exp_minus_anm_tag,
                    -1j * km * exp_anm_tag, -1j * km * exp_minus_anm_tag, 1j * kn * exp_anm_tag, 1j * kn * exp_minus_anm_tag,
                    0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, anm, -anm, 0, 0, -gamma / mur, gamma / mur, 0],
                    [0, 0, 0, 0, -anm, anm, 0, 0, gamma / mur, -gamma / mur, 0, 0, 0],
                    [0, 0, 0, 0, 1j * km, 1j * km, -1j * kn, -1j * kn, -1j * km, -1j * km, 1j * kn, 1j * kn, 0],
                    [0, 0, 0, 0, 1j * kn, 1j * kn, 1j * km, 1j * km, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1j * kn, 1j * kn, 1j * km, 1j * km, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gamma / mur / mu0 * exp_minus_gamma_tc, -gamma / mur / mu0 * exp_gamma_tc, -1j * kn * exp_minus_anm_tc],
                    [0, 0, 0, 0, 0, 0, 0, 0, -gamma / mur / mu0 * exp_minus_gamma_tc, gamma / mur / mu0 * exp_gamma_tc, 0, 0, -1j * km * exp_minus_anm_tc],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1j * km * exp_minus_gamma_tc, 1j * km * exp_gamma_tc, -1j * kn * exp_minus_gamma_tc, -1j * kn * exp_gamma_tc,
                    -anm * mu0 * exp_minus_anm_tc]
                ])

                if not back_iron:
                    A[0, :] = [1] + [0] * 12
                    A[1, :] = [0, 0, 1] + [0] * 10

                b_vec = np.zeros((13, 1), dtype=complex)
                b_vec[4, 0] = Sz

                coeffs = spsolve(csc_matrix(A), b_vec)

                exp_phase = exp(1j * (km * Y + kn * X))

                cxp, cxm = coeffs[6], coeffs[7]
                cyp, cym = coeffs[4], coeffs[5]


                bx += -anm * (cxp * exp_anm_tm - cxm * exp_minus_anm_tm) * exp_phase
                by += anm * (cyp * exp_anm_tm - cym * exp_minus_anm_tm) * exp_phase
                bz += (1j * kn * (cxp * exp_anm_tm + cxm * exp_minus_anm_tm) -
                    1j * km * (cyp * exp_anm_tm + cym * exp_minus_anm_tm)) * exp_phase

        forces = np.zeros(3)
        forces[1] = self.get_fy(by, bz)
        forces[2] = self.get_fz(bx, by, bz)
        return forces

    # def Fourier(self, back_iron=True, N=25, M=25):
    #     # A - phi formulation with a back-iron
    #     bx = 0j
    #     bz = 0j
    #     by = 0j
    #     t = self.air_gap
    #     for n in tqdm(range(-N, N + 1)):
    #         kn = 2 * n * np.pi / self.wc
    #         for m in range(-M, M + 1):
    #             km = 2 * m * np.pi / self.lc
    #             Sz = self.integ_cube(self.X, self.Y, self.Bz, kn, km)
    #             Sx = 0.0
    #             Sy = 0.0
    #             anm = np.sqrt(kn ** 2 + km ** 2)
    #             gamma = np.sqrt(anm ** 2 + 1j * self.mu0 * self.mur_plate * self.sigma * km * self.speed)
    #             if m == 0 and n == 0:
    #                 B = 0
    #             else:
    #                 exp_anm_pos = np.exp(anm * t)
    #                 exp_anm_neg = np.exp(-anm * t)
    #                 exp_tag_pos = np.exp(anm * self.tag)
    #                 exp_tag_neg = np.exp(-anm * self.tag)
    #                 exp_gamma_pos = np.exp(gamma * self.tc)
    #                 exp_gamma_neg = np.exp(-gamma * self.tc)
    #                 exp_tc_neg = np.exp(-anm * self.tc)
    #                 exp_tm_pos = np.exp(anm * self.tm)
    #                 exp_tm_neg = np.exp(-anm * self.tm)
    #                 A = np.array([[exp_anm_pos, exp_anm_neg, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                             [0, 0, exp_anm_pos, exp_anm_neg, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                             [0, 0, anm * exp_tag_pos, -anm * exp_tag_neg, 0, 0, -anm * exp_tag_pos, anm * exp_tag_neg, 0, 0, 0, 0, 0],
    #                             [-anm * exp_tag_pos, anm * exp_tag_neg, 0, 0, anm * exp_tag_pos, -anm * exp_tag_neg, 0, 0, 0, 0, 0, 0, 0],
    #                             [1j * km * exp_tag_pos, 1j * km * exp_tag_neg, -1j * kn * exp_tag_pos, -1j * kn * exp_tag_neg, -1j * km* exp_tag_pos, -1j * km* exp_tag_neg, 1j * kn* exp_tag_pos, 1j * kn* exp_tag_neg, 0, 0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0, 0, anm, -anm, 0, 0, -gamma/self.mur_plate, gamma/self.mur_plate, 0],
    #                             [0, 0, 0, 0, -anm, anm , 0, 0, gamma/self.mur_plate, -gamma/self.mur_plate, 0, 0, 0],
    #                             [0, 0, 0, 0, 1j * km, 1j * km, -1j * kn, -1j * kn, -1j * km, -1j * km, 1j * kn, 1j * kn, 0],
    #                             [0, 0, 0, 0, 1j * kn, 1j * kn, 1j * km, 1j * km, 0, 0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0, 0, 0, 0, 1j * kn, 1j * kn, 1j * km, 1j * km, 0],
    #                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gamma/self.mur_plate/self.mu0 * exp_gamma_neg, -gamma/self.mur_plate/self.mu0 * exp_gamma_pos, -1j * kn * exp_tc_neg],
    #                             [0, 0, 0, 0, 0, 0, 0, 0, -gamma/self.mur_plate/self.mu0 * exp_gamma_neg, gamma/self.mur_plate/self.mu0 * exp_gamma_pos, 0, 0, -1j * km * exp_tc_neg],
    #                             [0, 0, 0, 0, 0, 0, 0, 0, 1j * km * exp_gamma_neg, 1j * km * exp_gamma_pos, -1j * kn * exp_gamma_neg, -1j * kn * exp_gamma_pos, -anm * self.mu0 * exp_tc_neg]])
    #                 if not back_iron:
    #                     A[0, :] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #                     A[1, :] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #                 b_vec = np.array([[0, 0, 0, 0, Sz, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    #                 A = csc_matrix(A)
    #                 coefficient = spsolve(A, b_vec)
    #                 bx += (-anm * (coefficient[6] * exp_tm_pos - coefficient[7] * exp_tm_neg)) * np.exp(1j * km * self.Y) * np.exp(1j * kn * self.X)
    #                 by += (anm * (coefficient[4] * exp_tm_pos - coefficient[5] * exp_tm_neg)) * np.exp(1j * km * self.Y) * np.exp(1j * kn * self.X)
    #                 bz += (1j * kn * (coefficient[6] * exp_tm_pos + coefficient[7] * exp_tm_neg) - 1j * km * (coefficient[4] * exp_tm_pos + coefficient[5] * exp_tm_neg)) * np.exp(1j * km * self.Y) * np.exp(1j * kn * self.X)
    #     plt.figure()
    #     plt.scatter(self.X, self.Y, c=bz)
    #     plt.grid()
    #     forces = np.zeros(3)
    #     forces[1] = self.get_fy(by, bz)
    #     forces[2] = self.get_fz(bx, by, bz)
    #     return forces


    # def integ_cube(self, X, Y, b, hn, hm):
    #     return simpson(simpson(b * np.exp(-1j * (hn * X + hm * Y)),self.y),self.x)/ (self.wc * self.lc)



    def translateVariables(self):
        myModDim = self.myMEC.modDim
        myProbDim = self.myMEC.probDim
        myParams = self.myMEC.params

        air = 3000.0e-3
        self.tm = 0.0
        self.tc = self.myMEC.e_plate
        self.air_gap = myModDim["e_gap"]
        self.wc = 2*self.myMEC.L
        self.lc = self.myMEC.d_maglev + 2*air
        # self.wc = self.x[-1] - self.x[0]
        # self.lc = self.y[-1] - self.y[0]

        self.X, self.Y = np.meshgrid(self.x, self.y)

        
        self.mu0 = myParams['mu_0']
        self.mur_plate = myParams['mu_fer']
        # self.sigma = 3.5e7
        self.sigma = 1.12e7 # Pour le fer pure
        self.speed = self.vy
        # print(self.Bz.shape)
        Nx_new = 100
        Ny_new = 100
        self.wc = self.x[-1] - self.x[0]
        self.lc = self.y[-1] - self.y[0]
        dx = self.wc / Nx_new
        dy = self.lc / Ny_new
        x_min = -self.wc / 2
        x_max =  self.wc / 2
        y_min = -self.lc / 2
        y_max =  self.lc / 2
        x_u = np.linspace(x_min, x_max, Nx_new, endpoint=False)
        y_u = np.linspace(y_min, y_max, Ny_new, endpoint=False)
        X_u, Y_u = np.meshgrid(x_u, y_u)
        f_interp = RectBivariateSpline(self.y, self.x, self.Bz)
        Bz_u = f_interp(y_u, x_u)
        self.x = x_u
        self.y = y_u
        self.X = X_u
        self.Y = Y_u
        self.Bz = Bz_u

        self.dx = dx
        self.dy = dy

        # print("wc attendu:", self.wc, " -> dx*Nx =", self.dx * Nx_new)
        # print("lc attendu:", self.lc, " -> dy*Ny =", self.dy * Ny_new)



    def integ_cube(self, X, Y, b, hn, hm):
        return simpson(simpson(b * np.exp(-1j * (hn * X + hm * Y)),self.y, axis=0),self.x, axis=0)/ (self.wc * self.lc)
    
    def get_fz(self, bx, by, bz):
        return np.real(1/2*simpson(simpson(bz * np.conjugate(bz) - bx * np.conjugate(bx) - by * np.conjugate(by), self.y, axis=0), self.x, axis=0)/self.mu0)

    def get_fy(self, by, bz):
        return np.real(simpson(simpson(by * np.conjugate(bz), self.y, axis=0), self.x, axis=0)/self.mu0)
