import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, block_diag
import scipy.sparse as sp
from Utility import *
from scipy.interpolate import interp1d
import time
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator


class Network:


    def __init__(self, myMEC, myMesh):
        myPrint("----Network : Initialization of the network----\n")
        self.mesh = myMesh.mesh
        self.params = myMEC.params
        self.probDim = myMEC.probDim
        self.mu_abs = {"air": self.params["mu_0"], "fer": self.params["mu_0"]*self.params["mu_fer"], "coil": self.params["mu_0"]*self.params["mu_coil"]}
        self.R_x = lil_matrix((self.params['Nx'], self.params['Nx']))
        self.R_z = lil_matrix((self.params['Nz'], self.params['Nz']))
        self.S_x = np.zeros(self.params['Nx'])
        self.S_z = np.zeros(self.params['Nz'])
        self.mask_fer_x = np.zeros(self.params["Nx"])
        self.mask_fer_z = np.zeros(self.params["Nz"])
        self.Reluctance()
        self.MMF_Matrix()
        self.Xsi    = self.create_Xsi()

        self.mask_fer_bool = np.hstack([self.mask_fer_x.astype(bool), self.mask_fer_z.astype(bool)])

        
        myPrint("----Network : End----\n")


    def Reluctance(self, B=None, type=None):

        i = 0
        j = 0
        n = 0
        for key, elem in self.mesh.items():
            # Condition permettant d'éviter de recalculer la réluctance des zones
            # d'air et de bobinage qui ne sont pas impactés par la saturation
            if ((B is not None) and (elem["material"] != "fer")):
                n+=1
                if ((i % self.params['P']) == 0):
                    i += 1
                else:
                    i += 2

                if (j < self.params["p"]):
                    j += 1
                elif (j > self.params["Nz"] - self.params['p']-1):
                    j += 1
                else:
                    if ((j+1)%(self.params["p"]) == 0):
                        j += self.params["p"]+1
                    else:
                        j += 1
                continue

            k = n//self.params["p"]
            l = n%self.params["p"]

            if (B is not None):
                B_sat = B[k,l]
            else:
                B_sat = None

            mu = self.get_mu(elem["material"], B_sat, type)
            n+=1
            Sx = elem["Sx"]
            Sz = elem["Sz"]
            Rx = elem["lx"] / (mu * Sx)
            Rz = elem["lz"] / (mu * Sz)
            if ((i % self.params['P']) == 0):
                if ((B is None) and (elem["material"] == "fer")):
                    self.mask_fer_x[i] = 1
                self.R_x[i, i] = Rx
                self.S_x[i] = Sx

                i += 1
            else:
                if ((B is None) and (elem["material"] == "fer")):
                    self.mask_fer_x[i] = 1
                    self.mask_fer_x[i+1] = 1
                self.R_x[i, i] = Rx
                self.R_x[i + 1, i + 1] = Rx
                self.S_x[i] = Sx
                self.S_x[i+1] = Sx
                i += 2

            if (j < self.params["p"]):
                if ((B is None) and (elem["material"] == "fer")):
                    self.mask_fer_z[j] = 1
                self.R_z[j,j] = Rz
                self.S_z[j] = Sz

                j += 1
            elif (j > self.params["Nz"] - self.params['p']-1):
                if ((B is None) and (elem["material"] == "fer")):
                    self.mask_fer_z[j] = 1
                self.R_z[j,j] = Rz
                self.S_z[j] = Sz

                j+=1
            else:
                if ((B is None) and (elem["material"] == "fer")):
                    self.mask_fer_z[j] = 1
                    self.mask_fer_z[j+1] = 1
                self.R_z[j, j] = Rz
                self.R_z[j+self.params["p"], j+self.params["p"]] = Rz
                self.S_z[j] = Sz
                self.S_z[j+self.params["p"]] = Sz
                if ((j+1)%(self.params["p"]) == 0):
                    j += self.params["p"]+1
                else:
                    j += 1
        myPrint("R is done")
        self.S_x = self.S_x
        self.S_z = self.S_z
        self.R = block_diag((self.R_x, self.R_z), format='csr')

    def MMF_Matrix(self, I=None):

        MMF_x = np.zeros((self.params['Nx'], 1))
        MMF_z = np.zeros((self.params['Nz'], 1))

        if (I is not None):
            I_new = -I
        else:
            I_new = self.params["I"]

        ix = 0

        for key, elem in self.mesh.items():
            
            MMFx = elem["Nspire"] * I_new
            if ( (ix%self.params['P'] == 0) or (ix%self.params['P'] == self.params['P']-1)):
                MMF_x[ix] = MMFx
                ix+=1
            else:
                MMF_x[ix] = MMFx
                MMF_x[ix+1] = MMFx
                ix+=2
        
        self.MMF = np.vstack((MMF_x, MMF_z))
        myPrint("MMF is done")
        # return MMF
    

    def create_Xsi(self):

        # n = (4*(self.params["p"]-1)+2)*self.params["Npsi_z"]
        return sp.hstack([self.create_Xsix(), self.create_Xsiz()])

    def create_Xsix(self):
        id = sp.eye(self.params["Npsi_x"]-1)
        W = sp.csr_matrix([1,1])
        A = sp.kron(id, W)
        X = sp.block_diag((A, 1), format='csr')
        Y = sp.csr_matrix(self.Y_matrix(self.params['Npsi_z']))
        Xsi_x = sp.kron(Y, X)
        myPrint("Xsi is done")
        return Xsi_x

    def create_Xsiz(self):
        id = sp.eye(self.params["Npsi_z"])
        Y_z = sp.csr_matrix(self.Y_matrix(self.params["Npsi_x"]-1))
        Z = sp.csr_matrix(([-1], ([0], [Y_z.shape[1] - 1])), shape=(1, Y_z.shape[1]))
        Y_z = sp.vstack([Y_z, Z])
        Xsi_z = sp.kron(id, sp.hstack((Y_z, Y_z)))
        return Xsi_z

    def Y_matrix(self, size):
        Y = np.zeros((size, size + 1))
        for i in range(size):
            Y[i, i] = -1
            Y[i, i + 1] = 1
        return Y
        
    def get_mu(self, material, B_sat=None, type=None):
        if ((material == "fer") and ((B_sat is not None))):
            if (type == "diff"):
                mu_r = self.get_mu_r_diff(B_sat)
            elif (type == "app"):
                mu_r = self.get_mu_r(B_sat)
            return mu_r*self.params["mu_0"]
        else:
            return self.mu_abs[material]
        

    def get_mu_r_diff(self, B_sat):
        if (B_sat > 7):
            B_sat = 7
        dmu_dB = self.sigmoid(B_sat)[1]
        mu_r = self.sigmoid(B_sat)[0]
        mu_rd = mu_r/(1 - B_sat/mu_r*dmu_dB)
        return mu_rd

    def get_mu_r(self, B_sat):
        if (B_sat > 7):
            B_sat = 7
        return self.sigmoid(B_sat)[0]

    def sigmoid(self, B):
        L = 1638.3025031133686
        k = -4.503479204801756
        B0 = 1.3692966370327697
        
        exp_term = np.exp(-k * (B - B0))
        sig = L / (1 + exp_term)
        d_sig = (L * k * exp_term) / ((1 + exp_term)**2)
        return sig, d_sig





    # def get_mu_r_heatmap(self, B_sat):
    #     B = np.array([
    #         0.000000, 0.227065, 0.454130, 0.681195, 0.908260, 1.135330, 1.362390,
    #         1.589350, 1.812360, 2.010040, 2.133160, 2.199990, 2.254790, 2.299930,
    #         2.342510, 2.378760, 2.415010, 2.451260, 2.487500, 2.523750, 2.560000
    #     ])

    #     H = np.array([
    #         0.000000, 13.898400, 27.796700, 42.397400, 61.415700, 82.382400,
    #         144.669000, 897.760000, 4581.740000, 17736.200000, 41339.300000,
    #         68321.800000, 95685.500000, 123355.000000, 151083.000000, 178954.000000,
    #         206825.000000, 234696.000000, 262568.000000, 290439.000000, 318310.000000
    #     ])

    #     mu_0 = self.params["mu_0"]

    #     # H_sat = np.interp(B_sat, B, H)
    #     # mu_r = B_sat/(self.params["mu_0"]*H_sat)



    #     b_vec = np.linspace(B[0], B[-1], len(B))
    #     h_vec = np.interp(b_vec, B, H)

    #     a = (B[-2] - B[-1])/(H[-2] - H[-1])
    #     b = B[-1] - a*H[-1]

    #     h_vec_bis = np.linspace(H[-1], 3000000, 100)
    #     b_vec_bis = a*h_vec_bis + b

    #     h_final = np.hstack([h_vec, h_vec_bis])
    #     b_final = np.hstack([b_vec, b_vec_bis])

    #     h_mu = np.interp(B_sat, b_final, h_final)

    #     mu_r = B_sat/(mu_0*h_mu)

    #     return mu_r
    