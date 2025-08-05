import numpy as np
import matplotlib.pyplot as plt
from Utility import *
import time
import pandas as pd


from MEC import *
from Mesh import *
from Network import *
from Solve import *
from MagneticField import *
from Performance import *
from MaxwellFourier import *



class Maglev:

    def __init__(self, optiVariable=None, filename=None):
        self.filename = filename
        self.optiVariable = optiVariable
        start_time = time.perf_counter()

        param = np.array([400.0e-3])
        self.EM_module(param)

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        # print(f"Temps d'exécution : {execution_time:.9f} secondes")


    def EM_module(self, param):

        self.myMEC = MEC(param, optiVariable=self.optiVariable, filename=self.filename)
        self.W_mag_quad = self.myMEC.get_W()
        self.R_coil = self.myMEC.get_CoilResistance()
        self.myMesh = Mesh(self.myMEC)
        # self.myMesh.viewGeometry1()
        # self.myMesh.plotMesh()
        self.myNetwork = Network(self.myMEC, self.myMesh)
        self.mySolver = Solver(self.myMEC, self.myNetwork)
        self.myMagneticField = MagneticField(self.myMEC, mySolver=self.mySolver)
        self.myPerformance = Performance(self.myMEC, self.myMagneticField)
        Fz0_test = self.myPerformance.levitationForce2D()
        # print(f"Fz0_test = {Fz0_test}")
        # self.myMagneticField.plot_field()

        k = self.get_k(Fz0_test)
        # print(f"k = {k}")

        # For testing
        # k = 80

        tol = 1e-2
        Nmax = 20
        n=0
        psi = self.mySolver.psi*k
        MF = True
        while (True):
            n+=1
            if (n >= Nmax):
                self.P_tot = 1e9
                MF = False
                break
            B, psi = self.Newton(k, psi)
            myMagneticField_bis = MagneticField(self.myMEC, B=B)
            myPerformance_bis = Performance(self.myMEC, myMagneticField_bis)
            Fz0 = myPerformance_bis.levitationForce2D()
            if (Fz0 <= 0.0):
                self.P_tot = 1e9
                MF = False
                break
            # print(f"W = {self.W_mag_quad}")
            # print(f"Fz0 = {Fz0}")
            k_comp = self.get_k(Fz0)
            k = k_comp*k

            # break
            # ratio = 1-1/(Fz0/self.W_mag_quad)
            ratio = 1-(Fz0/self.W_mag_quad)
            if (np.abs(ratio) < tol):
                # print(f"converged in {n} iterations ! Force = {Fz0} and W = {self.W_mag_quad}\n"
                #       f"Error : {np.abs(100-Fz0/self.W_mag_quad*100)}%")
                # print(f"k = {k}")
                # print(f"Bmax = {np.max(B)}")
                break

        if (MF):
            myMaxwellFourier = MaxwellFourier(self.myMEC, myMagneticField_bis, 0.0, myMagneticField_bis.Bz_3D_bis)
            error_factor = 1-myMaxwellFourier.Fz/Fz0
            myMaxwellFourier = MaxwellFourier(self.myMEC, myMagneticField_bis, self.myMEC.v_mag, myMagneticField_bis.Bz_3D_bis)
            Fz = error_factor*myMaxwellFourier.Fz + myMaxwellFourier.Fz
            Fy = error_factor*myMaxwellFourier.Fy + myMaxwellFourier.Fy
            # print(f"error : {100-myMaxwellFourier.Fz/Fz0*100}")
            # print(f"Fy = {myMaxwellFourier.Fy}\n",
            #       f"Fz = {myMaxwellFourier.Fz}\n",
            #       f"Fz0 = {Fz0_test}\n",
            #       f"error = {100-100*myMaxwellFourier.Fz/Fz0_test}\n",
            #       f"v_mag = {self.myMEC.v_mag}\n")
            # print(f"Fy = {Fy}\n",
            #       f"Fz = {Fz}\n",
            #       f"Fz0 = {Fz0}\n",
            #       f"v_mag = {self.myMEC.v_mag}\n")
            self.P_tot = self.get_Losses(self.myMEC, Fy, Fz, Fz0, k)
        return Fz0, np.max(B)

    def Newton(self, I0, psi):
        start_time = time.perf_counter()

        S = np.hstack([self.myNetwork.S_x, self.myNetwork.S_z]).reshape(-1, 1)

        Xsi = csc_matrix(self.myNetwork.Xsi)
        Xsi_T = Xsi.transpose()
        self.myNetwork.MMF_Matrix(I0)
        F = Xsi @ self.myNetwork.MMF
        
        tol = 1e-2
        B_old = np.zeros(self.myNetwork.mask_fer_bool.sum())
        epsilon = np.inf
        n = 0
        Nmax = 100
        epsilon0 = 0
        psi = psi.reshape(-1, 1)
        phi = Xsi_T @ psi
        B = phi/S
        myTime = time.perf_counter()
        # print(myTime-start_time)
        while (epsilon > tol) and (n < Nmax):
            n+=1
            # print(n)
            self.get_B_2D(self.myMEC, B)
            self.myNetwork.Reluctance(self.B_2D, "diff")
            R = csc_matrix(self.myNetwork.R)
            J = Xsi @ R @ Xsi_T

            self.myNetwork.Reluctance(self.B_2D, "app")
            R = csc_matrix(self.myNetwork.R)
            A = Xsi @ R @ Xsi_T
            residu = F - A @ psi
            dpsi = spsolve(J, residu).reshape(-1, 1)
            psi = psi + dpsi
            phi = Xsi_T @ psi

            B = phi/S
            B_mask = B[self.myNetwork.mask_fer_bool].flatten()
            epsilon = np.linalg.norm(B_mask - B_old)/np.linalg.norm(B_mask)

            B_old = B_mask
            if (n == 1):
                epsilon0 = epsilon
            if (epsilon <= tol):
                break
            psi_old = psi

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        # print(f"Temps d'exécution (NRM) : {execution_time:.9f} secondes")
    

        # Evaluation des performances

        # ROC = (np.log10(epsilon0) - np.log10(epsilon))/(n)
        # print(f"nombre d'itérations : {n}\n",
            #   f"Taux d'erreur : {epsilon}\n",
            #   f"ROC : {ROC}\n")

        # df = pd.DataFrame({"x":myMagneticField.x, "B":myMagneticField.B_sat, "B_ag":myMagneticField.B_ag_mid})
        # df.to_csv("Tests/champ_B.csv", index=False)

        return B, psi

    def get_B_2D(self, myMEC, B):
        Nx  = myMEC.params['Nx']
        P   = myMEC.params["P"]
        p   = myMEC.params["p"]
        M   = myMEC.params["M"]
        m   = myMEC.params["m"]
        N   = myMEC.params["N"]

        B_x = B[:Nx]
        B_z = B[Nx:]
        B_x = np.reshape(B_x, (m, P))
        B_z = np.reshape(B_z, (M, p))

        B_x_new = np.zeros((m, p))
        B_z_new = np.zeros((m, p))

        j = 0
        for i in range(B_x_new.shape[1]):
            if (j == 0):
                B_x_new[:, i] = B_x[:, j].ravel()
                j += 1
            else:
                B_x_new[:, i] = (B_x[:, j].ravel() + B_x[:, j+1].ravel())/2.0
                j += 2

        j = 0
        for i in range(B_z_new.shape[0]):
            if (j == 0):
                B_z_new[i, :] = B_z[j, :].ravel()
                j += 1
            elif (j == B_z.shape[0] - 1):
                B_z_new[i, :] = B_z[j,:].ravel()
            else:
                B_z_new[i, :] = (B_z[j, :].ravel() + B_z[j+1, :].ravel())/2.0
                j += 2

        self.Bx_2D = B_x_new
        self.Bz_2D = B_z_new
        self.B_2D = np.sqrt(self.Bx_2D**2 + self.Bz_2D**2)

    
    def FixedPoint(self):

        myMEC = MEC(optiVariable=self.optiVariable, filename=self.filename)

        myMesh = Mesh(myMEC)
        # myMesh.plotMesh()
        myNetwork = Network(myMEC, myMesh)



        # df = pd.DataFrame(myMesh.mesh)
        # df.to_csv("Tests/Mesh.csv", index=False)

        delta_B = np.inf
        delta_B_old = np.inf
        tol = 1e-3
        Nmax = 5
        B_sat_old = 0.0
        n = 0
        B_2D = 0.0
        B_2D_old = 1e-6
        B_old = 0.0
        alpha = 1.0


        while (np.abs(delta_B) > tol and n < Nmax):
            n+=1
            # print(n)
            mySolver = Solver(myMEC, myNetwork)
            B = (1-alpha)*B_old + alpha*mySolver.B
            delta_B = np.linalg.norm(B - B_old)/np.linalg.norm(B)
            # if (delta_B_old < delta_B):
            #     break
            delta_B_old = delta_B
            B_old = B
            print(delta_B)
            myMagneticField = MagneticField(myMEC, mySolver, B)
            # B_2D = (1-alpha)*B_2D_old +  alpha*myMagneticField.B_2D
            # delta_B = np.linalg.norm(B_2D - B_2D_old)/np.linalg.norm(B_2D_old)
            # B_2D_old = B_2D
            # print(delta_B)
            myNetwork.Reluctance(myMagneticField.B_2D)
            # plt.plot(myMagneticField.x, myMagneticField.B_sat)

            # B_local = myMagneticField.B_2D[]

            z = myMagneticField.get_coordinate()[2][::-1]
            plt.figure(f"B Itération {n}")
            X, Z = np.meshgrid(myMagneticField.x, z)
            plt.axis("equal")
            plt.scatter(X, Z, c=myMagneticField.B_2D)
            plt.colorbar()

            z = myMagneticField.get_coordinate()[2][::-1]
            plt.figure(f"mu_r Itération {n}")
            X, Z = np.meshgrid(myMagneticField.x, z)
            plt.axis("equal")
            plt.scatter(X, Z, c=myNetwork.get_mu_r_heatmap(myMagneticField.B_2D))
            plt.colorbar()




            # B_sat = (1-alpha)*B_sat_old + alpha*myMagneticField.B_sat_mid
            # delta_B = B_sat_old - B_sat
            # B_sat_old = B_sat



        df = pd.DataFrame({"x":myMagneticField.x, "B":myMagneticField.B_sat, "B_ag":myMagneticField.B_ag})
        df.to_csv("Tests/champ_B.csv", index=False)
        plt.show()

        # myMEC = MEC(optiVariable=self.optiVariable, filename=self.filename)
        # self.W_mag_quad = myMEC.get_W()
        # # self.W_mag_quad = 41.3e3
        # self.R_coil = myMEC.get_CoilResistance()
        
        # myMesh = Mesh(myMEC)
        # # myMesh.plotMesh()
        # myNetwork = Network(myMEC, myMesh)

        # delta_B = np.inf
        # tol = 1e-3
        # Nmax = 20
        # n_k = 0
        # B_old = np.inf
        # delta_k = np.inf
        # self.B_2D_old = 0.0
        # alpha = 0.1


        # while (np.abs(delta_k) > tol and n_k < Nmax):

        #     n_k += 1
        #     print(f"n_k = {n_k}")
        #     mySolver_k = Solver(myMEC, myNetwork)
        #     myMagneticField_k = MagneticField(myMEC, mySolver_k)

        #     myPerformance = Performance(myMEC, myMagneticField_k)
        #     self.Fz0_test = myPerformance.levitationForce3D()
        #     k = self.get_k()
        #     self.B_2D = myMagneticField_k.B_2D*k
        #     n_B = 0
        #     B_old = 0
        #     self.B_2D_old = 0.0

        #     myNetwork.MMF_Matrix(k)
        #     while (np.abs(delta_B) > tol and n_B < Nmax):
        #         n_B += 1
        #         print(f"n_B = {n_B}")
        #         myNetwork.Reluctance(self.B_2D)
        #         mySolver_k = Solver(myMEC, myNetwork)
        #         myMagneticField_k = MagneticField(myMEC, mySolver_k)
        #         # plt.figure(figsize=(12,8))
        #         # plt.imshow(myMagneticField_k.B_2D)
        #         # plt.show()
        #         # plt.figure()
        #         # plt.plot(myMagneticField_k.x, myMagneticField_k.B_sat)
        #         # plt.show()
        #         B_sat = myMagneticField_k.B_sat_mid
        #         print(f"B_sat = {B_sat}")
        #         delta_B = B_sat - B_old
        #         print(delta_B)
        #         B_old = B_sat
        #         self.B_2D = alpha*myMagneticField_k.B_2D + (1-alpha)*self.B_2D_old
        #         self.B_2D_old = self.B_2D

        #     df = pd.DataFrame({"x":myMagneticField_k.x, "B":myMagneticField_k.B_sat, "B_ag":myMagneticField_k.B_ag})
        #     df.to_csv("Tests/champ_B.csv", index=False)
        #     print(k)
        #     break



        # while (np.abs(delta)>tol and n < Nmax):
        #     n+=1
        #     print(n)
        #     mySolver = Solver(myMEC, myNetwork)
        #     myMagneticField = MagneticField(myMEC, mySolver)
        #     myPerformance = Performance(myMEC, myMagneticField)
        #     self.Fz0_test = myPerformance.levitationForce3D()
        #     print(f"F1 = {self.Fz0_test}")
        #     k = self.get_k()
        #     print(f"k1 = {k}")



        #     # plt.figure(figsize=(12,8))
        #     # plt.imshow(myMagneticField.B_2D*k)
        #     # plt.show()

        #     myNetwork.Reluctance(myMagneticField.B_2D*k)
        #     mySolver = Solver(myMEC, myNetwork)
        #     myMagneticField = MagneticField(myMEC, mySolver)
        #     myPerformance = Performance(myMEC, myMagneticField)
        #     self.Fz0_test = myPerformance.levitationForce3D()
        #     print(f"F2 = {self.Fz0_test}")

        #     k = self.get_k()
        #     print(f"k2 = {k}")
        #     print(f"B_sat = {np.max(myMagneticField.B_sat)*k}")
        #     # delta = np.max(self.Bz0_final) - B_old
        #     # B_old = np.max(self.Bz0_final)



        #     myNetwork.Reluctance(myMagneticField.B_2D)
        # print(k)



        # plt.figure()
        # plt.plot(myMagneticField.x, myMagneticField.B_ag)
        # plt.figure()
        # plt.plot(myMagneticField.x, myMagneticField.B_sat)
        # plt.show()


        # X, Y = np.meshgrid(myMagneticField.get_coordinate()[0], myMagneticField.get_coordinate()[1], indexing='ij')
        # fig = plt.figure(figsize=(8, 6))
        # plt.title("Bz")
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X, Y, self.Bz0_final, cmap='viridis')
        # plt.show()

        # myMaxwellFourier = MaxwellFourier(myMEC, myMagneticField, myMEC.v_mag, self.Bz0)

        # self.P_tot = self.get_Losses(myMEC, myMaxwellFourier)

    def get_Losses(self, myMEC, Fy, Fz, Fz0, I0):
        I_tot = I0 * np.sqrt(2 - Fz/Fz0)
        P_J = self.R_coil*(I_tot)**2
        P_drag = np.abs(myMEC.v_mag*Fy)
        P_tot = P_drag + P_J
        S_fil = myMEC.params["S_fil"]

        # print(
        #     f"Pertes Joules [kW] : {P_J*1e-3}\n"
        #     f"Pertes par traînée magnétique [kW] : {P_drag*1e-3}\n"
        #     f"Pertes totales [kW] : {P_tot*1e-3}\n"
        #     f"Vitesse [m/s] : {myMEC.v_mag}\n"
        #     f"Courant statique [A]: {I0}\n"
        #     f"Courant total [A]: {I_tot}\n"
        #     f"Densité de courant [A/mm^2] : {I_tot/(S_fil*1e6)}\n"
        #     f"Résistance : {self.R_coil} Ohm\n"
        #     f"Fz : {Fz} N\n"
        #     f"Fy : {Fy} N\n")


        return P_tot

    def get_k(self, Fz):
        return np.sqrt( (self.W_mag_quad) / Fz)