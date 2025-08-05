import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy as sp

from Utility import *


class MagneticField:


    def __init__(self, myMEC, mySolver=None, B=None):
        self.myMEC = myMEC
        self.params = myMEC.params
        self.probDim= myMEC.probDim
        # self.B = mySolver.B
        if (B is not None):
            self.B = B 
        elif (mySolver is not None):
            self.B = mySolver.B
        else:
            raise ValueError(f"class \"MagneticField\" : mauvais arguments")

        
        self.get_B_2D()
        self.get_B_ag()
        self.get_Bz_3D()
        # self.get_B_3D()

    def get_B_2D(self):
        Nx  = self.params['Nx']
        P   = self.params["P"]
        p   = self.params["p"]
        M   = self.params["M"]
        m   = self.params["m"]

        B_x = np.reshape(self.B[:Nx], (m, P))
        B_z = np.reshape(self.B[Nx:], (M, p))


        Bx_2D = np.zeros((m, p))
        Bx_2D[:, 0] = B_x[:, 0]
        Bx_2D[:, 1:] = (B_x[:, 1:-1:2] + B_x[:, 2::2]) / 2.0

        Bz_2D = np.zeros((m, p))
        Bz_2D[0, :] = B_z[0, :]
        Bz_2D[1:-1, :] = (B_z[1:-2:2, :] + B_z[2:-1:2, :]) / 2.0
        if m > 1:
            Bz_2D[-1, :] = B_z[-1, :]

        self.Bx_2D = Bx_2D
        self.Bz_2D = Bz_2D
        self.B_2D = np.sqrt(Bx_2D**2 + Bz_2D**2)

    def get_B_ag(self):

        z0 = (self.myMEC.h6+self.myMEC.h7)/2
        z_cut_0 = np.argmin(np.abs(self.probDim["z_vec"] - z0))
        self.tag = self.probDim["z_vec"][z_cut_0]
        self.Bz_ag = self.Bz_2D[z_cut_0,:]
        self.Bx_ag = self.Bx_2D[z_cut_0,:]
        self.B_ag = self.B_2D[z_cut_0,:]
        
        # plt.figure()
        # plt.scatter(self.probDim["x_vec"], self.Bz_ag)
        # plt.grid()
        # plt.show()

        # df = pd.DataFrame({"x":self.probDim["x_vec"],
        #                    "Bz_0":self.Bz_ag, "Bx_0":self.Bx_ag})
        # df.to_csv("Tests/Compare/Test/champ_B_test.csv", index=False)
        
    def get_B_3D(self):

        df = pd.read_csv("dataR/magneticField - 3D/myBunit_long.csv")
        y_unit = df["x"].to_numpy()[::50]
        B_unit = df["B"].to_numpy()[::50]
        Bz_unit = df["Bz"].to_numpy()[::50]
        By_unit = df["By"].to_numpy()[::50]

        B_padding = np.zeros((len(self.B_ag), 10))
        Bx_padding = np.zeros((len(self.Bx_ag), 10))
        Bz_padding = np.zeros((len(self.Bz_ag), 10))
        By_padding = np.zeros((len(self.B_ag), 10))

        for i in range(B_padding.shape[1]):
            B_padding[:,i] = self.B_ag
            Bz_padding[:,i] = self.Bz_ag
            Bx_padding[:,i] = self.Bx_ag

        # On prend la moitié donc j'ai la branche qui fait 28e-3 et la patte qui dépasse de 10e-3 de chaque coté
        air = 3000e-3
        mid = 14.0e-3
        d = self.probDim["d_maglev"]-2*mid
        y_padding = np.linspace(0, d, int(B_padding.shape[1]))


        B_edge = np.zeros((len(self.B_ag), len(B_unit)))
        Bx_edge = np.zeros((len(self.Bx_ag), len(B_unit)))
        Bz_edge = np.zeros((len(self.Bz_ag), len(Bz_unit)))
        By_edge = np.zeros((len(self.B_ag), len(By_unit)))

        mask = (self.probDim["x_vec"]>=self.myMEC.l2) & (self.probDim["x_vec"]<=self.myMEC.l5)

        for i in range(len(self.B_ag)):
            B_edge[i, :] = B_unit*self.B_ag[i]
            # Bx_edge[i, :] = 0.0
            Bz_edge[i, :] = Bz_unit*self.Bz_ag[i]
            if(mask[i]):
                # By_edge[i, :] = By_unit*self.B_ag[i]
                # By_edge[i, :] = By_unit*np.max(self.B_ag)
                By_edge[i,:] = np.sqrt(B_edge[i,:]**2-Bz_edge[i, :]**2)
        
        self.B_3D = np.hstack([np.flip(B_edge)[::-1],B_padding, B_edge])
        self.Bx_3D = np.hstack([np.flip(Bx_edge)[::-1],Bx_padding, Bx_edge])
        self.Bz_3D = np.hstack([np.flip(Bz_edge)[::-1],Bz_padding, Bz_edge])
        self.By_3D = np.hstack([-np.flip(By_edge)[::-1],By_padding, By_edge])


        self.Bz_3D_bis = np.vstack([self.Bz_3D, -self.Bz_3D[::-1]])
        self.Bx_3D_bis = np.vstack([self.Bx_3D, self.Bx_3D[::-1]])
        self.By_3D_bis = np.vstack([self.By_3D, -self.By_3D[::-1]])
        self.B_3D_bis = np.vstack([self.B_3D, self.B_3D[::-1]])

        y_edge = y_unit
        self.y = np.hstack([y_edge, y_padding+y_edge[-1], y_edge+y_padding[-1]+y_edge[-1]])
        self.y = self.y - (air+self.probDim["d_maglev"]/2)

        X, Y = np.meshgrid(self.probDim["x_vec"],self.y)
        fig = plt.figure(figsize=(8, 6))
        plt.title("Bz_all")
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, self.Bz_3D, cmap='viridis')
        plt.show()


        
    def get_Bz_3D(self):
        sample_x = 1
        sample_y = 5

        # df = pd.read_csv("dataR/magneticField - 3D/myBunit_long.csv")
        # y_unit = df["x"].to_numpy()[::sample_y]
        # Bz_unit = df["Bz"].to_numpy()[::sample_y]

        df = pd.read_csv("dataR/magneticField - 3D/myBunit_test.csv")
        y_unit = df["x"].to_numpy()[::sample_y]
        Bz_unit = df["Bz"].to_numpy()[::sample_y]

        y_edge = y_unit
        Bz_ag_sampled = self.Bz_ag[::sample_x]
        Bz_edge = np.zeros((len(y_edge), len(Bz_ag_sampled)))
        for i in range(len(Bz_ag_sampled)):
            Bz_edge[:, i] = Bz_unit*Bz_ag_sampled[i]
            # Bz_edge[:, i] = 0.0

        Bz_padding = np.zeros((10, len(Bz_ag_sampled)))
        for i in range(Bz_padding.shape[0]):
            Bz_padding[i,:] = Bz_ag_sampled


        # # On prend la moitié donc j'ai la branche qui fait 28e-3 et la patte qui dépasse de 10e-3 de chaque coté
        air = 3000e-3
        mid = 5.0e-3
        d = self.probDim["d_maglev"]
        y_padding = np.linspace(d/(4*Bz_padding.shape[0]), d/2-mid, int(Bz_padding.shape[0]))
        self.y = np.hstack([y_padding, y_edge + d/2-mid])



        Bz_3D = np.vstack([Bz_padding, Bz_edge])
        Bz_3D_left = np.vstack([Bz_3D[::-1], Bz_3D])
        self.y = np.hstack([-self.y[::-1], self.y])

        self.Bz_3D_bis = np.hstack([Bz_3D_left, -np.flip(Bz_3D_left,axis=1)])

        x_left = self.probDim["x_vec"]-(self.probDim["x_vec"][-1] + (self.myMEC.L - self.probDim["x_vec"][-1]))
        x_left = x_left[::sample_x]
        x_right = -np.flip(x_left)

        # self.x = np.hstack([self.probDim["x_vec"]-(self.probDim["x_vec"][-1] + (self.myMEC.L - self.probDim["x_vec"][-1])), -np.flip(self.probDim["x_vec"])+self.probDim["x_vec"][-1]+(self.myMEC.L - self.probDim["x_vec"][-1])])
        self.x = np.hstack([x_left, x_right])
        
    def get_coordinate(self):
        x = self.x
        y = self.y
        z = self.probDim["z_vec"][::-1]
        
        return (x,y,z)

    def get_B_longitudinal(self):

        """
        Pour utiliser cette fonction, il faut mettre :
        l_air_p = 3000e-3
        e_branch = 100e-3
        """

        z0 = (self.myMEC.h6+self.myMEC.h7)/2
        z_cut_0 = np.argmin(np.abs(self.probDim["z_vec"] - z0))

        Bz_ag = self.Bz_2D[z_cut_0,:]
        By_ag = self.Bx_2D[z_cut_0,:]
        B_ag = self.B_2D[z_cut_0,:]
        x = self.probDim["x_vec"]

        plt.figure()
        plt.title("Entier")
        plt.plot(x, By_ag, label="Bx")
        plt.plot(x, Bz_ag, label="Bz")
        plt.plot(x, B_ag, label="B")
        plt.legend()
        plt.grid()

        air = 3000.0e-3
        branch = 100.0e-3
        foot = 0.0e-3
        ref = (2*air+branch + 2*foot)/2
        index = (x <= ref)
        x = x[index]
        By_ag = By_ag[index]
        Bz_ag = Bz_ag[index]
        B_ag = B_ag[index]

        B_ag_inv = B_ag[::-1]
        By_ag_inv = By_ag[::-1]
        Bz_ag_inv = Bz_ag[::-1]

        B_normalized = B_ag_inv[0]

        B_ag_inv = B_ag_inv/B_normalized
        By_ag_inv = By_ag_inv/B_normalized
        Bz_ag_inv = Bz_ag_inv/B_normalized


        df = pd.DataFrame({"x":x, "By":By_ag_inv, "Bz":Bz_ag_inv, "B":B_ag_inv})
        df.to_csv("myBunit.csv", index=False)


        plt.figure()
        plt.title("before")
        plt.plot(x, By_ag, label="Bx")
        plt.plot(x, Bz_ag, label="Bz")
        plt.plot(x, B_ag, label="B")
        plt.legend()
        plt.grid()


        plt.figure()
        plt.title("normalized")
        plt.plot(x, By_ag_inv, label="Bx")
        plt.plot(x, Bz_ag_inv, label="Bz")
        plt.plot(x, B_ag_inv, label="B")
        plt.legend()
        plt.grid()

        plt.show()

    def plot_field(self):
        x = self.get_coordinate()[0]
        y = self.get_coordinate()[1]

        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(8, 6))
        plt.title("Bz")

        # Contour rempli
        contour = plt.contourf(X, Y, self.Bz_3D_bis, levels=50, cmap='viridis')

        # Ajouter la barre de couleur
        plt.colorbar(contour, label='Bz [T]')

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')  # pour garder les proportions

        plt.show()

    
    def plot_transversal(self):
        z0 = (self.myMEC.h6+self.myMEC.h7)/2
        z_cut_0 = np.argmin(np.abs(self.probDim["z_vec"] - z0))
        self.tag = self.probDim["z_vec"][z_cut_0]
        self.Bz_ag = self.Bz_2D[z_cut_0,:]
        self.Bx_ag = self.Bx_2D[z_cut_0,:]
        self.B_ag = self.B_2D[z_cut_0,:]
        
        plt.figure()
        plt.scatter(self.probDim["x_vec"], self.Bz_ag)
        # plt.scatter(self.probDim["x_vec"], self.B_ag)
        plt.grid()
        plt.show()


    def plot_field_transversal(self):
        x = self.probDim["x_vec"]
        z = self.get_coordinate()[2]

        X,Z = np.meshgrid(x,z)
        plt.figure()
        plt.scatter(X,Z, c=self.B_2D[::-1])
        plt.axis("equal")
        plt.xlim([0,self.myMEC.l7])
        plt.ylim([0,self.myMEC.h9])
        plt.colorbar()
        plt.grid()

        plt.show()

    def plot_field_longitudinal(self):
        # Configuration :
        # l_air_p0 = 40
        # l_air_p1 = 40
        # Sinon le reste c'est pareil
        x,y,_ = self.get_coordinate()

        x0 = ((40.0e-3+80.0e-3)/2)
        x_cut_0 = np.argmin(np.abs(x - x0))
        Bz_0 = self.Bz_3D_bis[x_cut_0,:]
        B_0 = self.B_3D_bis[x_cut_0,:]
        print(f"{x0} : {x[x_cut_0]}")

        x1 = ((108e-3+80e-3)/2)
        x_cut_1 = np.argmin(np.abs(x - x1))
        Bz_1 = self.Bz_3D_bis[x_cut_1,:]
        B_1 = self.B_3D_bis[x_cut_1,:]
        print(f"{x1} : {x[x_cut_1]}")

        x2 = ((200.0e-3+118.0e-3)/2)
        x_cut_2 = np.argmin(np.abs(x - x2))
        Bz_2 = self.Bz_3D_bis[x_cut_2,:]
        B_2 = self.B_3D_bis[x_cut_2,:]
        print(f"{x2} : {x[x_cut_2]}")


        df = pd.DataFrame({"y":y, "Bz0":Bz_0, "Bz1":Bz_1, "Bz2":Bz_2, "B0":B_0, "B1":B_1, "B2":B_2})
        df.to_csv("Bz_long.csv")


        plt.figure()
        plt.title("z0")
        plt.plot(y, Bz_0)
        plt.grid()

        plt.figure()
        plt.title("z1")
        plt.plot(y, Bz_1)
        plt.grid()

        plt.figure()
        plt.title("z2")
        plt.plot(y, Bz_2)
        plt.grid()
        plt.show()

        plt.figure()
        plt.title("0")
        plt.plot(y, B_0)
        plt.grid()

        plt.figure()
        plt.title("1")
        plt.plot(y, B_1)
        plt.grid()

        plt.figure()
        plt.title("2")
        plt.plot(y, B_2)
        plt.grid()
        plt.show()

        

        


    def get_B_for_compare(self):
        z_vec = self.probDim["z_vec"]

        z0 = (self.myMEC.h1+self.myMEC.h2)/2
        z_cut_0 = np.argmin(np.abs(z_vec - z0))

        z1 = (self.myMEC.h2+self.myMEC.h3)/2
        z_cut_1 = np.argmin(np.abs(z_vec - z1))

        z2 = (self.myMEC.h3+self.myMEC.h5)/2
        z_cut_2 = np.argmin(np.abs(z_vec - z2))

        z3 = (self.myMEC.h6+self.myMEC.h7)/2
        z_cut_3 = np.argmin(np.abs(z_vec - z3))

        z4 = (self.myMEC.h7+self.myMEC.h8)/2
        z_cut_4 = np.argmin(np.abs(z_vec - z4))

        print(z0, z1, z2, z3, z4)
        print(z_vec[z_cut_0], z_vec[z_cut_1], z_vec[z_cut_2], z_vec[z_cut_3], z_vec[z_cut_4])

        Bz_0 = self.Bz_2D[z_cut_0,:]
        Bx_0 = self.Bx_2D[z_cut_0,:]
        B_0 = self.B_2D[z_cut_0,:]

        Bz_1 = self.Bz_2D[z_cut_1,:]
        Bx_1 = self.Bx_2D[z_cut_1,:]
        B_1 = self.B_2D[z_cut_1,:]
        
        Bz_2 = self.Bz_2D[z_cut_2,:]
        Bx_2 = self.Bx_2D[z_cut_2,:]
        B_2 = self.B_2D[z_cut_2,:]
        
        Bz_3 = self.Bz_2D[z_cut_3,:]
        Bx_3 = self.Bx_2D[z_cut_3,:]
        B_3 = self.B_2D[z_cut_3,:]
        
        Bz_4 = self.Bz_2D[z_cut_4,:]
        Bx_4 = self.Bx_2D[z_cut_4,:]
        B_4 = self.B_2D[z_cut_4,:]
        
        df = pd.DataFrame({"x":self.probDim["x_vec"],
                           "Bz_0":Bz_0, "Bx_0":Bx_0,
                           "Bz_1":Bz_1, "Bx_1":Bx_1, 
                           "Bz_2":Bz_2, "Bx_2":Bx_2,
                           "Bz_3":Bz_3, "Bx_3":Bx_3,
                           "Bz_4":Bz_4, "Bx_4":Bx_4})
        
        # df.to_csv("Validation/Compare/unsaturated/champ_B_1_test.csv", index=False)

    def heatmapField(self, k):
        X, Y = np.meshgrid(self.x, self.y)
        fig = plt.figure(figsize=(8, 6))
        plt.title("B")
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, k*self.B_3D, cmap='viridis')
        fig = plt.figure(figsize=(8, 6))
        plt.title("Bz")
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, k*self.Bz_3D, cmap='viridis')
        fig = plt.figure(figsize=(8, 6))
        plt.title("By")
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, k*self.By_3D, cmap='viridis')
        fig = plt.figure(figsize=(8, 6))
        plt.title("Bx")
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, k*self.Bx_3D, cmap='viridis')
        plt.show()


    def writeToCSV(self):
        myPrint(f"Writing in {self.filename}")
        df_B = pd.DataFrame(self.B)
        df_B.to_csv(self.filename, index=False)
        myPrint("End writing\n")

    def writeToCSV_ag(self):
        myPrint(f"Writing in Bz_ag")
        x = np.linspace(0, self.probDim["L"], len(self.Bz_ag))
        df_B_z = pd.DataFrame({"x": x, "Bz_ag": self.Bz_ag})
        df_B_x = pd.DataFrame({"x": x, "Bx_ag": self.Bx_ag})
        df_B = pd.DataFrame({"x": x, "B_ag": self.B_ag})
        df_B_z.to_csv("Bz_ag.csv", index=False)
        df_B_x.to_csv("Bx_ag.csv", index=False)
        df_B.to_csv("B_ag.csv", index=False)
        myPrint("End writing\n")