import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Utility import *
import matplotlib.patches as patches
from time import sleep

class Mesh:

    materials = ["air", "fer", "coil"]

    def __init__(self, myMEC):
        myPrint("----Mesh : Initialization of the mesh----\n")
        self.myMEC = myMEC
        self.probDim = self.myMEC.probDim
        self.params = self.myMEC.params
        self.L = myMEC.L
        self.H = myMEC.H
        self.d_solve = myMEC.d_solve

        self.mesh = self.createMesh()
        myPrint("----Mesh : End----\n")
    
    def createMesh(self):
        BD = {}
        
        index = 0
        index_m = 0
        x_vec = np.zeros(self.params["p"])
        z_vec = np.zeros(self.params["m"])
        dp = self.myMEC.dp
        dm = self.myMEC.dm[::-1]
        z = self.probDim["H"]
        while z > 1e-8:
            # dz = self.get_dz(z)
            dz = dm[index_m]
            zprime = z - dz / 2.0


            x = 0.0
            index_p = 0
            while index_p < self.params["p"]:
                
                # dx = self.get_dx(x)
                dx = dp[index_p]
                xprime = x + dx / 2.0

                if index_m == 0:
                    x_vec[index_p] = xprime

                Nspire = 0
                nb_coil = ((self.myMEC.l7 - self.myMEC.l6) / dx * 2)

                if self.condition_fer(xprime, zprime, dz):
                    material = self.materials[1]
                    if self.condition_spire(xprime, zprime):
                        Nspire = self.params["Nmax"] / nb_coil
                elif self.condition_coil(xprime, zprime):
                    material = self.materials[2]
                    if self.myMEC.h3 <= zprime <= self.myMEC.h4:
                        zsec = self.myMEC.h4 - zprime
                        Nspire = (self.params["Nmax"] / (self.myMEC.h2 - self.myMEC.h1) * zsec) / nb_coil
                    elif self.myMEC.h1 <= zprime <= self.myMEC.h2:
                        zsec = zprime - self.myMEC.h1
                        Nspire = (self.params["Nmax"] / (self.myMEC.h2 - self.myMEC.h1) * zsec) / nb_coil
                else:
                    material = self.materials[0]
                
                # if Nspire != 0:
                #     Nspire = np.round(Nspire, 2)
                
                lx = dx/2.0
                lz = dz/2.0
                Sx = self.d_solve*dz
                Sz = self.d_solve*dx
                
                BD[f"BD_{index}"] = {"material": material, "Nspire": Nspire, "x": x, "z": z, "dx": dx, "dz": dz, "lx": lx, "lz": lz, "Sx": Sx, "Sz": Sz}
                index += 1
                
                index_p += 1
                x += dx

            z -= dz
            z_vec[index_m] = zprime
            index_m += 1



        self.probDim["x_vec"] = x_vec
        self.probDim["z_vec"] = z_vec


        return BD
    
    def get_dz(self,z):
        z-=1e-6
        if (z < self.myMEC.h1 - self.myMEC.offset):             # m0
            return self.myMEC.hm0
        elif ((z < self.myMEC.h1) and (z > self.myMEC.h1-self.myMEC.offset)):
            return self.myMEC.hm_offset_down
        elif ((z < self.myMEC.h2) and (z > self.myMEC.h1)):    # m1
            return self.myMEC.hm1
        elif ((z < self.myMEC.h3) and (z > self.myMEC.h2)):    # m2
            return self.myMEC.hm2
        elif ((z < self.myMEC.h4) and (z > self.myMEC.h3)):    # m3
            return self.myMEC.hm3
        elif ((z < self.myMEC.h5) and (z > self.myMEC.h4)):    # m4
            return self.myMEC.hm4
        elif ((z < self.myMEC.h6) and (z > self.myMEC.h5)):    # m5
            return self.myMEC.hm5
        elif ((z < self.myMEC.h7) and (z > self.myMEC.h6)):    # m6
            return self.myMEC.hm6
        elif ((z < self.myMEC.h8) and (z > self.myMEC.h7)):    # m7
            return self.myMEC.hm7
        elif ((z < self.myMEC.h8+self.myMEC.offset) and (z > self.myMEC.h8)):
            return self.myMEC.hm_offset_up
        elif (z > self.myMEC.h8 + self.myMEC.offset):          # m8
            return self.myMEC.hm8
        else:                                                  
            raise ValueError(f"Position z={z:.6f} out of defined domain.")
        
    def get_dx(self,x):
        x+=1e-6
        if x < self.myMEC.l1:                                                   # p0
            return self.myMEC.lp0 
        elif ((x < self.myMEC.l2-self.myMEC.offset) and (x > self.myMEC.l1)):  # p1
            return self.myMEC.lp1
        elif ((x < self.myMEC.l2) and (x > self.myMEC.l2-self.myMEC.offset)):
            return self.myMEC.lp_offset
        elif ((x < self.myMEC.l5) and (x > self.myMEC.l2)):
            return self.myMEC.lp2
        elif ((x < self.myMEC.l7) and (x > self.myMEC.l5)):                    # p3
            return self.myMEC.lp3
        else:       
            raise ValueError(f"Position x={x:.6f} out of defined domain.")


    def condition_fer(self, x, z, dz):
        if ((x >= self.myMEC.l2) and (x <= self.myMEC.l3) and (z >= self.myMEC.h5) and (z <= self.myMEC.h6)):
            z_diag = ((self.myMEC.h5 - self.myMEC.h6) / (self.myMEC.l3 - self.myMEC.l2)) * (x - self.myMEC.l2) + self.myMEC.h6
            if z+dz/2 < z_diag:
                return False 
            else:
                return True
        elif ((x >= self.myMEC.l4) and (x <= self.myMEC.l5) and (z >= self.myMEC.h5) and (z <= self.myMEC.h6)):
            z_diag = ((self.myMEC.h6 - self.myMEC.h5) / (self.myMEC.l5 - self.myMEC.l4)) * (x - self.myMEC.l4) + self.myMEC.h5
            if z+dz/2 >= z_diag:
                return True
            else:
                return False
        elif ( (x >= self.myMEC.l3) and (x <= self.myMEC.l4) and (z >= self.myMEC.h2) and (z <= self.myMEC.h6)):
            return True
        elif ( (x >= self.myMEC.l4) and (x <= self.myMEC.l7) and (z >= self.myMEC.h2) and (z <= self.myMEC.h3)):
            return True
        elif ( (x >= self.myMEC.l1) and (x <= self.myMEC.l7) and (z >= self.myMEC.h7) and (z <= self.myMEC.h8)):
            return True
        else:
            return False

    def condition_coil(self, x, z):
        
        if ( (x >= self.myMEC.l6) and (x < self.myMEC.l7) and (z >= self.myMEC.h1) and (z < self.myMEC.h2)):
            return True
        elif ( (x >= self.myMEC.l6) and (x < self.myMEC.l7) and (z >= self.myMEC.h3) and (z < self.myMEC.h4)):
            return True
        else:
            return False
    
    def condition_spire(self, x, z):
        if ( (x >= self.myMEC.l6) and (x < self.myMEC.l7) and (z >= self.myMEC.h2) and (z < self.myMEC.h3)):
            return True
        else:
            return False


    def viewGeometry1(self):
        fig, ax = plt.subplots()

        line = 1

        x = [self.myMEC.l6, self.myMEC.l7]
        y = [self.myMEC.h1, self.myMEC.h1]
        plt.plot(x,y, color="red", linewidth=line)

        x = [self.myMEC.l6, self.myMEC.l6]
        y = [self.myMEC.h1, self.myMEC.h2]
        plt.plot(x,y, color="red", linewidth=line)

        x = [self.myMEC.l6, self.myMEC.l6]
        y = [self.myMEC.h3, self.myMEC.h4]
        plt.plot(x,y, color="red", linewidth=line)

        x = [self.myMEC.l6, self.myMEC.l7]
        y = [self.myMEC.h4, self.myMEC.h4]
        plt.plot(x,y, color="red", linewidth=line)


        # Domaine ------------
        x = [self.myMEC.l7, self.myMEC.l7]
        y = [0.0, self.myMEC.h9]
        plt.plot(x,y, color="black", linewidth=line)

        x = [0.0, self.myMEC.l7]
        y = [0.0, 0]
        plt.plot(x,y, color="black", linewidth=line)


        x = [0.0, 0.0]
        y = [0.0, self.myMEC.h9]
        plt.plot(x,y, color="black", linewidth=line)

        x = [0.0, self.myMEC.l7]
        y = [self.myMEC.h9, self.myMEC.h9]
        plt.plot(x,y, color="black", linewidth=line)
        # --------------------

        # Reactive plate -----
        x = [self.myMEC.l1, self.myMEC.l7]
        y = [self.myMEC.h7, self.myMEC.h7]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l1, self.myMEC.l7]
        y = [self.myMEC.h8, self.myMEC.h8]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l1, self.myMEC.l1]
        y = [self.myMEC.h7, self.myMEC.h8]
        plt.plot(x,y, color="blue", linewidth=line)
        # --------------------




        x = [self.myMEC.l7, self.myMEC.l3]
        y = [self.myMEC.h2, self.myMEC.h2]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l3, self.myMEC.l3]
        y = [self.myMEC.h2, self.myMEC.h5]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l3, self.myMEC.l2]
        y = [self.myMEC.h5, self.myMEC.h6]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l2, self.myMEC.l5]
        y = [self.myMEC.h6, self.myMEC.h6]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l5, self.myMEC.l4]
        y = [self.myMEC.h6, self.myMEC.h5]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l4, self.myMEC.l4]
        y = [self.myMEC.h5, self.myMEC.h3]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l4, self.myMEC.l7]
        y = [self.myMEC.h3, self.myMEC.h3]
        plt.plot(x,y, color="blue", linewidth=line)


        # plt.axhline((self.myMEC.h1+self.myMEC.h2)/(2), linestyle="--", color="black")
        # plt.axhline((self.myMEC.h2+self.myMEC.h3)/(2), linestyle="--", color="black")
        # plt.axhline((self.myMEC.h3+self.myMEC.h5)/(2), linestyle="--", color="black")
        # plt.axhline((self.myMEC.h6+self.myMEC.h7)/(2), linestyle="--", color="black")
        # plt.axhline((self.myMEC.h7+self.myMEC.h8)/(2), linestyle="--", color="black")

        plt.xlim([0.0,self.myMEC.l7])
        plt.ylim([0.0,self.myMEC.h9])
        ax.set_aspect('equal')  # Garder les proportions
        # plt.grid()
        plt.show()

    def viewGeometry2(self):
        fig, ax = plt.subplots()

        line = 1

        x = [self.myMEC.l6, self.myMEC.l7]
        y = [self.myMEC.h1, self.myMEC.h1]
        plt.plot(x,y, color="red", linewidth=line)

        x = [self.myMEC.l6, self.myMEC.l6]
        y = [self.myMEC.h1, self.myMEC.h2]
        plt.plot(x,y, color="red", linewidth=line)

        x = [self.myMEC.l6, self.myMEC.l6]
        y = [self.myMEC.h3, self.myMEC.h4]
        plt.plot(x,y, color="red", linewidth=line)

        x = [self.myMEC.l6, self.myMEC.l7]
        y = [self.myMEC.h4, self.myMEC.h4]
        plt.plot(x,y, color="red", linewidth=line)


        # Domaine ------------
        x = [self.myMEC.l7, self.myMEC.l7]
        y = [0.0, self.myMEC.h9]
        plt.plot(x,y, color="black", linewidth=line)

        x = [0.0, self.myMEC.l7]
        y = [0.0, 0]
        plt.plot(x,y, color="black", linewidth=line)


        x = [0.0, 0.0]
        y = [0.0, self.myMEC.h9]
        plt.plot(x,y, color="black", linewidth=line)

        x = [0.0, self.myMEC.l7]
        y = [self.myMEC.h9, self.myMEC.h9]
        plt.plot(x,y, color="black", linewidth=line)
        # --------------------

        # Reactive plate -----
        x = [self.myMEC.l1, self.myMEC.l7]
        y = [self.myMEC.h7, self.myMEC.h7]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l1, self.myMEC.l7]
        y = [self.myMEC.h8, self.myMEC.h8]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l1, self.myMEC.l1]
        y = [self.myMEC.h7, self.myMEC.h8]
        plt.plot(x,y, color="blue", linewidth=line)
        # --------------------




        x = [self.myMEC.l7, self.myMEC.l3]
        y = [self.myMEC.h2, self.myMEC.h2]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l3, self.myMEC.l3]
        y = [self.myMEC.h2, self.myMEC.h5]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l3, self.myMEC.l2]
        y = [self.myMEC.h5, self.myMEC.h6]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l2, self.myMEC.l5]
        y = [self.myMEC.h6, self.myMEC.h6]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l5, self.myMEC.l4]
        y = [self.myMEC.h6, self.myMEC.h5]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l4, self.myMEC.l4]
        y = [self.myMEC.h5, self.myMEC.h3]
        plt.plot(x,y, color="blue", linewidth=line)

        x = [self.myMEC.l4, self.myMEC.l7]
        y = [self.myMEC.h3, self.myMEC.h3]
        plt.plot(x,y, color="blue", linewidth=line)



        H0 = (self.myMEC.h1+self.myMEC.h2)/(2)
        H1 = (self.myMEC.h2+self.myMEC.h3)/(2)
        H2 = (self.myMEC.h3+self.myMEC.h5)/(2)
        H3 = (self.myMEC.h6+self.myMEC.h7)/(2)
        H4 = (self.myMEC.h7+self.myMEC.h8)/(2)
        plt.axhline(H0, linestyle="--", color="black", linewidth="1")
        plt.axhline(H1, linestyle="--", color="black", linewidth="1")
        plt.axhline(H2, linestyle="--", color="black", linewidth="1")
        plt.axhline(H3, linestyle="--", color="black", linewidth="1")
        plt.axhline(H4, linestyle="--", color="black", linewidth="1")

        dz = 2e-3
        plt.text(0,H0+dz,"$H_0$", fontsize=8)
        plt.text(0,H1+dz,"$H_1$", fontsize=8)
        plt.text(0,H2+dz,"$H_2$", fontsize=8)
        plt.text(0,H3-10e-3,"$H_3$", fontsize=8)
        plt.text(0,H4+dz,"$H_4$", fontsize=8)

        L0 = (self.myMEC.l2+self.myMEC.l1)/(2)
        L1 = (self.myMEC.l3+self.myMEC.l4)/(2)
        L2 = (self.myMEC.l7+self.myMEC.l4)/(2)
        plt.axvline(L0, linestyle="--", color="black", linewidth="1")
        plt.axvline(L1, linestyle="--", color="black", linewidth="1")
        plt.axvline(L2, linestyle="--", color="black", linewidth="1")

        dx = 1.0e-3
        dz = 5.0e-3
        plt.text(L0+dx,dz,"$L_0$", fontsize=8)
        plt.text(L1+dx,dz,"$L_1$", fontsize=8)
        plt.text(L2+dx,dz,"$L_2$", fontsize=8)



        plt.xlim([0.0,self.myMEC.l7])
        plt.ylim([0.0,self.myMEC.h9])
        ax.set_aspect('equal')  # Garder les proportions
        # plt.grid()
        plt.savefig("Validation1.pdf")
        plt.show()
        
    def plotMesh(self, folder_path=None):
        color_map = {"air": "lightblue", "fer": "gray", "coil": "orange"}
        fig, ax = plt.subplots(figsize=(8, 12))

        for i, (key, bd) in enumerate(self.mesh.items()):
            # Dessine le rectangle
            rect = plt.Rectangle((bd["x"], bd["z"]), bd["dx"], -bd["dz"],
                                facecolor=color_map[bd["material"]],
                                edgecolor="black", linewidth=0.5)
            ax.add_patch(rect)

            # Ajoute le texte au centre du rectangle
            # if (i <= 10):
            #     center_x = bd["x"] + bd["dx"] / 2
            #     center_z = bd["z"] - bd["dz"] / 2
            #     ax.text(center_x, center_z, f"BD{i}", ha='center', va='center', fontsize=6)

        ax.set_xlim(0, self.probDim["L"])
        ax.set_ylim(0, self.probDim["H"])
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title("Adaptive Mesh")
        ax.set_aspect("equal")

        # Option pour sauvegarder
        # if folder_path:
        #     filename = f"{folder_path}/Topology.pdf"
        #     plt.savefig(filename)
        # else:
        plt.show()



    # def plotMesh(self, folder_path=None):
    #     color_map = {"air": "lightblue", "fer": "gray", "coil": "orange"}
    #     fig, ax = plt.subplots(figsize=(8, 12))

    #     for key, bd in self.mesh.items():
    #         rect = plt.Rectangle((bd["x"], bd["z"]), bd["dx"], -bd["dz"], 
    #                             facecolor=color_map[bd["material"]], edgecolor="black", linewidth=0.5)
    #         ax.add_patch(rect)

    #     ax.set_xlim(0, self.probDim["L"])
    #     ax.set_ylim(0, self.probDim["H"])
    #     ax.set_xlabel("x (mm)")
    #     ax.set_ylabel("z (mm)")
    #     ax.set_title("Adaptive Mesh")
    #     ax.set_aspect("equal")
    #     # filename = f"{folder_path}/Topology.pdf"
    #     # plt.savefig("testMesh.png")
    #     plt.show()