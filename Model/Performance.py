import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Utility import *
from scipy.integrate import simpson


class Performance:

    def __init__(self, myMEC, myMagneticField):
        # self.myMagneticField = myMagneticField
        # self.Bz_ag = self.myMagneticField.Bz_ag
        # self.Bx_ag = self.myMagneticField.Bx_ag
        # self.B_ag = self.myMagneticField.B_ag.3








        

        # self.B_3D = self.myMagneticField.B_3D
        # self.Bx_3D = self.myMagneticField.Bx_3D
        # self.By_3D = self.myMagneticField.By_3D
        # self.Bz_3D = self.myMagneticField.Bz_3D
        # self.y = self.myMagneticField.y
        # self.x = self.myMagneticField.x
        # self.Bz_3D_bis = myMagneticField.Bz_3D_bis
        # self.B_3D_bis = myMagneticField.B_3D_bis
        self.x, self.y,_ = myMagneticField.get_coordinate()
        self.Bz_ag = myMagneticField.Bz_ag
        self.B_ag = myMagneticField.B_ag

        self.myMEC = myMEC
        self.params = myMEC.params
        self.probDim = myMEC.probDim
        
    # def levitationForce3D(self):
    #     Fz = np.trapz(np.trapz(self.Bz_3D_bis**2 - 0.5*self.B_3D_bis**2, self.x, axis=0), self.y, axis=0)
    #     # Fz = np.trapz(np.trapz(self.Bz_3D**2 - 0.5*(self.Bx_3D**2 + self.By_3D**2 + self.Bz_3D**2), self.x, axis=0), self.y, axis=0)

    #     # Fz = np.trapz(np.trapz(0.5*self.Bz_3D**2, self.x, axis=0), self.y, axis=0)
    #     # Fz = 2*Fz/self.params["mu_0"]
    #     # myPrint(f"F_3D = {Fz}")

    #     # Fz = 1/2*np.trapz(np.trapz(self.Bz_3D_bis**2, self.x, axis=0), self.y, axis=0)
    #     Fz = Fz/self.params["mu_0"]

    #     simp = simpson(simpson(self.Bz_3D_bis**2 - 0.5*self.B_3D_bis**2,self.y),self.x)
    #     print(simp/self.params["mu_0"])


    #     return Fz

    def levitationForce2D(self):
        x = self.probDim["x_vec"]
        # F_z1 = np.trapz(self.Bz_ag**2 - 1/2*self.B_ag**2, x)
        # F_z1 = np.trapz(1/2*self.Bz_ag**2, x)
        # F_z1 = 2*F_z1/self.params["mu_0"]*self.probDim["d_maglev"]
        # print(f"F_z_2D_premier = {F_z1}")

        F_z1 = simpson(self.Bz_ag**2 - 1/2*self.B_ag**2,x)
        F_z1 = 2*F_z1/self.params["mu_0"]*self.probDim["d_maglev"]
        
        return F_z1


    # def levitationForceSimple(self):
    #     S = 2*(self.myMEC.l5-self.myMEC.l2)*self.probDim["d_maglev"]
    #     Fz = S*(np.max(self.Bz_ag)*np.max(self.Bz_ag))/(2*self.params["mu_0"])

    #     myPrint(f"F_simple = {Fz}")
