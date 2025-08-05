import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy as sp

from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from Utility import *

class Solver:


    def __init__(self, myMEC, myNetwork):
        myPrint("----Solver : Initialization of the solver----\n")
        self.params = myMEC.params
        self.probDim= myMEC.probDim
        self.network = myNetwork
        self.Xsi = self.network.Xsi
        self.MMF = self.network.MMF
        self.R = self.network.R
        self.filename = f"dataR/B_{os.path.splitext(myMEC.filename)[0]}.csv"

        self.solve_system()
        myPrint("----Solver : End----\n")


    def solve_system(self):
        Xsi = csc_matrix(self.Xsi)
        Xsi_T = csc_matrix(sp.transpose(Xsi))
        R = csc_matrix(self.R)
        self.psi = spsolve(csc_matrix(Xsi @ (R @ Xsi_T)), csc_matrix(self.Xsi @ self.MMF))
        phi = Xsi_T @ self.psi
        S = np.hstack([self.network.S_x, self.network.S_z])
        self.B = np.divide(phi, S)


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