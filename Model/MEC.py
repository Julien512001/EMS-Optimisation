import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from configparser import ConfigParser
from Mesh import *
from Utility import myPrint
from scipy.optimize import fsolve
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

class MEC:

    def __init__(self, param, optiVariable=None, filename=None):
        myPrint("----MEC : Initialization of all variables----\n")
        self.param = param
        self.optiVariable = optiVariable
        self.filename = filename

        self.extractData()
        if (optiVariable):
            self.extractOptiVar()
        self.dimensions()
        self.get_meshParam()
        self.params = self.paramaters()
        self.probDim = self.problemDimension()
        self.modDim = self.moduleDimension()

    def extractData(self):
        config_path = self.filename
        config = ConfigParser()
        config.read(config_path)
        
        if (config.has_section("air")):
            self.l_air_p0 = float(config["air"]["l_air_p0"])
            self.l_air_p1 = float(config["air"]["l_air_p1"])
            self.l_air_m = float(config["air"]["l_air_m"])
            self.e_gap = float(config["air"]["e_gap"])

        if (config.has_section("params")):
            # self.Nmax = float(config["params"]["Nmax"])
            self.I = float(config["params"]["I"])
            self.d_solve = float(config["params"]["d_solve"])
            self.d_maglev = float(config["params"]["d_maglev"])
            self.M_mag = float(config["params"]["M_mag"])*1
            self.v_mag = float(config["params"]["v_mag"])

        if (config.has_section("permeability")):
            self.mu_0 = float(config["permeability"]["mu_0"])
            self.mu_fer = float(config["permeability"]["mu_fer"])
            self.mu_coil = float(config["permeability"]["mu_coil"])

        if (config.has_section("coilParams")):
            self.FF = float(config["coilParams"]["FF"])
            self.S_fil = float(config["coilParams"]["S_fil"])
            self.Nmax = float(config["coilParams"]["Nmax"])
            self.rho_fil = float(config["coilParams"]["rho_fil"])

        if (config.has_section("meshParams")):
            self.p0 = int(config["meshParams"]["p0"])
            self.p1 = int(config["meshParams"]["p1"])
            self.p_offset = int(config["meshParams"]["p_offset"])
            self.p2 = int(config["meshParams"]["p2"])
            self.p3 = int(config["meshParams"]["p3"])
            self.p4 = int(config["meshParams"]["p4"])
            self.p5 = int(config["meshParams"]["p5"])


            self.m0 = int(config["meshParams"]["m0"])
            self.m_offset_down = int(config["meshParams"]["m_offset_down"])
            self.m1 = int(config["meshParams"]["m1"])
            self.m2 = int(config["meshParams"]["m2"])
            self.m3 = int(config["meshParams"]["m3"])
            self.m4 = int(config["meshParams"]["m4"])
            self.m5 = int(config["meshParams"]["m5"])
            self.m6 = int(config["meshParams"]["m6"])
            self.m7 = int(config["meshParams"]["m7"])
            self.m_offset_up = int(config["meshParams"]["m_offset_up"])
            self.m8 = int(config["meshParams"]["m8"])



            self.offset = float(config["meshParams"]["offset"])


    def extractOptiVar(self):
        self.l_core = self.optiVariable["l_core"]
        self.h_core = self.optiVariable["h_core"]
        self.e_plate = self.optiVariable["e_plate"]
        self.d_maglev = self.optiVariable["d_maglev"]

        self.e_branch = self.optiVariable["a_branch"]*self.l_core/2.0

        self.e_base = self.optiVariable["a_base"]*(self.h_core)

        self.h_foot = self.optiVariable["a_hf"]*(self.h_core - self.e_base)
        self.l_foot = self.optiVariable["a_lf"]*(self.l_core - 2*self.e_branch)/2.0

        self.l_coil = self.optiVariable["a_lcoil"]*(self.l_core - 2*self.e_branch)
        self.h_coil = self.optiVariable["a_hcoil"]*(self.h_core - self.h_foot - self.e_base)

        self.v_mag = self.optiVariable["v_mag"]

        self.l_air_p1 = self.l_core*2.5
        self.l_air_m = self.param[0]

        # print(
        #     f"l_core   = {self.l_core}\n"
        #     f"h_core   = {self.h_core}\n"
        #     f"e_plate  = {self.e_plate}\n"
        #     f"e_branch = {self.e_branch}\n"
        #     f"l_coil   = {self.l_coil}\n"
        #     f"h_coil   = {self.h_coil}\n"
        #     f"h_foot   = {self.h_foot}\n"
        #     f"l_foot   = {self.l_foot}\n"
        #     f"e_base   = {self.e_base}"
        # )

    def dimensions(self):

        self.l1 = self.l_air_p0
        self.l2 = self.l1 + self.l_air_p1
        self.l3 = self.l2 + self.l_foot
        self.l4 = self.l3 + self.e_branch
        self.l5 = self.l4 + self.l_foot
        self.l6 = self.l4 + self.l_core/2 - self.l_coil/2 - self.e_branch
        self.l7 = self.l6 + self.l_coil/2.0

        self.h1 = self.l_air_m
        self.h2 = self.h1 + self.h_coil
        self.h3 = self.h2 + self.e_base
        self.h4 = self.h3 + self.h_coil
        self.h5 = self.h2 + self.h_core - self.h_foot
        self.h6 = self.h2 + self.h_core
        self.h7 = self.h6 + self.e_gap
        self.h8 = self.h7 + self.e_plate
        self.h9 = self.h8 + self.l_air_m

        self.l = np.array([self.l1,self.l2, self.l3,self.l4,self.l5,self.l6,self.l7])
        self.h = np.array([self.h1,self.h2, self.h3,self.h4,self.h5,self.h6,self.h7,self.h8,self.h9])
        self.L = self.l[-1]
        self.H = self.h[-1]

        # print(self.l)
        # print(self.h)

    def equation(self, r, d1, N, L):
        return d1 * (r**N - 1) / (r - 1) - L

    def get_meshParam(self):

        # Config for test : 
        # p2=1 ; p3=2; m0=2; m1=2; m2=2; m7=2
        # d1 = 1; Np = 10; Nm = 3


        p1_factor = 2
        p2_factor = 2
        p3_factor = 2

        m0_factor = 2
        m1_factor = 2
        m2_factor = 2
        m7_factor = 2

        self.p2 = int(np.round((self.l5-self.l2)*1000)/p2_factor)
        self.p3 = int(np.round((self.l7-self.l5)*1000/p3_factor))

        self.m1 = int(np.round((self.h2-self.h1)*1000/m1_factor))
        self.m2 = int(np.round((self.h3-self.h2)*1000/m2_factor))
        self.m3 = self.m1
        self.m4 = 5
        self.m5 = int(np.round((self.h6-self.h5)*1000))
        self.m6 = int(np.round((self.h7-self.h6)*1000))
        self.m7 = int(np.round((self.h8-self.h7)*1000/m7_factor))


        self.lp2 = (self.l5-self.l2)/self.p2
        self.lp3 = (self.l7-self.l5)/self.p3
        self.lp_offset = self.offset/self.p_offset

        self.hm1 = (self.h2-self.h1)/self.m1
        self.hm2 = (self.h3-self.h2)/self.m2
        self.hm3 = (self.h4-self.h3)/self.m3
        if (np.isclose(self.h4, self.h5, rtol=1e-9, atol=1e-12)):
            self.hm4 = 0.0
            self.m4 = 0
        else:
            self.hm4 = (self.h5-self.h4)/self.m4
        if (np.isclose(self.h5, self.h6, rtol=1e-9, atol=1e-12)):
            self.hm5 = 0.0
            self.m5 = 0
        else:
            self.hm5 = (self.h6-self.h5)/self.m5
        self.hm6 = (self.h7-self.h6)/self.m6
        self.hm7 = (self.h8-self.h7)/self.m7

        # print(f"hm1 = {self.hm1}")
        # print(f"hm7 = {self.hm7}")
        # print(f"lp2 = {self.lp2}")

        # d1 = 1.0e-3
        dx0 = self.lp2
        # print(f"dx0 = {dx0}")
        N_p = int(self.l_air_p1*100)
        N_p = int(N_p/p1_factor)
        L = self.l_air_p1
        r_p = fsolve(self.equation, 1.5, args=(dx0, N_p, L))[0]
        sizes_p = dx0 * r_p ** np.arange(N_p)


        # d1 = 1.0e-3
        dz0 = self.hm1
        # print(f"dz0 = {dz0}")
        # N_m = int(100*self.l_air_m)
        N_m = int(self.l_air_m*100)
        N_m = int(N_m/m0_factor)
        # print(f"N_m = {N_m}")
        H = self.l_air_m
        r_m = fsolve(self.equation, 1.5, args=(dz0, N_m, H))[0]
        sizes_m = dz0 * r_m ** np.arange(N_m)

        self.dp = np.hstack([sizes_p[::-1], np.ones(self.p2)*self.lp2, np.ones(self.p3)*self.lp3])
        self.dm = np.hstack([sizes_m[::-1], 
                             np.ones(self.m1)*self.hm1,
                             np.ones(self.m2)*self.hm2,
                             np.ones(self.m3)*self.hm3,
                             np.ones(self.m5)*self.hm5,
                             np.ones(self.m6)*self.hm6,
                             np.ones(self.m7)*self.hm7,
                             sizes_m])
        # print(sizes_p[0], sizes_p[-1])
        # print(sizes_m[0], sizes_m[-1])


        # print(f"p1 = {len(sizes_p)}")
        # print(f"p2 = {self.p2}")
        # print(f"p3 = {self.p3}")

        # print(f"m0 = {len(sizes_m)}")
        # print(f"m1 = {self.m1}")
        # print(f"m2 = {self.m2}")
        # print(f"m3 = {self.m3}")
        # print(f"m4 = {self.m4}")
        # print(f"m5 = {self.m5}")
        # print(f"m6 = {self.m6}")
        # print(f"m7 = {self.m7}")
        # print(f"m8 = {len(sizes_m)}")


    def paramaters(self):        
        self.p = len(self.dp)
        self.m = len(self.dm)

        Npsi_x = self.p
        Npsi_z = self.m-1
        Npsi = Npsi_x*Npsi_z

        P = 2*self.p-1
        M = 2*Npsi_z

        Nx = self.m*P
        Nz = self.p*M

        N_BD = self.p*self.m
        N = Nx + Nz


        # print(f"({self.m}x{self.p})")
        # print(N_BD)

        params = {
            "p": self.p,
            "m": self.m,
            "P": P,
            "M": M,
            "Nx": Nx,
            "Nz": Nz,
            "N": N,
            "Npsi_x": Npsi_x,
            "Npsi_z": Npsi_z,
            "Npsi": Npsi,
            "N_BD": N_BD,
            "I": -self.I,
            "Nmax": int(self.Nmax/2),
            "mu_0": self.mu_0,
            "mu_fer": self.mu_fer,
            "mu_coil": self.mu_coil,
            "FF": self.FF,
            "S_fil": self.S_fil
        }

        return params
    
    def moduleDimension(self):

        modDim = {"l_core": self.l_core,
                  "h_core": self.h_core,
                  "e_plate": self.e_plate,
                  "e_branch": self.e_branch,
                  "e_base": self.e_base,
                  "h_foot": self.h_foot,
                  "l_foot": self.l_foot,
                  "l_coil": self.l_coil,
                  "h_coil": self.h_coil,
                  "e_gap": self.e_gap,
                  "d_maglev": self.d_maglev
        }
    
        return modDim

    def problemDimension(self):

        probDim = {"L": self.L,
                    "H": self.H,
                    "d_maglev": self.d_maglev,
                    "d_solve": self.d_solve,
                    "x_vec": 0,
                    "z_vec": 0
                    }

        return probDim
    

    def get_W(self):
        rho_core = 7865
        rho_coil = 8950


        # Calcul du poids du bobinage
        V1 = self.h_coil*self.l_coil*self.d_maglev
        V2 = (self.h_coil*self.l_coil*(2*self.h_coil+self.e_base))
        m_coil = rho_coil*(2*V1+2*V2)

        # Calcul du poids du fer
        V1 = (self.h_foot*self.l_foot)*self.d_maglev
        V2 = (self.e_branch*(self.h_core))*self.d_maglev
        V3 = (((self.l_core/2)-self.e_branch)*self.e_base)*self.d_maglev
        m_core = rho_core*2*(V1+V2+V3)

        m_EM = m_core + m_coil

        m_tot = 4*m_EM + self.M_mag

        W_mag_quad = 1/4.0*m_tot*9.81

        return W_mag_quad
    
    def get_CoilResistance(self):
        return 2*self.rho_fil/(self.h_coil*self.l_coil)*(self.d_maglev+2*self.h_coil+self.e_base)*self.Nmax**2
    
    def get_speed(self):
        return self.v_mag