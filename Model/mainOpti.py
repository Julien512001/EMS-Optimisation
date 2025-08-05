import numpy as np
from Maglev import *
from Utility import *

IS_PRINT = False

def main():


    filename = "MagLev_opti.ini"

    # =========== Testing the optimisation loop =================
    
    # Basique
    e_plate = 28.0e-3
    l_core = 220.0e-3
    h_core = 136.0e-3
    l_f = 10.0e-3
    h_f = 10.0e-3

    e_branch = 28.0e-3
    e_base = 58.0e-3
    l_coil = 164.0e-3
    h_coil = 68.0e-3

    a_branch = 2*e_branch/l_core
    a_lcoil = l_coil/(l_core-2*e_branch)
    a_hcoil = h_coil/(h_core-h_f-e_base)
    a_hf = h_f/(h_core-e_base)
    a_lf = (2*l_f)/(l_core-2*e_branch)
    a_base = e_base/h_core

    d_maglev = 12
    v_mag = 700/3.6

    l_core   = 0.0954355682440714
    h_core   = 0.11611266806764645
    e_plate  = 0.05110724730739683

    a_branch = 0.537669499231554
    a_lcoil  = 1.0
    a_hcoil  = 1.0
    a_hf     = 0.10002348666565977
    a_lf     = 0.5006368115350897
    a_base   = 0.2722017217281425

    d_maglev = 12.0
    v_mag    = 700/3.6

    # l_core = 0.15
    # h_core = 0.11312696952727673
    # e_plate = 0.053888439305812004

    # a_branch = 0.5768982926022738
    # a_lcoil = 1.0
    # a_hcoil = 1.0
    # a_hf = 0.10482876110220377
    # a_lf = 0.5718881263729412
    # a_base = 0.2728256867840116

    # d_maglev = 12




    optiVar = {"a_branch": a_branch,
               "a_lcoil": a_lcoil,
               "a_hcoil": a_hcoil,
               "a_hf": a_hf,
               "a_lf": a_lf,
               "a_base": a_base,
               "e_plate": e_plate,
               "l_core": l_core,
               "h_core": h_core,
               "d_maglev": d_maglev,
               "v_mag":v_mag}

    Maglev(optiVariable=optiVar, filename=filename)
    # Maglev(filename=filename)

    return 0



if __name__=="__main__":
    main()