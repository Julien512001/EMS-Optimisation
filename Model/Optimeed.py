"""Now that you know everything about data and visualization, let's get started with optimization!
Optimeed provides high-level interface to perform optimization with visualization and data storage.
The Wiki gives more details about the optimization. To get started, you need the following key ingredients:

    - A device that contains the variables to be optimized ("Device") and other parameters you would like to save
    - A list of optimization variables ("OptimizationVariable")
    - An evaluation function ("Characterization")
    - One or more objective functions ("Objectives")
    - (optional) Eventual constraints ("Constraints")
    - An optimization algorithm ("Optimization Algorithm")
    - Something that will fill the "Device" object with the optimization variables coming from the optimization algorithm. ("MathsToPhysics")
        Don't get scared with this one, if you do not know how it can be useful, the proposition by default works perfectly fine.
    - Something that will link all the blocks together ("Optimizer")
"""


# These are what we need for the optimization
from optimeed.optimize.optiAlgorithms import MultiObjective_GA as OptimizationAlgorithm
from optimeed.optimize import Real_OptimizationVariable, InterfaceObjCons, InterfaceCharacterization, OptiHistoric, Integer_OptimizationVariable
from optimeed.optimize.optimizer import OptimizerSettings, run_optimization

# These are the high-level visualization tools
from optimeed.visualize.displayOptimization import OptimizationDisplayer
from optimeed.visualize import Onclick_representDevice, Represent_brut_attributes, start_qt_mainloop
from optimeed.core.evaluators import PermanentMultiprocessEvaluator, MultiprocessEvaluator, Evaluator
import time
import os

# Maintenant tu peux faire l'import comme s’il était à la racine
from Maglev import *  # ou ce que tu veux importer


class Device:
    """Define the Device to optimize."""
    l_core: float  
    h_core: float  
    e_plate: float
    d_maglev: float
    # Nmax: int

    a_branch: float  
    a_lcoil: float  
    a_hcoil: float  
    a_hf: float  
    a_lf: float  
    a_base: float

    v_mag: float

    def __init__(self):
        self.l_core = 300e-3/2
        self.h_core = 300e-3/2
        self.e_plate = 28e-3/2
        self.d_maglev = 12.0


        self.a_branch = 0.5
        self.a_lcoil = 1.0
        self.a_hcoil = 1.0
        self.a_hf = 0.5
        self.a_lf = 0.5
        self.a_base = 0.5


        self.v_mag = 700/3.6


    def __str__(self):
            return (
                "=== Device Parameters ===\n"
                f"(l_core)   : {self.l_core:.3f} [m]\n"
                f"(h_core)   : {self.h_core:.3f} [m]\n"
                f"(e_plate)  : {self.e_plate:.3f} [m]\n"
                f"(d_maglev) : {self.d_maglev:.3f} [m]\n"
                f"(a_branch) : {self.a_branch:.3f} \n"
                f"(a_lcoil)  : {self.a_lcoil:.3f} \n"
                f"(a_hcoil)  : {self.a_hcoil:.3f} \n"
                f"(a_hf)     : {self.a_hf:.3f}\n"
                f"(a_lf)     : {self.a_lf:.3f}\n"
                f"(a_base)   : {self.a_base:.3f} \n"
                f"(v_mag)   : {self.v_mag:.3f} \n"
            )

class Characterization(InterfaceCharacterization):
    """Define the Characterization scheme. In this case nothing is performed,
     but this is typically where model code will be executed and results saved inside 'theDevice'.
     The arguments time_initialization and time_evaluation are there to mimic the evaluation time of the model.
     They will be use to compare different evaluators
     """
    def __init__(self, time_initialization=1, time_evaluation=0.01):
        self.initialized = False
        self.time_initialization = time_initialization
        self.time_evaluation = time_evaluation

    def compute(self, thedevice):
        if not self.initialized:
            print("Characterization is initializing -> {} second penalty".format(self.time_initialization))
            time.sleep(self.time_initialization)
            self.initialized = True
        print("Process {} is active. The id of the characterization: {}".format(hex(os.getpid()), hex(id(self))))
        time.sleep(self.time_evaluation)
        optiVar = {"a_branch": thedevice.a_branch,
            "a_lcoil": thedevice.a_lcoil,
            "a_hcoil": thedevice.a_hcoil,
            "a_hf": thedevice.a_hf,
            "a_lf": thedevice.a_lf,
            "a_base": thedevice.a_base,
            "e_plate": thedevice.e_plate,
            "l_core": thedevice.l_core,
            "h_core": thedevice.h_core,
            "d_maglev": thedevice.d_maglev,
            "v_mag":thedevice.v_mag}

        optiVar = {key: round(val, 3) for key, val in optiVar.items()}
        # print(optiVar)

        myMaglev = Maglev(optiVariable=optiVar, filename="MagLev_opti.ini")
        thedevice.P_tot = myMaglev.P_tot

class MyObjective1(InterfaceObjCons):
    """First objective function (to be minimized)"""
    def compute(self, thedevice):
        return thedevice.P_tot


if __name__ == "__main__":  # This line is necessary to spawn new processes
    """Start the main code. Instantiate previously defined classes."""
    theDevice = Device()
    theAlgo = OptimizationAlgorithm()
    theAlgo.set_option(theAlgo.OPTI_ALGORITHM, "GA")  # You can change the algorithm if you need ;)

    theCharacterization = Characterization(time_initialization=0, time_evaluation=0.00)

    """Variable to be optimized"""
    optimizationVariables = list()
    optimizationVariables.append(Real_OptimizationVariable('l_core', 50.0e-3, 800.0e-3))  #
    optimizationVariables.append(Real_OptimizationVariable('h_core', 50.0e-3, 800.0e-3))  #
    optimizationVariables.append(Real_OptimizationVariable('e_plate', 10.0e-3, 100.0e-3))  #

    optimizationVariables.append(Real_OptimizationVariable('a_branch', 0.1, 0.9))  #
    # optimizationVariables.append(Real_OptimizationVariable('a_lcoil', 0.01, 1.0))  #
    # optimizationVariables.append(Real_OptimizationVariable('a_hcoil', 0.01, 1.0))  #
    optimizationVariables.append(Real_OptimizationVariable('a_hf', 0.1, 0.9))  #
    optimizationVariables.append(Real_OptimizationVariable('a_lf', 0.1, 0.9))  #
    optimizationVariables.append(Real_OptimizationVariable('a_base', 0.1, 0.9))  #

    """Objective and constraints"""
    listOfObjectives = [MyObjective1()]
    listOfConstraints = []

    """Set the optimizer"""
    theOptiSettings = OptimizerSettings(theDevice, listOfObjectives, listOfConstraints, optimizationVariables,
                                        theOptimizationAlgorithm=theAlgo, theCharacterization=theCharacterization)

    """The logger (to automatically save the points)"""
    theOptiHistoric = OptiHistoric(optiname="opti", autosave_timer=10, autosave=True, create_new_directory=True)

    """The evaluator. It is used to manage the evaluations of the parameters from the optimization algorithm, and the behaviour in parallel run.
    The first evaluator does not allow parallel run. You can see it with the constant process and id characterization. It is the recommended evaluator if you do not need parallelism. 
    The second evaluator allow parallel run. It is the recommended evaluator for parallelism. Notice each process getting its own id, but the characterization is forked at each call.
    In other words, if initialization of the models is long to execute (for instance, opening matlab), then you will pay the full price each time.
    The third evaluator also allows parallel run, but the characterizations are forked only once -> no extra penalty on initialization afterwards.
    """
    # theEvaluator = Evaluator(theOptiSettings)  # First evaluator -> One initialization, then proceeds in same thread
    theEvaluator = MultiprocessEvaluator(theOptiSettings, number_of_cores=8)  # Second evaluator -> Parallel run, initializes Charac() at each run.
    # theEvaluator = PermanentMultiprocessEvaluator(theOptiSettings, number_of_cores=2)  # Third evaluator -> Parallel run, initializes Charac() at startup.

    """Start the optimization"""
    hour = 4
    sec = hour*3600
    max_opti_time_sec = sec

    display_opti = True

    if display_opti:  # Display real-time graphs
        optiDisplayer = OptimizationDisplayer(theOptiSettings, theOptiHistoric, light_background=True)
        _, theDataLink, _ = optiDisplayer.generate_optimizationGraphs()

        # Here we set the actions on click.
        theActionsOnClick = list()
        theActionsOnClick.append(Onclick_representDevice(theDataLink, [Represent_brut_attributes()]))
        optiDisplayer.set_actionsOnClick(theActionsOnClick)

        resultsOpti, convergence = optiDisplayer.launch_optimization([theOptiSettings, theOptiHistoric],
                                                                     {"max_opti_time_sec": max_opti_time_sec, "evaluator": theEvaluator},
                                                                     refresh_time=0.1, max_nb_points_convergence=None)  # Refresh the graphs each nth seconds

    else:  # Otherwise just focus on results ... That can be helpful if you are confident the optimizations will converge and you need to launch several optimizations.
        resultsOpti, convergence = run_optimization(theOptiSettings, theOptiHistoric, max_opti_time_sec=max_opti_time_sec, evaluator=theEvaluator)

    """Gather results"""
    # Pro hint: you would probably never work with these next few lines of code, instead you would move to the next tutorial
    # to retrieve the results from the automatically saved files.
    print("Best individuals :")
    for device in resultsOpti:
        print(device)

    if display_opti:
        start_qt_mainloop()  # To keep windows alive

    """Note that the results are automatically saved if KWARGS_OPTIHISTO autosaved=True.
    In this case, optimization folder is automatically generated in Workspace/optiX. It contains five files:
    -> autosaved: contains all the devices evaluated during the optimization
    -> logopti: contains all the information relating to the optimization itself: objectives, constraints, evaluation time.
    -> opticonvergence: contains all the information relative to the convergence of the optimization (saved only at the end)
    -> results: all the best devices as decided by the optimization algorithm
    -> optimization_parameters: the class OptimizationParameters that can be reloaded using SingleObjectSaveLoad.load 
    -> summary.html: a summary of the optimization problem
    See other tutorials on how to save/load these information.
    """
