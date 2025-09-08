"""
Configuration parameters
"""
import numpy as np

# Experiment
BOUNDS = (-5,5) # Target xy-coordinate bounds
SIM_TIME = 120 # Simulation time
TARGETS = np.array(
    [[1,1],
     [0,2],
     [1,3],
     [0,4]]
    )

DESC = (
        "Simulate evolution with inherited knowledge.\n"
        "Don't forget to:\n"
        " - Check parameters in config.py.\n"
        " - Set the number of simulators to maximally utilize CPU for multithreaded simulation.\n(Recommend no. of threads minus 1)\n"
        )

# Body
DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS_BODY = 2 # No. of repetitions of the morphology optimization process
POPULATION_SIZE_BODY = 50
OFFSPRING_SIZE = 25
NUM_GENERATIONS_BODY = 12 # No. of body optimization generations

# Differential Evolution parameters
NUM_SIMULATORS_BRAIN = 20 # Directly controls the threaded environment for simulation, should be double num_pop_brain
NUM_GENERATIONS_BRAIN = 10
NUM_POPULATION_BRAIN = 10 # This value is doubled for threaded cpu count MODIFY ME FOR MORE JUICE
# 12 target vectors and 12 candidate vectors ^^^
PERTURB_SD_MOD = 2
P_CR = 0.7 # [0,1]
P_MU = 0.5 # [0,1]
F = 1.25 # [0,2]
UNIF_SAMPLE = 0.7

