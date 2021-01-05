from dataclasses import dataclass
from dataclasses import field
from typing import NamedTuple
from enum import Enum

class DepType(Enum):
    I_A = 0
    I_I = 1
    A_A = 2

class Cuts(Enum):
    IDEAL =  0
    XLAYER = 1
    LAYER = 2
    GROUP = 3

class CallbackFreq(Enum):
    DEFAULT = -1
    LOG = 0
    CONST = 1
    POW = 2
    
class EncType(Enum):
    IDEAL = 0
    MUL_CHOICE = 1
    BIG_M = 2

class Bounds(Enum):
    INT_ARITHMETIC = 0
    SYMBOLIC_INT_ARITHMETIC = 1
    CONST = 2

     
@dataclass
class Param:
    # Gurobi time limit per MILP in seconds
    # Default: -1 (No time limit)
    TIME_LIMIT: int = -1 
    # Frequency of Gurobi callbacks
    CALLBACK_FREQ: CallbackFreq = CallbackFreq.DEFAULT
    CALLBACK_FREQ_CONST: float = 1
    IDEAL_FREQ: CallbackFreq = CallbackFreq.POW
    IDEAL_FREQ_CONST: float = 1
    XLAYER_DEP_FREQ: CallbackFreq = CallbackFreq.POW
    XLAYER_DEP_FREQ_CONST: float = 1
    LAYER_DEP_FREQ: CallbackFreq = CallbackFreq.POW
    LAYER_DEP_FREQ_CONST: float = 1
    GROUP_DEP_FREQ: CallbackFreq = CallbackFreq.POW
    GROUP_DEP_FREQ_CONST: float = 1
   
    # up to which layer to calculate dependencies
    DEP_DEPTH = -1

    DEFAULT_CUTS: bool = False
    IDEAL_CUTS: bool = True

    XLAYER_DEP_CUTS: bool = True
    LAYER_DEP_CUTS: bool = True
    GROUP_DEP_CUTS: bool = False

    XLAYER_DEP_CONSTRS: bool = True
    LAYER_DEP_CONSTRS: bool = True
    GROUP_DEP_CONSTRS: bool = False

    BOUNDS_CONST: int = 300.
    BOUNDS: Bounds = Bounds.SYMBOLIC_INT_ARITHMETIC
    ENCODING: list = field(default_factory=list)
    DEBUG_MODE: bool = False
    PRINT_GUROBI_OUTPUT: bool = False

    WORKERS_NUMBER: int = 1
    OPTIMISE_BOUNDS: bool = False
    NUM_OPT_LAYERS = 1

    ENCODING: EncType = EncType.IDEAL
    # def __init__(self, n_layers=0):
        # self.ENCODING = [EncType.IDEAL for _ in range(n_layers)]


@dataclass
class SplittingParam:
    # parameters for determining when the splitting process can idle
    # because there are many unprocessed jobs in the jobs queue
    LARGE_N_OF_UNPROCESSED_JOBS: int = 500
    SLEEPING_INTERVAL: int = 3

    # the number of input dimensions still considered to be small
    # so that the best split can be chosen exhaustively
    SMALL_N_INPUT_DIMENSIONS: int = 20

    # the weight parameter when computing the easiness of a problem
    # this value will multiply the fixed ratio
    # and 1 - FIXED_RATIO_WEIGHT will multiply the dependency ratio
    FIXED_RATIO_WEIGHT: float = 1

    # the parameter of the depth exponent
    # bigger values encourage splitting
    DEPTH_POWER: float = 1

    # the value of fixed ratio above which the splitting can stop in any case
    FIXED_RATIO_CUTOFF: float = 0.7

    # the number of parallel splitting processes is 2^d where d is the number below
    STARTING_DEPTH_PARALLEL_SPLITTING_PROCESSES: int = 0

