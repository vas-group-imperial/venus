from src.formula import *
from gurobipy import *
from itertools import product
from enum import Enum
import numpy as np

all_networks = [item for sublist in
                         [[(i+1,j+1) for j in range(9)] for i in range(5)]
                     for item in sublist]
"""
Five inputs
    1) rho    0
    2) theta  1
    3) psi    2
    4) vown   3
    5) vint   4    #using the maximum upper bound of 30,000 feet per second or 33,000 km per hour
                   #as we can reasonably assume these to be valid upper bounds for airplane speed 

Five outputs:
    1) clear-of-conflict
    2) weak right
    3) strong right
    4) weak left
    5) strong left

5 x 9 networks for the pairs:

    Previous advisory
        1) clear-of-conflict
        2) weak left
        3) weak right
        4) strong left
        5) strong right
    
    Time until loss of vertical separation
        1) 0
        2) 1
        3) 5
        4) 10
        5) 20
        6) 40
        7) 60
        8) 80
        9) 100
"""

class AcasConstants:
    (COC, WL, WR, SL, SR) = (0, 1, 2, 3, 4)
    (RHO, THETA, PSI, VOWN, VINT) = (0, 1, 2, 3, 4)

    input_mean_values = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
    input_ranges = np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])
    output_mean = 7.5188840201005975
    output_range = 373.94992


def acas_normalise_input(values):
    return (values -AcasConstants.input_mean_values)/AcasConstants.input_ranges


def acas_denormalise_input(values):
    return values * AcasConstants.input_ranges + AcasConstants.input_mean_values


def acas_normalise_output(value):
    return (value - AcasConstants.output_mean) / AcasConstants.output_range


def acas_denormalise_output(value):
    return value * AcasConstants.output_range + AcasConstants.output_mean


acas_properties = [

    # 1: If the intruder is distant and is significantly slower than the ownship,
    # the score of a COC advisory will always be below a certain fixed threshold
    {
        "name": "Property 1",
        "input":
             NAryConjFormula([
                 VarConstConstraint(StateCoordinate(AcasConstants.RHO), GE, 55947.691),
                 VarConstConstraint(StateCoordinate(AcasConstants.VOWN), GE, 1145),
                 VarConstConstraint(StateCoordinate(AcasConstants.VINT), LE, 60)
             ]),
        "output":
             VarConstConstraint(StateCoordinate(AcasConstants.COC), LT, acas_normalise_output(1500)),
        "raw_bounds": {
            "lower": [55947.691, -3.141592, -3.141592, 1145, 0],
            "upper": [62000, 3.141592, 3.141592, 1200, 60]
        },
        "networks": all_networks,
    },

    # 2: If the intruder is distant and significantly slower than the ownship,
    # the score of a COC advisory will never be maximal
    #
    # Output constrains: the score for COC is not the maximal score
    {
        "name": "Property 2",
        "input":
             NAryConjFormula([
                 VarConstConstraint(StateCoordinate(AcasConstants.RHO), GE, 55947.691),
                 VarConstConstraint(StateCoordinate(AcasConstants.VOWN), GE, 1145),
                 VarConstConstraint(StateCoordinate(AcasConstants.VINT), LE, 60)
             ]),
        "output":
             NAryDisjFormula([
                 VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.WR)),
                 VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.SR)),
                 VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.WL)),
                 VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.SL))
             ]),
        "raw_bounds": {
            "lower": [55947.691, -3.141592, -3.141592, 1145, 0],
            "upper": [62000, 3.141592, 3.141592, 1200, 60]
        },
        "networks": [item for item in all_networks if item not in [(1, j + 1) for j in range(9)]],
    },

    # 3: If the intruder is directly ahead and is moving towards the ownship,
    # the score for COC will not be minimal
    #
    # Output constraints: the score for COC is not the minimal score
    {
        "name": "Property 3",
        "input":
            NAryConjFormula([
                VarConstConstraint(StateCoordinate(AcasConstants.RHO), GE, 1500), VarConstConstraint(StateCoordinate(AcasConstants.RHO), LE, 1800),
                VarConstConstraint(StateCoordinate(AcasConstants.THETA), GE, -0.06), VarConstConstraint(StateCoordinate(AcasConstants.THETA), LE, 0.06),
                VarConstConstraint(StateCoordinate(AcasConstants.PSI), GE, 3.10),
                VarConstConstraint(StateCoordinate(AcasConstants.VOWN), GE, 980),
                VarConstConstraint(StateCoordinate(AcasConstants.VINT), GE, 960)
            ]),
        "output":
            NAryDisjFormula([
                VarVarConstraint(StateCoordinate(AcasConstants.COC), GT, StateCoordinate(AcasConstants.WR)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), GT, StateCoordinate(AcasConstants.SR)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), GT, StateCoordinate(AcasConstants.WL)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), GT, StateCoordinate(AcasConstants.SL))
            ]),
        "raw_bounds": {
            "lower": [1500, -0.06, 3.10, 980, 960],
            "upper": [1800, 0.06, 3.141592, 1200, 1200]
        },
        "networks": [item for item in all_networks if item not in [(1,7), (1,8), (1,9)]],
    },

    # 4: If the intruder is directly ahead and is moving away from the ownship
    # but at a lower speed than that of the ownship,
    # the score for COC will not be minimal
    #
    # Output constraints: the score for COC is not the minimal score
    {
        "name": "Property 4",
        "input":
            NAryConjFormula([
                VarConstConstraint(StateCoordinate(AcasConstants.RHO), GE, 1500), VarConstConstraint(StateCoordinate(AcasConstants.RHO), LE, 1800),
                VarConstConstraint(StateCoordinate(AcasConstants.THETA), GE, -0.06), VarConstConstraint(StateCoordinate(AcasConstants.THETA), LE, 0.06),
                VarConstConstraint(StateCoordinate(AcasConstants.PSI), EQ, 0),
                VarConstConstraint(StateCoordinate(AcasConstants.VOWN), GE, 1000),
                VarConstConstraint(StateCoordinate(AcasConstants.VINT), GE, 700), VarConstConstraint(StateCoordinate(AcasConstants.VINT), LE, 800)
            ]),
        "output":
            NAryDisjFormula([
                VarVarConstraint(StateCoordinate(AcasConstants.COC), GT, StateCoordinate(AcasConstants.WR)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), GT, StateCoordinate(AcasConstants.SR)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), GT, StateCoordinate(AcasConstants.WL)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), GT, StateCoordinate(AcasConstants.SL))
            ]),
        "raw_bounds": {
            "lower": [1500, -0.06, 0, 1000, 700],
            "upper": [1800, 0.06, 3.141592, 1200, 800]
        },
        "networks": [item for item in all_networks if item not in [(1, 7), (1, 8), (1, 9)]],
    },

    # 5: If the intruder is near and approaching from the left,
    # the network advises "strong right"
    #
    # Output constraints: the score for "strong right" is the minimal score
    {
        "name": "Property 5",
        "input":
            NAryConjFormula([
                VarConstConstraint(StateCoordinate(AcasConstants.RHO), GE, 250), VarConstConstraint(StateCoordinate(AcasConstants.RHO), LE, 400),
                VarConstConstraint(StateCoordinate(AcasConstants.THETA), GE, 0.2), VarConstConstraint(StateCoordinate(AcasConstants.THETA), LE, 0.4),
                VarConstConstraint(StateCoordinate(AcasConstants.PSI), GE, -3.141592), VarConstConstraint(StateCoordinate(AcasConstants.PSI), LE, -3.141592+0.005),
                VarConstConstraint(StateCoordinate(AcasConstants.VOWN), GE, 100), VarConstConstraint(StateCoordinate(AcasConstants.VOWN), LE, 400),
                VarConstConstraint(StateCoordinate(AcasConstants.VINT), GE, 0), VarConstConstraint(StateCoordinate(AcasConstants.VINT), LE, 400)
            ]),
        "output":
            NAryConjFormula([
                VarVarConstraint(StateCoordinate(AcasConstants.SR), LT, StateCoordinate(AcasConstants.COC)),
                VarVarConstraint(StateCoordinate(AcasConstants.SR), LT, StateCoordinate(AcasConstants.WR)),
                VarVarConstraint(StateCoordinate(AcasConstants.SR), LT, StateCoordinate(AcasConstants.WL)),
                VarVarConstraint(StateCoordinate(AcasConstants.SR), LT, StateCoordinate(AcasConstants.SL))
            ]),
        "raw_bounds": {
            "lower": [250, 0.2, -3.141592, 100, 0],
            "upper": [400, 0.4, -3.141592+0.005, 400, 400]
        },
        "networks": [(1,1)]
    },

    # 6: If the intruder is sufficiently far away,
    # the network advises COC
    #
    # Output constraints: the score for COC is the minimal score
    {
        "name": "Property 6a",
        "input":
            NAryConjFormula([
                VarConstConstraint(StateCoordinate(AcasConstants.RHO), GE, 12000), VarConstConstraint(StateCoordinate(AcasConstants.RHO), LE, 62000),
                VarConstConstraint(StateCoordinate(AcasConstants.THETA), GE, 0.7), VarConstConstraint(StateCoordinate(AcasConstants.THETA), LE, 3.141592),
                VarConstConstraint(StateCoordinate(AcasConstants.PSI), GE, -3.141592), VarConstConstraint(StateCoordinate(AcasConstants.PSI), LE, -3.141592 + 0.005),
                VarConstConstraint(StateCoordinate(AcasConstants.VOWN), GE, 100), VarConstConstraint(StateCoordinate(AcasConstants.VOWN), LE, 1200),
                VarConstConstraint(StateCoordinate(AcasConstants.VINT), GE, 0), VarConstConstraint(StateCoordinate(AcasConstants.VINT), LE, 1200)
            ]),
        "output":
            NAryConjFormula([
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.WR)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.SR)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.WL)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.SL))
            ]),
        "raw_bounds": {
            "lower": [12000, 0.7, -3.141592, 100, 0],
            "upper": [62000, 3.141592, -3.141592 + 0.005, 1200, 1200]
        },
        "networks": [(1, 1)]
    },
    {
        "name": "Property 6b",
        "input":
            NAryConjFormula([
                VarConstConstraint(StateCoordinate(AcasConstants.RHO), GE, 12000), VarConstConstraint(StateCoordinate(AcasConstants.RHO), LE, 62000),
                VarConstConstraint(StateCoordinate(AcasConstants.THETA), GE, -3.141592), VarConstConstraint(StateCoordinate(AcasConstants.THETA), LE, -0.7),
                VarConstConstraint(StateCoordinate(AcasConstants.PSI), GE, -3.141592), VarConstConstraint(StateCoordinate(AcasConstants.PSI), LE, -3.141592 + 0.005),
                VarConstConstraint(StateCoordinate(AcasConstants.VOWN), GE, 100), VarConstConstraint(StateCoordinate(AcasConstants.VOWN), LE, 1200),
                VarConstConstraint(StateCoordinate(AcasConstants.VINT), GE, 0), VarConstConstraint(StateCoordinate(AcasConstants.VINT), LE, 1200)
            ]),
        "output":
            NAryConjFormula([
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.WR)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.SR)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.WL)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.SL))
            ]),
        "raw_bounds": {
            "lower": [12000, -3.141592, -3.141592, 100, 0],
            "upper": [62000, -0.7, -3.141592 + 0.005, 1200, 1200]
        },
        "networks": [(1, 1)]
    },

    # 7: If vertical separation is large, the network will never advise a strong turn
    #
    # Output constraints: the scores for "strong right" and "strong left" are never the minimal scores
    {
        "name": "Property 7",
        "input":
            NAryConjFormula([
                VarConstConstraint(StateCoordinate(AcasConstants.RHO), GE, 0), VarConstConstraint(StateCoordinate(AcasConstants.RHO), LE, 60760),
                VarConstConstraint(StateCoordinate(AcasConstants.THETA), GE, -3.141592), VarConstConstraint(StateCoordinate(AcasConstants.THETA), LE, 3.141592),
                VarConstConstraint(StateCoordinate(AcasConstants.PSI), GE, -3.141592), VarConstConstraint(StateCoordinate(AcasConstants.PSI), LE, 3.141592),
                VarConstConstraint(StateCoordinate(AcasConstants.VOWN), GE, 100), VarConstConstraint(StateCoordinate(AcasConstants.VOWN), LE, 1200),
                VarConstConstraint(StateCoordinate(AcasConstants.VINT), GE, 0), VarConstConstraint(StateCoordinate(AcasConstants.VINT), LE, 1200)
            ]),
        "output":
            ConjFormula(
                NAryDisjFormula([
                    VarVarConstraint(StateCoordinate(AcasConstants.SR), GT, StateCoordinate(AcasConstants.COC)),
                    VarVarConstraint(StateCoordinate(AcasConstants.SR), GT, StateCoordinate(AcasConstants.WR)),
                    VarVarConstraint(StateCoordinate(AcasConstants.SR), GT, StateCoordinate(AcasConstants.WL))]),
                NAryDisjFormula([
                    VarVarConstraint(StateCoordinate(AcasConstants.SL), GT, StateCoordinate(AcasConstants.COC)),
                    VarVarConstraint(StateCoordinate(AcasConstants.SL), GT, StateCoordinate(AcasConstants.WR)),
                    VarVarConstraint(StateCoordinate(AcasConstants.SL), GT, StateCoordinate(AcasConstants.WL))])
            ),
        "raw_bounds": {
            "lower": [0, -3.141592, -3.141592, 100, 0],
            "upper": [60760, 3.141592, 3.141592, 1200, 1200]
        },
        "networks": [(1, 9)]
    },

    # 8: For a large vertical separation and a previous "weak left" advisory,
    # the network will either output COC or continue advising "weak left".
    #
    # Output constraints: the score for "weak left" is minimal or
    # the score for COC is minimal (both can have the same minimal value at the same time).
    {
        "name": "Property 8",
        "output":
            DisjFormula(
                NAryConjFormula([
                    VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.WR)),
                    VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.SR)),
                    VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.SL))]),
                NAryConjFormula([
                    VarVarConstraint(StateCoordinate(AcasConstants.WL), LT, StateCoordinate(AcasConstants.WR)),
                    VarVarConstraint(StateCoordinate(AcasConstants.WL), LT, StateCoordinate(AcasConstants.SR)),
                    VarVarConstraint(StateCoordinate(AcasConstants.WL), LT, StateCoordinate(AcasConstants.SL))])
            ),
        "raw_bounds": {
            "lower": [0, -3.141592, -0.1, 600, 600],
            "upper": [60760, -0.75*3.141592, 0.1, 1200, 1200]
        },
        "networks": [(2, 9)]
    },

    # 9: Even if the previous advisory was "weak right",
    # the presence of a nearby intruder will cause
    # the network to output a "strong left" advisory instead.
    #
    # Output constraints: the score for "strong left" is minimal.
    {
        "name": "Property 9",
        "output":
            NAryConjFormula([
                VarVarConstraint(StateCoordinate(AcasConstants.SL), LT, StateCoordinate(AcasConstants.WR)),
                VarVarConstraint(StateCoordinate(AcasConstants.SL), LT, StateCoordinate(AcasConstants.SR)),
                VarVarConstraint(StateCoordinate(AcasConstants.SL), LT, StateCoordinate(AcasConstants.WL)),
                VarVarConstraint(StateCoordinate(AcasConstants.SL), LT, StateCoordinate(AcasConstants.COC))]),
        "raw_bounds": {
            "lower": [2000, -0.4, -3.141592, 100, 0],
            "upper": [7000, -0.14, -3.141592+0.01, 150, 150]
        },
        "networks": [(3, 3)]
    },
    # 10: For a far away intruder, the network advises COC.
    #
    # Output constraints: the score for COC is minimal.
    {
        "name": "Property 10",
        "output":
            NAryConjFormula([
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.WR)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.SR)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.WL)),
                VarVarConstraint(StateCoordinate(AcasConstants.COC), LT, StateCoordinate(AcasConstants.SL))]),
        "raw_bounds": {
            "lower": [36000, 0.7, -3.141592, 900, 600],
            "upper": [60760, 3.141592, -3.141592 + 0.01, 1200, 1200]
        },
        "networks": [(4, 5)]
    }
]


for prop in acas_properties:
    raw_lower = prop["raw_bounds"]["lower"]
    raw_upper = prop["raw_bounds"]["upper"]
    norm_lower = acas_normalise_input(raw_lower)#[(raw_lower[i] - input_mean_values[i])/input_ranges[i] for i in range(len(raw_lower))]
    norm_upper = acas_normalise_input(raw_upper)#[(raw_upper[i] - input_mean_values[i])/input_ranges[i] for i in range(len(raw_upper))]

    prop["bounds"] = {"lower": norm_lower, "upper": norm_upper}

