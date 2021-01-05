import sys
sys.path.append('.')

import argparse
import numpy as np
import pickle
import random
from timeit import default_timer as timer

#### to stop seeing various warning and system messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
####

from resources.acas.acasprop import acas_properties, acas_denormalise_input, acas_normalise_input
from src.parameters import SplittingParam, Param
from src.specifications import GenSpec, LRob
from src.venusverifier import VenusVerifier

verifier_result_to_string = {"True": "NOT Satis", "False": "Satisfied", "Interrupted": "Interrupt", "Timeout": "Timed Out"}

def verify_acas_property(options):
    random.seed(options.acas_prop)

    prop = acas_properties[options.acas_prop]

    input_bounds = (np.array(prop['bounds']['lower']), np.array(prop['bounds']['upper']))
    spec = GenSpec(input_bounds, prop['output'])

    encoder_params = Param()
    encoder_params.TIME_LIMIT = options.timeout
    encoder_params.XLAYER_DEP_CONSTRS = options.offline_dep
    encoder_params.LAYER_DEP_CONSTRS = options.offline_dep
    encoder_params.XLAYER_DEP_CUTS = options.online_dep
    encoder_params.LAYER_DEP_CUTS = options.online_dep
    encoder_params.IDEAL_CUTS = options.ideal_cuts
    encoder_params.WORKERS_NUMBER = options.workers

    splitting_params = SplittingParam()
    splitting_params.FIXED_RATIO_CUTOFF = options.st_ratio
    splitting_params.DEPTH_POWER = options.depth_power
    splitting_params.STARTING_DEPTH_PARALLEL_SPLITTING_PROCESSES = options.splitters
    
    verifier = VenusVerifier(options.net, spec, encoder_params, splitting_params, options.print)

    start = timer()
    result, job_id, extra = verifier.verify()
    end = timer()
    runtime = end - start

    result = verifier_result_to_string[result]

    print("{} over {}".format(prop['name'], options.net),
          "is", result, "in {:9.4f}s".format(runtime), "job n", job_id)
    if result == True:
        denormalised_ctx = acas_denormalise_input(extra)
        ctx = extra.reshape(1, -1)
        network_output = nmodel.predict(x=ctx, batch_size=1)
        print("\t\tCounter-example:", list(extra))
        print("\t\tDenormalised   :", list(denormalised_ctx))
        print("\t\tNetwork output :", list(network_output[0]))
    print("")

    return result, runtime, job_id


def verify_local_robustness(options):
    # load image
    with open(options.lrob_input, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    image = data[0]
    label = data[1]
    num_classes = 10
    # range of pixel values
    min_value = 0
    max_value = 1

    # create specification
    spec = LRob(image, label, num_classes, options.lrob_radius, min_value, max_value)

    # set verifier's parameters
    encoder_params = Param()
    encoder_params.TIME_LIMIT = options.timeout
    encoder_params.XLAYER_DEP_CONSTRS = options.offline_dep
    encoder_params.LAYER_DEP_CONSTRS = options.offline_dep
    encoder_params.XLAYER_DEP_CUTS = options.online_dep
    encoder_params.LAYER_DEP_CUTS = options.online_dep
    encoder_params.IDEAL_CUTS = options.ideal_cuts
    encoder_params.WORKERS_NUMBER = options.workers

    # set splitter's parameters
    splitting_params = SplittingParam()
    splitting_params.FIXED_RATIO_CUTOFF = options.st_ratio
    splitting_params.DEPTH_POWER = options.depth_power
    splitting_params.STARTING_DEPTH_PARALLEL_SPLITTING_PROCESSES = options.splitters

    # create verifier
    verifier = VenusVerifier(options.net, spec, encoder_params, splitting_params, options.print)

    start = timer()
    result, job_id, extra = verifier.verify()
    end = timer()
    runtime = end - start

    result = verifier_result_to_string[result]

    print("Local robustness for input {} perturbed by {} over {}".format(options.lrob_input, options.lrob_radius, options.net),
          "is", result, "in {:9.4f}s".format(runtime), "job n", job_id)
    if result == True:
        ctx = extra.reshape(1, -1)
        network_output = nmodel.predict(x=ctx, batch_size=1)
        print("\t\tCounter-example:", list(extra))
        print("\t\tNetwork output :", list(network_output[0]))
        print("\t\tExpected label :", label)
    print("")

    return result, runtime, job_id


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser(description="Venus Example",
                                     epilog="Exactly one of the parameters --acas_prop or --lrob_input is required.")
    parser.add_argument("--property", choices=["acas", "lrob"], required=True,
                        help="Verification property, one of acas or lrob (local robustness).")
    parser.add_argument("--net", type=str, required=True,
                        help="Path to the neural network in Keras format.")
    parser.add_argument("--acas_prop", type=int, default=None,
                        help="Acas property number from 0 to 10. Default value is 1.")
    parser.add_argument("--lrob_input", type=str, default=None,
                        help="Path to the original input and the correct label for the local robustness property in the pickle format.")
    parser.add_argument("--lrob_radius", default=0.1, type=float,
                        help="Perturbation radius for L_inifinity norm. Default value is 0.1.")
    parser.add_argument("--st_ratio", default=0.5, type=float,
                        help="Cutoff value of the stable ratio during the splitting procedure. Default value is 0.5.")
    parser.add_argument("--depth_power", default=1.0, type=float,
                        help="Parameter for the splitting depth. Higher values favour splitting. Default value is 1.")
    parser.add_argument("--splitters", default=0, type=int,
                        help="Determines the number of splitting processes = 2^splitters. Default value is 0.")
    parser.add_argument("--workers", default=1, type=int,
                        help="Number of worker processes. Default value is 1.")
    parser.add_argument("--offline_dep", default=True, type=boolean_string,
                        help="Whether to include offline dependency cuts (before starting the solver) or not. Default value is True.")
    parser.add_argument("--online_dep", default=True, type=boolean_string,
                        help="Whether to include online dependency cuts (through solver callbacks) or not. Default value is True.")
    parser.add_argument("--ideal_cuts", default=True, type=boolean_string,
                        help="Whether to include online ideal cuts (through solver callbacks) or not. Default value is True.")
    parser.add_argument("--opt_bounds", default=False, type=boolean_string,
                        help="Whether to optimise bounds using linear relaxation or not. Default value is False.")
    parser.add_argument("--timeout", default=3600, type=int,
                        help="Timeout in seconds. Default value is 3600.")
    parser.add_argument("--logfile", default=None, type=str,
                        help="Path to logging file.")
    parser.add_argument("--print", default=False, type=boolean_string,
                        help="Print extra information or not. Default value is False.")

    ARGS = parser.parse_args()

    if ARGS.acas_prop is None and ARGS.lrob_input is None:
        parser.print_help()
        sys.exit(1)

    if ARGS.property == "acas":
        result, runtime, job_id = verify_acas_property(ARGS)
    else:
        result, runtime, job_id = verify_local_robustness(ARGS)

    if not ARGS.logfile is None:
        log = open(ARGS.logfile, 'a')
        log.write('{},'.format(ARGS.net.split(os.path.sep)[-1]))
        if ARGS.property == "acas":
            log.write('{},'.format(acas_properties[ARGS.acas_prop]['name']))
        else:
            log.write('{},{},'.format(ARGS.lrob_input.split(os.path.sep)[-1].split('.')[0], ARGS.lrob_radius))
        log.write('{},{:9.4f}\n'.format(result, runtime))
        log.close()

if __name__ == "__main__":
    main()
