import datetime
import random
import time
from multiprocessing import Process

import numpy as np

from src.formula import Constraint, VarVarConstraint, LT, VarConstConstraint, GT, ConjFormula, NAryConjFormula, \
    DisjFormula, NAryDisjFormula
from src.model import VModel


class SplittingProcess(Process):
    MAX_SPLIT_DEPTH = 1000
    MAX_N_SPLITS = 1048576#65536

    TOTAL_JOBS_NUMBER_STRING = "Total_jobs_n"

    def __init__(self, id, lmodel, spec, splitting_params, encoder_params,
                 jobs_queue, reporting_queue, print=False):
        super(SplittingProcess, self).__init__()

        self.id = id

        """
        lmodel is our internal representation of a neural network model.
        It is independent of the format of the network and also is pickle-able
        and multiprocessing-friendly (unlike, e.g., keras models).
        """
        self.lmodel = lmodel

        self.spec = spec
        self.params = splitting_params
        self.encoder_params = encoder_params

        # the queue to which jobs will be added
        """
        Job descriptions are vmodels themselves without gmodel
        so that it could be serialised and shared through a queue
        """
        self.jobs_queue = jobs_queue

        # the queue to communicate with the main process
        # splitter with report to it the total number of splits once they all have been computed
        self.reporting_queue = reporting_queue

        self.PRINT_TO_CONSOLE = print
        self.DEBUG_MODE = False

    def get_initial_splits(self):
        # Compute the initial fixed ratio
        vmodel, initial_fixed_ratio = self.get_fixed_nodes_percentage(self.lmodel, self.spec)

        if self.PRINT_TO_CONSOLE:
            print("Initial fixed ratio", initial_fixed_ratio)

        regions_to_split = [self.spec.getInputBounds()]

        split_depth = 0
        while split_depth < self.params.STARTING_DEPTH_PARALLEL_SPLITTING_PROCESSES:
            split_subregions = []

            for i in range(len(regions_to_split)):
                subproblems = self.get_best_split_middle(self.lmodel, self.spec, regions_to_split[i])
                split_subregions.extend([subproblems[0][1], subproblems[1][1]])

            split_depth += 1
            regions_to_split = split_subregions

        return [(self.lmodel, self.spec.clone_new_input_bounds(region)) for region in regions_to_split]

    def run(self):

        if self.PRINT_TO_CONSOLE:
            print("running splitting process", self.id)

        self.compute_splits_depth_first_stop_on_the_fly(self.lmodel)

        if self.PRINT_TO_CONSOLE:
            print("Splitting process", self.id, "finished")

    def compute_splits_depth_first_stop_on_the_fly(self, lmodel):
        vmodel, initial_fixed_ratio = self.get_fixed_nodes_percentage(lmodel, self.spec)
        # if self.PRINT_TO_CONSOLE:
        #     print("Initial fixed ratio", initial_fixed_ratio)

        # initialise the regions to be split
        sent_jobs_count = 0
        regions_to_split = [(0, initial_fixed_ratio, self.spec.getInputBounds(), vmodel)]

        # treat regions_to_split as a stack
        while len(regions_to_split) > 0:
            # to ease the load on the processor, if there are
            # already many jobs in the queue,
            # the splitting process can rest
            if self.jobs_queue.qsize() < self.params.LARGE_N_OF_UNPROCESSED_JOBS:

                instance = regions_to_split.pop()

                worth, subproblems = self.is_worth_splitting(instance, lmodel, initial_fixed_ratio)
                if worth:
                    """
                    worth splitting
                    i.e., continue splitting as it creates substantially simpler subproblems
                    """
                    regions_to_split.extend(subproblems)
                else:
                    """
                    not worth splitting
                    i.e., either splitting does not create substantially simpler subproblems,
                    or the problem is already simple enough 
                    """
                    if self.PRINT_TO_CONSOLE:
                        print("Added job", "{}/{}".format(self.id, sent_jobs_count+1), round(instance[1], 4), datetime.datetime.now())
                    self.jobs_queue.put(("{}/{}".format(self.id, sent_jobs_count+1), instance[-1]))
                    sent_jobs_count += 1

            else:
                time.sleep(self.params.SLEEPING_INTERVAL)

        # communicate to the main process the total number of splits
        # so that it would know when to stop
        self.reporting_queue.put((sent_jobs_count, self.TOTAL_JOBS_NUMBER_STRING, self.id, None))

    def count_splits_on_the_fly(self):

        # Compute the initial fixed ratio
        vmodel, initial_fixed_ratio = self.get_fixed_nodes_percentage(self.lmodel, self.spec)

        if self.PRINT_TO_CONSOLE:
            print("Initial fixed ratio", initial_fixed_ratio)

        # initialise the regions to be split
        sent_jobs_count = 0
        regions_to_split = [(0, initial_fixed_ratio, self.spec.getInputBounds(), vmodel)]

        # treat regions_to_split as a stack
        while len(regions_to_split) > 0:
            if self.PRINT_TO_CONSOLE:
                if sent_jobs_count % 1000 == 0 and sent_jobs_count > 0:
                    print(datetime.datetime.now())

            instance = regions_to_split.pop()

            worth, subproblems = self.is_worth_splitting(instance, self.lmodel, initial_fixed_ratio)
            if worth:
                """
                worth splitting
                i.e., continue splitting as it creates substantially simpler subproblems
                """
                regions_to_split.extend(subproblems)
            else:
                """
                not worth splitting
                i.e., either splitting does not create substantially simpler subproblems,
                or the problem is already simple enough 
                """

                if self.PRINT_TO_CONSOLE:
                    print("Added job", sent_jobs_count, round(instance[1], 4))

                sent_jobs_count += 1

        print("Total number of jobs", sent_jobs_count)
        return sent_jobs_count

    def get_fixed_nodes_percentage(self, lmodel, spec):
        vmodel = VModel(lmodel, spec, self.encoder_params)
        vmodel.initialise_layers()

        return vmodel, vmodel.get_n_fixed_nodes() / vmodel.get_n_relu_nodes()

    def is_worth_splitting(self, problem, lmodel, initial_fixed_ratio):

        split_depth, fixed_ratio, region, vmodel = problem

        # if the instance already satisfies the spec,
        # we can discard it. That's why we return True and an empty set of subproblems
        if self.instance_satisfies_spec(vmodel):
            if self.PRINT_TO_CONSOLE:
                print(self.id, "Instance", round(fixed_ratio, 4), "discarded as already satisfies specification")

            return True, []

        # if the fixed ratio is already very high, we stop splitting
        if fixed_ratio < self.params.FIXED_RATIO_CUTOFF:

            # compute a split
            subproblems = self.get_best_split_middle(lmodel, self.spec, region)
            subproblems[0] = (split_depth+1,) + subproblems[0]
            subproblems[1] = (split_depth+1,) + subproblems[1]

            # compute the scores of all 3 problems
            score = self.estimate_problem_score(problem, initial_fixed_ratio)
            score1 = self.estimate_problem_score(subproblems[0], initial_fixed_ratio)
            score2 = self.estimate_problem_score(subproblems[1], initial_fixed_ratio)

            if self.PRINT_TO_CONSOLE and self.DEBUG_MODE:
                print("\t\t", split_depth,
                      "{:.4f}".format(initial_fixed_ratio), "{:.4f}".format(problem[1]),
                      "{:.4f}".format(subproblems[0][1]), "{:.4f}".format(subproblems[1][1]),
                      "{:.4f}".format(score), "{:.4f}".format(score1), "{:.4f}".format(score2),
                      datetime.datetime.now())

            max_s_score = max(score1, score2)
            min_s_score = min(score1, score2)

            # score is maximal, so we stop splitting
            if score > max_s_score:
                return False, None

            # score is minimal, so we continue splitting
            if min_s_score > score:
                return True, subproblems

            # min_s_score < score < max_s_score
            # otherwise we check that the average of score1 and score2 is better
            # than the score
            if (score1 + score2)/2 > score:
                return True, subproblems

        return False, None

    def get_best_split_middle(self, lmodel, spec, input_bounds):
        best_ratio_dim = -1
        best_ratio = 0
        best_region1_info = None
        best_region2_info = None

        input_shape = len(input_bounds[0])

        if input_shape < self.params.SMALL_N_INPUT_DIMENSIONS:
            """
            If the number of input dimensions is not big,
            we choose the best dimension to split
            """
            for i in range(input_shape):
                ratio, region1_info, region2_info = \
                    self.compute_fixed_ratio_by_split(lmodel, spec, input_bounds, i,
                                                      (input_bounds[0][i] + input_bounds[1][i]) / 2)
                if (ratio > best_ratio):
                    best_ratio = ratio
                    best_ratio_dim = i
                    best_region1_info = region1_info
                    best_region2_info = region2_info
        else:
            """
            Otherwise we split randomly as 
            the overhead from the above loop will be too much
            """
            dim = random.randint(0, input_shape - 1)
            _, best_region1_info, best_region2_info = \
                self.compute_fixed_ratio_by_split(lmodel, spec, input_bounds, dim,
                                                  (input_bounds[0][dim] + input_bounds[1][dim]) / 2)
            best_ratio = (best_region1_info[0] + best_region2_info[0])/2
            best_ratio_dim = dim

        return [best_region1_info, best_region2_info]

    def compute_fixed_ratio_by_split(self, lmodel, spec, input_bounds, dim, where):
        input_lower, input_upper = input_bounds

        lower1 = np.array(input_lower)
        lower1[dim] = where
        region1 = (lower1, input_upper)

        vmodel1, fixed_ratio1 = self.get_fixed_nodes_percentage(lmodel, spec.clone_new_input_bounds(region1))

        upper2 = np.array(input_upper)
        upper2[dim] = where
        region2 = (input_lower, upper2)

        vmodel2, fixed_ratio2 = self.get_fixed_nodes_percentage(lmodel, spec.clone_new_input_bounds(region2))

        return SplittingProcess.get_ratio_score(fixed_ratio1, fixed_ratio2), \
               (fixed_ratio1, region1, vmodel1), \
               (fixed_ratio2, region2, vmodel2)

    @staticmethod
    def get_ratio_score(fixed_ratio1, fixed_ratio2):
        return (fixed_ratio1 + fixed_ratio2) / 2

    def estimate_problem_score(self, problem, initial_fixed_ratio):
        return (problem[1] - initial_fixed_ratio) / ((problem[0]+1) ** (1/self.params.DEPTH_POWER))

    def instance_satisfies_spec(self, vmodel):
        output_formula = vmodel.spec.output_formula

        output_bounds = vmodel.lmodel.layers[-1].bounds['out']
        lower_bounds = output_bounds['l']
        upper_bounds = output_bounds['u']

        return self.check_formula_satisfaction(output_formula, lower_bounds, upper_bounds)

    def check_formula_satisfaction(self, formula, lower_bounds, upper_bounds):
        if isinstance(formula, Constraint):
            sense = formula.sense
            if sense == LT:
                if isinstance(formula, VarVarConstraint):
                    return upper_bounds[formula.op1.i] < lower_bounds[formula.op2.i]
                if isinstance(formula, VarConstConstraint):
                    return upper_bounds[formula.op1.i] < formula.op2
            elif sense == GT:
                if isinstance(formula, VarVarConstraint):
                    return lower_bounds[formula.op1.i] > upper_bounds[formula.op2.i]
                if isinstance(formula, VarConstConstraint):
                    return lower_bounds[formula.op1.i] > formula.op2

        if isinstance(formula, ConjFormula):
            return self.check_formula_satisfaction(formula.left, lower_bounds, upper_bounds) and \
                   self.check_formula_satisfaction(formula.right, lower_bounds, upper_bounds)

        if isinstance(formula, NAryConjFormula):
            for clause in formula.clauses:
                if not self.check_formula_satisfaction(clause, lower_bounds, upper_bounds):
                    return False
            return True

        if isinstance(formula, DisjFormula):
            return self.check_formula_satisfaction(formula.left, lower_bounds, upper_bounds) or \
                   self.check_formula_satisfaction(formula.right, lower_bounds, upper_bounds)

        if isinstance(formula, NAryDisjFormula):
            for clause in formula.clauses:
                if self.check_formula_satisfaction(clause, lower_bounds, upper_bounds):
                    return True
            return False

