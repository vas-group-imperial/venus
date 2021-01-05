from src.linearfunctions import compute_lower, compute_upper
import numpy as np
import operator


class Literal:
    def __init__(self, i):
        self.i = i

    def get_i(self):
        return self.i

    def __lt__(self, other):
        return self.i < other.i

    def __repr__(self):
        return self.__str__()


class PosLiteral(Literal):
    def __init__(self, i):
        super(PosLiteral, self).__init__(i)

    def negated(self):
        return NegLiteral(self.i)

    def __eq__(self, other):
        if isinstance(other, PosLiteral):
            return self.i == other.i
        return False

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        return str(self.i)


class NegLiteral(Literal):
    def __init__(self, i):
        super(NegLiteral, self).__init__(i)

    def negated(self):
        return PosLiteral(self.i)

    def __eq__(self, other):
        if isinstance(other, NegLiteral):
            return self.i == other.i
        return False

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        return "~" + str(self.i)


class Implication:
    def __init__(self, from_lit, to_lit):
        self.from_lit = from_lit
        self.to_lit = to_lit

    def __str__(self):
        return self.from_lit.__str__() + " -> " + self.to_lit.__str__()

    def __repr__(self):
        return str(self)


class FirstLayerDependencies:
    def __init__(self, layer, input_bounds, output_bounds,
                 runtime=False, binary_vars=np.empty(0)):
        self.fixed, \
        self.dep_stats, \
        self.dep_per_var = \
            self.analyse_dependencies(layer, input_bounds,
                                      output_bounds, runtime=runtime,
                                      binary_vars=binary_vars)

    def analyse_dependencies(self, layer, input_bounds, output_bounds,
                             runtime=False, binary_vars = np.empty(0)):
        """
        :returns
            fixed: a list of literals that are fixed
            dependencies_stats: a list of pairs (var, m) where var is node number, and
                                                               m the number of dependencies from var
                                sorted in decreasing order by m (the first element has highest m)
            dependencies_per_variable: a dictionary that for each key var has a pair (neg, pos) of lists pos and neg
                                        such that when var is inactive, all literals in neg are implied and
                                                  when var is active, all literals in pos are implied
        """
        fixed = []

        output_lower_bounds = output_bounds['l']
        output_upper_bounds = output_bounds['u']

        matrix = layer.weights
        offset = layer.bias

        pair_of_eq = np.zeros((2,matrix.shape[1]))
        pair_of_bias = np.zeros(2)

        size = matrix.shape[0]

        dependencies_stats = np.zeros(size, dtype=int)
        dependencies_per_variable = [([],[]) for i in range(size)]
        for i in range(size):
            if output_upper_bounds[i] <= 0 or \
                    (runtime and binary_vars[i]==0):
                # fixed.append(NegLiteral(i))
                continue
            if output_lower_bounds[i] >= 0 or \
                    (runtime and binary_vars[i]==1):
                # fixed.append(PosLiteral(i))
                continue

            pair_of_eq[0] = matrix[i]
            pair_of_bias[0] = offset[i]
            for j in range(i+1,size):
                if output_upper_bounds[j] <= 0 or \
                output_lower_bounds[j] >= 0 or \
                (runtime and binary_vars[j] == 0) or \
                (runtime and binary_vars[j] == 1):
                    continue

                pair_of_eq[1] = matrix[j]
                pair_of_bias[1] = offset[j]

                result = self.analyse_pair(pair_of_eq, pair_of_bias, input_bounds)
                dependency_added = 1
                if result[0] == 1:
                    # i -> ~j or j -> ~i
                    dependencies_per_variable[i][1].append(NegLiteral(j))
                    dependencies_per_variable[j][1].append(NegLiteral(i))
                elif result[0] == 2:
                    # ~i -> j or ~j -> i
                    dependencies_per_variable[i][0].append(PosLiteral(j))
                    dependencies_per_variable[j][0].append(PosLiteral(i))
                elif result[0] == 3:
                    # i -> j or ~j -> ~i
                    dependencies_per_variable[i][1].append(PosLiteral(j))
                    dependencies_per_variable[j][0].append(NegLiteral(i))
                elif result[0] == 4:
                    # j -> i or ~i -> ~j
                    dependencies_per_variable[j][1].append(PosLiteral(i))
                    dependencies_per_variable[i][0].append(NegLiteral(j))
                else:
                    dependency_added = 0

                dependencies_stats[i] += dependency_added
                dependencies_stats[j] += dependency_added

        ## we are not using stats now, so don't perform this extra computation
        # dependencies_stats_dict = {}
        # for i in range(size):
        #     if dependencies_stats[i] != 0:
        #         dependencies_stats_dict[i] = dependencies_stats[i]
        #
        #dependencies_stats = sorted(dependencies_stats_dict.items(), key=operator.itemgetter(1), reverse=True)

        dependencies_per_variable_dict = {}
        for i in range(size):
            if dependencies_per_variable[i] != ([],[]):
                dependencies_per_variable_dict[i] = dependencies_per_variable[i]

        return fixed, dependencies_stats, dependencies_per_variable_dict

    def analyse_pair(self, pair_of_eq, pair_of_bias, input_bounds):
        min0, max0 = self.compute_extremums(pair_of_eq, pair_of_bias, input_bounds, 0)
        min1, max1 = self.compute_extremums(pair_of_eq, pair_of_bias, input_bounds, 1)

        if max0 < 0 and max1 < 0:
            return 1, (max0, max1)
        if min0 > 0 and min1 > 0:
            return 2, (min0, min1)
        if max0 < 0 and min1 > 0:
            return 3, (max0, min1)
        if min0 > 0 and max1 < 0:
            return 4, (max1, min0)
        return -1, min0

    def compute_extremums(self, pair_of_eq, pair_of_bias, input_bounds, index):

        nonzero_index = 0
        while pair_of_eq[1-index][nonzero_index] == 0:
            nonzero_index += 1

        new_equation = pair_of_eq[index] - (pair_of_eq[index][nonzero_index]/pair_of_eq[1-index][nonzero_index])*pair_of_eq[1-index]
        new_offset = pair_of_bias[index] - (pair_of_eq[index][nonzero_index]/pair_of_eq[1-index][nonzero_index])*pair_of_bias[1-index]

        weights_plus = np.maximum(new_equation, np.zeros(new_equation.shape))
        weights_minus = np.minimum(new_equation, np.zeros(new_equation.shape))

        min = compute_lower(weights_minus, weights_plus, input_bounds['l'], input_bounds['u']) + new_offset
        max = compute_upper(weights_minus, weights_plus, input_bounds['l'], input_bounds['u']) + new_offset

        return min, max

    @staticmethod
    def complete_fixed_variables(fixed, dependencies):
        fixed_set = set(fixed)

        to_explore = fixed
        while True:
            new = set()
            for lit in to_explore:
                var = lit.get_i()
                if var in dependencies.dep_per_var:
                    if isinstance(lit, PosLiteral):
                        new.update(dependencies.dep_per_var[var][1])
                    elif isinstance(lit, NegLiteral) and var in dependencies.dep_per_var:
                        new.update(dependencies.dep_per_var[var][0])
                    else:
                        raise Exception("Unexpected literal", lit)

            if new.issubset(fixed_set):
                break

            to_explore = new.difference(fixed_set)
            fixed_set.update(new)

        return fixed_set

    @staticmethod
    def compute_A_and_B(dependencies):
        """
        This method computes two sets A and B of literals (interpreted as conjunctions)
        such that ~A implies B (and equivalently, ~B implies A)
        """
        # dependencies_stats is sorted with the variable with maximum number of dependencies being the first one
        var_with_max_n_dep = dependencies.dep_stats[0][0]
        # choose the dependencies from the positive or from the negative value
        max_i = 1 if len(dependencies.dep_per_var[var_with_max_n_dep][1]) > \
                     len(dependencies.dep_per_var[var_with_max_n_dep][0]) else 0
        B = set(dependencies.dep_per_var[var_with_max_n_dep][max_i])

        A_indices = {var_with_max_n_dep}
        notA = {PosLiteral(var_with_max_n_dep)} if max_i == 1 else {NegLiteral(var_with_max_n_dep)}

        while True:

            print("~A", sorted(notA))
            print("B", sorted(B))
            print("")

            max_intersection_size = 0
            max_intersection = {}
            index_of_max = -1
            literal_of_max = None
            for j in range(len(dependencies.dep_stats)):
                # stop when we know that we cannot improve (dependencies_stats is sorted)
                if dependencies.dep_stats[j][1] <= max_intersection_size:
                    break

                if dependencies.dep_stats[j][0] not in A_indices:
                    var = dependencies.dep_stats[j][0]
                    max_i = 1 if len(dependencies.dep_per_var[var][1]) > len(dependencies.dep_per_var[var][0]) else 0
                    intersection = set(B).intersection(set(dependencies.dep_per_var[var][max_i]))

                    if len(intersection) > max_intersection_size:
                        max_intersection_size = len(intersection)
                        max_intersection = intersection
                        index_of_max = var
                        literal_of_max = PosLiteral(var) if max_i == 1 else NegLiteral(var)

            # stop when failed to find a new intersection
            if max_intersection_size == 0:
                break

            # if adding the new literal makes the proportion of the sizes of A and B further from 1
            # we stop
            if abs(1 - len(A_indices) / len(B)) < abs(1 - (len(A_indices) + 1) / max_intersection_size):
                break

            B = max_intersection
            A_indices.add(index_of_max)
            notA.add(literal_of_max)

        A = {lit.negated() for lit in notA}

        return A, B


