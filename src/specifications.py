import numpy as np
from gurobipy import *

from src.formula import *


class Specification(object):
    def __init__(self, input_shape):
        """
        :args: input_shape: dimensionality of the input
        """
        self.input_shape = input_shape
        self.input_lower_bounds = np.full(input_shape, -GRB.INFINITY)
        self.input_upper_bounds = np.full(input_shape, GRB.INFINITY)

    def isOutRange(self):
        return False

    def isLRob(self):
        return False

    def isMaxLRob(self):
        return False

    def isGenSpec(self):
        return False

    def getInputBounds(self):
        return self.input_lower_bounds, self.input_upper_bounds


class OutRange(Specification):
    def __init__(self, input_bounds,output_bounds):
        """
        :args: input_bounds: np array of pairs (lower,upper) of bounds of
            the input.
        :args: outputbounds: np array of pairs (lower,upper) of bounds of
            the output.
        """
        super(OutRange, self).__init__(input_bounds.shape[0])

        self.input = input_bounds
        self.output = output_bounds

        self.input_lower_bounds = input_bounds[:, 0]
        self.input_upper_bounds = input_bounds[:, 1]

    def isOutRange(self):
        return True


    def getOutputConstrs(self, output_vars, lower_bounds, upper_bounds, delta_vars, max_var):
        constr = []

        for i in itertools.product(*[range(j) for j in output_vars.shape]):
            out = output_vars[i]
            min_val = self.output[i][0]
            max_val = self.output[i][1]
            constr.append(out >= min_val)
            constr.append(out <= max_val)

        return constr


class GenSpec(Specification):
    def __init__(self, input_bounds, output_formula):
        """
        :args: input_formula: formula encoding input constraints.
        :args: output_formula: formula encoding output constraints.
        :args: input_shape: dimensionality of the input.
        """
        super(GenSpec, self).__init__(len(input_bounds[0]))
        self.output_formula = output_formula

        self.input_lower_bounds = input_bounds[0]
        self.input_upper_bounds = input_bounds[1]

    def getOutputConstrs(self, gmodel, output_vars, lower_bounds, upper_bounds, delta_vars, max_var):

        negated_output_formula = NegationFormula(self.output_formula).to_NNF()
        # print(negated_output_formula)

        constrs = self.getConstrs(gmodel, negated_output_formula, output_vars)
        gmodel.update()

        return constrs

    def getConstrs(self, gmodel, formula, vars):
        if isinstance(formula, Constraint):
            # Note: Constraint is a leaf (terminating) node.
            return [self.getAtomicConstr(formula, vars)]

        if isinstance(formula, ConjFormula):
            return self.getConstrs(gmodel, formula.left, vars) + \
                   self.getConstrs(gmodel, formula.right, vars)

        if isinstance(formula, NAryConjFormula):
            constrs = []
            for subformula in formula.clauses:
                constrs += self.getConstrs(gmodel, subformula, vars)

            return constrs

        if isinstance(formula, DisjFormula):

            split_var = gmodel.addVar(vtype=GRB.BINARY)
            clause_vars = [gmodel.addVars(len(vars), lb=-GRB.INFINITY),
                           gmodel.addVars(len(vars), lb=-GRB.INFINITY)]

            constr_sets = [self.getConstrs(gmodel, formula.left, clause_vars[0]),
                           self.getConstrs(gmodel, formula.right, clause_vars[1])]

            constrs = []
            for i in [0, 1]:
                for j in range(len(vars)):
                    constrs.append((split_var == i) >> (vars[j] == clause_vars[i][j]))

                for disj_constr in constr_sets[i]:
                    constrs.append((split_var == i) >> disj_constr)

            return constrs

        if isinstance(formula, NAryDisjFormula):

            clauses = formula.clauses
            split_vars = gmodel.addVars(len(clauses), vtype=GRB.BINARY)
            clause_vars = [gmodel.addVars(len(vars), lb=-GRB.INFINITY) for _ in range(len(clauses))]

            constr_sets = []
            constrs = []
            for i in range(len(clauses)):
                constr_sets.append(self.getConstrs(gmodel, clauses[i], clause_vars[i]))

                for j in range(len(vars)):
                    constrs.append((split_vars[i] == 1) >> (vars[j] == clause_vars[i][j]))

                for disj_constr in constr_sets[i]:
                    constrs.append((split_vars[i] == 1) >> disj_constr)

            # exactly one variable must be true
            constrs.append(quicksum(split_vars) == 1)

            return constrs

        raise Exception("unexpected formula", formula)

    def getAtomicConstr(self, constraint, vars):
        sense = constraint.sense
        if isinstance(constraint, VarVarConstraint):
            op1 = vars[constraint.op1.i]
            op2 = vars[constraint.op2.i]
        elif isinstance(constraint, VarConstConstraint):
            op1 = vars[constraint.op1.i]
            op2 = constraint.op2
        elif isinstance(constraint, LinExprConstraint):
            op1 = 0
            for i, c in constraint.op1.coord_coeff_map.items():
                op1 += c * vars[i]
            op2 = constraint.op2

        else:
            raise Exception("Unexpected type of atomic constraint", constraint)

        if sense == GE:
            return op1 >= op2
        elif sense == LE:
            return op1 <= op2
        elif sense == EQ:
            return op1 == op2
        else:
            raise Exception("Unexpected type of sense", sense)

    def isGenSpec(self):
        return True

    def clone_new_input_bounds(self, new_input_bounds):
        return GenSpec(new_input_bounds, self.output_formula)


class AdvRob(Specification):
    """
    Abstract class for all local robustness specifications.
    Computes the l_infty ball of all perturbations of
    net_input within the radius
    """
    def __init__(self, net_input, radius, valid_lower=-GRB.INFINITY, valid_upper=GRB.INFINITY):
        """
        :args: net_input: the input to the network
        :args: radius: the radius of the perturbation norm ball
        """
        super(AdvRob, self).__init__(net_input.shape[0])
        self.input = net_input
        self.radius = radius

        self.input_lower_bounds = np.maximum(self.input - self.radius, valid_lower)
        self.input_upper_bounds = np.minimum(self.input + self.radius, valid_upper)


class LRob(GenSpec):
    def __init__(self,net_input,label,output_dim,radius, valid_lower=-GRB.INFINITY, valid_upper=GRB.INFINITY):
        """
        :args: net_input: the input to the network
        :args: label: the label of the input
        :args: output_dim: the number of outputs
        :args: radius: the radius of the perturbation norm ball
        """
        self.input = net_input
        self.radius = radius

        self.label = label
        self.output_dim = output_dim

        input_lower_bounds = np.maximum(self.input - self.radius, valid_lower)
        input_upper_bounds = np.minimum(self.input + self.radius, valid_upper)
        output_formula = self.get_output_formula(label, output_dim)
        super(LRob, self).__init__((input_lower_bounds, input_upper_bounds), output_formula)

    @staticmethod
    def get_output_formula(label, output_dim):
        coordinates = [StateCoordinate(i) for i in range(output_dim)]
        atoms = [VarVarConstraint(coordinates[i], LT, coordinates[label]) for i in range(output_dim) if i not in [label]]
        return NAryConjFormula(atoms)

    def isLRob(self):
        return True

    def getOutputConstrs(self, gmodel, output_vars, lower_bounds, upper_bounds, delta_vars, max_var):
        constr = []

        out_max = np.max(upper_bounds)
        d = LinExpr()
        for i in range(self.output_dim):
            if not i == self.label:
                constr.append(max_var >= output_vars[i])
                constr.append(max_var <= output_vars[i] + \
                    (out_max - lower_bounds[i]) * (1 - delta_vars[i]))
                d.addTerms(1, delta_vars[i])
        constr.append(d == 1)
        constr.append(max_var >= output_vars[self.label])

        return constr


class MaxLRob(AdvRob):
    def __init__(self,net_input,label1,label2,radius, valid_lower=-GRB.INFINITY, valid_upper=GRB.INFINITY):
        """
        :args: net_input: the input to the network
        :args: label1: the label of the input
        :args: label2:
        :args: radius: the radius of the perturbation norm ball
        """
        super(MaxLRob, self).__init__(net_input, radius, valid_lower, valid_upper)
        self.label1 = label1
        self.label2 = label2

    def isMaxLRob(self):
        return True


