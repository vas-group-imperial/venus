from gurobipy import *
from src.linearfunctions import *
from src.parameters import *
from src.fldependencies import NegLiteral, FirstLayerDependencies, PosLiteral


class Layer(object):
    def __init__(self, output_shape, weights, bias, depth):
        self.vars = {'out': [], 'delta': []}
        self.bounds = {'in': {'l': [], 'u': []},
                       'out': {'l': [], 'u': []}}
        self.rnt_bounds = {'in': {'l': [], 'u': []},
                           'out': {'l': [], 'u': []}}
        self.rnt_bound_equations = {'in': {'l': None, 'u': None},
                                    'out': {'l': None, 'u': None}}
        self.bound_equations = {'in': {'l': None, 'u': None},
                                'out': {'l': None, 'u': None}}
        self.output_shape = output_shape 
        self.weights = weights
        self.bias = bias
        self.depth = depth

    def add_vars(self, gmodel, linear_approximation=False):
        self.vars['out'] = np.empty(shape=self.output_shape, dtype=Var)
        for i in range(self.output_shape):
            self.vars['out'][i] = \
                gmodel.addVar(lb=self.bounds['out']['l'][i], ub=self.bounds['out']['u'][i])
            # if linear_approximation:
                # self.vars['out'][i] = \
                # gmodel.addVar(lb=0, ub=self.bounds['out']['u'][i])
            # else:
                # self.vars['out'][i] = \
                # gmodel.addVar(lb=self.bounds['out']['l'][i], ub=self.bounds['out']['u'][i])

    def clean_vars(self):
        self.vars = {'out': [], 'delta': []}

    def getPrimeLBound(self, node, p_layer, p_node):
        lbound = p_layer.bounds['out']['l'][p_node]
        if self.weights[node][p_node] < 0:
            lbound = p_layer.bounds['out']['u'][p_node]
        return lbound

    def getPrimeUBound(self, node, p_layer, p_node):
        ubound = p_layer.bounds['out']['u'][p_node]
        if self.weights[node][p_node] < 0:
            ubound = p_layer.bounds['out']['l'][p_node]
        return ubound

    def getActiveWeights(self, node):
        ind = np.array(range(self.weights.shape[1]))
        w = ind[(self.weights[node] != 0)]
        w = list(w)
        return w


class Input(Layer):
    def __init__(self, spec):
        self.vars = {'out': []}
        self.bounds = {'out': {'l': [], 'u': []}}
        self.bound_equations = {'in': {'l': None, 'u': None},
                                'out': {'l': None, 'u': None}}
        self.rnt_bound_equations = {'in': {'l': None, 'u': None},
                                    'out': {'l': None, 'u': None}}
        self.spec = spec
        self.depth = 0
        self.output_shape = spec.input_shape

    def add_vars(self, gmodel):
        self.vars['out'] = np.empty(shape=self.output_shape, dtype=Var)
        lowerBounds, upperBounds = self.spec.getInputBounds()
        for i in range(self.output_shape):
            self.vars['out'][i] = gmodel.addVar(lb=lowerBounds[i],
                                                ub=upperBounds[i])

    def clean_vars(self):
        self.vars = {'out': []}

    def compute_bounds(self):
        l, u = self.spec.getInputBounds()
        self.bounds['out']['l'] = l
        self.bounds['out']['u'] = u
        self.error = {'out': [0 for i in range(self.output_shape)]}
        size = self.spec.input_shape
        self.bound_equations['out']['l'] = LinearFunctions(np.identity(size), np.zeros(size))
        self.bound_equations['out']['u'] = LinearFunctions(np.identity(size), np.zeros(size))
        self.rnt_bounds = self.bounds
        self.rnt_bound_equations = self.bound_equations

    def add_constrs(self, gmodel):
        pass

    def clone(self):
        return Input(self.spec)


class Output(Layer):
    def __init__(self, spec, depth):
        """
        :args input_shape: size of the input.
        :args gmodel: a gurobi model
        """
        self.vars = {'delta': {}}
        self.max_var = None
        self.spec = spec
        self.depth = depth

    def add_vars(self, gmodel):
        if self.spec.isLRob():
            for i in range(self.spec.output_dim):
                if not i == self.spec.label:
                    self.vars['delta'][i] = gmodel.addVar(vtype=GRB.BINARY)
                    self.vars['delta'][i].setAttr(GRB.Attr.BranchPriority, self.depth)
            self.max_var = gmodel.addVar(lb=-GRB.INFINITY)

    def clean_vars(self):
        self.vars = {'delta': {}}

    def add_constrs(self, p_layer, gmodel):
        # out_max = np.max(p_layer.bounds['out']['u'])
        # out_min = np.min(p_layer.bounds['out']['l'])
        # d = LinExpr()
        # for i in range(self.spec.labels):
            # if not i == self.spec.label:
                # gmodel.addConstr(self.max_var >= p_layer.vars['out'][i])
                # gmodel.addConstr(self.max_var <= p_layer.vars['out'][i] + \
                    # (out_max-out_min) * (1 - self.vars['delta'][i]))
                # d.addTerms(1, self.vars['delta'][i])
        # gmodel.addConstr(d == 1)
        # gmodel.addConstr(self.max_var >= p_layer.vars['out'][self.spec.label])

        constrs = self.spec.getOutputConstrs(gmodel,
                                    p_layer.vars['out'],
                                    p_layer.bounds['out']['l'],
                                    p_layer.bounds['out']['u'],
                                    self.vars['delta'],
                                    self.max_var)
        for constr in constrs:
            gmodel.addConstr(constr)

    def clone(self):
        return Output(self.spec, self.depth)


class Relu(Layer):

    def __init__(self, output_shape, weights, bias, depth):
        super().__init__(output_shape, weights, bias, depth)
        self.layer_deps = None
        self.xlayer_deps = [
            [] for i in range(self.output_shape)
        ]
        self.group_deps = {}
        self.error = {'in': [], 'out': []}

    def add_vars(self, p_layer, gmodel, linear_approximation=False):
        super().add_vars(gmodel, linear_approximation)
        if not linear_approximation:
            self.vars['delta'] = np.empty(shape=self.output_shape, dtype=Var)
            for i in range(self.output_shape):
                self.vars['delta'][i] = gmodel.addVar(vtype=GRB.BINARY)
                self.vars['delta'][i].setAttr(GRB.Attr.BranchPriority, self.depth)


    def is_active(self, node):
        if self.bounds['in']['l'][node] >= 0:
            return True
        else:
            return False

    def is_inactive(self, node):
        if self.bounds['in']['u'][node] <= 0:
            return True
        else:
            return False

    def is_fixed(self, node):
        if self.bounds['in']['l'][node] < 0 and \
                self.bounds['in']['u'][node] > 0:
            return False
        else:
            return True

    def get_active(self):
        act = []
        for i in range(self.output_shape):
            if self.is_active(i):
                act.append(i)

        return act

    def get_inactive(self):
        inact = []
        for i in range(self.output_shape):
            if self.is_inactive(i):
                inact.append(i)

        return inact

    def get_not_fixed(self, binary_vars=np.empty(0)):
        """
        :return nodes that are not fixed to either the active or the
        inactive state.
        """
        nf = []
        if len(binary_vars) == 0:
            for i in range(self.output_shape):
                if not self.is_fixed(i):
                    nf.append(i)
        else:
            for i in range(self.output_shape):
                if not self.is_fixed(i) and not binary_vars[i] == 0 and not \
                        binary_vars[i] == 1:
                    nf.append(i)
        return nf

    def get_fixed(self, binary_vars=None):
        """
        :return nodes that are fixed to either the active or the
        inactive state.
        """
        fx = []
        if binary_vars == None:
            for i in range(self.output_shape):
                if self.is_fixed(i):
                    fx.append(i)
        else:
            for i in range(self.output_shape):
                if self.is_fixed(i) and not binary_vars[i] == 0 and not \
                        binary_vars[i] == 1:
                    fx.append(i)
        return fx

    def get_total_dep_count(self):
        dep_count = 0
        if self.layer_deps is not None:
            dep_count = sum([n for n in self.layer_deps.dep_stats])

        for var_deps in self.xlayer_deps:
            dep_count += len(var_deps)

        for group_ds in self.group_deps:
            dep_count += len(group_ds)

        return dep_count

    """
    Bounds 
    """

    def compute_bounds(self, method, p_layer, runtime=False,
                       binary_vars=[], input_bounds=[], const=None):
        if method == Bounds.INT_ARITHMETIC:
            if runtime:
                if len(binary_vars)==0:
                    raise Exception("""Missing binary_vars parameter for runtime
                    calculation of bounds""")
                else:
                    self._compute_runtime_bounds_ia(p_layer, binary_vars)
            else:
                self._compute_bounds_ia(p_layer)
        elif method == Bounds.SYMBOLIC_INT_ARITHMETIC:
            if len(input_bounds)==0:
                raise Exception("""Missing input-bounds parameter for
                                symbolic-based calculation of
                                bounds""")
            else:
                if runtime:
                    if len(binary_vars)==0:
                        raise Exception("""Missing binary_vars parameter for
                        runtime calculation of bounds""")
                    else:
                        self._compute_runtime_bounds_sia(p_layer, input_bounds, binary_vars)
                else:
                    self._compute_bounds_sia(p_layer, input_bounds)
        elif method == Bounds.CONST:
            if not isinstance(const, float):
                raise Exception("""Missing float value for constant
                bounds""")
            else:
                self._compute_bounds_const(const)

        self.error['in'] = self.weights.dot(p_layer.error['out'])
        self.error['out'] = np.empty(self.output_shape)


        for i in range(self.output_shape):
            ub = self.bounds['in']['u'][i]
            lb = self.bounds['in']['l'][i]

            if self.is_fixed(i):
                self.error['out'][i] =  0
            else:
                act_area = (pow(ub,2)/2) - (pow(ub,3)/(2*(ub-lb)))
                inact_area = (pow(lb,2) * ub) / (2*(ub-lb))
                area1 = act_area + inact_area
                area2 = pow(ub,2)/2
                whole_area =  (ub*(ub-lb))-(lb*ub)

                if area2 < area1:
                    self.error['out'][i] = area2 / whole_area
                else:
                    self.error['out'][i] = area1 / whole_area



    """ 
    Bounds  - Interval Arithmetic - Pre-computation
    """ 


    def _compute_bounds_ia(self, p_layer):
        self.bounds['in'] = self._compute_in_bounds_ia( \
            p_layer.bounds['out']['l'], \
            p_layer.bounds['out']['u'])
        self.bounds['out'] = self._compute_out_bounds_ia(self.bounds['in'])


    def _compute_in_bounds_ia(self, p_low, p_up):
        weights_plus = np.maximum(self.weights, np.zeros(self.weights.shape))
        weights_minus = np.minimum(self.weights, np.zeros(self.weights.shape))

        lower = np.array([
            weights_plus[i].dot(p_low) + \
            weights_minus[i].dot(p_up) + \
            self.bias[i] for i in range(self.output_shape)
        ])

        upper = np.array([
            weights_plus[i].dot(p_up) + \
            weights_minus[i].dot(p_low) + \
            self.bias[i] for i in range(self.output_shape)
        ])


        return {'l': lower, 'u': upper}

    def _compute_out_bounds_ia(self, in_bounds):
        lower = np.maximum(in_bounds['l'], np.zeros(self.output_shape))
        upper = np.maximum(in_bounds['u'], np.zeros(self.output_shape))

        return {'l': lower, 'u': upper}

    """
    Bounds  - Interval Arithmetic - Runtime computation
    """

    def _compute_runtime_bounds_ia(self, p_layer, binary_vars):
        self.rnt_bounds['in'] = self._compute_runtime_in_bounds_ia( \
            p_layer.rnt_bounds['out']['l'], \
            p_layer.rnt_bounds['out']['u'])
        self.rnt_bounds['out'] = self._compute_runtime_out_bounds_ia( \
            self.rnt_bounds['in'], \
            binary_vars)

    def _compute_runtime_in_bounds_ia(self,p_low, p_up):
        return self._compute_in_bounds_ia(p_low, p_up)

    def _compute_runtime_out_bounds_ia(self,in_bounds, binary_vars):
        bounds = self._compute_out_bounds_ia(in_bounds)
        bounds['l'][(binary_vars == 0)] = 0
        bounds['u'][(binary_vars == 0)] = 0

        return bounds


    """
    Bounds - Symbolic Interval Arithmetic - Pre-computation
    """

    def _compute_bounds_sia(self, p_layer, input_bounds):
        # set input bounds equations
        self.bound_equations['in'] = self._compute_in_bound_eqs( \
            p_layer.bound_equations['out']['l'],
            p_layer.bound_equations['out']['u'])

        # set concrete input bounds
        self.bounds['in']['l'] = \
            self.bound_equations['in']['l'].compute_min_values(input_bounds)
        self.bounds['in']['u'] = \
            self.bound_equations['in']['u'].compute_max_values(input_bounds)

        # set output bounds equatuons
        self.bound_equations['out'] = self._compute_out_bound_eqs( \
            self.bound_equations['in'], \
            input_bounds)

        # set concrete output bounds
        self.bounds['out']['l'] = \
            self.bound_equations['out']['l'].compute_min_values(input_bounds)
        self.bounds['out']['u'] = \
            self.bound_equations['out']['u'].compute_max_values(input_bounds)
        # make sure the bounds are not below zero
        self.bounds['out']['l'] = np.maximum(self.bounds['out']['l'], \
                                             np.zeros(self.output_shape))
        self.bounds['out']['u'] = np.maximum(self.bounds['out']['u'], \
                                             np.zeros(self.output_shape))

    def _compute_in_bound_eqs(self, p_low_eq, p_up_eq):
        weights_plus = np.maximum(self.weights, np.zeros(self.weights.shape))
        weights_minus = np.minimum(self.weights, np.zeros(self.weights.shape))

        # get coefficients for the input bound equations
        p_l_coeffs = p_low_eq.matrix
        p_u_coeffs = p_up_eq.matrix
        l_coeffs = weights_plus.dot(p_l_coeffs) + weights_minus.dot(p_u_coeffs)
        u_coeffs = weights_plus.dot(p_u_coeffs) + weights_minus.dot(p_l_coeffs)

        # get constants for the input bound equations
        p_l_const = p_low_eq.offset
        p_u_const = p_up_eq.offset
        l_const = weights_plus.dot(p_l_const) + weights_minus.dot(p_u_const) + self.bias
        u_const = weights_plus.dot(p_u_const) + weights_minus.dot(p_l_const) + self.bias

        # return input bound equations
        return {'l': LinearFunctions(l_coeffs, l_const),
                'u': LinearFunctions(u_coeffs, u_const)}

    def _compute_out_bound_eqs(self, in_eqs, input_bounds):
        # return out bound equations
        return {'l': in_eqs['l'].getLowerReluRelax(input_bounds),
                'u': in_eqs['u'].getUpperReluRelax(input_bounds)}


    """
    Bounds - Symbolic Interval Arithmetic - Runtime computation
    """


    def _compute_runtime_bounds_sia(self, p_layer, input_bounds, binary_vars):
        # set input runtime bounds equations
        self.rnt_bound_equations['in'] = self._compute_runtime_in_bound_eqs( \
            p_layer.rnt_bound_equations['out']['l'],
            p_layer.rnt_bound_equations['out']['u'])
        # set concrete input runtime bounds
        self.rnt_bounds['in']['l'] = \
            self.rnt_bound_equations['in']['l'].compute_min_values(input_bounds)
        self.rnt_bounds['in']['u'] = \
            self.rnt_bound_equations['in']['u'].compute_max_values(input_bounds)

        # set output bounds equations
        self.rnt_bound_equations['out'] = self._compute_runtime_out_bound_eqs( \
            self.rnt_bound_equations['in'],
            input_bounds,
            binary_vars)
        # set concrete output bounds
        self.rnt_bounds['out']['l'] = \
            self.rnt_bound_equations['out']['l'].compute_min_values(input_bounds)
        self.rnt_bounds['out']['u'] = \
            self.rnt_bound_equations['out']['u'].compute_max_values(input_bounds)
        # make sure the bounds are not below zero
        self.rnt_bounds['out']['l'] = np.maximum(self.rnt_bounds['out']['l'], \
                                                 np.zeros(self.output_shape))
        self.rnt_bounds['out']['u'] = np.maximum(self.rnt_bounds['out']['u'], \
                                                 np.zeros(self.output_shape))

    def _compute_runtime_in_bound_eqs(self, p_low_eq, p_up_eq):
        return self._compute_in_bound_eqs(p_low_eq, p_up_eq)

    def _compute_runtime_out_bound_eqs(self, in_eqs, input_bounds, binary_vars):
        # set out bound equations
        eqs = self._compute_out_bound_eqs(in_eqs, input_bounds)

        # set bound functions to zero for inactive nodes
        eqs['l'].matrix[(binary_vars == 0), :] = 0
        eqs['l'].offset[(binary_vars == 0)] = 0
        eqs['u'].matrix[(binary_vars == 0), :] = 0
        eqs['u'].offset[(binary_vars == 0)] = 0
        # set bound functions to input ones for active nodes
        eqs['l'].matrix[(binary_vars == 1), :] = in_eqs['l'].matrix[(binary_vars == 1), :]
        eqs['l'].offset[(binary_vars == 1)] = in_eqs['l'].offset[(binary_vars == 1)]
        eqs['u'].matrix[(binary_vars == 1), :] = in_eqs['u'].matrix[(binary_vars == 1), :]
        eqs['u'].offset[(binary_vars == 1)] = in_eqs['u'].offset[(binary_vars == 1)]

        return eqs


    """ 
    Bounds - Const
    """

    def _compute_bounds_const(self, const):
        self.bounds['in']['l'] = np.ones(self.output_shape) * const * -1
        self.bounds['in']['u'] = np.ones(self.output_shape) * const
        self.bounds['out']['l'] = np.zeros(self.output_shape)
        self.bounds['out']['u'] = np.ones(self.output_shape) * const 

    """ 
    Dependencies between pairs of nodes
    """

    def compute_xlayer_deps(self, next_layer, runtime=False,
                     binary_vars=np.empty(0),
                     next_binary_vars=np.empty(0)):
        if runtime==True:
            if len(binary_vars) == 0:
                raise Exception("""Missing the layer's binary variables
                for the runtime computation of group dependencies""")
            if len(next_binary_vars) == 0:
                raise Exception("""Missing the next layer's binary
                variables for the runtime computation of group
                               dependencies""")
            self._compute_runtime_xlayer_deps(next_layer, binary_vars, \
                                             next_binary_vars)
        else:
            self._compute_xlayer_deps(next_layer)


    def _compute_xlayer_deps(self, next_layer):
        nf = self.get_not_fixed()
        nfp = next_layer.get_not_fixed()

        self.xlayer_deps = [
            [] for i in range(self.output_shape)
        ]

        for i in nf:
            for j in nfp:
                if next_layer.weights[j][i] >= 0:
                    """
                    inactive -> inactive dependency
                    """
                    ubp = next_layer.bounds['in']['u'][j] - \
                          next_layer.weights[j][i] * \
                          self.bounds['in']['u'][i]
                    if ubp <= 0:
                        self.xlayer_deps[i].append((j, DepType.I_I))
                else:
                    """
                    inactive -> active dependency
                    """
                    lbp = next_layer.bounds['in']['l'][j] - \
                          next_layer.weights[j][i] * \
                          self.bounds['in']['u'][i]
                    if lbp >= 0:
                        self.xlayer_deps[i].append((j, DepType.I_A))

    def _compute_runtime_xlayer_deps(self, next_layer, binary_vars, next_binary_vars):
        nf = self.get_not_fixed(binary_vars=binary_vars)
        nfp = next_layer.get_not_fixed(binary_vars=next_binary_vars)


        self.xlayer_deps = [
            [] for i in range(self.output_shape)
        ]

        for i in nf:
            for j in nfp:
                if next_layer.weights[j][i] >= 0:
                    """
                    inactive -> inactive dependency
                    """
                    ubp = next_layer.rnt_bounds['in']['u'][j] - \
                          next_layer.weights[j][i] * \
                          self.rnt_bounds['in']['u'][i]
                    if ubp <= 0:
                        self.xlayer_deps[i].append((j, DepType.I_I))
                else:
                    """
                    inactive -> active dependency
                    """
                    lbp = next_layer.rnt_bounds['in']['l'][j] - \
                          next_layer.weights[j][i] * \
                          self.rnt_bounds['in']['u'][i]
                    if lbp >= 0:
                        self.xlayer_deps[i].append((j, DepType.I_A))

    def compute_f_deps_cl(self, next_layer):
        """
        Compute forward chains of dependencies      
        """
        self.deps_f_cl = {
            i: set() for i in range(self.output_shape)
        }

        nf = self.get_not_fixed()
        nfp = next_layer.get_not_fixed()

        for i in nf:
            for j in nfp:
                if (Deps.I_I, j) in self.xlayer_deps[i]:
                    self.deps_f_cl[i] = self.deps_f_cl[i] | next_layer.deps_f_cl[j]
                    self.deps_f_cl[i].add((next_layer.depth, j))
                if (Deps.I_A, j) in self.xlayer_deps[i]:
                    self.deps_f_cl[i].add((next_layer.depth, j))

    def compute_b_deps_cl(self, p_layer):
        """
        Compute backward chains of dependencies      
        """

        deps_b_cl_aa = {
            i: set() for i in range(self.output_shape)
        }
        deps_b_cl_ia = {
            i: set() for i in range(self.output_shape)
        }
        self.deps_b_cl = {'aa': deps_b_cl_aa, 'ia': deps_b_cl_ia}

        nf = self.get_not_fixed()
        nfp = p_layer.get_not_fixed()

        for i in nf:
            for j in nfp:
                if (Deps.I_I, i) in p_layer.xlayer_deps[j]:
                    self.deps_b_cl[Deps.A_A][i] = \
                        self.deps_b_cl[Deps.A_A][i] | \
                        p_layer.deps_b_cl[Deps.A_A][j]
                    self.deps_b_cl[Deps.A_A][i].add((p_layer.depth, j))
                if (Deps.I_A, i) in p_layer.deps[j]:
                    self.deps_b_cl[Deps.I_A][i] = \
                        self.deps_b_cl[Deps.I_A][i] | \
                        p_layer.deps_b_cl[Deps.A_A][j]
                    self.deps_b_cl[Deps.I_A][i].add((p_layer.depth, j))


    def compute_group_deps(self, next_layer, runtime=False,
                           binary_vars=np.empty(0), \
                           next_binary_vars=np.empty(0)):
        if runtime==True:
            if len(binary_vars) == 0:
                raise Exception("""Missing the layer's binary variables
                for the runtime computation of group dependencies""")
            if len(next_binary_vars) == 0:
                raise Exception("""Missing the next layer's binary
                variables for the runtime computation of group
                               dependencies""")
            self._compute_runtime_group_deps(next_layer, binary_vars, \
                                             next_binary_vars)
        else:
            self._compute_group_deps(next_layer)


    def _compute_runtime_group_deps(self, next_layer, binary_vars,
                           next_binary_vars):

        nf = self.get_not_fixed(binary_vars=binary_vars)
        nfp = next_layer.get_not_fixed(binary_vars=next_binary_vars)

        self.group_deps = {'group': nf, 'nodes': []}

        for j in nfp: 
            ubp = next_layer.rnt_bounds['in']['u'][j]
            lbp = next_layer.rnt_bounds['in']['l'][j]

            for i in nf:
                if next_layer.weights[j][i] >= 0:
                    """
                    inactive -> inactive dependency
                    """
                    ubp -= next_layer.weights[j][i] * \
                        self.rnt_bounds['in']['u'][i]
                else:
                    """
                    inactive -> active dependency
                    """
                    lbp -= next_layer.weights[j][i] * \
                        self.rnt_bounds['in']['u'][i]

            if ubp <= 0:
                self.group_deps['nodes'].append((j,DepType.I_I))
            if lbp >= 0:
                self.group_deps['nodes'].append((j,DepType.I_A))


    def compute_layer_deps(self, p_layer, runtime=False,
                           binary_vars=np.empty(0)):
        if runtime==True:
            if len(binary_vars) == 0:
                raise Exception("""Missing the layer's binary variables
                for the runtime computation of layer dependencies""")
            self.layer_deps = \
                    FirstLayerDependencies(self, p_layer.rnt_bounds['out'],
                                    self.rnt_bounds['in'], runtime=True,
                                    binary_vars=binary_vars)
        else:
            self.layer_deps = FirstLayerDependencies(self,
                                                     p_layer.bounds['out'],
                                                     self.bounds['in'])


    def add_xlayer_dep_constrs(self, next_layer, gmodel):
        for nd in self.get_not_fixed():
            v = self.vars['delta'][nd]
            for (n_nd,dep_type) in self.xlayer_deps[nd]:
                if dep_type == DepType.I_A:
                    n_v = next_layer.vars['delta'][n_nd] 
                    gmodel.addConstr( 1 - n_v <= v )
                elif dep_type == DepType.I_I:
                    n_v = next_layer.vars['delta'][n_nd]
                    gmodel.addConstr(  n_v <= v )

    def add_layer_dep_constrs(self, gmodel):
        for nd, (neg,pos) in self.layer_deps.dep_per_var.items():
            v = self.vars['delta'][nd]
            for x in neg:
                v2 = self.vars['delta'][x.i]
                if isinstance(x, PosLiteral):
                    gmodel.addConstr( 1 -  v2 <= v )
                else:
                    gmodel.addConstr( v2 <= v )
            for x in pos:
                v2 = self.vars['delta'][x.i]
                if isinstance(x, PosLiteral):
                    gmodel.addConstr( 1 -  v2 <= 1 - v )
                else:
                    gmodel.addConstr( v2 <= 1 - v )
    

    def add_group_dep_constrs(self, next_layer, gmodel):
        for dep in self.group_deps: 
            s = quicksum([ self.vars['delta'][k] for k in dep['group'] ])
            for node, dep_type in dep['nodes']:
                if dep_type == DepType.I_I:
                    v = next_layer.vars['delta'][node]
                    gmodel.addConstr( v <= s)
                elif dep_type == DepType.I_A:
                    v = next_layer.vars['delta'][node]
                    gmodel.addConstr( 1 - v <= s)

    def _compute_group_deps(self, next_layer):
 
        self.group_deps = [] 

        ii_group = [ self._get_dep_ii_group(next_layer,i) for i in
                  range(next_layer.output_shape) ]
        aa_group = [ self._get_dep_ia_group(next_layer,i) for i in
                  range(next_layer.output_shape) ]

        groups = ii_group + aa_group


        for gr in range(len(groups)):
            if len(groups[gr]) < 2:
                continue

            self.group_deps.append({'group' : groups[gr], 'nodes': []})


            for j in next_layer.get_not_fixed():
                ubp = next_layer.bounds['in']['u'][j]
                lbp = next_layer.bounds['in']['l'][j]

                for i in groups[gr]:
                    if next_layer.weights[j][i] >= 0:
                        """
                        inactive -> inactive dependency
                        """
                        ubp -= next_layer.weights[j][i] * \
                               self.bounds['in']['u'][i]
                    else:
                        """
                        inactive -> active dependency
                        """
                        lbp -= next_layer.weights[j][i] * \
                               self.bounds['in']['u'][i]

                if ubp <= 0:
                    self.group_deps[-1]['nodes'].append((j,DepType.I_I))
                if lbp > 0:
                    self.group_deps[-1]['nodes'].append((j,DepType.I_A))



    def _get_dep_ii_group(self, next_layer, node):
        nf = self.get_not_fixed()
        ub = next_layer.bounds['in']['u'][node]
        w = next_layer.weights[node, nf] * self.bounds['in']['u'][nf]
        s_w = np.argsort(w)
        group = []

        if np.sum(w[(w > 0)]) >= ub:
            _sum = 0
            i = len(s_w) - 1
            while _sum < ub:
                _sum += w[s_w[i]]
                group.append(nf[s_w[i]])
                i -= 1

        return group


    def _get_dep_ia_group(self, next_layer, node):
        nf = self.get_not_fixed()
        lb = next_layer.bounds['in']['l'][node]
        w = next_layer.weights[node, nf] * self.bounds['in']['u'][nf]
        s_w = np.argsort(w)

        group = []

        if np.sum(w[(w < 0)]) <= lb:
            _sum = 0
            i = 0
            while _sum > lb:
                _sum += w[s_w[i]]
                group.append(nf[s_w[i]])
                i += 1

        return group


    def _get_ii_group(self, next_layer):
        nf = self.get_not_fixed()
        nfp = next_layer.get_not_fixed()

        if len(nf) == 0 or len(nfp) == 0:
            return []

        nfp_bounds = [next_layer.bounds['in']['u'][i] \
                      for i in nfp]
        max_ind = np.argmax(nfp_bounds)
        max_ub = nfp_bounds[max_ind]
        w = next_layer.weights[nfp[max_ind], nf] * self.bounds['in']['u'][nf]
        s_w = np.argsort(w)

        group = []

        if np.sum(w[(w > 0)]) >= max_ub:
            _sum = 0
            i = len(s_w) - 1
            while _sum < max_ub:
                _sum += w[s_w[i]]
                group.append(nf[s_w[i]])
                i -= 1

        return group

    def _get_ia_group(self, next_layer):
        nf = self.get_not_fixed()
        nfp = next_layer.get_not_fixed()

        if len(nf) == 0 or len(nfp) == 0:
            return []

        nfp_bounds = [next_layer.bounds['in']['l'][i] \
                      for i in nfp]

        min_ind = np.argmin(nfp_bounds)
        min_lb = nfp_bounds[min_ind]
        w = next_layer.weights[nfp[min_ind], nf] * self.bounds['in']['u'][nf]
        s_w = np.argsort(w)

        group = []

        if np.sum(w[(w < 0)]) <= min_lb:
            _sum = 0
            i = 0
            while _sum > min_lb:
                _sum += w[s_w[i]]
                group.append(nf[s_w[i]])
                i += 1

        return group


class ReluBigM(Relu):

    def add_constrs(self, p_layer, gmodel, linear_approximation=False):
        out = self.vars['out']
        l_bounds = self.bounds['in']['l']
        u_bounds = self.bounds['in']['u']
        dot_product = [
            self.weights[i].dot(p_layer.vars['out']) + self.bias[i] \
            for i in range(self.output_shape)
        ]
        delta = self.vars['delta']
        for i in range(self.output_shape):
            if l_bounds[i] >= 0:
                gmodel.addConstr(out[i] == dot_product[i])
            elif u_bounds[i] <= 0:
                gmodel.addConstr(out[i] == 0)
            else:
                if linear_approximation:
                    gmodel.addConstr( out[i] >= dot_product[i] )
                    expr = u_bounds[i] / (u_bounds[i] - l_bounds[i])
                    expr = expr * (dot_product[i]-l_bounds[i])
                    gmodel.addConstr( out[i] <= expr )
                else:
                    gmodel.addConstr(out[i] >= dot_product[i])
                    gmodel.addConstr(out[i] <= dot_product[i] - l_bounds[i] * (1 - delta[i]))
                    gmodel.addConstr(out[i] <= u_bounds[i] * delta[i])

    def optimise_bound(self, p_layer, node, gmodel):
        var = gmodel.addVar(lb=self.bounds['in']['l'][node], \
                            ub=self.bounds['in']['u'][node])
        dot_product = self.weights[node].dot(p_layer.vars['out']) + self.bias[node]
        gmodel.addConstr( var == dot_product )

        return var

    def clone(self):
        return ReluBigM(self.output_shape, self.weights, self.bias, self.depth)


class ReluIdeal(ReluBigM):
    def clone(self):
        return ReluIdeal(self.output_shape, self.weights, self.bias, self.depth)


class ReluMulChoice(Relu):


    def add_vars(self, p_layer, gmodel):
        super().add_vars(p_layer, gmodel)

        self.vars['in_aux'] = np.empty(shape=(self.output_shape, p_layer.output_shape, 2), dtype=Var)
        self.vars['out_aux'] = np.empty(shape=(self.output_shape, 2), dtype=Var)
        for i in range(self.output_shape):
            self.vars['out_aux'][i][0] = gmodel.addVar(lb=0, ub=0)
            self.vars['out_aux'][i][1] = gmodel.addVar(lb=self.bounds['out']['l'][i], ub=self.bounds['out']['u'][i])
            for j in range(p_layer.output_shape):
                self.vars['in_aux'][i][j][0] = gmodel.addVar(lb=0, ub=p_layer.bounds['out']['u'][j])
                self.vars['in_aux'][i][j][1] = gmodel.addVar(lb=0, ub=p_layer.bounds['out']['u'][j])

    def add_constrs(self, p_layer, gmodel):
        out = self.vars['out']
        out_aux = self.vars['out_aux']
        in_aux = self.vars['in_aux']
        delta = self.vars['delta']
        _in = p_layer.vars['out']

        for i in range(self.output_shape):

            dot_product_0 = self.weights[i, :].dot(in_aux[i, :, 0]) + self.bias[i] * (1 - delta[i])
            dot_product_1 = self.weights[i, :].dot(in_aux[i, :, 1]) + self.bias[i] * delta[i]

            for j in range(p_layer.output_shape):
                gmodel.addConstr(_in[j] == in_aux[i][j][0] + in_aux[i][j][1])
                gmodel.addConstr(in_aux[i][j][0] >= \
                                 p_layer.bounds['out']['l'][j] * (1 - delta[i]))
                gmodel.addConstr(in_aux[i][j][0] <= \
                                 p_layer.bounds['out']['u'][j] * (1 - delta[i]))
                gmodel.addConstr(in_aux[i][j][1] <= \
                                 p_layer.bounds['out']['u'][j] * delta[i])
                gmodel.addConstr(in_aux[i][j][1] >= \
                                 p_layer.bounds['out']['l'][j] * delta[i])

            gmodel.addConstr(out[i] == out_aux[i][0] + out_aux[i][1])
            # gmodel.addConstr( out_aux[i][0] == 0)
            gmodel.addConstr(out_aux[i][0] >= dot_product_0)
            gmodel.addConstr(out_aux[i][1] >= 0)
            gmodel.addConstr(out_aux[i][1] == dot_product_1)

    def clone(self):
        return ReluMulChoice(self.output_shape, self.weights, self.bias, self.depth)


class Linear(Layer):

    def add_vars(self, p_layer, gmodel):
        super().add_vars(gmodel)

    def add_constrs(self, p_layer, gmodel):
        out = self.vars['out']
        dot_product = [
            self.weights[i].dot(p_layer.vars['out']) + self.bias[i] \
            for i in range(self.output_shape)
        ]
        for i in range(self.output_shape):
            gmodel.addConstr(out[i] == dot_product[i])

    def compute_bounds(self, method, p_layer, input_bounds=None, const=None):
        if method == Bounds.INT_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
        elif method == Bounds.SYMBOLIC_INT_ARITHMETIC:
            if input_bounds == None:
                raise Exception("""Missing input-bounds parameter for
                                symbolic-based calculation of
                                bounds""")
            else:
                self._compute_bounds_sia(p_layer, input_bounds)
        elif method == Bounds.CONST:
            if not isinstance(const, float):
                raise Exception("""Missing float value for constant
                bounds""")
            else:
                self._compute_bounds_const(const)

    def _compute_bounds_sia(self, p_layer, input_bounds):
        weights_plus = np.maximum(self.weights, np.zeros(self.weights.shape))
        weights_minus = np.minimum(self.weights, np.zeros(self.weights.shape))

        # get coefficients for the bound equations
        p_l_coeffs = p_layer.bound_equations['out']['l'].matrix
        p_u_coeffs = p_layer.bound_equations['out']['u'].matrix
        l_coeffs = weights_plus.dot(p_l_coeffs) + weights_minus.dot(p_u_coeffs)
        u_coeffs = weights_plus.dot(p_u_coeffs) + weights_minus.dot(p_l_coeffs)

        # get constants for the bound equations
        p_l_const = p_layer.bound_equations['out']['l'].offset
        p_u_const = p_layer.bound_equations['out']['u'].offset
        l_const = weights_plus.dot(p_l_const) + weights_minus.dot(p_u_const) + self.bias
        u_const = weights_plus.dot(p_u_const) + weights_minus.dot(p_l_const) + self.bias

        # set bound equations
        self.bound_equations['out']['l'] = LinearFunctions(l_coeffs, l_const)
        self.bound_equations['out']['u'] = LinearFunctions(u_coeffs, u_const)

        # set concrete output bounds
        self.bounds['out']['l'] = \
            self.bound_equations['out']['l'].compute_min_values(input_bounds)
        self.bounds['out']['u'] = \
            self.bound_equations['out']['u'].compute_max_values(input_bounds)

    def _compute_bounds_ia(self, p_layer):
        weights_plus = np.maximum(self.weights, np.zeros(self.weights.shape))
        weights_minus = np.minimum(self.weights, np.zeros(self.weights.shape))

        self.bounds['out']['l'] = np.array([
            weights_plus[i].dot(p_layer.bounds['out']['l']) + \
            weights_minus[i].dot(p_layer.bounds['out']['u']) + \
            self.bias[i] for i in range(self.output_shape)
        ])

        self.bounds['out']['u'] = np.array([
            weights_plus[i].dot(p_layer.bounds['out']['u']) + \
            weights_minus[i].dot(p_layer.bounds['out']['l']) + \
            self.bias[i] for i in range(self.output_shape)
        ])

    def _compute_bounds_const(self, const):
        self.bounds['out']['l'] = np.ones(self.output_shape) * const * -1
        self.bounds['out']['u'] = np.ones(self.output_shape) * const

    def clone(self):
        return Linear(self.output_shape, self.weights, self.bias, self.depth)
