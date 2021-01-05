import keras.models

from src.fldependencies import FirstLayerDependencies
from src.layers import *
from src.layersmodel import LayersModel
from src.specifications import *
from src.parameters import *
import math
from src.solver import solver


class VModel(object):
    def __init__(self, lmodel, spec, params):
        # the internal representation of a neural network as instance of LayersModel
        self.lmodel = lmodel.clone(spec)
        self.spec = spec

        self.params = params

        # gmodel will be initialised when encode() is called
        self.gmodel = None

        self.layers_initialised = False
        self.dependecies_computed = False
        self.dependecies_count = -1
        self.max_dependecies_count = -1
        self.fixed_nodes_count = -1
        self.relu_nodes_count = -1
        self.bounds_termination = False


    def disable_cuts(self):
        self.gmodel.setParam('PreCrush', 1)
        self.gmodel.setParam(GRB.Param.CoverCuts,0)
        self.gmodel.setParam(GRB.Param.CliqueCuts,0)
        self.gmodel.setParam(GRB.Param.FlowCoverCuts,0)
        self.gmodel.setParam(GRB.Param.FlowPathCuts,0)
        self.gmodel.setParam(GRB.Param.GUBCoverCuts,0)
        self.gmodel.setParam(GRB.Param.ImpliedCuts,0)
        self.gmodel.setParam(GRB.Param.InfProofCuts,0)
        self.gmodel.setParam(GRB.Param.MIPSepCuts,0)
        self.gmodel.setParam(GRB.Param.MIRCuts,0)
        self.gmodel.setParam(GRB.Param.ModKCuts,0)
        self.gmodel.setParam(GRB.Param.NetworkCuts,0)
        self.gmodel.setParam(GRB.Param.ProjImpliedCuts,0)
        self.gmodel.setParam(GRB.Param.StrongCGCuts,0)
        self.gmodel.setParam(GRB.Param.SubMIPCuts,0)
        self.gmodel.setParam(GRB.Param.ZeroHalfCuts,0)
        self.gmodel.setParam(GRB.Param.GomoryPasses,0)


    def add_layers(self):
        # input constraints
        self.input = Input(self.spec)
        # layers of the network
        for i in range(len(self.lmodel.layers)):
            l = self.lmodel.layers[i]
            if l.activation == keras.activations.relu:
                if self.params.ENCODING[i] == EncType.BIG_M:
                    self.layers.append(ReluBigM(
                        l.output_shape[1],
                        l.get_weights()[0].T,
                        l.get_weights()[1],
                        i+1))
                elif self.params.ENCODING[i]==EncType.IDEAL:
                    self.layers.append(ReluIdeal(
                        l.output_shape[1],
                        l.get_weights()[0].T,
                        l.get_weights()[1],
                        i+1))
                elif self.params.ENCODING[i] == EncType.MUL_CHOICE:
                    self.layers.append(ReluMulChoice(
                        l.output_shape[1],
                        l.get_weights()[0].T,
                        l.get_weights()[1],
                        i+1))
                else:
                    raise Exception('   Error: invalid encoding', self.params.ENCODING[i])
            elif l.activation == keras.activations.linear:
                self.layers.append(Linear(
                    l.output_shape[1],
                    l.get_weights()[0].T,
                    l.get_weights()[1],
                    i+1))
        # output constraints
        self.output = Output(self.spec, len(self.layers)+1)

        # we set nmodel to None to enable pickle-ability of vmodel
        # after having added the layers, nmodel is no longer needed
        self.lmodel = None

    def initialise_layers(self, compute_dep=False, start=-1, end=-1):
        self.compute_bounds(start=start,end=end)
        if compute_dep:
            self.compute_all_deps(end=self.params.DEP_DEPTH)
        # turn on the flag only if all layers have been initialised
        if end==-1:
            self.layers_initialised = True

    def get_n_deps(self):
        if not self.dependecies_computed:
            return 0

        if self.dependecies_count == -1:
            dep_count = 0
            for layer_n in range(len(self.lmodel.layers) - 1):
                dep_count += self.lmodel.layers[layer_n].get_total_dep_count()

            self.dependecies_count = dep_count
        return self.dependecies_count

    def get_max_n_deps(self):
        if self.max_dependecies_count == -1:
            dep_count = 0
            group_dep_included = 1 if self.params.GROUP_DEP_CONSTRS or self.params.GROUP_DEP_CUTS else 0
            for i in range(len(self.lmodel.layers) - 1):
                size_i = self.lmodel.layers[i].output_shape
                dep_count += size_i * self.lmodel.layers[i + 1].output_shape + \
                             size_i * 2 * group_dep_included + \
                             size_i * (size_i-1) / 2
            self.max_dependecies_count = dep_count
        return self.max_dependecies_count

    def get_n_relu_nodes(self, start=-1, end=-1):
        if end==-1:
            # end computation at the output layer
            end = len(self.lmodel.layers)

        if self.relu_nodes_count == -1:
            total_n = 0
            for layer_n in range(start+1, end):
                if isinstance(self.lmodel.layers[layer_n],Relu):
                    total_n += self.lmodel.layers[layer_n].output_shape

            self.relu_nodes_count = total_n
        return self.relu_nodes_count

    def get_n_fixed_nodes(self, start=-1, end=-1):
        if end==-1:
            # end computation at the output layer
            end = len(self.lmodel.layers)
        if self.fixed_nodes_count == -1:
            fixed_n = 0
            for layer_n in range(start+1, end):
                if isinstance(self.lmodel.layers[layer_n],Relu):
                    fixed_n += len(self.lmodel.layers[layer_n].get_fixed())
            self.fixed_nodes_count = fixed_n
        return self.fixed_nodes_count

    def encode(self):
        if not self.layers_initialised:
            self.initialise_layers()


        if isinstance(self.spec, LRob):
            out = self.lmodel.layers[-1]
            l = self.spec.label
            flag = False
            for i in  range(out.output_shape):
                if not i == label and \
                out.bounds['out']['l'][label] <= \
                out.bounds['out']['u'][i]:
                    flag = True
            if not flag:
                self.bounds_termination = True
                

        if self.params.OPTIMISE_BOUNDS:
            self.optimise_bounds(linear_approximation=True)


        # gmodel is created only now so as to enable pickle-ability of vmodel
        # before this method has been called
        self.gmodel = Model()
        if self.params.PRINT_GUROBI_OUTPUT:
            self.gmodel.setParam('OutputFlag', 1)
        else:
            self.gmodel.setParam('OutputFlag', 0)

        if self.params.TIME_LIMIT != -1:
            self.gmodel.setParam('TimeLimit', self.params.TIME_LIMIT)

        # add MIP variables
        self.lmodel.input.add_vars(self.gmodel)
        p_layer = self.lmodel.input
        for i in self.lmodel.layers:
            i.add_vars(p_layer,self.gmodel)
            p_layer = i
        self.lmodel.output.add_vars(self.gmodel)

        # add MIP constraints 
        p_layer = self.lmodel.input
        for i in self.lmodel.layers:
            i.add_constrs(p_layer,self.gmodel)
            p_layer = i
        self.lmodel.output.add_constrs(p_layer, self.gmodel)

        # add dependency constrains
        if self.params.GROUP_DEP_CONSTRS or self.params.XLAYER_DEP_CONSTRS or self.params.LAYER_DEP_CONSTRS:
            if self.params.DEP_DEPTH==-1:
                end = len(self.lmodel.layers) - 1
            else:
                end = self.params.DEP_DEPTH - 1
            p_l = self.lmodel.input
            for i in range(end):
                l = self.lmodel.layers[i]
                n_l = self.lmodel.layers[i + 1]
                if isinstance(l, Relu) and isinstance(n_l, Relu):
                    if self.params.GROUP_DEP_CONSTRS:
                        l.compute_group_deps(n_l)                        
                        l.add_group_dep_constrs(n_l, self.gmodel)
                    if self.params.XLAYER_DEP_CONSTRS:
                        l.compute_xlayer_deps(n_l)                        
                        l.add_xlayer_dep_constrs(n_l, self.gmodel)
                if isinstance(l, Relu) and self.params.LAYER_DEP_CONSTRS:
                        l.compute_layer_deps(p_l)
                        l.add_layer_dep_constrs(self.gmodel)
                p_l = l


        self.gmodel.update()

        # set optimisation objective for MaxLRob specification
        if self.spec.isMaxLRob():
            label1 = self.spec.label1
            label2 = self.spec.label2
            out1 = self.lmodel.layers[-1].vars['out'][label1]
            out2 = self.lmodel.layers[-1].vars['out'][label2]
            obj =  out2 - out1
            self.gmodel.setObjective(obj,GRB.MAXIMIZE)

    def compute_bounds(self, runtime=False, binary_vars=None, \
                       start=-1, end=-1):
        if start==-1:
            # begin computation from the input layer
            self.lmodel.input.compute_bounds()
            p_l = self.lmodel.input
        else:
            # begin computation from the specified hidden layer
            p_l = self.lmodel.layers[start]
        if end==-1:
            # end computation at the output layer
            end = len(self.lmodel.layers) - 1
        for l in range(start+1, end+1):
            # bounds for relu nodes
            if isinstance(self.lmodel.layers[l], Relu):
                if runtime == True:
                    b_v = binary_vars[l]
                else:
                    b_v = []
                self.lmodel.layers[l].compute_bounds(self.params.BOUNDS, \
                                                     p_l, \
                                                     runtime = runtime, \
                                                     binary_vars = b_v, \
                                                     input_bounds = self.lmodel.input.bounds, \
                                                     const = self.params.BOUNDS_CONST)
            else:
                #bounds for linear nodes
                self.lmodel.layers[l].compute_bounds(self.params.BOUNDS, \
                                                     p_l, \
                                                     input_bounds=self.lmodel.input.bounds, \
                                                     const=self.params.BOUNDS_CONST)
            p_l = self.lmodel.layers[l]


    def compute_all_deps(self, runtime=False, binary_vars=np.empty(0), end=-1):
        if not self.dependecies_computed:
            if self.params.LAYER_DEP_CONSTRS:
                self.compute_layer_deps(runtime, binary_vars, end)
            if self.params.XLAYER_DEP_CONSTRS:
                self.compute_xlayer_deps(runtime, binary_vars, end)
            if self.params.GROUP_DEP_CONSTRS:
                self.compute_group_deps(runtime, binary_vars, end)
            self.dependecies_computed = True


    def compute_layer_deps(self, runtime=False, binary_vars=np.empty(0), end=-1): 
        if end==-1:
            # end computation at the output layer
            end = len(self.lmodel.layers)

        p_l = self.lmodel.input
        for i in range(end):
            l = self.lmodel.layers[i]
            if isinstance(l,Relu):
                if runtime:
                    bv = binary_vars[i]
                    l.compute_layer_deps(p_l, runtime=True,
                                         binary_vars=bv)
                else:
                    l.compute_layer_deps(p_l)
            p_l = l


    def compute_xlayer_deps(self, runtime=False,
                            binary_vars=np.empty(0), end=-1):
        if end==-1:
            # end computation at the output layer
            end = len(self.lmodel.layers)-1
 
        for i in range(end):
            l = self.lmodel.layers[i]
            n_l = self.lmodel.layers[i + 1]
            if isinstance(l,Relu) and isinstance(n_l, Relu):
                if runtime:
                    bv = binary_vars[i]
                    n_bv = binary_vars[i+1]
                    l.compute_xlayer_deps(n_l, runtime=True,
                                          binary_vars=bv,
                                          next_binary_vars=n_bv)
                else:
                    l.compute_xlayer_deps(n_l)

    def compute_group_deps(self, runtime=False,
                           binary_vars=np.empty(0)):
        if end==-1:
            # end computation at the output layer
            end = len(self.lmodel.layers)

        for i in range(end):
            l = self.lmodel.layers[i]
            n_l = self.lmodel.layers[i + 1]
            if isinstance(l, Relu) and isinstance(n_l, Relu):
                if runtime:
                    bv = binary_vars[i]
                    n_bv = binary_vars[i+1]
                    l.compute_group_deps(n_l, runtime=True,
                                         binary_vars=bv,
                                         next_binary_vars=n_bv)
                else:
                    l.compute_group_deps(n_l)

    def optimise_bounds(self, linear_approximation=False):
        # Initialise gurobi model 
        gmodel = Model()
        gmodel.setParam('OutputFlag', 0)
        # Add input variables
        self.lmodel.input.add_vars(gmodel)
        p_l = self.lmodel.input
        # Add first layer vars and constraints
        self.lmodel.layers[0].add_vars(p_l,gmodel,linear_approximation)
        self.lmodel.layers[0].add_constrs(p_l,gmodel,linear_approximation)
        p_l = self.lmodel.layers[0]


        for i in range(1,self.params.NUM_OPT_LAYERS+1):
            l = self.lmodel.layers[i]
            if not isinstance(l,Relu):
                # add MILP encoding but do not optimise
                l.add_vars(p_l,gmodel)
                l.add_constrs(p_l,gmodel)
                p_l = l
                continue
            
            # mark whether to propagate optimised bounds
            flag = False

            for nd in range(l.output_shape):
                # optimise only if the bounds are smaller than the
                # average overapproximation error
                cond1 = l.error['out'][nd] < l.bounds['in']['l'][nd]
                cond2 = l.error['out'][nd] > l.bounds['in']['u'][nd]
                if l.is_fixed(nd) or (not cond1 and not cond2):
                    continue

                flag = True

                l.bounds['in']['l'][nd], l.bounds['in']['u'][nd] =  \
                self.optimise_bound(i, nd, gmodel, linear_approximation)

                               
            if flag:
                # propagage the optimised bounds
                self.compute_bounds(start=i,end=i+1)

            # Add layer's vars and constraints
            l.add_vars(p_l,gmodel,linear_approximation)
            l.add_constrs(p_l,gmodel,linear_approximation)
            p_l = l


        # we need to remove the variables that were added for an auxiliary gmodel
        # required for pickle-ability and
        # also because they are useless
        self.lmodel.clean_vars()

    def optimise_bound(self, layer, node, gmodel, linear_approximation=False):
        # node's to be optimised layer 
        l = self.lmodel.layers[layer]
        lb = l.bounds['in']['l'][node]
        ub = l.bounds['in']['u'][node]
        # previous layer
        p_l = self.lmodel.layers[layer-1]
        # weights and bias of the node to be optimised
        w = l.weights[node,:]
        b = np.array([l.bias[node]])
 
        # add variable and constraint expressing the input to the node
        obj = gmodel.addVar(lb=lb, ub=ub)
        dot_product = w.dot(p_l.vars['out']) +  b
        gmodel.addConstr( obj == dot_product )

        # update model
        gmodel.update()

        # optimise lower bound 
        gmodel.setObjective(obj,GRB.MINIMIZE)
        gmodel.optimize()
        if gmodel.status == GRB.Status.OPTIMAL:
            lb = obj.X 
  
        # update model
        gmodel.update()

        # optimise upper bound 
        gmodel.setObjective(obj,GRB.MAXIMIZE)
        gmodel.optimize()
        if gmodel.status == GRB.Status.OPTIMAL:
            ub = obj.X 

        # remove optimisation var and constraint
        gmodel.remove(gmodel.getVars()[-1])
        gmodel.remove(gmodel.getConstrs()[-1])

        return lb, ub



    def get_var_indices(self, layer, var_type):
        layers = [self.lmodel.input] + self.lmodel.layers
        start = 0
        end = 0
        for l in range(layer):
            start += len(layers[l].vars['out'])
            if 'delta' in layers[l].vars.keys():
                start += len(layers[l].vars['delta'])
        if var_type == 'out':
            end = start + len(layers[layer].vars['out'])
        elif var_type == 'delta':
            start += len(layers[layer].vars['out'])
            end = start + len(layers[layer].vars['delta'])
        return (start, end)

 

    def verify(self):
        """
        :return (False,None) if the MILP has no solution 
        :return (True,Cex) if the MILP has a solution Cex 
       """
        # self.gmodel.write("program.lp")
        if self.bounds_termination:
            return (False,None)
        else:
            (sat,cex) = solver(self)
            return (sat,cex)

