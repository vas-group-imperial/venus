import keras
import os
import onnx
import onnx.numpy_helper
from src.layers import Output, ReluMulChoice, ReluBigM, ReluIdeal, Input, Linear
from src.parameters import EncType


class LayersModel(object):
    def __init__(self, encoding):
        self.input = None
        self.layers = []
        self.output = None
        self.ENCODING = encoding


    def load(self, path, spec):
        _,model_format = os.path.splitext(path)
        if model_format == '.h5': 
            keras_model = keras.models.load_model(path,compile=False)
            self.parse_keras(keras_model, spec)
        elif model_format == '.onnx':
            onnx_model = onnx.load(path)
            self.parse_onnx(onnx_model, spec)
        else:
            raise Exception("Unsupportted model format")

    def parse_onnx(self, model, spec):
        # input constraints
        self.input = Input(spec)
        i = 1
        nodes = model.graph.node
        for j in range(len(nodes)):
            node = nodes[j]
            next_node = nodes[j+1] if j < len(nodes) - 1 else nodes[j]
            if node.op_type in ['Flatten','Relu']:
                pass
            elif node.op_type=='Gemm':
                [weights] = [onnx.numpy_helper.to_array(t) for t in model.graph.initializer if t.name == node.input[1]]
                [bias] = [onnx.numpy_helper.to_array(t) for t in model.graph.initializer if t.name == node.input[2]]
                if next_node.op_type=='Relu':
                    self.layers.append(ReluIdeal(weights.shape[0], weights, bias, i))
                else: 
                    self.layers.append(Linear(weights.shape[0], weights, bias, i))
                i += 1
            else:
                pass

    def parse_keras(self, model, spec):
        # input constraints
        self.input = Input(spec)

        # layers of the network
        for i in range(len(model.layers)):
            l = model.layers[i]
            if l.activation == keras.activations.relu:
                if self.ENCODING == EncType.BIG_M:
                    self.layers.append(ReluBigM(
                        l.output_shape[1],
                        l.get_weights()[0].T,
                        l.get_weights()[1],
                        i + 1))
                elif self.ENCODING == EncType.IDEAL:
                    self.layers.append(ReluIdeal(
                        l.output_shape[1],
                        l.get_weights()[0].T,
                        l.get_weights()[1],
                        i + 1))
                elif self.ENCODING == EncType.MUL_CHOICE:
                    self.layers.append(ReluMulChoice(
                        l.output_shape[1],
                        l.get_weights()[0].T,
                        l.get_weights()[1],
                        i + 1))
                else:
                    raise Exception('   Error: invalid encoding', self.ENCODING)
            elif l.activation == keras.activations.linear:
                self.layers.append(Linear(
                    l.output_shape[1],
                    l.get_weights()[0].T,
                    l.get_weights()[1],
                    i + 1))

        # output constraints
        self.output = Output(spec, len(self.layers) + 1)

    def clone(self, spec=None):
        new_model = LayersModel(self.ENCODING)
        for layer in self.layers:
            new_model.layers.append(layer.clone())
        if spec is None:
            new_model.output = self.output.clone()
            new_model.input = self.input.clone()
        else:
            new_model.input = Input(spec)
            new_model.output = Output(spec, len(new_model.layers) + 1)
        return new_model

    def clean_vars(self):
        for layer in self.layers + [self.input] + [self.output]:
            layer.clean_vars()
