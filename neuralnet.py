import math
import random


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x, screen=None):
        for layer in self.layers:
            x = layer.forward(x, screen=screen)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_params(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, "update_params"):
                layer.update_params(learning_rate)

class Layer:
    def __init__(self, num_neurons, input_size):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def forward(self, x, screen=None):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(x, screen=screen))
        return outputs

    def backward(self, grad):
        grads = [0] * len(self.neurons[0].weights)
        for neuron, g in zip(self.neurons, grad):
            neuron_grads = neuron.backward(g)
            grads = [sum(x) for x in zip(grads, neuron_grads)]
        return grads

    def update_params(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_params(learning_rate)

class Neuron:
    def __init__(self, input_size):
        scale = 1.0 / math.sqrt(max(1, input_size))
        self.weights = [random.uniform(-1.0, 1.0) * scale for _ in range(input_size)]
        self.bias = random.uniform(-0.1, 0.1)
        self.activation = None
        self.last_letter = "A"

    @staticmethod
    def output_to_letter(value, temperature=1.0):
        # Squash arbitrary activations into 0..1 before mapping to A..Z.
        temp = max(1e-6, float(temperature))
        scaled_value = value / temp
        if scaled_value >= 0:
            exp_term = math.exp(-scaled_value)
            normalized = 1.0 / (1.0 + exp_term)
        else:
            exp_term = math.exp(scaled_value)
            normalized = exp_term / (1.0 + exp_term)
        index = min(25, max(0, int(normalized * 26)))
        return chr(65 + index)

    def forward(self, x, screen=None):
        self.activation = sum(w * i for w, i in zip(self.weights, x)) + self.bias
        self.last_input = x
        self.last_output = self.activation
        self.last_letter = self.output_to_letter(self.last_output)
        if screen is not None:
            if hasattr(screen, "display"):
                screen.display(self.last_letter)
        return self.last_output

    def backward(self, grad):
        self.grad_weights = [grad * i for i in self.last_input]
        self.grad_bias = grad
        return [grad * w for w in self.weights]

    def update_params(self, learning_rate):
        self.weights = [w - learning_rate * gw for w, gw in zip(self.weights, self.grad_weights)]
        self.bias -= learning_rate * self.grad_bias


def create_neuron(input_size):
    return Neuron(input_size)

def create_layer(num_neurons, input_size):
    return Layer(num_neurons, input_size)

def create_network():
    return NeuralNetwork()

