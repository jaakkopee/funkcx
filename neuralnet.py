import math
import random

import numpy as np


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
        self.weights = np.vstack([neuron.weights for neuron in self.neurons]).astype(float)
        self.biases = np.array([neuron.bias for neuron in self.neurons], dtype=float)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

    def _sync_neurons_from_matrix(self):
        for neuron, weights, bias in zip(self.neurons, self.weights, self.biases):
            neuron.weights = weights.copy()
            neuron.bias = float(bias)

    def forward(self, x, screen=None):
        x_array = np.asarray(x, dtype=float)
        self.last_input = x_array
        outputs = self.weights @ x_array + self.biases

        if screen is not None and hasattr(screen, "display"):
            for neuron, output in zip(self.neurons, outputs):
                neuron.activation = float(output)
                neuron.last_input = x_array
                neuron.last_output = float(output)
                neuron.last_letter = neuron.output_to_letter(neuron.last_output)
                screen.display(neuron.last_letter)

        self.last_output = outputs
        return outputs

    def backward(self, grad):
        grad_array = np.asarray(grad, dtype=float)
        self.grad_weights = np.outer(grad_array, self.last_input)
        self.grad_biases = grad_array
        return self.weights.T @ grad_array

    def update_params(self, learning_rate):
        self.weights = self.weights - (learning_rate * self.grad_weights)
        self.biases = self.biases - (learning_rate * self.grad_biases)
        self._sync_neurons_from_matrix()

class Neuron:
    def __init__(self, input_size):
        scale = 1.0 / math.sqrt(max(1, input_size))
        self.weights = np.random.uniform(-1.0, 1.0, size=input_size) * scale
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
        x_array = np.asarray(x, dtype=float)
        self.activation = float(np.dot(self.weights, x_array) + self.bias)
        self.last_input = x_array
        self.last_output = self.activation
        self.last_letter = self.output_to_letter(self.last_output)
        if screen is not None:
            if hasattr(screen, "display"):
                screen.display(self.last_letter)
        return self.last_output

    def backward(self, grad):
        grad_value = float(grad)
        self.grad_weights = grad_value * self.last_input
        self.grad_bias = grad_value
        return grad_value * self.weights

    def update_params(self, learning_rate):
        self.weights = self.weights - (learning_rate * self.grad_weights)
        self.bias -= learning_rate * self.grad_bias


def create_neuron(input_size):
    return Neuron(input_size)

def create_layer(num_neurons, input_size):
    return Layer(num_neurons, input_size)

def create_network():
    return NeuralNetwork()

