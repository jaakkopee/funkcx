import math
import random

import numpy as np

try:
    import metal_backend
except Exception:
    metal_backend = None


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

    def _forward_batch_numpy(self, x_batch):
        return x_batch @ self.weights.T + self.biases

    def _forward_batch_metal(self, x_batch):
        if metal_backend is None:
            return None
        is_available = getattr(metal_backend, "is_available", None)
        dense_forward = getattr(metal_backend, "dense_forward", None)
        if not callable(is_available) or not callable(dense_forward) or not is_available():
            return None
        try:
            outputs = dense_forward(
                self.weights.astype(np.float32, copy=False),
                self.biases.astype(np.float32, copy=False),
                x_batch.astype(np.float32, copy=False),
            )
            return np.asarray(outputs, dtype=float)
        except Exception:
            return None

    def _sync_neurons_from_matrix(self):
        for neuron, weights, bias in zip(self.neurons, self.weights, self.biases):
            neuron.weights = weights.copy()
            neuron.bias = float(bias)

    def forward(self, x, screen=None):
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            self.last_input = x_array
            outputs = self._forward_batch_metal(x_array[np.newaxis, :])
            if outputs is None:
                outputs = self._forward_batch_numpy(x_array[np.newaxis, :])
            outputs = np.asarray(outputs[0], dtype=float)
        elif x_array.ndim == 2:
            self.last_input = x_array
            outputs = self._forward_batch_metal(x_array)
            if outputs is None:
                outputs = self._forward_batch_numpy(x_array)
            outputs = np.asarray(outputs, dtype=float)
        else:
            raise ValueError("Layer.forward expects a 1D or 2D input array.")

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
        if metal_backend is not None:
            dense_backward = getattr(metal_backend, "dense_backward", None)
            if callable(dense_backward) and self.last_input is not None:
                try:
                    grad_input, grad_weights, grad_biases = dense_backward(
                        self.weights.astype(np.float32, copy=False),
                        np.asarray(self.last_input, dtype=np.float32),
                        grad_array.astype(np.float32, copy=False),
                    )
                    self.grad_weights = np.asarray(grad_weights, dtype=float)
                    self.grad_biases = np.asarray(grad_biases, dtype=float)
                    return np.asarray(grad_input, dtype=float)
                except Exception:
                    pass
        self.grad_weights = np.outer(grad_array, self.last_input)
        self.grad_biases = grad_array
        return self.weights.T @ grad_array

    def update_params(self, learning_rate):
        if metal_backend is not None:
            dense_apply_update = getattr(metal_backend, "dense_apply_update", None)
            if callable(dense_apply_update):
                try:
                    self.weights, self.biases = dense_apply_update(
                        self.weights.astype(np.float32, copy=False),
                        self.biases.astype(np.float32, copy=False),
                        np.asarray(self.grad_weights, dtype=np.float32),
                        np.asarray(self.grad_biases, dtype=np.float32),
                        float(learning_rate),
                    )
                    self.weights = np.asarray(self.weights, dtype=float)
                    self.biases = np.asarray(self.biases, dtype=float)
                    self._sync_neurons_from_matrix()
                    return
                except Exception:
                    pass
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

