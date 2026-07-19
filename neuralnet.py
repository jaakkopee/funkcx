import pygame


class NeuralNet:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_params(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, "update_params"):
                layer.update_params(learning_rate)
                
class Neuron:
    def __init__(self, input_size):
        self.weights = [0.0] * input_size
        self.bias = 0.0

    def forward(self, x):
        self.last_input = x
        return sum(w * i for w, i in zip(self.weights, x)) + self.bias

    def backward(self, grad):
        self.grad_weights = [grad * i for i in self.last_input]
        self.grad_bias = grad
        return [grad * w for w in self.weights]

    def update_params(self, learning_rate):
        self.weights = [w - learning_rate * gw for w, gw in zip(self.weights, self.grad_weights)]
        self.bias -= learning_rate * self.grad_bias

class Layer:
    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, x):
        self.last_input = x
        return [neuron.forward(x) for neuron in self.neurons]

    def backward(self, grad):
        grads = [neuron.backward(g) for neuron, g in zip(self.neurons, grad)]
        return [sum(g[i] for g in grads) for i in range(len(grads[0]))]

    def update_params(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_params(learning_rate)


def create_neural_net():
    return NeuralNet()

def create_layer(input_size, output_size):
    return Layer(input_size, output_size)

def create_neuron(input_size):
    return Neuron(input_size)

def main():
    nn = create_neural_net()
    layer1 = create_layer(128, 8)
    layer2 = create_layer(8, 4)
    layer3 = create_layer(4, 1)
    nn.add_layer(layer1)
    nn.add_layer(layer2)
    nn.add_layer(layer3)
    
    x = [1.0] * 128
    output = nn.forward(x)
    print("Forward output:", output)
    
    grad = [0.1]  # Adjusted to match the output size of the last layer
    nn.backward(grad)
    nn.update_params(0.01)
    print("Updated parameters after backward pass.")
    output_after_update = nn.forward(x)
    print("Forward output after parameter update:", output_after_update)
    
if __name__ == "__main__":
    main()