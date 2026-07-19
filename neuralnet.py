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
    def forward(self, x, screen=None):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update_params(self, learning_rate):
        pass

class Neuron:
    def __init__(self, input_size):
        self.weights = [0.0] * input_size
        self.bias = 0.0

    def forward(self, x, screen=None):
        self.last_input = x
        self.last_output = sum(w * i for w, i in zip(self.weights, x)) + self.bias
        if screen is not None:
            screen.display(self.last_output)
        return self.last_output

    def backward(self, grad):
        self.grad_weights = [grad * i for i in self.last_input]
        self.grad_bias = grad
        return [grad * w for w in self.weights]

    def update_params(self, learning_rate):
        self.weights = [w - learning_rate * gw for w, gw in zip(self.weights, self.grad_weights)]
        self.bias -= learning_rate * self.grad_bias


