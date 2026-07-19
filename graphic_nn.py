import pygame
import warnings
import neuralnet
import random

nn = neuralnet.NeuralNetwork()
nn.add_layer(neuralnet.Neuron(128))
nn.add_layer(neuralnet.Neuron(16))
nn.add_layer(neuralnet.Neuron(4))  # Example: adding another neuron with 4 inputs

def init_font():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pygame.font.init()
        return pygame.font.Font(None, 36)
    except Exception:
        return None


def display_entire_network(screen, network, x, font, start_x=50, start_y=50, layer_spacing=100, neuron_spacing=100):
    current_x = start_x
    current_y = start_y
    for layer in network.layers:
        if isinstance(layer, neuralnet.Neuron):
            neurons = [layer]
        else:
            neurons = layer.neurons
        for neuron in neurons:
            output = neuron.forward(x, screen=None)
            neuron_rect = pygame.Rect(current_x, current_y, 50, 50)
            pygame.draw.ellipse(screen, (255, 255, 255), neuron_rect)
            if font is not None:
                neuron_output_surface = font.render(f"{output:.2f}", True, (255, 255, 255))
                screen.blit(neuron_output_surface, (current_x, current_y))
            else:
                print(f"Neuron output at ({current_x}, {current_y}): {output:.2f}")
            current_y += neuron_spacing
        current_x += layer_spacing
        current_y = start_y


def main():
    pygame.init()
    font = init_font()
    if font is None:
        print("Warning: pygame font module is unavailable in this environment.")

    screen = pygame.display.set_mode((800, 600))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        random_input = [random.random() for _ in range(128)]
        display_entire_network(screen, nn, random_input, font)  # Example input
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()