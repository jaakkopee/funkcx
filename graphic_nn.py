import pygame
import warnings
import neuralnet
import random
import time 

nn = neuralnet.create_network()
for _ in range(10):
    nn.add_layer(neuralnet.create_layer(27, 27))

LETTER_TEMPERATURE = 0.8

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
            letter = neuron.output_to_letter(output, temperature=LETTER_TEMPERATURE)
            neuron_rect = pygame.Rect(current_x, current_y, 50, 50)
            pygame.draw.ellipse(screen, (max(0, min(127, int(output * 127))), max(0, min(127, int(output * 127))), max(0, min(127, int(output * 127)))), neuron_rect)
            if font is not None:
                neuron_output_surface = font.render(letter, True, (255, 0, 127))
                screen.blit(neuron_output_surface, (current_x, current_y))
            else:
                print(f"Neuron output at ({current_x}, {current_y}): {output:.2f} -> {letter}")
            current_y += neuron_spacing
        current_x += layer_spacing
        current_y = start_y
    time.sleep(0.3)


def main():
    pygame.init()
    font = init_font()
    if font is None:
        print("Warning: pygame font module is unavailable in this environment.")

    screen = pygame.display.set_mode((1000, 800))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        alphabet_input = [random.uniform(-1.0, 1.0) for _ in range(27)]
        display_entire_network(screen, nn, alphabet_input, font)  # Example input
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()