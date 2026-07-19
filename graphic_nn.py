import pygame
import warnings
import random
import os

from language_model import CharLanguageModel


MODEL_PATH = "char_lm_weights.json"
LM_TEMPERATURE = 0.9
TOP_K = 5

CORPUS = (
    "hello neural network. "
    "this tiny language model is built from a sketch network. "
    "it learns next character probabilities and paints predictions on screen. "
)


def load_or_train_char_model():
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from {MODEL_PATH} ...")
            return CharLanguageModel.load(MODEL_PATH)
        except Exception as exc:
            print(f"Failed to load saved model: {exc}")

    print("Training new character model ...")
    model = CharLanguageModel(CORPUS)
    model.train(epochs=100, learning_rate=0.12, print_every=25)
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    return model

def init_font():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pygame.font.init()
        return pygame.font.Font(None, 36)
    except Exception:
        return None


def display_distribution(screen, font, model, current_char, temperature):
    logits, probs = model.predict_logits_and_probs(current_char, temperature=temperature)

    cols = 8
    cell_w = 115
    cell_h = 70
    start_x = 30
    start_y = 130

    for idx, token in enumerate(model.vocab):
        row = idx // cols
        col = idx % cols
        x = start_x + col * cell_w
        y = start_y + row * cell_h

        prob = probs[idx]
        intensity = max(30, min(255, int(prob * 1200)))
        color = (intensity, 40, max(10, 220 - intensity // 2))

        rect = pygame.Rect(x, y, 54, 54)
        pygame.draw.ellipse(screen, color, rect)

        if font is not None:
            label = token if token != " " else "<sp>"
            text = font.render(label, True, (255, 255, 255))
            screen.blit(text, (x + 60, y + 10))
            score_text = font.render(f"{prob:.2f}", True, (200, 200, 200))
            screen.blit(score_text, (x + 60, y + 30))

    return logits, probs


def draw_text_lines(screen, font, lines, x, y, line_gap=28, color=(255, 220, 120)):
    if font is None:
        return
    for i, line in enumerate(lines):
        surface = font.render(line, True, color)
        screen.blit(surface, (x, y + i * line_gap))


def main():
    pygame.init()
    font = init_font()
    if font is None:
        print("Warning: pygame font module is unavailable in this environment.")

    model = load_or_train_char_model()

    screen = pygame.display.set_mode((1080, 820))
    pygame.display.set_caption("Sketch Network Language Model Visualizer")

    clock = pygame.time.Clock()
    current_char = random.choice([c for c in model.vocab if c.strip()])
    generated = current_char
    frame_index = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        top_k = model.top_k_predictions(current_char, k=TOP_K, temperature=LM_TEMPERATURE)
        _, probs = display_distribution(screen, font, model, current_char, LM_TEMPERATURE)

        if font is not None:
            draw_text_lines(
                screen,
                font,
                [
                    f"Current char: {repr(current_char)}",
                    f"Temperature: {LM_TEMPERATURE}",
                ],
                x=30,
                y=20,
            )

            top_lines = [f"Top {TOP_K} predictions:"]
            top_lines.extend([f"{tok!r}: {prob:.3f}" for tok, prob in top_k])
            draw_text_lines(screen, font, top_lines, x=720, y=20)

            tail = generated[-120:]
            draw_text_lines(screen, font, [f"Generated: {tail}"], x=30, y=760, line_gap=24)
        elif frame_index % 20 == 0:
            print(f"current={current_char!r} top={top_k}")

        next_char = model.sample_next(current_char, temperature=LM_TEMPERATURE)
        generated += next_char
        current_char = next_char

        pygame.display.flip()
        frame_index += 1
        clock.tick(4)

    pygame.quit()

if __name__ == "__main__":
    main()