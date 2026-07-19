import pygame
import warnings
import random
import os

from language_model import WordLanguageModel


MODEL_PATH = "nsoe.json"
CORPUS_PATH = "nsoe.txt"
LM_TEMPERATURE = 0.9
TOP_K = 5


def load_corpus_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def summarize_corpus(text, model):
    print(
        f"Corpus loaded: {len(text):,} characters, "
        f"{len(model.tokens):,} tokens, vocab {model.vocab_size}")
    sample = " ".join(model.tokens[:24])
    print(f"Corpus preview: {sample}")


def load_or_train_word_model():
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from {MODEL_PATH} ...")
            return WordLanguageModel.load(MODEL_PATH)
        except Exception as exc:
            print(f"Failed to load saved model: {exc}")

    print("Training new word model ...")
    corpus = load_corpus_text(CORPUS_PATH)
    model = WordLanguageModel(corpus)
    summarize_corpus(corpus, model)
    model.train(epochs=140, learning_rate=0.08, print_every=10, progress_chunks=12)
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


def display_distribution(screen, font, model, current_token, temperature):
    logits, probs = model.predict_logits_and_probs(current_token, temperature=temperature)

    cols = 5
    cell_w = 200
    cell_h = 58
    start_x = 24
    start_y = 132

    for idx, token in enumerate(model.vocab):
        row = idx // cols
        col = idx % cols
        x = start_x + col * cell_w
        y = start_y + row * cell_h

        prob = probs[idx]
        intensity = max(28, min(255, int(prob * 1500)))
        color = (intensity, 70, max(25, 200 - intensity // 3))

        rect = pygame.Rect(x, y, 18, 18)
        pygame.draw.ellipse(screen, color, rect)

        if font is not None:
            label = token
            if len(label) > 14:
                label = label[:11] + "..."
            token_text = font.render(label, True, (245, 245, 245))
            screen.blit(token_text, (x + 28, y - 2))
            prob_text = font.render(f"{prob:.3f}", True, (180, 180, 180))
            screen.blit(prob_text, (x + 138, y - 2))

    return logits, probs


def draw_text_lines(screen, font, lines, x, y, line_gap=24, color=(255, 220, 120)):
    if font is None:
        return
    for i, line in enumerate(lines):
        surface = font.render(line, True, color)
        screen.blit(surface, (x, y + i * line_gap))


def compact_tokens(tokens):
    text = " ".join(tokens)
    for punct in [".", ",", "!", "?", ";", ":"]:
        text = text.replace(f" {punct}", punct)
    return text


def main():
    pygame.init()
    font = init_font()
    if font is None:
        print("Warning: pygame font module is unavailable in this environment.")

    model = load_or_train_word_model()
    screen = pygame.display.set_mode((1060, 860))
    pygame.display.set_caption("Sketch Network Word-LM Visualizer")

    clock = pygame.time.Clock()
    current_token = random.choice([token for token in model.vocab if token.isalpha()])
    generated_tokens = [current_token]
    frame_index = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        top_k = model.top_k_predictions(current_token, k=TOP_K, temperature=LM_TEMPERATURE)
        display_distribution(screen, font, model, current_token, LM_TEMPERATURE)

        if font is not None:
            draw_text_lines(
                screen,
                font,
                [
                    f"Current token: {current_token}",
                    f"Temperature: {LM_TEMPERATURE}",
                    f"Vocab size: {model.vocab_size}",
                ],
                x=24,
                y=20,
            )

            top_lines = [f"Top {TOP_K} predictions:"]
            top_lines.extend([f"{tok}: {prob:.3f}" for tok, prob in top_k])
            draw_text_lines(screen, font, top_lines, x=660, y=20)

            preview = compact_tokens(generated_tokens[-45:])
            draw_text_lines(screen, font, [f"Generated: {preview}"], x=24, y=814, line_gap=22)
        elif frame_index % 20 == 0:
            print(f"current={current_token!r} top={top_k}")

        next_token = model.sample_next(current_token, temperature=LM_TEMPERATURE)
        generated_tokens.append(next_token)
        current_token = next_token

        pygame.display.flip()
        frame_index += 1
        clock.tick(3)

    pygame.quit()


if __name__ == "__main__":
    main()

