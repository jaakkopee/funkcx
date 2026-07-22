import argparse
import colorsys
import os
import pygame
import random
import warnings

from language_model import WordLanguageModel


DEFAULT_MODEL_PATH = "nsoe.json"
DEFAULT_CORPUS_PATH = "nsoe.txt"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 5
DEFAULT_CONTEXT_SIZE = 4
DEFAULT_EPOCHS = 140
DEFAULT_LEARNING_RATE = 0.08
DEFAULT_PRINT_EVERY = 10
DEFAULT_PROGRESS_CHUNKS = 12
DEFAULT_BATCH_SIZE = 8
DEFAULT_FPS = 3
MIN_ZOOM = 0.4
MAX_ZOOM = 3.5
TOP_PANEL_HEIGHT = 160
BOTTOM_PANEL_HEIGHT = 60
ARROW_PAN_SCREEN_FRACTION = 0.25


def parse_args():
    parser = argparse.ArgumentParser(description="Word LM visualizer and trainer")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--corpus-path", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--print-every", type=int, default=DEFAULT_PRINT_EVERY)
    parser.add_argument("--progress-chunks", type=int, default=DEFAULT_PROGRESS_CHUNKS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    return parser.parse_args()


def load_corpus_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def summarize_corpus(text, model):
    print(
        f"Corpus loaded: {len(text):,} characters, "
        f"{len(model.tokens):,} tokens, vocab {model.vocab_size}")
    sample = " ".join(model.tokens[:24])
    print(f"Corpus preview: {sample}")


def load_or_train_word_model(args):
    if os.path.exists(args.model_path) and not args.force_retrain:
        try:
            print(f"Loading model from {args.model_path} ...")
            model = WordLanguageModel.load(args.model_path)
            if getattr(model, "context_size", 1) >= args.context_size:
                print(
                    "Loaded existing model; training arguments are ignored unless "
                    "--force-retrain is provided."
                )
                return model
            print("Saved model context is too small, retraining with larger context...")
        except Exception as exc:
            print(f"Failed to load saved model: {exc}")

    print("Training new word model ...")
    corpus = load_corpus_text(args.corpus_path)
    model = WordLanguageModel(corpus, context_size=args.context_size)
    summarize_corpus(corpus, model)
    print(
        "Training args: "
        f"context={args.context_size}, epochs={args.epochs}, lr={args.learning_rate}, "
        f"print_every={args.print_every}, progress_chunks={args.progress_chunks}, "
        f"batch_size={args.batch_size}"
    )
    model.train(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        print_every=args.print_every,
        progress_chunks=args.progress_chunks,
        batch_size=args.batch_size,
    )
    model.save(args.model_path)
    print(f"Saved trained model to {args.model_path}")
    return model

def init_font():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pygame.font.init()
        return pygame.font.Font(None, 36)
    except Exception:
        return None


def world_to_screen(x, y, camera):
    zoom = camera["zoom"]
    sx = int(x * zoom + camera["offset_x"])
    sy = int(y * zoom + camera["offset_y"])
    return sx, sy


def zoom_at(camera, factor, mouse_pos):
    old_zoom = camera["zoom"]
    new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, old_zoom * factor))
    if abs(new_zoom - old_zoom) < 1e-9:
        return

    mx, my = mouse_pos
    world_x = (mx - camera["offset_x"]) / old_zoom
    world_y = (my - camera["offset_y"]) / old_zoom

    camera["zoom"] = new_zoom
    camera["offset_x"] = mx - world_x * new_zoom
    camera["offset_y"] = my - world_y * new_zoom


def activation_to_color(activation):
    # Special cases first: very low activity is black, strong firing is white.
    if activation <= 0.02:
        return (0, 0, 0)
    if activation >= 0.98:
        return (255, 255, 255)

    # Map activation to a visible spectrum via HSV hue.
    t = max(0.0, min(1.0, float(activation)))
    hue = (1.0 - t) * 0.66  # blue -> red
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def display_distribution(screen, font, model, context_tokens, temperature, camera):
    logits, probs = model.predict_logits_and_probs_from_context(context_tokens, temperature=temperature)
    probs_array = list(float(p) for p in probs)
    p_min = min(probs_array)
    p_max = max(probs_array)
    p_span = max(1e-12, p_max - p_min)

    cols = 5
    cell_w = 200
    cell_h = 58
    start_x = 24
    start_y = 132

    for idx, token in enumerate(model.vocab):
        row = idx // cols
        col = idx % cols
        wx = start_x + col * cell_w
        wy = start_y + row * cell_h
        x, y = world_to_screen(wx, wy, camera)

        prob = probs[idx]
        normalized_activation = (float(prob) - p_min) / p_span
        color = activation_to_color(normalized_activation)

        node_size = max(6, int(18 * camera["zoom"]))
        rect = pygame.Rect(x, y, node_size, node_size)
        pygame.draw.ellipse(screen, color, rect)

        if font is not None:
            label = token
            if len(label) > 14:
                label = label[:11] + "..."
            token_text = font.render(label, True, (245, 245, 245))
            screen.blit(token_text, (x + node_size + 10, y - 2))
            prob_text = font.render(f"{prob:.3f}", True, (180, 180, 180))
            screen.blit(prob_text, (x + node_size + 120, y - 2))

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
    args = parse_args()

    pygame.init()
    font = init_font()
    if font is None:
        print("Warning: pygame font module is unavailable in this environment.")

    model = load_or_train_word_model(args)
    if args.train_only:
        print("Training complete (--train-only), exiting without launching GUI.")
        return

    screen = pygame.display.set_mode((1060, 860))
    pygame.display.set_caption("Sketch Network Word-LM Visualizer")

    clock = pygame.time.Clock()
    current_token = random.choice([token for token in model.vocab if token.isalpha()])
    generated_tokens = [current_token]
    context_tokens = [current_token]
    frame_index = 0
    camera = {"zoom": 1.0, "offset_x": 0.0, "offset_y": 0.0}
    panning = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    panning = True
                elif event.button == 4:
                    zoom_at(camera, 1.12, event.pos)
                elif event.button == 5:
                    zoom_at(camera, 1 / 1.12, event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    panning = False
            elif event.type == pygame.MOUSEMOTION and panning:
                dx, dy = event.rel
                camera["offset_x"] += dx
                camera["offset_y"] += dy
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                camera["zoom"] = 1.0
                camera["offset_x"] = 0.0
                camera["offset_y"] = 0.0

        keys = pygame.key.get_pressed()
        pan_x = 0.0
        pan_y = 0.0
        if keys[pygame.K_LEFT]:
            pan_x += 1.0
        if keys[pygame.K_RIGHT]:
            pan_x -= 1.0
        if keys[pygame.K_UP]:
            pan_y += 1.0
        if keys[pygame.K_DOWN]:
            pan_y -= 1.0

        if pan_x != 0.0 or pan_y != 0.0:
            width, height = screen.get_size()
            if pan_x != 0.0 and pan_y != 0.0:
                scale = 0.7071
                pan_x *= scale
                pan_y *= scale
            camera["offset_x"] += pan_x * (width * ARROW_PAN_SCREEN_FRACTION)
            camera["offset_y"] += pan_y * (height * ARROW_PAN_SCREEN_FRACTION)

        screen.fill((0, 0, 0))
        top_k = model.top_k_predictions_from_context(
            context_tokens,
            k=max(1, args.top_k),
            temperature=args.temperature,
        )
        display_distribution(screen, font, model, context_tokens, args.temperature, camera)

        # Cover network visuals under static text areas.
        width, height = screen.get_size()
        pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, width, TOP_PANEL_HEIGHT))
        pygame.draw.rect(
            screen,
            (0, 0, 0),
            pygame.Rect(0, height - BOTTOM_PANEL_HEIGHT, width, BOTTOM_PANEL_HEIGHT),
        )

        if font is not None:
            draw_text_lines(
                screen,
                font,
                [
                    f"Current token: {current_token}",
                    f"Temperature: {args.temperature}",
                    f"Vocab size: {model.vocab_size}",
                    f"Zoom: {camera['zoom']:.2f}",
                    "Pan: right-drag or arrows | Zoom: wheel | Reset: R",
                ],
                x=24,
                y=20,
            )

            top_lines = [f"Top {max(1, args.top_k)} predictions:"]
            top_lines.extend([f"{tok}: {prob:.3f}" for tok, prob in top_k])
            draw_text_lines(screen, font, top_lines, x=660, y=20)

            preview = compact_tokens(generated_tokens[-45:])
            draw_text_lines(screen, font, [f"Generated: {preview}"], x=24, y=814, line_gap=22)
        elif frame_index % 20 == 0:
            print(f"current={current_token!r} top={top_k}")

        next_token = model.sample_next_from_context(context_tokens, temperature=args.temperature)
        generated_tokens.append(next_token)
        current_token = next_token
        context_tokens.append(next_token)
        if len(context_tokens) > model.context_size:
            context_tokens.pop(0)

        pygame.display.flip()
        frame_index += 1
        clock.tick(max(1, args.fps))

    pygame.quit()


if __name__ == "__main__":
    main()

