import argparse
import os
import pygame
import random
import warnings

from gematria_engine import GematriaEngine
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
TOP_PANEL_HEIGHT = 160
BOTTOM_PANEL_HEIGHT = 60
PROMPT_BOX_HEIGHT = 46
GENERATED_PREVIEW_MARGIN = 220
GENERATED_PREVIEW_WORDS = 7
NO_LEADING_SPACE_TOKENS = {".", ",", "!", "?", ";", ":"}


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
    parser.add_argument("--generation-log-file", default="")
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


def format_reduction_path(path):
    return "->".join(str(value) for value in path)


def format_token_with_reduction(token, gematria_engine, reduction_cache):
    if token in NO_LEADING_SPACE_TOKENS:
        return token
    if token in reduction_cache:
        return reduction_cache[token]

    # Keep the original token text, but normalize for mapping lookup.
    lookup_token = token.upper()
    path = gematria_engine.numerological_reduction_path(lookup_token)
    annotated = f"{token}({format_reduction_path(path)})"
    reduction_cache[token] = annotated
    return annotated


def annotate_tokens(tokens, gematria_engine, reduction_cache):
    return [format_token_with_reduction(token, gematria_engine, reduction_cache) for token in tokens]


def fit_token_tail(font, tokens, max_width, min_tokens=GENERATED_PREVIEW_WORDS):
    if font is None or not tokens:
        return tokens[:]

    min_tokens = max(1, int(min_tokens))
    if len(tokens) <= min_tokens:
        return tokens[:]

    tail_tokens = tokens[-min_tokens:]
    preview = compact_tokens(tail_tokens)
    if font.size(preview)[0] <= max_width:
        return tail_tokens
    return tail_tokens


def fit_annotated_token_tail(font, tokens, max_width, gematria_engine, reduction_cache, min_tokens=2):
    if font is None or not tokens:
        return annotate_tokens(tokens, gematria_engine, reduction_cache)

    min_tokens = max(1, int(min_tokens))
    tail_tokens = tokens[-max(min_tokens, GENERATED_PREVIEW_WORDS):]
    annotated_tail = annotate_tokens(tail_tokens, gematria_engine, reduction_cache)

    while len(annotated_tail) > min_tokens and font.size(compact_tokens(annotated_tail))[0] > max_width:
        tail_tokens = tail_tokens[1:]
        annotated_tail = annotate_tokens(tail_tokens, gematria_engine, reduction_cache)

    return annotated_tail


def append_token_to_log(log_file, token, line_has_content, split_on_period=False):
    if token in NO_LEADING_SPACE_TOKENS or not line_has_content:
        log_file.write(token)
    else:
        log_file.write(f" {token}")
    if split_on_period and token == ".":
        log_file.write("\n")
        return False
    return True


def append_token_with_reduction_to_log(
    log_file,
    token,
    line_has_content,
    gematria_engine,
    reduction_cache,
    split_on_period=False,
):
    printable = format_token_with_reduction(token, gematria_engine, reduction_cache)
    if token in NO_LEADING_SPACE_TOKENS or not line_has_content:
        log_file.write(printable)
    else:
        log_file.write(f" {printable}")
    if split_on_period and token == ".":
        log_file.write("\n")
        return False
    return True


def append_tokens_to_log(log_file, tokens):
    line_has_content = False
    for token in tokens:
        line_has_content = append_token_to_log(log_file, token, line_has_content)
    return line_has_content


def append_tokens_with_reduction_to_log(log_file, tokens, gematria_engine, reduction_cache):
    line_has_content = False
    for token in tokens:
        line_has_content = append_token_with_reduction_to_log(
            log_file,
            token,
            line_has_content,
            gematria_engine,
            reduction_cache,
        )
    return line_has_content


def build_seed_state(model, prompt_text):
    seed_tokens = model.tokenize(prompt_text)
    valid_tokens = [token for token in seed_tokens if token in model.stoi]
    if not valid_tokens:
        fallback = random.choice([token for token in model.vocab if token.isalpha()])
        return [fallback], [fallback], fallback

    generated_tokens = valid_tokens[:]
    context_tokens = valid_tokens[-model.context_size:]
    current_token = valid_tokens[-1]
    return generated_tokens, context_tokens, current_token


def main():
    args = parse_args()
    generation_log = None
    log_line_has_content = False
    gematria_engine = GematriaEngine()
    reduction_cache = {}

    if args.generation_log_file:
        log_path = os.path.abspath(args.generation_log_file)
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        generation_log = open(log_path, "a", encoding="utf-8")
        print(f"Logging generated text to {log_path}")

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
    prompt_text = current_token
    input_active = True

    if generation_log is not None:
        log_line_has_content = append_tokens_with_reduction_to_log(
            generation_log,
            generated_tokens,
            gematria_engine,
            reduction_cache,
        )
        generation_log.flush()

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    width, height = screen.get_size()
                    prompt_rect = pygame.Rect(24, height - 108, width - 48, PROMPT_BOX_HEIGHT)
                    input_active = prompt_rect.collidepoint(event.pos)
                elif event.type == pygame.KEYDOWN and input_active:
                    if event.key == pygame.K_RETURN:
                        generated_tokens, context_tokens, current_token = build_seed_state(model, prompt_text)
                        if generation_log is not None:
                            generation_log.write("\n")
                            log_line_has_content = append_tokens_with_reduction_to_log(
                                generation_log,
                                generated_tokens,
                                gematria_engine,
                                reduction_cache,
                            )
                            generation_log.flush()
                    elif event.key == pygame.K_BACKSPACE:
                        prompt_text = prompt_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        prompt_text = ""
                    elif event.unicode and event.unicode.isprintable():
                        prompt_text += event.unicode

            screen.fill((0, 0, 0))
            logits, probs = model.predict_logits_and_probs_from_context(
                context_tokens,
                temperature=args.temperature,
            )
            top_k = sorted(
                ((model.itos[i], float(probs[i])) for i in range(model.vocab_size)),
                key=lambda item: item[1],
                reverse=True,
            )[: max(1, args.top_k)]

            width, height = screen.get_size()
            pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, width, TOP_PANEL_HEIGHT))
            pygame.draw.rect(
                screen,
                (0, 0, 0),
                pygame.Rect(0, height - BOTTOM_PANEL_HEIGHT, width, BOTTOM_PANEL_HEIGHT),
            )
            prompt_rect = pygame.Rect(24, height - 108, width - 48, PROMPT_BOX_HEIGHT)
            pygame.draw.rect(screen, (20, 20, 20), prompt_rect)
            pygame.draw.rect(screen, (180, 180, 180) if input_active else (90, 90, 90), prompt_rect, width=2)

            if font is not None:
                draw_text_lines(
                    screen,
                    font,
                    [
                        f"Current token: {current_token}",
                        f"Temperature: {args.temperature}",
                        f"Vocab size: {model.vocab_size}",
                        f"FPS: {max(1, args.fps)}",
                        "Prompt field active. Press Enter to reseed generation.",
                    ],
                    x=24,
                    y=20,
                )

                top_lines = [f"Top {max(1, args.top_k)} predictions:"]
                top_lines.extend(
                    [
                        f"{format_token_with_reduction(tok, gematria_engine, reduction_cache)}: {prob:.3f}"
                        for tok, prob in top_k
                    ]
                )
                draw_text_lines(screen, font, top_lines, x=660, y=20)

                preview_tokens = fit_annotated_token_tail(
                    font,
                    generated_tokens,
                    max(1, width - 48 - GENERATED_PREVIEW_MARGIN),
                    gematria_engine,
                    reduction_cache,
                )
                preview = compact_tokens(preview_tokens)
                draw_text_lines(screen, font, [f"Generated: {preview}"], x=24, y=814, line_gap=22)
                prompt_label = "Prompt / seed text:"
                prompt_display = prompt_text if prompt_text else ""
                draw_text_lines(screen, font, [prompt_label], x=24, y=height - 142, line_gap=22)
                cursor = "|" if input_active and (frame_index // 15) % 2 == 0 else ""
                prompt_surface = font.render(prompt_display + cursor, True, (235, 235, 235))
                screen.blit(prompt_surface, (prompt_rect.x + 12, prompt_rect.y + 8))
            elif frame_index % 20 == 0:
                print(f"current={current_token!r} top={top_k}")

            next_token = model.itos[model._sample_from_probs(probs)]
            generated_tokens.append(next_token)
            current_token = next_token
            context_tokens.append(next_token)
            if len(context_tokens) > model.context_size:
                context_tokens.pop(0)

            if generation_log is not None:
                log_line_has_content = append_token_with_reduction_to_log(
                    generation_log,
                    next_token,
                    log_line_has_content,
                    gematria_engine,
                    reduction_cache,
                    split_on_period=True,
                )
                generation_log.flush()

            pygame.display.flip()
            frame_index += 1
            clock.tick(max(1, args.fps))
    finally:
        if generation_log is not None:
            generation_log.write("\n")
            generation_log.close()
        pygame.quit()


if __name__ == "__main__":
    main()

