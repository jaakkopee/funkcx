import json
import math
import random
import re
import sys
from typing import Dict, List, Tuple

import numpy as np

import neuralnet


class OneStepLanguageModel:
    """One-step next-token language model built on the sketch network."""

    model_type = "base"

    def __init__(self, tokens: List[str]):
        if len(tokens) < 2:
            raise ValueError("Training tokens must contain at least 2 entries.")

        self.tokens = tokens[:]
        self.vocab = sorted(set(tokens))
        self.vocab_size = len(self.vocab)

        self.stoi: Dict[str, int] = {token: i for i, token in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: token for token, i in self.stoi.items()}

        # One linear layer maps one-hot(current token) -> logits(next token).
        self.net = neuralnet.create_network()
        self.net.add_layer(neuralnet.create_layer(self.vocab_size, self.vocab_size))

        self.pairs = np.array(
            [(self.stoi[tokens[i]], self.stoi[tokens[i + 1]]) for i in range(len(tokens) - 1)],
            dtype=np.int32,
        )

        self._one_hot_cache = np.eye(self.vocab_size, dtype=np.float32)

    def one_hot(self, index: int) -> List[float]:
        return self._one_hot_cache[index]

    @staticmethod
    def _softmax(logits: List[float], temperature: float = 1.0) -> np.ndarray:
        temp = max(1e-6, float(temperature))
        logits_array = np.asarray(logits, dtype=np.float64) / temp
        logits_array = logits_array - np.max(logits_array)
        exps = np.exp(logits_array)
        return exps / np.sum(exps)

    @staticmethod
    def _sample_from_probs(probs: List[float]) -> int:
        r = random.random()
        cdf = 0.0
        for i, p in enumerate(probs):
            cdf += p
            if r <= cdf:
                return i
        return len(probs) - 1

    @staticmethod
    def _progress_bar(progress: float, width: int = 28) -> str:
        progress = max(0.0, min(1.0, progress))
        filled = int(round(progress * width))
        filled = min(width, max(0, filled))
        empty = width - filled
        return f"[{('=' * filled)}{('-' * empty)}]"

    @staticmethod
    def _epoch_header(epoch: int, epochs: int) -> str:
        return f"epoch {epoch:4d}/{epochs} {OneStepLanguageModel._progress_bar(epoch / max(1, epochs))}"

    @staticmethod
    def _write_progress_line(text: str) -> None:
        sys.stdout.write("\r" + text + "\x1b[K")
        sys.stdout.flush()

    def _train_batch(self, input_indices: np.ndarray, target_indices: np.ndarray, learning_rate: float) -> float:
        layer = self.net.layers[0]
        x_batch = self._one_hot_cache[input_indices]
        logits = x_batch @ layer.weights.T + layer.biases

        logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits)
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        batch_size = max(1, input_indices.shape[0])
        row_indices = np.arange(batch_size)
        batch_loss = -np.log(np.maximum(1e-12, probs[row_indices, target_indices])).mean()

        grad_logits = probs
        grad_logits[row_indices, target_indices] -= 1.0
        grad_logits /= batch_size

        grad_weights = grad_logits.T @ x_batch
        grad_biases = grad_logits.sum(axis=0)

        layer.weights -= learning_rate * grad_weights
        layer.biases -= learning_rate * grad_biases
        layer._sync_neurons_from_matrix()

        return float(batch_loss)

    def train(
        self,
        epochs: int = 80,
        learning_rate: float = 0.1,
        print_every: int = 10,
        progress_chunks: int = 10,
        batch_size: int = 64,
    ) -> None:
        total_pairs = len(self.pairs)
        if total_pairs == 0:
            raise ValueError("Training data does not contain any adjacent token pairs.")

        if total_pairs >= 200:
            print(
                f"Training {self.model_type} model on {total_pairs} pairs "
                f"({len(self.tokens)} tokens, vocab {self.vocab_size}) for {epochs} epochs"
            )
            self._write_progress_line("Progress: [----------------------------] 0.0%")

        batch_size = max(1, int(batch_size))
        chunk_count = max(1, int(progress_chunks))
        total_batches = max(1, (total_pairs + batch_size - 1) // batch_size)
        progress_stride = max(1, total_batches // chunk_count)

        for epoch in range(1, epochs + 1):
            np.random.shuffle(self.pairs)
            total_loss = 0.0
            self._write_progress_line(f"{self._epoch_header(epoch, epochs)} start")

            for batch_number, start in enumerate(range(0, total_pairs, batch_size), start=1):
                batch_pairs = self.pairs[start:start + batch_size]
                input_indices = batch_pairs[:, 0]
                target_indices = batch_pairs[:, 1]
                batch_loss = self._train_batch(input_indices, target_indices, learning_rate)
                total_loss += batch_loss * len(batch_pairs)

                if total_pairs >= 200 and (batch_number % progress_stride == 0 or start + batch_size >= total_pairs):
                    seen = min(total_pairs, start + len(batch_pairs))
                    pct = (seen / total_pairs) * 100.0
                    running_loss = total_loss / seen
                    bar = self._progress_bar(seen / total_pairs)
                    self._write_progress_line(
                        f"epoch {epoch:4d}/{epochs} {self._progress_bar(epoch / epochs)} | sub {bar} {pct:5.1f}% "
                        f"| {seen:6d}/{total_pairs:<6d} | batch loss {batch_loss:.4f} | running loss {running_loss:.4f}"
                    )

            if epoch % max(1, print_every) == 0 or epoch == 1 or epoch == epochs:
                avg_loss = total_loss / len(self.pairs)
                self._write_progress_line(
                    f"epoch {epoch:4d}/{epochs} {self._progress_bar(epoch / epochs)} | sub {self._progress_bar(1.0)} 100.0% "
                    f"| avg loss {avg_loss:.4f}"
                )

        sys.stdout.write("\n")
        sys.stdout.flush()

    def predict_logits_and_probs(self, current_token: str, temperature: float = 1.0) -> Tuple[List[float], List[float]]:
        if current_token not in self.stoi:
            raise ValueError(f"Token {current_token!r} not in vocabulary.")

        x = self._one_hot_cache[self.stoi[current_token]]
        logits = self.net.forward(x)
        probs = self._softmax(logits, temperature=temperature)
        return logits, probs

    def predict_next_distribution(self, current_token: str, temperature: float = 1.0) -> List[Tuple[str, float]]:
        _, probs = self.predict_logits_and_probs(current_token, temperature=temperature)
        return [(self.itos[i], float(probs[i])) for i in range(self.vocab_size)]

    def top_k_predictions(self, current_token: str, k: int = 5, temperature: float = 1.0) -> List[Tuple[str, float]]:
        dist = self.predict_next_distribution(current_token, temperature=temperature)
        return sorted(dist, key=lambda item: item[1], reverse=True)[: max(1, k)]

    def sample_next(self, current_token: str, temperature: float = 1.0) -> str:
        _, probs = self.predict_logits_and_probs(current_token, temperature=temperature)
        next_idx = self._sample_from_probs(probs)
        return self.itos[next_idx]

    def _serialize(self) -> Dict[str, object]:
        layer = self.net.layers[0]
        return {
            "model_type": self.model_type,
            "tokens": self.tokens,
            "vocab": self.vocab,
            "weights": layer.weights.tolist(),
            "biases": layer.biases.tolist(),
        }

    @classmethod
    def _init_from_saved_tokens(cls, tokens: List[str]):
        model = cls.__new__(cls)
        OneStepLanguageModel.__init__(model, tokens)
        return model

    def save(self, path: str) -> None:
        payload = self._serialize()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def _from_payload(cls, payload: Dict[str, object]):
        tokens = payload["tokens"]
        model = cls._init_from_saved_tokens(tokens)

        layer = model.net.layers[0]
        saved_weights = payload["weights"]
        saved_biases = payload["biases"]

        if len(saved_weights) != len(layer.neurons):
            raise ValueError("Saved model shape does not match current network shape.")

        for i, neuron in enumerate(layer.neurons):
            neuron.weights = np.asarray(saved_weights[i], dtype=np.float64)
            neuron.bias = float(saved_biases[i])

        layer.weights = np.asarray(saved_weights, dtype=np.float64)
        layer.biases = np.asarray(saved_biases, dtype=np.float64)

        return model


class CharLanguageModel(OneStepLanguageModel):
    model_type = "char"

    def __init__(self, text: str):
        if len(text) < 2:
            raise ValueError("Training text must contain at least 2 characters.")
        self.text = text
        super().__init__(list(text))

    def generate(self, seed: str, length: int = 120, temperature: float = 1.0) -> str:
        if length <= 0:
            return seed

        if not seed:
            seed = random.choice(self.vocab)

        for ch in seed:
            if ch not in self.stoi:
                raise ValueError(f"Seed contains unknown char: {ch!r}")

        out = list(seed)
        current = out[-1]

        for _ in range(length):
            next_char = self.sample_next(current, temperature=temperature)
            out.append(next_char)
            current = next_char

        return "".join(out)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("model_type") != cls.model_type:
            raise ValueError("Saved model is not a character model.")
        model = cls._from_payload(payload)
        model.text = "".join(model.tokens)
        return model


class WordLanguageModel(OneStepLanguageModel):
    model_type = "word"

    def __init__(self, text: str):
        tokens = self.tokenize(text)
        if len(tokens) < 2:
            raise ValueError("Training text must contain at least 2 word tokens.")
        self.text = text
        super().__init__(tokens)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Keep punctuation as separate tokens so generation has sentence rhythm.
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def generate(self, seed: str, length: int = 40, temperature: float = 1.0) -> str:
        seed_tokens = self.tokenize(seed)
        if not seed_tokens:
            seed_tokens = [random.choice(self.vocab)]

        for token in seed_tokens:
            if token not in self.stoi:
                raise ValueError(f"Seed contains unknown token: {token!r}")

        out = seed_tokens[:]
        current = out[-1]

        for _ in range(length):
            next_token = self.sample_next(current, temperature=temperature)
            out.append(next_token)
            current = next_token

        # Compact punctuation spacing.
        text = " ".join(out)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("model_type") != cls.model_type:
            raise ValueError("Saved model is not a word model.")
        model = cls._from_payload(payload)
        model.text = " ".join(model.tokens)
        return model


def demo() -> None:
    corpus = (
        "hello neural network. "
        "this tiny language model is built from a sketch network. "
        "it learns next token probabilities."
    )

    print("Character model")
    char_model = CharLanguageModel(corpus)
    print(f"vocab size: {char_model.vocab_size}")
    print(f"vocab: {''.join(char_model.vocab)}")
    char_model.train(epochs=80, learning_rate=0.12, print_every=20)
    char_model.save("char_lm_weights.json")
    loaded_char = CharLanguageModel.load("char_lm_weights.json")
    print("sample t=0.8:")
    print(loaded_char.generate(seed="h", length=140, temperature=0.8))

    print("\nWord model")
    word_model = WordLanguageModel(corpus)
    print(f"vocab size: {word_model.vocab_size}")
    word_model.train(epochs=120, learning_rate=0.08, print_every=30)
    word_model.save("word_lm_weights.json")
    loaded_word = WordLanguageModel.load("word_lm_weights.json")
    print("sample t=1.0:")
    print(loaded_word.generate(seed="hello", length=28, temperature=1.0))

def demon() -> None:
    #read the model from the file
    loaded_word = WordLanguageModel.load("nsoe.json")
    print("sample t=1.0:")
    print(loaded_word.generate(seed="hello", length=28, temperature=1.0))

if __name__ == "__main__":
    demon()
