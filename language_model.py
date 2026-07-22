import json
import math
import random
import re
import sys
from typing import Dict, List, Tuple

import numpy as np

import neuralnet

try:
    import metal_backend
except Exception:
    metal_backend = None

try:
    import torch_backend
except Exception:
    torch_backend = None


METAL_TILE_SIZE = 16
MAX_METAL_LAYER_BYTES = 64 * 1024 * 1024
ONE_HOT_CACHE_MAX_VOCAB = 4096


class OneStepLanguageModel:
    """Next-token language model with configurable context window."""

    model_type = "base"

    def __init__(self, tokens: List[str], context_size: int = 1):
        if len(tokens) < 2:
            raise ValueError("Training tokens must contain at least 2 entries.")

        self.context_size = max(1, int(context_size))
        self.tokens = tokens[:]
        self.vocab = sorted(set(tokens))
        self.vocab_size = len(self.vocab)

        self.stoi: Dict[str, int] = {token: i for i, token in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: token for token, i in self.stoi.items()}

        # A linear layer maps concatenated one-hot context -> logits(next token).
        self.net = neuralnet.create_network()
        self.net.add_layer(
            neuralnet.create_layer(self.vocab_size, self.vocab_size * self.context_size)
        )

        token_ids = np.array([self.stoi[token] for token in tokens], dtype=np.int32)
        sample_count = len(token_ids) - 1
        self.contexts = np.full((sample_count, self.context_size), -1, dtype=np.int32)
        self.targets = np.empty(sample_count, dtype=np.int32)

        for sample_idx in range(sample_count):
            token_pos = sample_idx + 1
            start_pos = max(0, token_pos - self.context_size)
            history = token_ids[start_pos:token_pos]
            if history.size > 0:
                self.contexts[sample_idx, -history.size:] = history
            self.targets[sample_idx] = token_ids[token_pos]

        self.sample_count = sample_count

        if self.vocab_size <= ONE_HOT_CACHE_MAX_VOCAB:
            self._one_hot_cache = np.eye(self.vocab_size, dtype=np.float32)
        else:
            self._one_hot_cache = None

    def one_hot(self, index: int) -> List[float]:
        if self._one_hot_cache is not None:
            return self._one_hot_cache[index]
        vec = np.zeros((self.vocab_size,), dtype=np.float32)
        vec[index] = 1.0
        return vec

    def _context_indices_to_input(self, context_indices: np.ndarray) -> np.ndarray:
        context_input = np.zeros((self.context_size, self.vocab_size), dtype=np.float32)
        valid = context_indices >= 0
        if np.any(valid):
            if self._one_hot_cache is not None:
                context_input[valid] = self._one_hot_cache[context_indices[valid]]
            else:
                row_indices = np.where(valid)[0]
                token_indices = context_indices[row_indices]
                context_input[row_indices, token_indices] = 1.0
        return context_input.reshape(self.context_size * self.vocab_size)

    def _context_batch_to_inputs(self, context_batch: np.ndarray) -> np.ndarray:
        batch_size = context_batch.shape[0]
        context_input = np.zeros((batch_size, self.context_size, self.vocab_size), dtype=np.float32)
        valid = context_batch >= 0
        if np.any(valid):
            if self._one_hot_cache is not None:
                context_input[valid] = self._one_hot_cache[context_batch[valid]]
            else:
                batch_rows, ctx_rows = np.where(valid)
                token_indices = context_batch[batch_rows, ctx_rows]
                context_input[batch_rows, ctx_rows, token_indices] = 1.0
        return context_input.reshape(batch_size, self.context_size * self.vocab_size)

    def _context_tokens_to_indices(self, context_tokens: List[str]) -> np.ndarray:
        context_indices = np.full((self.context_size,), -1, dtype=np.int32)
        tail_tokens = context_tokens[-self.context_size:]
        for i, token in enumerate(tail_tokens, start=self.context_size - len(tail_tokens)):
            if token not in self.stoi:
                raise ValueError(f"Token {token!r} not in vocabulary.")
            context_indices[i] = self.stoi[token]
        return context_indices

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

    @staticmethod
    def _print_progress_line(text: str) -> None:
        print(text, flush=True)

    def _train_batch(
        self,
        context_batch: np.ndarray,
        target_indices: np.ndarray,
        learning_rate: float,
        torch_trainer=None,
    ) -> Tuple[float, str]:
        layer = self.net.layers[0]
        x_batch = self._context_batch_to_inputs(context_batch)
        batch_size = max(1, context_batch.shape[0])

        if torch_trainer is not None:
            batch_loss = torch_trainer.train_batch(
                inputs=x_batch.astype(np.float32, copy=False),
                target_indices=np.asarray(target_indices, dtype=np.int64),
            )
            return float(batch_loss), torch_trainer.backend_name

        if torch_backend is not None:
            train_dense_batch = getattr(torch_backend, "train_dense_batch", None)
            if callable(train_dense_batch):
                try:
                    updated_weights, updated_biases, batch_loss, device_name = train_dense_batch(
                        layer.weights.astype(np.float32, copy=False),
                        layer.biases.astype(np.float32, copy=False),
                        x_batch.astype(np.float32, copy=False),
                        np.asarray(target_indices, dtype=np.int64),
                        float(learning_rate),
                    )
                    layer.weights = np.asarray(updated_weights, dtype=float)
                    layer.biases = np.asarray(updated_biases, dtype=float)
                    layer._sync_neurons_from_matrix()
                    return float(batch_loss), f"torch ({device_name})"
                except Exception:
                    pass

        use_metal = (
            metal_backend is not None
            and layer.weights.nbytes <= MAX_METAL_LAYER_BYTES
            and layer.biases.nbytes <= MAX_METAL_LAYER_BYTES
            and x_batch.nbytes <= MAX_METAL_LAYER_BYTES
        )

        if use_metal:
            dense_forward_tile = getattr(metal_backend, "dense_forward_tile", None)
            batch_grad_params_tile = getattr(metal_backend, "dense_batch_grad_params_tile", None)
            dense_apply_update_tile = getattr(metal_backend, "dense_apply_update_tile", None)
            if callable(dense_forward_tile) and callable(batch_grad_params_tile) and callable(dense_apply_update_tile):
                tile_size = max(1, int(METAL_TILE_SIZE))
                logits_tiles = []
                grad_weights_tiles = []
                grad_biases_tiles = []
                for tile_start in range(0, layer.weights.shape[0], tile_size):
                    tile_end = min(layer.weights.shape[0], tile_start + tile_size)
                    weights_tile = layer.weights[tile_start:tile_end]
                    biases_tile = layer.biases[tile_start:tile_end]
                    tile_logits = dense_forward_tile(
                        weights_tile.astype(np.float32, copy=False),
                        biases_tile.astype(np.float32, copy=False),
                        x_batch.astype(np.float32, copy=False),
                        int(tile_start),
                    )
                    logits_tiles.append(np.asarray(tile_logits, dtype=np.float64))

                logits = np.concatenate(logits_tiles, axis=1)
                logits = logits - np.max(logits, axis=1, keepdims=True)
                exps = np.exp(logits)
                probs = exps / np.sum(exps, axis=1, keepdims=True)

                row_indices = np.arange(batch_size)
                batch_loss = -np.log(np.maximum(1e-12, probs[row_indices, target_indices])).mean()

                grad_logits = probs
                grad_logits[row_indices, target_indices] -= 1.0
                grad_logits /= batch_size

                for tile_start in range(0, layer.weights.shape[0], tile_size):
                    tile_end = min(layer.weights.shape[0], tile_start + tile_size)
                    tile_grad_logits = grad_logits[:, tile_start:tile_end].astype(np.float32, copy=False)
                    tile_grad_weights, tile_grad_biases = batch_grad_params_tile(
                        x_batch.astype(np.float32, copy=False),
                        tile_grad_logits,
                    )
                    grad_weights_tiles.append(np.asarray(tile_grad_weights, dtype=float))
                    grad_biases_tiles.append(np.asarray(tile_grad_biases, dtype=float))

                updated_weights = []
                updated_biases = []
                for tile_index, tile_start in enumerate(range(0, layer.weights.shape[0], tile_size)):
                    tile_end = min(layer.weights.shape[0], tile_start + tile_size)
                    weights_tile = layer.weights[tile_start:tile_end]
                    biases_tile = layer.biases[tile_start:tile_end]
                    tile_updated_weights, tile_updated_biases = dense_apply_update_tile(
                        weights_tile.astype(np.float32, copy=False),
                        biases_tile.astype(np.float32, copy=False),
                        grad_weights_tiles[tile_index].astype(np.float32, copy=False),
                        grad_biases_tiles[tile_index].astype(np.float32, copy=False),
                        float(learning_rate),
                    )
                    updated_weights.append(np.asarray(tile_updated_weights, dtype=float))
                    updated_biases.append(np.asarray(tile_updated_biases, dtype=float))

                layer.weights = np.vstack(updated_weights)
                layer.biases = np.concatenate(updated_biases)
                layer.grad_weights = np.vstack(grad_weights_tiles)
                layer.grad_biases = np.concatenate(grad_biases_tiles)
                layer._sync_neurons_from_matrix()
                return float(batch_loss), "metal"

        logits = np.asarray(layer.forward(x_batch), dtype=np.float64)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits)
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        row_indices = np.arange(batch_size)
        batch_loss = -np.log(np.maximum(1e-12, probs[row_indices, target_indices])).mean()

        grad_logits = probs
        grad_logits[row_indices, target_indices] -= 1.0
        grad_logits /= batch_size

        grad_weights = grad_logits.T @ x_batch
        grad_biases = grad_logits.sum(axis=0)

        layer.grad_weights = np.asarray(grad_weights, dtype=float)
        layer.grad_biases = np.asarray(grad_biases, dtype=float)
        layer.update_params(learning_rate)

        return float(batch_loss), "cpu"

    def train(
        self,
        epochs: int = 80,
        learning_rate: float = 0.1,
        print_every: int = 10,
        progress_chunks: int = 10,
        batch_size: int = 64,
    ) -> None:
        total_pairs = self.sample_count
        if total_pairs == 0:
            raise ValueError("Training data does not contain any adjacent token pairs.")

        if total_pairs >= 200:
            print(
                f"Training {self.model_type} model on {total_pairs} pairs "
                f"({len(self.tokens)} tokens, vocab {self.vocab_size}) for {epochs} epochs"
            )
            self._print_progress_line("Epoch progress: [----------------------------] 0.0%")

        batch_size = max(1, int(batch_size))
        chunk_count = max(1, int(progress_chunks))
        total_batches = max(1, (total_pairs + batch_size - 1) // batch_size)
        progress_stride = 1

        torch_trainer = None
        if torch_backend is not None:
            create_dense_trainer = getattr(torch_backend, "create_dense_trainer", None)
            if callable(create_dense_trainer):
                layer = self.net.layers[0]
                try:
                    torch_trainer = create_dense_trainer(
                        weights=layer.weights.astype(np.float32, copy=False),
                        biases=layer.biases.astype(np.float32, copy=False),
                        learning_rate=float(learning_rate),
                    )
                except Exception:
                    torch_trainer = None

        if torch_trainer is not None:
            # Keep only tiny placeholders while torch owns the large parameter tensors.
            layer = self.net.layers[0]
            layer.weights = np.empty((0, 0), dtype=np.float32)
            layer.biases = np.empty((0,), dtype=np.float32)

        for epoch in range(1, epochs + 1):
            permutation = np.random.permutation(total_pairs)
            total_loss = 0.0
            self._print_progress_line(f"{self._epoch_header(epoch, epochs)}")
            backend_reported = False

            for batch_number, start in enumerate(range(0, total_pairs, batch_size), start=1):
                batch_indices = permutation[start:start + batch_size]
                batch_contexts = self.contexts[batch_indices]
                batch_targets = self.targets[batch_indices]
                batch_loss, backend_name = self._train_batch(
                    batch_contexts,
                    batch_targets,
                    learning_rate,
                    torch_trainer=torch_trainer,
                )
                if not backend_reported:
                    print(f"Training backend: {backend_name}", flush=True)
                    backend_reported = True
                total_loss += batch_loss * len(batch_indices)

                if total_pairs >= 200 and (batch_number % progress_stride == 0 or start + batch_size >= total_pairs):
                    seen = min(total_pairs, start + len(batch_indices))
                    pct = (seen / total_pairs) * 100.0
                    running_loss = total_loss / seen
                    bar = self._progress_bar(seen / total_pairs)
                    self._write_progress_line(
                        f"  batch {batch_number:4d}/{total_batches:<4d} {bar} {pct:5.1f}% "
                        f"| {seen:6d}/{total_pairs:<6d} | batch loss {batch_loss:.4f} | running loss {running_loss:.4f}"
                    )

            if epoch % max(1, print_every) == 0 or epoch == 1 or epoch == epochs:
                avg_loss = total_loss / total_pairs
                self._print_progress_line(
                    f"  epoch end {self._progress_bar(epoch / epochs)} | avg loss {avg_loss:.4f}"
                )

        if torch_trainer is not None:
            layer = self.net.layers[0]
            updated_weights, updated_biases = torch_trainer.export_parameters()
            layer.weights = np.asarray(updated_weights, dtype=np.float32)
            layer.biases = np.asarray(updated_biases, dtype=np.float32)
            layer._sync_neurons_from_matrix()

        sys.stdout.write("\n")
        sys.stdout.flush()

    def predict_logits_and_probs_from_context(
        self,
        context_tokens: List[str],
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        context_indices = self._context_tokens_to_indices(context_tokens)
        x = self._context_indices_to_input(context_indices)
        logits = self.net.forward(x)
        probs = self._softmax(logits, temperature=temperature)
        return np.asarray(logits, dtype=np.float64), probs

    def predict_logits_and_probs(self, current_token: str, temperature: float = 1.0) -> Tuple[List[float], List[float]]:
        return self.predict_logits_and_probs_from_context([current_token], temperature=temperature)

    def predict_next_distribution_from_context(
        self,
        context_tokens: List[str],
        temperature: float = 1.0,
    ) -> List[Tuple[str, float]]:
        _, probs = self.predict_logits_and_probs_from_context(context_tokens, temperature=temperature)
        return [(self.itos[i], float(probs[i])) for i in range(self.vocab_size)]

    def predict_next_distribution(self, current_token: str, temperature: float = 1.0) -> List[Tuple[str, float]]:
        return self.predict_next_distribution_from_context([current_token], temperature=temperature)

    def top_k_predictions_from_context(
        self,
        context_tokens: List[str],
        k: int = 5,
        temperature: float = 1.0,
    ) -> List[Tuple[str, float]]:
        dist = self.predict_next_distribution_from_context(context_tokens, temperature=temperature)
        return sorted(dist, key=lambda item: item[1], reverse=True)[: max(1, k)]

    def top_k_predictions(self, current_token: str, k: int = 5, temperature: float = 1.0) -> List[Tuple[str, float]]:
        return self.top_k_predictions_from_context([current_token], k=k, temperature=temperature)

    def sample_next_from_context(self, context_tokens: List[str], temperature: float = 1.0) -> str:
        _, probs = self.predict_logits_and_probs_from_context(context_tokens, temperature=temperature)
        next_idx = self._sample_from_probs(probs)
        return self.itos[next_idx]

    def sample_next(self, current_token: str, temperature: float = 1.0) -> str:
        return self.sample_next_from_context([current_token], temperature=temperature)

    def _serialize(self) -> Dict[str, object]:
        layer = self.net.layers[0]
        return {
            "model_type": self.model_type,
            "context_size": self.context_size,
            "tokens": self.tokens,
            "vocab": self.vocab,
            "weights": layer.weights.tolist(),
            "biases": layer.biases.tolist(),
        }

    @classmethod
    def _init_from_saved_tokens(cls, tokens: List[str], context_size: int):
        model = cls.__new__(cls)
        OneStepLanguageModel.__init__(model, tokens, context_size=context_size)
        return model

    def save(self, path: str) -> None:
        payload = self._serialize()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def _from_payload(cls, payload: Dict[str, object]):
        tokens = payload["tokens"]
        context_size = int(payload.get("context_size", 1))
        model = cls._init_from_saved_tokens(tokens, context_size=context_size)

        layer = model.net.layers[0]
        saved_weights = payload["weights"]
        saved_biases = payload["biases"]

        if len(saved_weights) != layer.weights.shape[0]:
            raise ValueError("Saved model shape does not match current network shape.")
        if len(saved_weights) > 0 and len(saved_weights[0]) != layer.weights.shape[1]:
            raise ValueError("Saved model input width does not match model context size.")

        if getattr(layer, "keep_neuron_mirror", False):
            for i, neuron in enumerate(layer.neurons):
                neuron.weights = np.asarray(saved_weights[i], dtype=np.float32)
                neuron.bias = float(saved_biases[i])

        layer.weights = np.asarray(saved_weights, dtype=np.float32)
        layer.biases = np.asarray(saved_biases, dtype=np.float32)
        layer._sync_neurons_from_matrix()

        return model


class CharLanguageModel(OneStepLanguageModel):
    model_type = "char"

    def __init__(self, text: str, context_size: int = 6):
        if len(text) < 2:
            raise ValueError("Training text must contain at least 2 characters.")
        self.text = text
        super().__init__(list(text), context_size=context_size)

    def generate(self, seed: str, length: int = 120, temperature: float = 1.0) -> str:
        if length <= 0:
            return seed

        if not seed:
            seed = random.choice(self.vocab)

        for ch in seed:
            if ch not in self.stoi:
                raise ValueError(f"Seed contains unknown char: {ch!r}")

        out = list(seed)
        context_tokens = out[-self.context_size:]

        for _ in range(length):
            next_char = self.sample_next_from_context(context_tokens, temperature=temperature)
            out.append(next_char)
            context_tokens.append(next_char)
            if len(context_tokens) > self.context_size:
                context_tokens.pop(0)

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

    def __init__(self, text: str, context_size: int = 4):
        tokens = self.tokenize(text)
        if len(tokens) < 2:
            raise ValueError("Training text must contain at least 2 word tokens.")
        self.text = text
        super().__init__(tokens, context_size=context_size)

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
        context_tokens = out[-self.context_size:]

        for _ in range(length):
            next_token = self.sample_next_from_context(context_tokens, temperature=temperature)
            out.append(next_token)
            context_tokens.append(next_token)
            if len(context_tokens) > self.context_size:
                context_tokens.pop(0)

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
