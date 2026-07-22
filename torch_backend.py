"""Optional PyTorch backend for dense-batch training.

This backend is intentionally narrow: it only handles the linear layer used by
funkcx's language model training step. When PyTorch is available it will prefer
MPS on macOS, otherwise it will fall back to CPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - optional dependency.
    torch = None
    F = None
    _import_error = exc
else:
    _import_error = None


def is_available() -> bool:
    return torch is not None


def _select_device() -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch is not installed.") from _import_error

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_dense_batch(
    weights: np.ndarray,
    biases: np.ndarray,
    inputs: np.ndarray,
    target_indices: np.ndarray,
    learning_rate: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if torch is None or F is None:
        raise RuntimeError(
            "PyTorch backend is not available. Install torch to enable the MPS/CPU training path."
        ) from _import_error

    device = _select_device()
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
    biases_t = torch.as_tensor(biases, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
    inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(target_indices, dtype=torch.long, device=device)

    logits = F.linear(inputs_t, weights_t, biases_t)
    loss = F.cross_entropy(logits, target_t)
    loss.backward()

    with torch.no_grad():
        updated_weights = weights_t - (float(learning_rate) * weights_t.grad)
        updated_biases = biases_t - (float(learning_rate) * biases_t.grad)

    return (
        updated_weights.detach().cpu().numpy(),
        updated_biases.detach().cpu().numpy(),
        float(loss.item()),
    )
