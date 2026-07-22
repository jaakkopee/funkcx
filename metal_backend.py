"""Optional Python wrapper for the native Metal dense-layer backend."""

import numpy as np

try:
    from _metal_nn import (
        available as _native_available,
        dense_apply_update_tile as _native_dense_apply_update_tile,
        dense_batch_grad_params as _native_dense_batch_grad_params,
        dense_batch_grad_params_tile as _native_dense_batch_grad_params_tile,
        dense_apply_update as _native_dense_apply_update,
        dense_backward as _native_dense_backward,
        dense_forward as _native_dense_forward,
        dense_forward_tile as _native_dense_forward_tile,
    )
except Exception as exc:  # pragma: no cover - native build is optional.
    _native_available = None
    _native_dense_apply_update_tile = None
    _native_dense_batch_grad_params = None
    _native_dense_batch_grad_params_tile = None
    _native_dense_apply_update = None
    _native_dense_backward = None
    _native_dense_forward = None
    _native_dense_forward_tile = None
    _import_error = exc
else:
    _import_error = None


MAX_NATIVE_DENSE_BYTES = 64 * 1024 * 1024


def can_offload_dense(*arrays) -> bool:
    for array in arrays:
        if array is None:
            return False
        nbytes = getattr(array, "nbytes", None)
        if nbytes is None:
            try:
                nbytes = np.asarray(array).nbytes
            except Exception:
                return False
        if nbytes > MAX_NATIVE_DENSE_BYTES:
            return False
    return True


def is_available() -> bool:
    if _native_available is None:
        return False
    try:
        return bool(_native_available())
    except Exception:
        return False


def dense_forward(weights, biases, inputs):
    if _native_dense_forward is None:
        raise RuntimeError(
            "Metal backend is not built. Run `python setup.py build_ext --inplace` on macOS with Xcode tools installed."
        ) from _import_error
    if not can_offload_dense(weights, biases, inputs):
        raise RuntimeError("Dense workload exceeds the safe Metal threshold.")
    return _native_dense_forward(weights, biases, inputs)


def dense_forward_tile(weights_tile, biases_tile, inputs, output_offset):
    if _native_dense_forward_tile is None:
        raise RuntimeError(
            "Metal backend is not built. Run `python setup.py build_ext --inplace` on macOS with Xcode tools installed."
        ) from _import_error
    return _native_dense_forward_tile(weights_tile, biases_tile, inputs, output_offset)


def dense_backward(weights, last_input, grad_output):
    if _native_dense_backward is None:
        raise RuntimeError(
            "Metal backend is not built. Run `python setup.py build_ext --inplace` on macOS with Xcode tools installed."
        ) from _import_error
    if not can_offload_dense(weights, last_input, grad_output):
        raise RuntimeError("Dense workload exceeds the safe Metal threshold.")
    return _native_dense_backward(weights, last_input, grad_output)


def dense_apply_update(weights, biases, grad_weights, grad_biases, learning_rate):
    if _native_dense_apply_update is None:
        raise RuntimeError(
            "Metal backend is not built. Run `python setup.py build_ext --inplace` on macOS with Xcode tools installed."
        ) from _import_error
    if not can_offload_dense(weights, biases, grad_weights, grad_biases):
        raise RuntimeError("Dense workload exceeds the safe Metal threshold.")
    return _native_dense_apply_update(weights, biases, grad_weights, grad_biases, learning_rate)


def dense_apply_update_tile(weights_tile, biases_tile, grad_weights, grad_biases, learning_rate):
    if _native_dense_apply_update_tile is None:
        raise RuntimeError(
            "Metal backend is not built. Run `python setup.py build_ext --inplace` on macOS with Xcode tools installed."
        ) from _import_error
    return _native_dense_apply_update_tile(weights_tile, biases_tile, grad_weights, grad_biases, learning_rate)


def dense_batch_grad_params(inputs, grad_output):
    if _native_dense_batch_grad_params is None:
        raise RuntimeError(
            "Metal backend is not built. Run `python setup.py build_ext --inplace` on macOS with Xcode tools installed."
        ) from _import_error
    if not can_offload_dense(inputs, grad_output):
        raise RuntimeError("Dense workload exceeds the safe Metal threshold.")
    return _native_dense_batch_grad_params(inputs, grad_output)


def dense_batch_grad_params_tile(inputs, grad_output):
    if _native_dense_batch_grad_params_tile is None:
        raise RuntimeError(
            "Metal backend is not built. Run `python setup.py build_ext --inplace` on macOS with Xcode tools installed."
        ) from _import_error
    return _native_dense_batch_grad_params_tile(inputs, grad_output)