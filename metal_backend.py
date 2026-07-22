"""Optional Python wrapper for the native Metal dense-layer backend."""

try:
    from _metal_nn import available as _native_available, dense_forward as _native_dense_forward
except Exception as exc:  # pragma: no cover - native build is optional.
    _native_available = None
    _native_dense_forward = None
    _import_error = exc
else:
    _import_error = None


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
    return _native_dense_forward(weights, biases, inputs)