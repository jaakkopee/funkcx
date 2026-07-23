# funkcx

This is a small test project for experimenting with a hand-built language model and simple neural-network visualization ideas.

## What it does

The project trains lightweight character-level and word-level language models on text data and can visualize word prediction probabilities with a pygame window.

## Tech used

- Python: main application code, training loop, model orchestration, and scripts.
- NumPy: vectorized math for the dense layer, minibatch training, and softmax computations.
- Pygame: basic desktop visualization for model predictions and generated output.
- Custom neural network code: a simple in-repo implementation in `neuralnet.py` and `language_model.py` rather than an external ML framework.
- JSON persistence: saves and loads model weights such as `char_lm_weights.json`, `word_lm_weights.json`, and `nsoe.json`.
- Text corpus input: training data can be read from `nsoe.txt` for the word-level model.
- Optional Metal backend: an Objective-C++/pybind11 extension can accelerate dense-layer forward passes on macOS.
- Optional PyTorch backend: if `torch` is installed, the training step can use MPS on macOS and fall back to CPU otherwise.

## Main files

- `language_model.py`: character and word language model logic, training, batching, and save/load.
- `neuralnet.py`: lightweight dense-layer and neuron implementation.
- `graphic_nn.py`: character-level visualization.
- `funkcx.py`: word-level visualization using `nsoe.txt` and saved model weights.
- `metal_backend.mm` / `metal_dense.metal`: optional native Metal dense-layer implementation.
- `metal_backend.py`: Python wrapper that falls back cleanly when the native module is unavailable.
- `torch_backend.py`: optional PyTorch training backend for the dense batch step.
- `setup.py`: build script for the optional native extension.

## How to run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install numpy pygame
```

If you want the PyTorch training path, also install `torch`:

```bash
pip install torch
```

3. Run the word-level visualizer with `nsoe.txt`:

```bash
python funkcx.py
```

4. Run the character-level visualizer:

```bash
python graphic_nn.py
```

5. Run the language-model demo/training script directly:

```bash
python language_model.py
```

## Optional Metal backend

On macOS with Xcode command line tools installed, you can build the native backend in place:

```bash
python setup.py build_ext --inplace
```

The Python code will use the Metal path automatically when the compiled module is present; otherwise it falls back to NumPy.

If PyTorch is installed, large training batches will prefer the PyTorch MPS/CPU path before Metal.

## Notes

This repository is intended as an experiment and test bed, not a production-ready language model implementation.
