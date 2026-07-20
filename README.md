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

## Main files

- `language_model.py`: character and word language model logic, training, batching, and save/load.
- `neuralnet.py`: lightweight dense-layer and neuron implementation.
- `graphic_nn.py`: character-level visualization.
- `graphic_nn_wordLM.py`: word-level visualization using `nsoe.txt` and saved model weights.

## How to run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install numpy pygame
```

3. Run the word-level visualizer with `nsoe.txt`:

```bash
python graphic_nn_wordLM.py
```

4. Run the character-level visualizer:

```bash
python graphic_nn.py
```

5. Run the language-model demo/training script directly:

```bash
python language_model.py
```

## Notes

This repository is intended as an experiment and test bed, not a production-ready language model implementation.
