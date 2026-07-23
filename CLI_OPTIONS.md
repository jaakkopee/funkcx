# CLI Options

This document tracks command-line options for this project.

## Scope

Implemented CLI options below refer to `funkcx.py`.

## Implemented Options

| Option | Type | Default | Used In | Description |
|---|---|---|---|---|
| --model-path | string | nsoe.json | train/load | Path to model checkpoint file. |
| --corpus-path | string | nsoe.txt | train | Path to training corpus text file. |
| --context-size | int | 4 | train/model build | Context window size for word model training. |
| --epochs | int | 140 | train | Number of training epochs. |
| --learning-rate | float | 0.08 | train | Learning rate for optimizer/training step. |
| --print-every | int | 10 | train logging | Epoch logging frequency. |
| --progress-chunks | int | 12 | train progress | Progress bar chunk setting passed to training. |
| --batch-size | int | 8 | train | Minibatch size for training. |
| --temperature | float | 0.9 | inference/visualization | Sampling temperature for next-token prediction. |
| --top-k | int | 5 | visualization | Number of top predictions to show in UI. |
| --fps | int | 3 | visualization | UI refresh/update frame rate. |
| --generation-log-file | string | (empty) | visualization/logging | Append generated text stream to this file; each reseed starts a new line. |
| --force-retrain | flag | false | train/load | Force retraining even if model file exists. |
| --train-only | flag | false | runner | Train then exit without launching pygame UI. |

## Planned But Not Implemented

These were planned and are good candidates, but are not currently wired as CLI options.

### Backend And Device Control

| Option | Type | Suggested Default | Purpose |
|---|---|---|---|
| --backend | choice(auto,torch,metal,cpu) | auto | Force or auto-select training backend. |
| --device | choice(auto,mps,cpu,cuda) | auto | Select torch device explicitly. |
| --pin-backend | flag | false | Disallow runtime fallback backend switching. |
| --oom-fallback-policy | choice(drop-to-cpu,reduce-batch,stop) | reduce-batch | Behavior when out-of-memory is detected. |

### Memory And Stability

| Option | Type | Suggested Default | Purpose |
|---|---|---|---|
| --metal-tile-size | int | 16 | Tile size for custom Metal tiled kernels. |
| --max-metal-layer-bytes | int | 67108864 | Size threshold for Metal offload safety gate. |
| --one-hot-cache-max-vocab | int | 4096 | Disable giant one-hot cache beyond this vocab size. |
| --max-train-samples | int | unlimited | Cap training set size for quick/low-memory runs. |
| --gradient-accumulation-steps | int | 1 | Simulate larger effective batch with lower memory. |
| --dtype | choice(fp32,fp16,bf16) | fp32 | Numeric dtype policy where backend supports it. |

### Vocabulary And Token Filtering

| Option | Type | Suggested Default | Purpose |
|---|---|---|---|
| --vocab-limit | int | unlimited | Limit vocab size to reduce parameter count. |
| --min-token-frequency | int | 1 | Drop very rare tokens to reduce memory/noise. |
| --tokenizer-mode | choice(word,char) | word | Select tokenizer strategy from CLI. |

### Training Control And Reproducibility

| Option | Type | Suggested Default | Purpose |
|---|---|---|---|
| --seed | int | none | Reproducible randomness for training/sampling. |
| --shuffle | choice(on,off) | on | Enable/disable per-epoch sample shuffling. |
| --eval-interval | int | disabled | Run periodic evaluation during training. |
| --early-stopping-patience | int | disabled | Stop when metric stops improving. |

### Checkpointing

| Option | Type | Suggested Default | Purpose |
|---|---|---|---|
| --resume | flag | false | Resume training from checkpoint. |
| --save-every-n-epochs | int | disabled | Periodic checkpointing by epoch. |
| --save-every-n-batches | int | disabled | Periodic checkpointing by batch count. |
| --keep-last-n-checkpoints | int | 1 | Retain recent checkpoints only. |
| --autosave-on-interrupt | flag | true | Save checkpoint on Ctrl-C/termination. |

### Logging And Reporting

| Option | Type | Suggested Default | Purpose |
|---|---|---|---|
| --log-interval-batches | int | 1 | Batch-level logging frequency. |
| --show-backend | flag | true | Print selected backend and device at start. |
| --progress-style | choice(compact,verbose) | compact | Progress output verbosity style. |
| --memory-report-interval | int | disabled | Print memory usage periodically. |

## Notes

- Current implemented CLI lives in `funkcx.py`.
- `language_model.py` has internal constants affecting backend and memory behavior, but most are not yet exposed as CLI flags.
- If model already exists and `--force-retrain` is not used, training options are ignored because the model is loaded instead of retrained.
