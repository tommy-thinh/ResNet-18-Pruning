# ResNet-18 Pruning

## Overview

This project compares unstructured magnitude pruning,
hardware-aligned 2:4 sparsity, channel pruning, and Lottery Ticket Hypothesis
(LTH) resetting on multi-label MNIST using TensorFlow/Keras. The model uses ResNet-18 blocks and ten
sigmoid outputs; the dataset contains 50,000 training, 10,000 validation and
10,000 test images at 64 × 64 resolution.

The work covers persistent masks, residual-compatible channel removal,
fine-tuning, ONNX export and matched TensorRT benchmarking. Each experiment has
its own command; shared implementation lives in `scripts/pruning/`.

## Results

Exact match requires all ten labels to be correct. Deployment rows below use
TensorRT 10.13.3.9, FP16-enabled inference on A30, batch 64, 200 warmups and
5,000 repetitions. GPU medians use resident buffers and CUDA graphs, excluding
transfers and preprocessing. Parameters count the allocated Keras architecture,
not just nonzero weights.

| Model | Test exact match | Parameters (M) | GPU median (ms/batch) | GPU speedup |
| --- | ---: | ---: | ---: | ---: |
| Dense | 81.66% | 11.253 | 0.4577 | 1.00× |
| Unstructured, 93.75% sparse | 82.17% | 11.253 | 0.4557 | 1.00× |
| 2:4, sparse tactics enabled | 82.08% | 11.253 | 0.4547 | 1.01× |
| Channels, 25% removed | 80.47% | 6.348 | 0.3768 | 1.21× |
| Channels, 50% removed | 77.22% | 2.838 | 0.2458 | 1.86× |
| LTH original initialization (TF FP32) | 81.99% | 11.253 | — | — |
| LTH random initialization (TF FP32) | 81.50% | 11.253 | — | — |

LTH retains 40.96% of eligible convolution weights; its TensorFlow dense reference
scores 81.63%. No LTH TensorRT timing is claimed. Full histories, all five
unstructured densities, before/after accuracy and same-weight sparse-tactic
controls are preserved in [results/](results/). The generated
[summary CSV](results/summary.csv) separates runtimes and batch sizes.

![Accuracy versus allocated parameters and measured GPU latency](figures/efficiency_tradeoffs.png)

![LTH discovery and original-versus-random initialization](figures/lottery_ticket.png)

## Reproduction

Create the environment on the training machine. GPU execution requires
Linux/WSL2 and a compatible NVIDIA driver. TensorRT dependencies are optional;
CPU validation uses `--device cpu`. Defaults are batch 64, GPU 0 and seed 42.

```bash
conda env create -f environment.yml
conda activate cnn
python -m pip install -r requirements-tensorrt.txt  # optional, for inference/export
python scripts/download_assets.py --model-repo OWNER/MODEL_REPO --dataset-repo OWNER/DATASET_REPO
```

Replace the repository IDs after publication; assets have not been uploaded by
this cleanup. Publish `models/` contents at the model-repository root and `data/`
contents at the dataset-repository root. [artifacts.json](artifacts.json) records
the retained files and checksums. Downloads verify hashes and refuse conflicting
local files; revision flags accept commit hashes.

```bash
# Other retained densities: 0.75, 0.5, 0.25, 0.125; use a fresh output for each.
python scripts/prune_unstructured.py --checkpoint models/dense/model.keras --density 0.0625 --output results/rerun-unstructured.json
python scripts/prune_2to4.py --checkpoint models/dense/model.keras --output results/rerun-semi24.json
python scripts/prune_channels.py --checkpoint models/dense/model.keras --remove 0.5 --output results/rerun-channels.json  # also 0.25
python scripts/lottery_ticket.py --checkpoint models/dense/model.keras --initialization models/dense/initial.keras --baseline results/baseline.json --rounds 4 --prune-rate 0.2 --epochs 200 --learning-rate 0.1 --output results/rerun-lth.json
# Optional fresh baseline; never overwrite the published initialization.
python scripts/train_baseline.py --model-dir models/dense-new --output results/baseline-new.json
```

Fine-tuning defaults to 100 epochs at LR 0.01, with early stopping and plateau
LR reduction. Override flags for new experiments; existing outputs are protected.
For a newly trained baseline, pass its checkpoint and `--baseline` JSON together.

```bash
# TensorFlow evaluation, or matched TensorRT inference for any checkpoint:
python scripts/benchmark.py --checkpoint models/dense/model.keras --output results/rerun-baseline.json
python scripts/benchmark_inference.py --checkpoint models/channels-50/model.keras --warmup 200 --repeats 5000 --output results/rerun-latency.json
# Existing engines: use --engine PATH instead of --checkpoint.
python scripts/export_onnx.py --checkpoint models/dense/model.keras --device cpu --output results/export-dense.json
python scripts/export_onnx.py --checkpoint models/semi24/model.keras --device cpu --output results/export-semi24.json
python scripts/benchmark_tensorrt.py --dense-onnx models/export-dense/model.onnx --sparse-onnx models/export-semi24/model.onnx --batch-size 64 --warmup 200 --repeats 5000 --output results/a30-b64.json
# Repeat the comparison with --batch-size 256 --repeats 3000 and a fresh output.
python scripts/plot_results.py  # four figures + summary.csv; no GPU/models needed
```

## Takeaways

Channel pruning delivers real acceleration: 50% width removal cuts MACs by
73.36%, with a 4.44-point accuracy cost. Its host-input/output speedup is only
1.21×; that path includes transfers and runner overhead and does not use CUDA graphs.
Unstructured zeros preserve accuracy but do not shrink dense computation.
2:4 reaches 1.096× same-weight GPU speedup at batch 256, not a universal 2× gain.

LTH's 0.49-point original-initialization advantage is modest, and the random
control reaches its best epoch sooner. These are single-seed results, without
an unpruned fine-tuning control, batch-1 measurements or energy measurements.
They demonstrate engineering trade-offs, not universal pruning benefits.
