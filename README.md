# ResNet-18 Pruning

ResNet-18 pruning experiments on multi-label MNIST: unstructured magnitude
pruning, 2:4 sparsity, channel pruning, and Lottery Ticket Hypothesis (LTH)
resetting. The TensorFlow/Keras model has ten sigmoid outputs and takes 64 × 64
images. The dataset has 50,000 training, 10,000 validation and 10,000 test samples.

Each experiment has a separate script. Shared code in `scripts/pruning/` handles
persistent masks, residual-compatible channel removal, fine-tuning, ONNX export
and TensorRT benchmarking.

## Results

Deployment measurements use TensorRT 10.13.3.9 with FP16 enabled on an A30:
batch 64, 200 warmups and 5,000 repetitions. GPU latency uses resident buffers
and CUDA graphs; it excludes transfers and preprocessing. Exact match counts
samples with all ten labels correct. Parameter counts include zero weights.

| Model | Test exact match | Parameters (M) | GPU median (ms/batch) | GPU speedup |
| --- | ---: | ---: | ---: | ---: |
| Dense | 81.66% | 11.253 | 0.4577 | 1.00× |
| Unstructured, 93.75% sparse | 82.17% | 11.253 | 0.4557 | 1.00× |
| 2:4, sparse tactics enabled | 82.08% | 11.253 | 0.4547 | 1.01× |
| Channels, 25% removed | 80.47% | 6.348 | 0.3768 | 1.21× |
| Channels, 50% removed | 77.22% | 2.838 | 0.2458 | 1.86× |
| LTH original initialization (TF FP32) | 81.99% | 11.253 | — | — |
| LTH random initialization (TF FP32) | 81.50% | 11.253 | — | — |

Removing 50% of channel width cuts MACs by 73.36% and costs 4.44 percentage
points of exact-match accuracy. The GPU speedup is 1.86×, but host-input/output
speedup is 1.21×. The latter includes transfers and runner overhead, without
CUDA graphs. Unstructured pruning preserves accuracy without reducing dense
computation. At batch 256, enabling sparse tactics for the same 2:4 weights
gives 1.096× GPU speedup—not the theoretical 2×.

LTH retains 40.96% of eligible convolution weights. Its TensorFlow dense
reference scores 81.63%; original initialization beats the random control by
0.49 points, though the random control reaches its best epoch sooner. LTH was
not timed in TensorRT.

These are single-seed runs. There is no unpruned fine-tuning control, batch-1
benchmark or energy measurement. [Results](results/) retain full histories,
all five unstructured densities, before/after accuracy and same-weight
sparse-tactic controls. [summary.csv](results/summary.csv) separates runtimes
and batch sizes.

![Accuracy versus allocated parameters and measured GPU latency](figures/efficiency_tradeoffs.png)

![LTH discovery and original-versus-random initialization](figures/lottery_ticket.png)

## Reproduction

Set up Conda on the training machine. GPU execution requires Linux/WSL2 and a
compatible NVIDIA driver. TensorRT dependencies are optional. Use `--device cpu`
for CPU validation; defaults are batch 64, GPU 0 and seed 42.

```bash
conda env create -f environment.yml
conda activate cnn
python -m pip install -r requirements-tensorrt.txt  # optional, for inference/export
python scripts/download_assets.py
```

Checkpoints, pruning masks and dataset splits are hosted together on
[Hugging Face](https://huggingface.co/datasets/fucthin/ResNet-18-Pruning-Datasets-and-Models).
The download script restores `models/` and `data/`, checks hashes against
[artifacts.json](artifacts.json), and refuses to overwrite conflicting files.
Use `--repo OWNER/NAME` to change the dataset repository, or `--revision COMMIT`
to pin a version.

```bash
# Other retained densities: 0.75, 0.5, 0.25, 0.125; use a fresh output for each.
python scripts/prune_unstructured.py --checkpoint models/dense/model.keras --density 0.0625 --output results/rerun-unstructured.json
python scripts/prune_2to4.py --checkpoint models/dense/model.keras --output results/rerun-semi24.json
python scripts/prune_channels.py --checkpoint models/dense/model.keras --remove 0.5 --output results/rerun-channels.json  # also 0.25
python scripts/lottery_ticket.py --checkpoint models/dense/model.keras --initialization models/dense/initial.keras --baseline results/baseline.json --rounds 4 --prune-rate 0.2 --epochs 200 --learning-rate 0.1 --output results/rerun-lth.json
# Optional fresh baseline; never overwrite the published initialization.
python scripts/train_baseline.py --model-dir models/dense-new --output results/baseline-new.json
```

Fine-tuning defaults to a 100-epoch budget at LR 0.01, with early stopping and
plateau LR reduction. Change these through CLI flags. Use a fresh output path
for each run; existing outputs are protected. When training a new baseline,
pass its checkpoint and matching `--baseline` JSON to the pruning scripts.

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
