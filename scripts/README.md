# Pruning experiments on NVIDIA A30

Each experiment has its own Python script. Each command saves one JSON to the `--output`
path and checkpoints to `--model-dir`. The default checkpoint directory is
`models/<output filename without .json>`. No suite, run registry or experiment IDs.

## Setup

Run from the repository root, using Python 3.11:

```bash
conda create -n cnn python=3.11 -y
conda activate cnn
python -m pip install -r requirements.txt
# Only needed for ONNX export and TensorRT benchmarking:
python -m pip install -r scripts/requirements.txt
```

The target stack remains TensorFlow 2.20.0 / Keras 3.12.0, NumPy 2.2.6 and
CUDA 12. A30 supports Ampere 2:4 sparsity; TensorRT 10.13.3.9 is the optional
FP16 inference runtime. This hardware change does not require replacing TensorFlow
or changing the 2:4 grouping rule. Native Windows runs TensorFlow on CPU; use
Linux/WSL2 for NVIDIA GPU execution. See [research and sources](RESEARCH.md).

Provide the original `data/train.pt`, `data/val.pt`, and `data/test.pt` files.
The CLI memory-maps their uint8 images and normalizes/replicates channels per
batch, avoiding a second full RGB copy or a full-image shuffle buffer in RAM.
Missing data fails explicitly. Tests use synthetic inputs only outside this repo.

Check your GPU and driver before a GPU run:

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Training, fine-tuning, validation and benchmarking default explicitly to GPU 0
(`--device gpu:0`, TensorFlow `/GPU:0`). `gpu` remains an alias for `gpu:0`.
GPU 0 means the first GPU visible to TensorFlow; `CUDA_VISIBLE_DEVICES` can remap
physical indices. Pass `--device cpu` for CPU execution. GPU requests fail if no
GPU is available instead of silently falling back to CPU.
TensorFlow GPU memory grows on demand by default, including in the notebook,
baseline training, pruning fine-tuning, Lottery Ticket retraining and benchmarking.
Larger batches can use more VRAM without changing a fixed memory budget. There is
no default 4 GB cap; `--gpu-memory-mb` is an optional explicit cap instead of growth.
Growth is bounded by available VRAM and does not automatically reduce the batch
size if it cannot fit. TensorFlow may retain allocated memory for reuse.
Restart the process (or notebook kernel) for a changed memory policy to take effect.
TensorRT runtime buffers follow the engine's batch-size profile; its separate
builder workspace limit (up to 1 GiB) is not a cap on total runtime GPU memory.
A30 has 24 GB when exposed as a full GPU; a MIG slice can be smaller.

Shared training/benchmark settings live in `scripts/pruning/config.py`.
The scripts' CLI defaults/flags supply epochs, learning rate, batch size and device;
use `--epochs`, `--learning-rate`, `--batch-size` and `--device` to override a run.
Other training defaults, such as momentum and weight decay, come from the config.
File edits affect the next process, not training already running. Lottery Ticket
training keeps the saved baseline budget while using the selected current device.
In the notebook, GPU 0 is configured in the setup cell; change that cell to `cpu`
for CPU execution and restart the kernel before changing device visibility.

## Train or reuse the baseline

The notebook remains the interactive entry point. It uses the shared modules
under `scripts/pruning/` and produces the same baseline JSON as the CLI:

```bash
jupyter notebook resnet-18.ipynb
# Or train headlessly:
python scripts/train_baseline.py --model-dir models/dense --output results/baseline.json
```

Use one entry point for a given output path. The reference settings remain seed
42, batch size 16, at most 200 epochs, SGD LR 0.1, momentum 0.9, weight decay
0.005, EMA enabled, BCE, EarlyStopping and ReduceLROnPlateau. Override epochs/LR
explicitly when needed. The initialization is saved before compiling/training.

`initial.keras` holds the original weights and BatchNorm state. `best.keras` is
the raw best-validation epoch checkpoint, including optimizer state. After fit
finalizes EMA, training reloads that epoch and saves `model.keras` without the
optimizer. Use **model.keras** for independent pruning and comparable model sizes.
Every fine-tuning or Lottery Ticket training run starts with fresh optimizer state.

To benchmark an existing checkpoint without retraining:

```bash
python scripts/benchmark.py --checkpoint models/dense/model.keras --output results/benchmark.json
```

Existing compatible `.keras` checkpoints can also be supplied. Lottery Ticket
experiments additionally require their matching untouched initialization and the
new baseline JSON; a trained model alone cannot recover its original initialization.

Use the already saved baseline metrics as the unpruned comparison. There is no
pruning or extra fine-tuning run at density 1.0. Pruning scripts read
`results/baseline.json` by default and verify that it matches the checkpoint.
If you benchmarked an existing model separately, pass its JSON with
`--baseline results/benchmark.json`. That benchmark is a one-time measurement,
not a prerequisite to repeat for every pruning experiment.

## Run individual pruning experiments

```bash
# Global unstructured pruning: choose the density you want to evaluate.
python scripts/prune_unstructured.py --checkpoint models/dense/model.keras --density 0.75 --output results/unstructured-75.json
python scripts/prune_unstructured.py --checkpoint models/dense/model.keras --density 0.5 --output results/unstructured-50.json
python scripts/prune_unstructured.py --checkpoint models/dense/model.keras --density 0.25 --output results/unstructured-25.json
python scripts/prune_unstructured.py --checkpoint models/dense/model.keras --density 0.125 --output results/unstructured-12_5.json
python scripts/prune_unstructured.py --checkpoint models/dense/model.keras --density 0.0625 --output results/unstructured-6_25.json

# Hardware-aligned 2:4 sparsity.
python scripts/prune_2to4.py --checkpoint models/dense/model.keras --output results/semi24.json

# Compact channel-pruned models.
python scripts/prune_channels.py --checkpoint models/dense/model.keras --remove 0.2 --output results/channels-20.json
python scripts/prune_channels.py --checkpoint models/dense/model.keras --remove 0.4 --output results/channels-40.json
```

Each command loads the supplied dense checkpoint independently, records immediate
pruning performance, fine-tunes for at most 30 epochs at LR 0.01, and records the
selected model's performance. Override with `--epochs` and `--learning-rate`.
Choose another `--density` or `--remove` and output filename for each comparison.
Unstructured pruning requires `0 < density < 1`; density 1 is rejected before any
model loading or training. Each JSON reuses the dense metrics from the baseline
alongside the pruned model's before/after measurements.

Unstructured pruning ranks eligible kernel magnitudes globally. 2:4 retains two
weights per four consecutive input channels at each kernel position/output
channel, excluding the three-input-channel stem. Both enforce persistent masks
in gradients and weight updates, including SGD momentum/EMA. Biases, BatchNorm
and the ten-label output kernel are excluded. Channel pruning constructs a smaller
dense graph with aligned residual/projection/BN channel indices.

## Lottery Ticket comparison

```bash
python scripts/lottery_ticket.py --checkpoint models/dense/model.keras --initialization models/dense/initial.keras --baseline results/baseline.json --rounds 4 --output results/lth-round4.json
```

Each round removes 20% of surviving convolutional weights by default, resets all
weights/BatchNorm state to the original initialization, and retrains. Projection
and Dense kernels are excluded, following the paper's ResNet experiment.
`--rounds 4` gives approximately 0.8^4 = 40.96% retained eligible weights.
`--prune-rate` changes the per-round fraction. The final mask is also trained with
a random initialization, using the same training budget, shuffle and Dropout seed.

The baseline JSON verifies checkpoint/initialization checksums and supplies the
reference training configuration. Keep batch size matched; incompatible explicit
epoch/LR overrides are rejected. Discovery uses validation for selection; the
original/random subnetworks are test-evaluated only at the requested final round.
Histories for all discovery rounds are included in the one result JSON.
The dense reference metrics are read from the baseline JSON, without retraining
or re-evaluating that model.

## Export and measure A30 sparse inference

```bash
python scripts/export_onnx.py --checkpoint models/dense/model.keras --output results/export-dense.json
python scripts/export_onnx.py --checkpoint models/semi24/model.keras --output results/export-semi24.json
python scripts/benchmark_tensorrt.py --dense-onnx models/export-dense/model.onnx --sparse-onnx models/export-semi24/model.onnx --output results/tensorrt-a30.json
```

Export checks ONNX validity, numerical parity on 32 real examples, and sparsity
with Polygraphy. TensorRT builds three comparable FP16-enabled engines: original
dense weights, pruned weights with sparse tactics disabled, and the same pruned
weights with sparse tactics enabled. It evaluates full-test accuracy and reports
the selected sparse-layer count and same-weight latency ratio. Detailed engine
logs and layer descriptions stay beside the engines under `models/`.

Rebuild engines on the A30; existing engines are generally GPU/runtime-specific.
The implementation targets TensorRT 10.x and rejects 11.x's changed precision API.
FP16-enabled does not guarantee FP16 for every layer. INT8 requires calibration
and is not enabled here; BF16 sparse execution is not assumed.

## Results and code layout

JSON contains the method, checkpoint, seed, short runtime description, accuracy
metrics, parameter/MAC count, eligible-weight sparsity, model size, median/p95
latency, throughput and allocator peak where supported. Fine-tuned experiments
have `before`, `after` and training curves; LTH has `dense`, `original`, `random`
and discovery histories. There are no CSV tables, environment dumps or automatic
reports. The `loss` metric is BCE; MACs count Conv/Dense only (FLOPs = 2 x MACs).
Runtime descriptions include the exact device placement (`/GPU:0` or `/CPU:0`)
and the hardware name.
`dense_runtime` preserves the saved baseline's measurement conditions. Compare
latencies only when GPU/runtime/precision/batch size match the current run;
reusing saved accuracy does not turn a different GPU's timings into A30 timings.

Warmup and repetitions default to 10/50, adjustable with `--warmup`/`--repeats`.
TensorFlow timings use resident inputs and synchronize through host output
materialization. TensorRT uses synchronized host input/output inference; compare
dense/sparse TensorRT engines to isolate sparsity acceleration, not TF/TRT timings.
Zero-valued dense Keras tensors do not automatically execute faster.

Output paths and nonempty model directories are protected against overwriting.
Intermediate pruning/LTH progress updates the same JSON with `status: incomplete`;
successful commands finish with `status: completed`. To stop a foreground command,
press Ctrl+C. Nothing starts another experiment afterward.

Each script owns its argument parser and experiment implementation:
`train_baseline.py`, `benchmark.py`, `prune_unstructured.py`, `prune_2to4.py`,
`prune_channels.py`, `lottery_ticket.py`, `export_onnx.py` and `benchmark_tensorrt.py`.
`pruning/` shares model, data, mask, channel, training, evaluation and runtime helpers.
`requirements.txt` in this directory adds only optional inference dependencies.
Keep this README as the single command reference when adding or changing experiments.
Commit measured JSON files under root `results/`; data and model binaries remain ignored.
