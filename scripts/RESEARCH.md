# A30 sparsity and experimental design

## Hardware and dependency choices

NVIDIA lists A30 as an Ampere GPU with 24 GB HBM2, third-generation Tensor Cores
and sparse FP16/INT8 throughput. Its documentation confirms sparse execution
support; TensorRT documents Ampere's 2:4 requirement. The mask algorithm therefore
remains the same: Conv weights [R,S,C,K] group along C, and Dense [C,K] along C.
Do not group arbitrary flattened weights. The C=3 stem cannot satisfy this layout.

- [A30 specifications and sparse throughput](https://www.nvidia.com/en-us/data-center/products/a30-gpu/)
- [TensorRT 10.x structured sparsity rules](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/io-formats-sparsity.html)

The existing TensorFlow 2.20.0 / Keras 3.12.0 stack remains suitable. TensorFlow's
tested build table lists CUDA 12.5 and cuDNN 9.3 for 2.20.0; Linux GPU installation
uses its and-cuda extra. Optional TensorRT CUDA 12 10.13.3.9, tf2onnx 1.17.0 and
Polygraphy 0.53.6 remain pinned. Switching GPU models alone does not require a
framework downgrade. Driver/library compatibility must be checked on the actual
machine; this local refactor does not establish a new A30 hardware measurement.

- [TensorFlow installation](https://www.tensorflow.org/install/pip)
- [TensorFlow tested build configurations](https://www.tensorflow.org/install/source#gpu)
- [TensorRT 10.x support and engine portability](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html)
- [Keras ONNX export](https://keras.io/api/models/model_saving_apis/export/)
- [tf2onnx compatibility](https://github.com/onnx/tensorflow-onnx#supported-versions)

The TensorRT path uses FP16-enabled and SPARSE_WEIGHTS flags. A valid mask is
necessary but does not prove that TensorRT chooses sparse tactics, or that they
improve latency. Builder logs and the same-weights, sparse-tactics-off control
provide the required evidence. Rebuild engines for the target GPU/runtime.
INT8 calibration and BF16 sparse paths are outside these commands.

## Reference model and measurements

The plain model uses [2,2,2,2] BasicBlocks, stage widths [64,128,256,512], projection
shortcuts, global average pooling, Dense(128), Dropout(0.5), and ten sigmoid outputs.
The original external dataset splits, /255 scaling, RGB replication and metrics
are preserved. Exact match requires all ten labels at threshold >=0.5; Keras
binary accuracy, precision, recall and BCE remain available. The scalar outputs
are weighted correctly for partial batches.

Memory-mapped uint8 inputs are expanded only per batch through
[Keras PyDataset](https://keras.io/api/utils/preprocessing_utils/#pydataset-class),
with two workers and a two-batch queue. This bounds prefetching memory while
preserving the same input tensors and preprocessing.

Training keeps SGD/EMA and the original schedule. Checkpoints are selected using
validation exact match, then explicitly reloaded after EMA finalization. The
inference-only model is saved separately so storage comparisons exclude optimizer
slots. Every independent training run resets optimizer and Dropout generator state.

- [Keras SGD and EMA](https://keras.io/api/optimizers/sgd/)
- [Validation checkpoint selection](https://keras.io/api/callbacks/model_checkpoint/)
- [Persistent weight constraints](https://keras.io/api/layers/constraints/)
- [GPU 0 selection and explicit device placement](https://www.tensorflow.org/guide/gpu)
- [On-demand GPU memory growth](https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth)

Training scopes model/optimizer creation, training and validation with
`tf.device(config.device)`. Evaluation/inference uses the same selected placement.
The default is `/GPU:0`; explicit CPU execution uses `/CPU:0`. Device availability
is checked so a GPU request cannot silently become a CPU benchmark. The notebook
and CLI configure GPU visibility before model construction.

The shared device setup enables TensorFlow memory growth by default, before GPU
initialization. Training, fine-tuning and evaluation therefore allocate VRAM as
needed for their model and batch size, rather than sharing a fixed 4 GB cap.
An explicit `--gpu-memory-mb` uses a logical-device hard limit instead; TensorFlow
does not support combining memory growth with configured virtual devices.

Structured channels share a survivor set within each residual stage. Stem/stage-2
identity connections, transition projections, BN arrays and downstream Dense inputs
are sliced consistently. Channel-removal fraction is not parameter/FLOP reduction;
measure the resulting model. Unstructured/2:4 masks keep dense tensor shapes, so
dense-equivalent MACs remain unchanged.

## Lottery Ticket adaptation

[Frankle and Carbin, ICLR 2019](https://arxiv.org/abs/1803.03635), Section 4 and
Appendix I.3, globally prune 20% of surviving Conv weights per round for their
ResNet experiment, excluding projection and output weights. Surviving values reset
to their original initialization. The same mask with random initialization is an
essential control. This implementation also restores all BN state and resets
optimizer state; training and validation histories support learning-speed comparisons.

The paper's CIFAR10 model is much narrower, uses Gaussian Glorot initialization
and a fixed iteration schedule, and is sensitive to learning rate/warmup. Our
wider Keras model, Multi-label MNIST, initializer defaults, LR 0.1, plateau schedule
and EMA differ. This is an adaptation, not an exact reproduction. No winning-ticket
claim follows from reasonable sparse accuracy alone. Compare matched training
budgets, dense accuracy, original versus random initialization, and learning speed.
Multiple seeds and uncontended hardware are needed for stronger conclusions.

Sources checked on 2026-11-06. No A30 speedup or final accuracy is claimed before
the corresponding command produces measured results.
