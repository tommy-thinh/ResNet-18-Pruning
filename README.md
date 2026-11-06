# CNN experiments

Multi-label MNIST digit classification with TensorFlow/Keras. The current notebook
uses a single-output ResNet-18 with ten sigmoid outputs.

- [resnet-18.ipynb](resnet-18.ipynb): interactive model training, visualization and evaluation.
- [scripts/README.md](scripts/README.md): separate experiment scripts, copyable commands,
  shared Python modules, A30 setup instructions and methodology.
- `data/`: external `train.pt`, `val.pt` and `test.pt` splits; excluded from Git.
- `models/`: initialization, trained/pruned checkpoints and inference engines; excluded from Git.
- `results/`: one JSON per experiment; intended to be committed to Git.

Run commands from the repository root. Start with the setup and checkpoint
instructions in [scripts/README.md](scripts/README.md).

Keep future experiment-specific commands and supporting code under `scripts/`;
keep commands in `scripts/README.md`, notebooks at the root and measured result JSONs under `results/`.
