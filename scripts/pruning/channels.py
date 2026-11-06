"""Compact channel pruning with shared survivor indices in each residual stage."""
import numpy as np
from tensorflow import keras

from .model import build_resnet18_model


def prune_channels(model: keras.Model, removal: float) -> tuple[keras.Model, dict]:
    if not np.isfinite(removal) or not 0 <= removal < 1:
        raise ValueError("Channel removal fraction must be in [0,1).")
    widths = [model.get_layer(f"conv{s}_block1_conv2").filters for s in range(2, 6)]
    # Restrict to the inspected architecture; never silently operate on another graph.
    reference = build_resnet18_model(tuple(model.input_shape[1:]), tuple(widths))
    if [(l.name, [tuple(w.shape) for w in l.weights]) for l in model.layers if l.weights] != [
        (l.name, [tuple(w.shape) for w in l.weights]) for l in reference.layers if l.weights
    ]:
        raise ValueError("Channel pruning supports only the plain single-output ResNet-18 architecture.")
    indices = []
    for stage, width in zip(range(2, 6), widths):
        producers = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)
                     and ((l.name.startswith(f"conv{stage}_block") and l.filters == width)
                          or (stage == 2 and l.name == "conv1"))]
        # Mean L1 magnitude per output filter; normalize each producer's scale.
        score = np.zeros(width)
        for layer in producers:
            importance = np.mean(np.abs(layer.kernel.numpy()), axis=(0, 1, 2))
            score += importance / max(float(importance.mean()), np.finfo(float).eps)
        keep = max(1, int(np.floor(width * (1 - removal) + 0.5)))
        indices.append(np.sort(np.argsort(-score, kind="stable")[:keep]))
    compact = build_resnet18_model(tuple(model.input_shape[1:]), tuple(map(len, indices)))
    for layer in compact.layers:
        old = model.get_layer(layer.name)
        if not old.weights:
            continue
        if layer.name == "conv1":
            layer.set_weights([old.kernel.numpy()[:, :, :, indices[0]]])
        elif layer.name == "bn_conv1":
            layer.set_weights([w[indices[0]] for w in old.get_weights()])
        elif layer.name.startswith("conv") and "_block" in layer.name:
            stage = int(layer.name[4]) - 2
            out_idx = indices[stage]
            if isinstance(layer, keras.layers.Conv2D):
                transition = "_block1_" in layer.name and stage > 0 and (
                    layer.name.endswith("_conv1") or layer.name.endswith("_proj_conv"))
                in_idx = indices[stage - 1] if transition else out_idx
                kernel = np.take(np.take(old.kernel.numpy(), in_idx, axis=2), out_idx, axis=3)
                layer.set_weights([kernel])
            elif isinstance(layer, keras.layers.BatchNormalization):
                layer.set_weights([w[out_idx] for w in old.get_weights()])
        elif layer.name == "fc1":
            kernel, bias = old.get_weights()
            layer.set_weights([kernel[indices[-1]], bias])
        else:
            layer.set_weights(old.get_weights())
    plan = {"criterion": "sum of normalized mean-L1 filter scores from stage producers",
            "policy": "same sorted channel indices in stem/stage-2 and both branches of every stage",
            "original_widths": widths, "compact_widths": list(map(len, indices)),
            "surviving_channels": [idx.tolist() for idx in indices], "target_channel_removal": removal}
    return compact, plan
