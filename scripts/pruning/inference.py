"""Optional ONNX export and matched TensorRT 10 FP16 engine comparison."""
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from .benchmark import runtime_info, timing_summary
from .config import BenchmarkConfig
from .data import load_data_pt, preprocess


def inspect_onnx_sparsity(path: Path) -> dict:
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(path)).graph
    initializers = {w.name: numpy_helper.to_array(w) for w in graph.initializer}
    details = []
    for node in graph.node:
        if node.op_type not in ("Conv", "MatMul", "Gemm"):
            continue
        if len(node.input) < 2 or node.input[1] not in initializers:
            details.append({"name": node.name, "eligible": False, "reason": "nonconstant weight input"})
            continue
        weight = initializers[node.input[1]]
        if node.op_type == "Conv":
            axis = 1  # ONNX weights [K,C,R,S].
        elif node.op_type == "MatMul":
            axis = weight.ndim - 2
        else:
            transpose = next((a.i for a in node.attribute if a.name == "transB"), 0)
            axis = 1 if transpose else 0
        if weight.shape[axis] % 4:
            details.append({"name": node.name, "eligible": False, "reason": "reduction dimension not divisible by four"})
            continue
        grouped = np.moveaxis(weight, axis, -1).reshape(-1, 4)
        violations = int(np.count_nonzero(np.count_nonzero(grouped, axis=-1) > 2))
        details.append({"name": node.name, "op": node.op_type, "eligible": True,
                        "groups": len(grouped), "violating_groups": violations, "valid": violations == 0})
    return {"layers": details, "eligible_layers": sum(d["eligible"] for d in details),
            "valid_layers": sum(d.get("valid", False) for d in details)}


def export_onnx(checkpoint: Path, destination: Path, test_path: Path) -> dict:
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    from .model import load_model

    model = load_model(checkpoint)
    # Keras' official ONNX export API uses tf2onnx with the TensorFlow backend.
    # https://keras.io/api/models/model_saving_apis/export/
    model.export(str(destination), format="onnx", opset_version=17,
                 input_signature=[tf.TensorSpec((None, 64, 64, 3), tf.float32, name="images")])
    onnx.checker.check_model(str(destination))
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=session_options, providers=["CPUExecutionProvider"])
    x, _, _ = load_data_pt(test_path, normalized=False)
    batch = preprocess(x[:32])
    reference = model(batch, training=False).numpy()
    converted = session.run(None, {session.get_inputs()[0].name: batch})[0]
    np.testing.assert_allclose(converted, reference, rtol=1e-3, atol=1e-4)
    pattern = inspect_onnx_sparsity(destination)
    inspection = subprocess.run(["polygraphy", "inspect", "sparsity", str(destination)],
                                 capture_output=True, text=True, timeout=120, check=True)
    (destination.parent / "polygraphy-sparsity.log").write_text(inspection.stdout + inspection.stderr, encoding="utf-8")
    return {"method": "export", "checkpoint": str(checkpoint), "onnx": str(destination),
            "eligible_layers": pattern["eligible_layers"], "sparse_layers": pattern["valid_layers"],
            "parity_samples": min(32, len(x)),
            "max_absolute_error": float(np.max(np.abs(reference - converted)))}


def _engine(onnx_path: Path, destination: Path, sparse: bool, batch: int, log_path: Path):
    import tensorrt as trt

    if int(trt.__version__.split(".")[0]) != 10:
        raise RuntimeError("This verified builder targets TensorRT 10.x; TensorRT 11 changes typing/precision flags.")
    messages = []

    class Logger(trt.ILogger):
        def __init__(self):
            super().__init__()

        def log(self, severity, message):
            messages.append(f"{severity}: {message}")
            # Retain live build evidence even if a long builder process is interrupted.
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"{severity}: {message}\n")

    logger = Logger()
    try:
        with trt.Builder(logger) as builder, builder.create_network(0) as network, trt.OnnxParser(network, logger) as parser:
            if not parser.parse_from_file(str(onnx_path)):
                raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            config.clear_flag(trt.BuilderFlag.TF32)
            if sparse:
                config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            shape = (batch, 64, 64, 3)
            profile.set_shape(input_tensor.name, shape, shape, shape)
            config.add_optimization_profile(profile)
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("TensorRT failed to build engine; inspect builder log.")
            destination.write_bytes(bytes(serialized))
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(destination.read_bytes())
        if engine is None:
            raise RuntimeError("TensorRT could not deserialize the newly built engine.")
        inspector = engine.create_engine_inspector()
        layer_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        destination.with_suffix(".layers.json").write_text(layer_info, encoding="utf-8")
        # Keep runtime/logger alive for engine lifetime.
        return engine, runtime, logger, messages
    finally:
        log_path.write_text("\n".join(messages), encoding="utf-8")


def _accuracy(runner, input_name: str, x, y, batch: int) -> dict:
    import tensorflow as tf
    from tensorflow import keras

    metrics = [keras.metrics.BinaryAccuracy(name="binary_accuracy"), keras.metrics.Precision(name="precision"),
               keras.metrics.Recall(name="recall")]
    exact, loss = 0, 0.0
    for start in range(0, len(x), batch):
        count = min(batch, len(x) - start)
        inputs = preprocess(x[start:start + count])
        if count < batch:
            inputs = np.concatenate([inputs, np.repeat(inputs[-1:], batch - count, axis=0)])
        pred = next(iter(runner.infer({input_name: inputs}).values()))[:count].copy()
        if not np.isfinite(pred).all():
            raise RuntimeError("TensorRT returned nonfinite predictions.")
        labels = y[start:start + count]
        for metric in metrics:
            metric.update_state(labels, pred)
        exact += int(np.all((pred >= .5) == labels, axis=1).sum())
        loss += float(tf.reduce_sum(keras.losses.binary_crossentropy(labels, pred)))
    return {**{m.name: float(m.result()) for m in metrics}, "exact_match_accuracy": exact / len(x), "loss": loss / len(x)}


def tensorrt_comparison(dense: Path, sparse: Path, test_path: Path, model_dir: Path,
                        config: BenchmarkConfig) -> dict:
    import tensorrt as trt
    from polygraphy.backend.trt import TrtRunner

    sparse_check = inspect_onnx_sparsity(sparse)
    convs = [l for l in sparse_check["layers"] if l.get("op") == "Conv" and l["eligible"]]
    if not convs or not all(l["valid"] for l in convs):
        raise ValueError("Sparse ONNX has invalid 2:4 convolution groups.")
    x, y, _ = load_data_pt(test_path, normalized=False)
    if len(x) < config.batch_size:
        raise ValueError("Test set is smaller than the inference batch size.")
    measurements = {}
    # The same pruned ONNX with sparse tactics OFF isolates acceleration from
    # any unrelated weight-dependent engine optimizations in the dense baseline.
    for label, path, enable_sparse in (("dense", dense, False),
                                      ("sparse_dense_tactics", sparse, False), ("sparse", sparse, True)):
        engine_path = model_dir / f"{label}.engine"
        engine, runtime, logger, messages = _engine(path, engine_path, enable_sparse, config.batch_size,
                                                     model_dir / f"{label}-builder.log")
        with TrtRunner(engine) as runner:
            name = next(iter(runner.get_input_metadata()))
            accuracy = _accuracy(runner, name, x, y, config.batch_size)
            inputs = {name: preprocess(x[:config.batch_size])}
            for _ in range(config.warmup):
                runner.infer(inputs)
            samples = []
            for _ in range(config.repeats):
                start = time.perf_counter_ns()
                runner.infer(inputs)  # Polygraphy copies outputs to host and synchronizes its stream.
                samples.append((time.perf_counter_ns() - start) / 1e6)
        chosen_counts = re.findall(r"\(Sparsity\) Chose (\d+) layer", "\n".join(messages))
        measurements[label] = {**accuracy, "model_mb": engine_path.stat().st_size / 1024**2,
                               "latency": timing_summary(samples, config.batch_size),
                               "sparse_layers_selected": int(chosen_counts[-1]) if chosen_counts else None}
        del runner, engine, runtime, logger
    dense_result, sparse_result = measurements["dense"], measurements["sparse"]
    controlled_ratio = measurements["sparse_dense_tactics"]["latency"]["median_ms"] / sparse_result["latency"]["median_ms"]
    return {"method": "tensorrt", "runtime": {**runtime_info(config), "framework": f"TensorRT {trt.__version__}",
                                               "precision": "fp16-enabled"},
            "dense_onnx": str(dense), "sparse_onnx": str(sparse), **measurements,
            "sparse_tactic_speedup": controlled_ratio,
            "exact_match_delta": sparse_result["exact_match_accuracy"] - dense_result["exact_match_accuracy"]}
