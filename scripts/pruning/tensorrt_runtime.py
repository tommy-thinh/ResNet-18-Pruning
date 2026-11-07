"""Resident-buffer CUDA-event timing, separate from host input/output latency."""
from contextlib import ExitStack
import ctypes as ct
import time

import numpy as np

from .config import normalize_device


def select_cuda_device(device) -> None:
    from polygraphy import cuda

    device = normalize_device(device)
    if device == "/CPU:0":
        raise ValueError("TensorRT requires a GPU device.")
    _call(cuda.wrapper(), "cudaSetDevice", ct.c_int(int(device.rsplit(":", 1)[1])))


def _call(runtime, name, *args) -> None:
    status = getattr(runtime.handle, name)(*args)
    if status != 0:
        raise RuntimeError(f"{name} failed with CUDA error {status}.")


def _enqueue(context, stream) -> None:
    if not context.execute_async_v3(stream.ptr):
        raise RuntimeError("TensorRT execute_async_v3 failed.")


def _gpu_samples(runner, config, cuda_graph: bool) -> list[float]:
    from polygraphy import cuda

    runtime = cuda.wrapper()
    stream = ct.c_void_p(runner.stream.ptr)
    with ExitStack() as cleanup:
        start, stop = ct.c_void_p(), ct.c_void_p()
        for event in (start, stop):
            _call(runtime, "cudaEventCreate", ct.byref(event))
            cleanup.callback(_call, runtime, "cudaEventDestroy", event)
        launch = lambda: _enqueue(runner.context, runner.stream)
        if cuda_graph:
            graph, executable = ct.c_void_p(), ct.c_void_p()
            # Flush shape/buffer updates before capture; pointers stay alive until replay ends.
            # https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html#cuda-graphs
            launch()
            runner.stream.synchronize()
            _call(runtime, "cudaStreamBeginCapture", stream, ct.c_int(1))
            try:
                launch()
            finally:
                status = runtime.handle.cudaStreamEndCapture(stream, ct.byref(graph))
                if graph.value:
                    cleanup.callback(_call, runtime, "cudaGraphDestroy", graph)
            if status != 0:
                raise RuntimeError(f"CUDA graph capture failed ({status}); use --no-cuda-graph for every comparison.")
            _call(runtime, "cudaGraphInstantiateWithFlags", ct.byref(executable), graph, ct.c_ulonglong(0))
            cleanup.callback(_call, runtime, "cudaGraphExecDestroy", executable)
            launch = lambda: _call(runtime, "cudaGraphLaunch", executable, stream)
        cleanup.callback(runner.stream.synchronize)
        for _ in range(config.warmup):
            launch()
        runner.stream.synchronize()
        samples = []
        # Every sample is GPU timestamped, not a Python wall-clock interval.
        # https://docs.nvidia.com/cuda/archive/12.5.0/cuda-runtime-api/group__CUDART__EVENT.html
        for _ in range(config.repeats):
            elapsed = ct.c_float()
            _call(runtime, "cudaEventRecord", start, stream)
            launch()
            _call(runtime, "cudaEventRecord", stop, stream)
            _call(runtime, "cudaEventSynchronize", stop)
            _call(runtime, "cudaEventElapsedTime", ct.byref(elapsed), start, stop)
            samples.append(float(elapsed.value))
        return samples


def _layer_profile(runner, repeats: int) -> list[dict]:
    import tensorrt as trt

    class Profiler(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.timings = {}

        def report_layer_time(self, name, milliseconds):
            self.timings.setdefault(name, []).append(milliseconds)

    profiler = Profiler()
    runner._latency_profiler = profiler  # TensorRT holds a non-owning callback pointer.
    runner.context.profiler = profiler
    runner.context.enqueue_emits_profile = True
    for _ in range(min(repeats, 100)):
        _enqueue(runner.context, runner.stream)
        runner.stream.synchronize()
    return [{"name": name, "mean_ms": float(np.mean(values))}
            for name, values in profiler.timings.items()]


def measure_latency(runner, batch: np.ndarray, config, *, cuda_graph=False, profile=False) -> dict:
    from polygraphy import cuda
    import tensorrt as trt
    from .benchmark import timing_summary

    names = list(runner.get_input_metadata())
    if len(names) != 1:
        raise ValueError("This benchmark requires a single image input.")
    name = names[0]
    runner.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
    if hasattr(runner.context, "set_all_tensors_debug_state"):
        runner.context.set_all_tensors_debug_state(False)
    feed = {name: batch}
    for _ in range(config.warmup):
        runner.infer(feed)
    host_samples = []
    for _ in range(config.repeats):
        start = time.perf_counter_ns()
        runner.infer(feed)  # Includes H2D, D2H and stream synchronization.
        host_samples.append((time.perf_counter_ns() - start) / 1e6)
    reference = {key: value.copy() for key, value in runner.infer(feed).items()}
    with cuda.DeviceArray(shape=batch.shape, dtype=batch.dtype) as resident:
        resident.copy_from(batch)
        # Initialize all TensorRT buffers before direct enqueue/capture; no copies in GPU timing.
        outputs = runner.infer({name: resident}, copy_outputs_to_host=False)
        gpu_samples = _gpu_samples(runner, config, cuda_graph)
        if not np.isfinite(gpu_samples).all() or min(gpu_samples) <= 0:
            raise RuntimeError("CUDA events returned invalid GPU timings.")
        for key, output in outputs.items():
            np.testing.assert_allclose(output.numpy(), reference[key], rtol=1e-4, atol=1e-5,
                                       err_msg="Resident-buffer inference differs from host inference")
        result = {"gpu_latency": timing_summary(gpu_samples, config.batch_size),
                  "host_latency": timing_summary(host_samples, config.batch_size)}
        if profile:
            result["layer_profile"] = _layer_profile(runner, config.repeats)
    return result
