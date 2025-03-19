from __future__ import annotations

__all__ = [
    "load",
]

from typing import Literal, Mapping, Sequence
import numpy as np
import onnxruntime as ort


class _WrappedSession(ort.InferenceSession):
    """
    A wrapper around the ONNX Runtime InferenceSession to provide a more user-friendly
    interface for running inference on ONNX models.
    """

    def __init__(self, *args, device: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def __call__(self, *inputs: np.ndarray):
        input_names = [inp.name for inp in self.get_inputs()]
        ort_inputs = {
            name: ort.OrtValue.ortvalue_from_numpy(inp, self.device)
            for name, inp in zip(input_names, inputs)
        }
        output_names = [out.name for out in self.get_outputs()]
        ort_outputs = self.run_with_ort_values(output_names, ort_inputs)
        return [output.numpy() for output in ort_outputs]


def _get_providers(device: str) -> tuple[str, ...]:
    if device == "cpu":
        return ("CPUExecutionProvider",)
    if device == "cuda":
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    raise ValueError(f"Unsupported device: {device}")


def _get_execution_order(order: str) -> ort.ExecutionOrder:
    orders = {
        "default": ort.ExecutionOrder.DEFAULT,
        "priority_based": ort.ExecutionOrder.PRIORITY_BASED,
        "memory_efficient": ort.ExecutionOrder.MEMORY_EFFICIENT,
    }
    if order not in orders:
        raise ValueError(f"Unsupported execution order: {order}")
    return orders[order]


def _get_graph_optimization_level(level: str) -> ort.GraphOptimizationLevel:
    levels = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if level not in levels:
        raise ValueError(f"Unsupported graph optimization level: {level}")
    return levels[level]


def _get_severity_level(level: str) -> int:
    levels = {
        "info": 1,
        "warning": 2,
        "error": 3,
        "fatal": 4,
    }
    if level not in levels:
        raise ValueError(f"Unsupported severity level: {level}")
    return levels[level]


def load(
    model_path: str,
    /,
    device: Literal["cpu", "cuda"] = "cpu",
    # TODO: Support device ID
    *,
    enable_cpu_mem_arena: bool = True,
    enable_mem_pattern: bool = True,
    enable_mem_reuse: bool = True,
    enable_profiling: bool = False,
    execution_order: Literal[
        "default", "priority_based", "memory_efficient"
    ] = "default",
    graph_optimization_level: Literal[
        "disabled",
        "basic",
        "extended",
        "all",
    ] = "all",
    inter_op_num_threads: int = 0,
    intra_op_num_threads: int = 0,
    log_severity_level: Literal["info", "warning", "error", "fatal"] = "error",
    log_verbosity_level: int = 0,
    profile_file_prefix: str | None = None,
    custom_ops_libraries: Sequence[str] = (),
    use_deterministic_compute: bool = False,
    external_initializers: Mapping[str, np.ndarray] | None = None,
    optimized_model_filepath: str | None = None,
):
    """
    Load a model from a file.

    Args:
        model_path: Path to the model file.
        device: Device to run the model on. Can be "cpu" or "cuda".

    Returns:
        An inference session for the model.
    """
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = enable_cpu_mem_arena
    opts.enable_mem_pattern = enable_mem_pattern
    opts.enable_mem_reuse = enable_mem_reuse
    opts.enable_profiling = enable_profiling
    opts.execution_order = _get_execution_order(execution_order)
    opts.graph_optimization_level = _get_graph_optimization_level(
        graph_optimization_level
    )
    opts.inter_op_num_threads = inter_op_num_threads
    opts.intra_op_num_threads = intra_op_num_threads
    opts.log_severity_level = _get_severity_level(log_severity_level)
    opts.log_verbosity_level = log_verbosity_level
    if profile_file_prefix is not None:
        opts.profile_file_prefix = profile_file_prefix
    opts.use_deterministic_compute = use_deterministic_compute
    if external_initializers is not None:
        names, values = zip(*external_initializers.items())
        ort_values = [
            ort.OrtValue.ortvalue_from_numpy(value, device) for value in values
        ]
        opts.add_external_initializers(names, ort_values)
    if optimized_model_filepath is not None:
        opts.optimized_model_filepath = optimized_model_filepath
    for library in custom_ops_libraries:
        opts.register_custom_ops_library(library)

    return _WrappedSession(
        model_path,
        sess_options=opts,
        providers=_get_providers(device),
        device=device,
    )
