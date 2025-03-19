"""ONNX Runtime made easy."""

from __future__ import annotations

import contextlib

__all__ = [
    "EasySession",
    "load",
]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol

import ml_dtypes
import numpy as np
import onnxruntime as ort
import onnxruntime.capi._pybind_state as _ort_c

if TYPE_CHECKING:
    import numpy.typing as npt


class DLPackCompatible(Protocol):
    def __dlpack__(self) -> object: ...


__version__ = "0.0.2"


_BFLOAT16_TYPE = 16
_FLOAT8E4M3FN_TYPE = 17
_FLOAT8E4M3FNUZ_TYPE = 18
_FLOAT8E5M2_TYPE = 19
_FLOAT8E5M2FNUZ_TYPE = 20
_UINT4_TYPE = 21
_INT4_TYPE = 22
_FLOAT4E2M1_TYPE = 23


def _ml_dtypes_to_onnx_type(dtype: np.dtype) -> int | None:  # noqa: PLR0911
    """Convert a NumPy dtype to an ONNX type."""
    if dtype == ml_dtypes.bfloat16:
        return _BFLOAT16_TYPE
    if dtype == ml_dtypes.float8_e4m3fn:
        return _FLOAT8E4M3FN_TYPE
    if dtype == ml_dtypes.float8_e4m3fnuz:
        return _FLOAT8E4M3FNUZ_TYPE
    if dtype == ml_dtypes.float8_e5m2:
        return _FLOAT8E5M2_TYPE
    if dtype == ml_dtypes.float8_e5m2fnuz:
        return _FLOAT8E5M2FNUZ_TYPE
    if dtype == ml_dtypes.uint4:
        return _UINT4_TYPE
    if dtype == ml_dtypes.int4:
        return _INT4_TYPE
    if dtype == ml_dtypes.float4_e2m1fn:
        return _FLOAT4E2M1_TYPE
    return None


def _to_ort_value(value: npt.ArrayLike | DLPackCompatible, device: str) -> ort.OrtValue:
    """Convert a NumPy array or a DLPack-compatible object to an ONNX Runtime OrtValue."""
    # TODO: Update this call when dlpack support in OrtValue is improved
    if hasattr(value, "__dlpack__"):
        return ort.OrtValue(
            _ort_c.OrtValue.from_dlpack(value.__dlpack__(), False), value
        )
    if isinstance(value, np.ndarray):
        maybe_onnx_type = _ml_dtypes_to_onnx_type(value.dtype)
        if maybe_onnx_type is not None:
            return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                value, onnx_element_type=maybe_onnx_type
            )
    return ort.OrtValue.ortvalue_from_numpy(np.asarray(value), device)


class EasySession(ort.InferenceSession):
    """An inference session where everything is easy.

    This is a wrapper around the ONNX Runtime InferenceSession to provide a
    more user-friendly interface for running inference on ONNX models. It makes
    the model callable and supports Pythonic argument passing.

    Inputs can be anything that is convertible to a NumPy array or a DLPack-compatible
    object. The outputs are returned as NumPy arrays.

    Example usage::
        import onnxruntime_easy as ort_easy
        session = ort_easy.load("model.onnx", device="cuda")
        result = session(input1, input2, input3)
        # result is a list of NumPy arrays corresponding to the model outputs.
    """

    def __init__(  # noqa: D107
        self,
        *args,
        device: str,
        log_severity_level: Literal["info", "warning", "error", "fatal"] = "error",
        log_verbosity_level: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        # Internal state to store the requested output names
        self._requested_outputs: tuple[str, ...] | None = None
        # Run options for the session
        self._run_options = ort.RunOptions()
        self._run_options.log_severity_level = _get_severity_level(log_severity_level)
        self._run_options.log_verbosity_level = log_verbosity_level

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}(device={self.device}, "
            f"inputs={[inp.name for inp in self.get_inputs()]}, "
            f"outputs={[out.name for out in self.get_outputs()]})"
        )

    def __call__(
        self,
        *args: npt.ArrayLike | DLPackCompatible,
        **kwargs: npt.ArrayLike | DLPackCompatible,
    ) -> Sequence[npt.NDArray]:
        """Run inference on the model with the given inputs.

        Inputs can be anything that is convertible to a NumPy array or a DLPack-compatible
        object. The outputs are returned as NumPy arrays.
        """
        input_names = [inp.name for inp in self.get_inputs()]
        ort_inputs = {
            name: _to_ort_value(inp, self.device)
            for name, inp in zip(input_names, args)
        }
        ort_inputs.update(
            {name: _to_ort_value(inp, self.device) for name, inp in kwargs.items()}
        )
        ort_outputs = self.run_with_ort_values(
            self._requested_outputs, ort_inputs, run_options=self._run_options
        )
        return [output.numpy() for output in ort_outputs]

    @contextlib.contextmanager
    def set_outputs(self, *output_names: str):
        """Temporarily set the output names for the next inference call.

        Use this context manager to specify which outputs you want to retrieve
        from the model. This is useful for models with multiple outputs where
        you only want to get a subset of them.

        Example::
            with session.set_outputs("output1", "output2"):
                results = session(input_data)
                assert len(results) == 2
                # Only output1 and output2 will be returned.
        """
        prev_outputs = self._requested_outputs
        self._requested_outputs = output_names
        try:
            yield
        finally:
            self._requested_outputs = prev_outputs


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


def load(  # noqa: D417
    model_path: str,
    /,
    device: Literal["cpu", "cuda"] = "cpu",
    # TODO: Support device ID
    *,
    providers: Sequence[str] = (),
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
    external_initializers: Mapping[str, npt.ArrayLike | DLPackCompatible] | None = None,
    optimized_model_filepath: str | None = None,
) -> EasySession:
    """Load a model from a file.

    Args:
        model_path: Path to the model file.
        device: Device to run the model on. Can be "cpu" or "cuda". Overridden when
            providers are specified.

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
        ort_values = [_to_ort_value(value, device) for value in values]
        opts.add_external_initializers(names, ort_values)
    if optimized_model_filepath is not None:
        opts.optimized_model_filepath = optimized_model_filepath
    for library in custom_ops_libraries:
        opts.register_custom_ops_library(library)

    return EasySession(
        model_path,
        sess_options=opts,
        providers=providers if providers else _get_providers(device),
        device=device,
        log_severity_level=log_severity_level,
        log_verbosity_level=log_verbosity_level,
    )
