from typing import Literal, Mapping, Sequence
import numpy as np
import onnxruntime as ort


class WrappedSession(ort.InferenceSession):
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


def load(
    model_path: str,
    /,
    device: Literal["cpu", "cuda"] = "cpu",
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
    log_severity_level: int = 2,
    log_verbosity_level: int = 0,
    profile_file_prefix: str = "",
    custom_ops_libraries: Sequence[str] = (),
    use_deterministic_compute: bool = False,
    external_initializers: Mapping[str, ort.OrtValue] | None = None,
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
    return WrappedSession(
        model_path,
        providers=_get_providers(device),
        device="cuda",
    )
