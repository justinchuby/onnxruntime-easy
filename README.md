# onnxruntime-easy

Simplified APIs for onnxruntime

## Usage

```py
import onnxruntime_easy as ort_easy
import numpy as np
import ml_dtypes

# Simple `load` method that handles setting up ONNX Runtime inference session
# All session options discoverable in the load function.
model = ort_easy.load("model.onnx", device="cpu")  # You can control the providers if the default is not what you need
# Supports all ONNX dtypes via ml_dtypes or dlpack
input = ort_easy.ort_value(np.random.rand(1, 3, 299, 299).astype(ml_dtypes.bfloat16))
output = model(input)

# Works with any ndarray that implements the __array__ interface
# Or automatically share data on device (like cuda) with dlpack
import torch
model = ort_easy.load("model.onnx", device="cuda")
input_tensor = ort_easy.ort_value(torch.rand(1, 3, 299, 299, device="cuda"))
output = model(input_tensor)

# Use a context manager to control the outputs you get
with model.set_outputs("output1"):
    output1 = model(input_tensor)
```
