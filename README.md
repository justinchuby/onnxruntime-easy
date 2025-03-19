# onnxruntime-easy

Simplified APIs for onnxruntime

## Usage

```py
import onnxruntime_easy as ort_easy
import numpy as np
import ml_dtypes

# Simple `load` method that handles setting up ONNX Runtime inference session
model = ort_easy.load("model.onnx", device="cpu")
# Supports all ONNX dtypes via ml_dtypes or dlpack
input = np.random.rand(1, 3, 299, 299).astype(ml_dtypes.bfloat16)
output = model(input)

# Works with torch tensors and any ndarray that implements the __array__ interface
import torch
input_tensor = torch.rand(1, 3, 299, 299)
output = model(input)

with model.set_outputs("output1"):
    output1 = model(input)
```
