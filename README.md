# onnxruntime-easy

Simplified APIs for onnxruntime

## Usage

```py
import onnxruntime_easy as ort
import numpy as np

model = ort.load("model.onnx", device="cpu")
input = np.random.rand(1, 3, 299, 299).astype(np.float32)
output = model(input)

import torch
input_tensor = torch.rand(1, 3, 299, 299)
output = model(input)
```
