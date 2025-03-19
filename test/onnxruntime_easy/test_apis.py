import os
import tempfile
import unittest

import numpy as np
import onnx
import onnxscript

import onnxruntime_easy as ort_easy


def create_test_model():
    # Define a simple ONNX model using onnxscript
    @onnxscript.script(default_opset=onnxscript.opset18)
    def add_model(
        x: onnxscript.FLOAT["N"], y: onnxscript.FLOAT["N"]
    ) -> onnxscript.FLOAT["N"]:
        return x + y

    return add_model.to_model_proto()


class TestAPIs(unittest.TestCase):
    def setUp(self):
        self.model_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.model_dir.name, "model.onnx")
        onnx.save(create_test_model(), self.model_path)

    def test_load(self):
        model = ort_easy.load(self.model_path)
        outputs = model(
            np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)
        )
        np.testing.assert_equal(outputs[0], np.array([3.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
