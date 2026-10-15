"""Generate tiny synthetic graphs for contract/integration tests, never ship as trained models."""
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

output = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
output.mkdir(parents=True, exist_ok=True)
for kind in ["detector", "classifier"]:
    count = 5 if kind == "detector" else 2
    weights = np.zeros((count, 3), np.float32)
    bias = np.zeros(count, np.float32)
    if kind == "detector":
        bias[:4] = 16
        weights[4] = 1 / 3
    else:
        weights[0], weights[1] = 1 / 3, -1 / 3
        bias[1] = 1
    initializers = [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")]
    nodes = [helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
             helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
             helper.make_node("Gemm", ["flat", "weights", "bias"], ["scores"], transB=1)]
    shape = [1, count]
    result = "scores"
    if kind == "detector":
        initializers.append(numpy_helper.from_array(np.array([1, count, 1], np.int64), "shape"))
        nodes.append(helper.make_node("Reshape", ["scores", "shape"], ["output"]))
        result, shape = "output", [1, count, 1]
    graph = helper.make_graph(nodes, "synthetic-" + kind,
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
                              [helper.make_tensor_value_info(result, TensorProto.FLOAT, shape)], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, output / f"{kind}.onnx")
    print(output / f"{kind}.onnx")
