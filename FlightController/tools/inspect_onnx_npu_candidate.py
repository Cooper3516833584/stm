"""Statically inspect an ONNX model before spending time generating .nb.

This is a fast preflight check.  It cannot prove that ST Cloud will emit an
INT8 NPU .nb, but it quickly separates FP32 models from QDQ/QOperator models
and highlights graph features that commonly lead to fallback or failed compile.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


RISKY_OPS = {
    "ConvTranspose": "often falls back or compiles poorly on this target",
    "NonMaxSuppression": "postprocess op; avoid inside NPU graph",
    "NonMaxPool": "dilation variants are not supported by VSINPU",
    "RoiAlign": "detection/instance-seg postprocess; avoid for NPU path",
    "TopK": "postprocess-style op; verify compiler support",
}

SHAPE_OPS = {
    "Shape",
    "Gather",
    "GatherElements",
    "GatherND",
    "Slice",
    "Concat",
    "Reshape",
    "Unsqueeze",
    "Squeeze",
    "Expand",
}

QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}
QOPERATOR_PREFIXES = ("QLinear", "DynamicQuantizeLinear")
QOPERATOR_OPS = {
    "ConvInteger",
    "MatMulInteger",
    "IntegerConv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ONNX graph contract and quantization markers."
    )
    parser.add_argument("model", help="Path to .onnx model.")
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of most common ops to print.",
    )
    parser.add_argument(
        "--expect-input",
        default="1,3,256,256",
        help="Expected NCHW input shape, or empty string to skip.",
    )
    parser.add_argument(
        "--expect-output",
        default="1,2,256,256",
        help="Expected output shape, or empty string to skip.",
    )
    return parser.parse_args()


def _parse_shape(text: str) -> list[int | str]:
    if not text:
        return []
    out: list[int | str] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            out.append(part)
    return out


def _dtype_name(elem_type: int) -> str:
    import onnx

    return onnx.TensorProto.DataType.Name(elem_type)


def _value_info_shape(value_info) -> list[int | str]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    shape: list[int | str] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(str(dim.dim_param))
        else:
            shape.append("?")
    return shape


def _value_info_dtype(value_info) -> str:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        return "UNKNOWN"
    return _dtype_name(tensor_type.elem_type)


def _format_io(items: Iterable) -> list[str]:
    return [
        f"{item.name} shape={_value_info_shape(item)} dtype={_value_info_dtype(item)}"
        for item in items
    ]


def _shape_matches(actual: list[int | str], expected: list[int | str]) -> bool:
    return bool(expected) and actual == expected


def _initializer_bytes(initializer) -> int:
    if initializer.raw_data:
        return len(initializer.raw_data)
    # Fallback for models storing typed repeated fields instead of raw_data.
    elem_count = 1
    for dim in initializer.dims:
        elem_count *= int(dim)
    elem_sizes = {
        1: 4,  # FLOAT
        2: 1,  # UINT8
        3: 1,  # INT8
        4: 2,  # UINT16
        5: 2,  # INT16
        6: 4,  # INT32
        7: 8,  # INT64
        10: 2,  # FLOAT16
        11: 8,  # DOUBLE
        12: 4,  # UINT32
        13: 8,  # UINT64
    }
    return elem_count * elem_sizes.get(int(initializer.data_type), 4)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[FAIL] model not found: {model_path}")
        return 2

    try:
        import onnx
    except ImportError:
        print("[FAIL] missing dependency: onnx")
        print("Install or activate an environment with the onnx Python package.")
        return 2

    model = onnx.load(str(model_path))
    graph = model.graph
    nodes = list(graph.node)
    op_counts = Counter(node.op_type for node in nodes)
    initializer_names = {item.name for item in graph.initializer}
    real_inputs = [item for item in graph.input if item.name not in initializer_names]
    outputs = list(graph.output)

    qdq_count = sum(op_counts[op] for op in QDQ_OPS)
    qoperator_count = 0
    for op_type, count in op_counts.items():
        if op_type.startswith(QOPERATOR_PREFIXES) or op_type in QOPERATOR_OPS:
            qoperator_count += count

    weight_type_counts = Counter(_dtype_name(init.data_type) for init in graph.initializer)
    weight_bytes = sum(_initializer_bytes(init) for init in graph.initializer)

    producers: dict[str, str] = {}
    consumers: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for name in node.output:
            producers[name] = node.op_type
        for name in node.input:
            consumers[name].append(node.op_type)

    print("=" * 72)
    print("  ONNX NPU candidate inspection")
    print("=" * 72)
    print(f"model: {model_path}")
    print(f"size : {model_path.stat().st_size / 1024 / 1024:.2f} MiB")
    print(f"ir_version: {model.ir_version}")
    print(
        "opsets: "
        + ", ".join(
            f"{opset.domain or 'ai.onnx'}={opset.version}"
            for opset in model.opset_import
        )
    )
    print(f"nodes: {len(nodes)}")
    print(f"initializers: {len(graph.initializer)} ({weight_bytes / 1024 / 1024:.2f} MiB)")

    print()
    print("inputs:")
    for line in _format_io(real_inputs):
        print(f"  - {line}")
    print("outputs:")
    for line in _format_io(outputs):
        print(f"  - {line}")

    print()
    print("initializer_dtypes:")
    for dtype, count in sorted(weight_type_counts.items()):
        print(f"  - {dtype}: {count}")

    print()
    print("top_ops:")
    for op_type, count in op_counts.most_common(max(1, args.top)):
        print(f"  - {op_type}: {count}")

    print()
    print("quantization_markers:")
    print(f"  QuantizeLinear: {op_counts.get('QuantizeLinear', 0)}")
    print(f"  DequantizeLinear: {op_counts.get('DequantizeLinear', 0)}")
    print(f"  QOperator-like: {qoperator_count}")

    print()
    print("risk_scan:")
    risk_count = 0
    for op_type, reason in sorted(RISKY_OPS.items()):
        count = op_counts.get(op_type, 0)
        if count:
            risk_count += count
            print(f"  - {op_type}: {count} ({reason})")
    shape_count = sum(op_counts.get(op_type, 0) for op_type in SHAPE_OPS)
    print(f"  - shape/control-ish ops total: {shape_count}")
    if risk_count == 0:
        print("  - no high-risk ops from the local denylist")

    print()
    print("boundary_quantization:")
    for item in real_inputs:
        consumer_ops = sorted(set(consumers.get(item.name, [])))
        print(f"  input {item.name}: first_consumers={consumer_ops[:6]}")
    for item in outputs:
        producer_op = producers.get(item.name, "<graph input or unknown>")
        print(f"  output {item.name}: producer={producer_op}")

    expected_input = _parse_shape(args.expect_input)
    expected_output = _parse_shape(args.expect_output)
    issues: list[str] = []
    warnings: list[str] = []

    if expected_input and real_inputs:
        actual = _value_info_shape(real_inputs[0])
        if not _shape_matches(actual, expected_input):
            issues.append(f"first input shape {actual} != expected {expected_input}")
    if expected_output and outputs:
        actual = _value_info_shape(outputs[0])
        if not _shape_matches(actual, expected_output):
            issues.append(f"first output shape {actual} != expected {expected_output}")

    input_dtypes = {_value_info_dtype(item) for item in real_inputs}
    output_dtypes = {_value_info_dtype(item) for item in outputs}
    if input_dtypes - {"FLOAT", "FLOAT16"}:
        warnings.append(f"non-float graph input dtype(s): {sorted(input_dtypes)}")
    if output_dtypes - {"FLOAT", "FLOAT16"}:
        warnings.append(f"non-float graph output dtype(s): {sorted(output_dtypes)}")

    if qdq_count == 0 and qoperator_count == 0:
        warnings.append(
            "no QDQ/QOperator markers; direct ST Optimize is likely FP16/fallback, not INT8"
        )
    elif qdq_count > 0:
        first_consumers = {
            op
            for item in real_inputs
            for op in consumers.get(item.name, [])
        }
        output_producers = {producers.get(item.name, "") for item in outputs}
        if "QuantizeLinear" not in first_consumers:
            warnings.append("QDQ exists but graph input is not immediately quantized")
        if "DequantizeLinear" not in output_producers:
            warnings.append("QDQ exists but graph output is not produced by DequantizeLinear")

    for op_type, reason in sorted(RISKY_OPS.items()):
        if op_counts.get(op_type, 0):
            warnings.append(f"{op_type} present: {reason}")

    print()
    print("assessment:")
    if qdq_count == 0 and qoperator_count == 0:
        print("  class: FP32_OR_FP16_ONNX")
        print("  likely .nb result: float16 unless ST Cloud quantization is applied")
    elif qdq_count > 0:
        print("  class: QDQ_QUANTIZED_ONNX")
        print("  likely .nb result: candidate for INT8; still must verify generated .nb")
    else:
        print("  class: QOPERATOR_OR_INTEGER_QUANTIZED_ONNX")
        print("  likely .nb result: uncertain; verify ST compiler support")

    if issues:
        print("  hard_issues:")
        for issue in issues:
            print(f"    - {issue}")
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")

    if issues:
        print("[FAIL] ONNX contract does not match expected deployment shape")
        return 1
    print("[OK] static inspection complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
