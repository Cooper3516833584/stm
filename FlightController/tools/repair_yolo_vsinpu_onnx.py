"""Rewrite YOLO ONNX graphs into a shape friendlier to STM32MP257 VSINPU.

This pass is intentionally narrow and conservative:
* Drop MaxPool dilations attributes when they are the default [1, 1].
* Replace constant-size Split nodes with Slice nodes.
* Replace ConvTranspose(kernel=2, stride=2, pad=0, group=1) with an exact
  sub-pixel Conv + Reshape + Transpose + Reshape sequence.

The ConvTranspose rewrite is exact for the pattern exported by the current
YOLO11 segmentation prototype head; it is not a generic deconvolution rewrite.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "FlightController" / "Solutions" / "model"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "npu_quantization"


@dataclass
class RewriteStats:
    maxpool_dilations_removed: int = 0
    split_to_slice: int = 0
    convtranspose_to_subpixel: int = 0
    unsupported_convtranspose: list[str] = field(default_factory=list)
    unsupported_split: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite YOLO ONNX ops that are awkward for VSINPU."
    )
    parser.add_argument("input", type=Path, help="Input FP32 ONNX model.")
    parser.add_argument("output", type=Path, help="Output rewritten ONNX model.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults to OUTPUT.with_suffix('.json').",
    )
    return parser.parse_args()


def tensor_name(base: str, suffix: str) -> str:
    return f"{base}{suffix}".replace(":", "_")


def get_attr(node: onnx.NodeProto, name: str, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def collect_constant_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                arrays[node.output[0]] = numpy_helper.to_array(attr.t)
                break
    return arrays


def collect_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = shape_inference.infer_shapes(model)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(
        inferred.graph.output
    )
    shapes: dict[str, list[int]] = {}
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        complete = True
        for dim in tensor_type.shape.dim:
            if dim.dim_value <= 0:
                complete = False
                break
            dims.append(int(dim.dim_value))
        if complete:
            shapes[value.name] = dims
    return shapes


def make_int64_initializer(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def make_node_like_without_attr(
    node: onnx.NodeProto, removed_attr_names: set[str]
) -> onnx.NodeProto:
    new_node = helper.make_node(
        node.op_type,
        inputs=list(node.input),
        outputs=list(node.output),
        name=node.name,
        domain=node.domain,
    )
    for attr in node.attribute:
        if attr.name not in removed_attr_names:
            new_node.attribute.append(attr)
    return new_node


def split_to_slices(
    node: onnx.NodeProto,
    constants: dict[str, np.ndarray],
    new_initializers: list[onnx.TensorProto],
    stats: RewriteStats,
) -> list[onnx.NodeProto] | None:
    if len(node.input) < 2 or node.input[1] not in constants:
        stats.unsupported_split.append(node.name or node.output[0])
        return None

    split_sizes = np.asarray(constants[node.input[1]], dtype=np.int64).reshape(-1)
    if len(split_sizes) != len(node.output):
        stats.unsupported_split.append(node.name or node.output[0])
        return None

    axis = int(get_attr(node, "axis", 0))
    offset = 0
    slice_nodes: list[onnx.NodeProto] = []
    base = node.name or node.output[0]
    for index, (size, output_name) in enumerate(zip(split_sizes, node.output)):
        start = int(offset)
        end = int(offset + int(size))
        offset = end

        starts_name = tensor_name(base, f"_slice_{index}_starts")
        ends_name = tensor_name(base, f"_slice_{index}_ends")
        axes_name = tensor_name(base, f"_slice_{index}_axes")
        steps_name = tensor_name(base, f"_slice_{index}_steps")
        new_initializers.extend(
            [
                make_int64_initializer(starts_name, [start]),
                make_int64_initializer(ends_name, [end]),
                make_int64_initializer(axes_name, [axis]),
                make_int64_initializer(steps_name, [1]),
            ]
        )
        slice_nodes.append(
            helper.make_node(
                "Slice",
                inputs=[node.input[0], starts_name, ends_name, axes_name, steps_name],
                outputs=[output_name],
                name=f"{base}/Slice_{index}",
            )
        )

    stats.split_to_slice += 1
    return slice_nodes


def convtranspose_to_subpixel(
    node: onnx.NodeProto,
    constants: dict[str, np.ndarray],
    shapes: dict[str, list[int]],
    new_initializers: list[onnx.TensorProto],
    stats: RewriteStats,
) -> list[onnx.NodeProto] | None:
    if len(node.input) < 2 or node.input[1] not in constants:
        stats.unsupported_convtranspose.append(node.name or node.output[0])
        return None

    strides = list(get_attr(node, "strides", [1, 1]))
    kernel_shape = list(get_attr(node, "kernel_shape", []))
    pads = list(get_attr(node, "pads", [0, 0, 0, 0]))
    dilations = list(get_attr(node, "dilations", [1, 1]))
    group = int(get_attr(node, "group", 1))

    if (
        strides != [2, 2]
        or kernel_shape != [2, 2]
        or pads != [0, 0, 0, 0]
        or dilations != [1, 1]
        or group != 1
    ):
        stats.unsupported_convtranspose.append(node.name or node.output[0])
        return None

    weight = constants[node.input[1]]
    if weight.ndim != 4:
        stats.unsupported_convtranspose.append(node.name or node.output[0])
        return None
    cin, cout, kh, kw = weight.shape
    if (kh, kw) != (2, 2):
        stats.unsupported_convtranspose.append(node.name or node.output[0])
        return None

    input_shape = shapes.get(node.input[0])
    output_shape = shapes.get(node.output[0])
    if (
        input_shape is None
        or output_shape is None
        or len(input_shape) != 4
        or len(output_shape) != 4
    ):
        stats.unsupported_convtranspose.append(node.name or node.output[0])
        return None

    batch, _, height, width = input_shape
    out_batch, out_channels, out_height, out_width = output_shape
    if out_batch != batch or out_channels != cout or out_height != height * 2 or out_width != width * 2:
        stats.unsupported_convtranspose.append(node.name or node.output[0])
        return None

    conv_weight = np.empty((cout * kh * kw, cin, 1, 1), dtype=weight.dtype)
    for co in range(cout):
        for sub_h in range(kh):
            for sub_w in range(kw):
                dst = (co * kh + sub_h) * kw + sub_w
                conv_weight[dst, :, 0, 0] = weight[:, co, sub_h, sub_w]

    base = node.name or node.output[0]
    conv_weight_name = tensor_name(base, "_subpixel_conv_weight")
    new_initializers.append(numpy_helper.from_array(conv_weight, conv_weight_name))

    conv_inputs = [node.input[0], conv_weight_name]
    if len(node.input) >= 3 and node.input[2] in constants:
        bias = np.asarray(constants[node.input[2]], dtype=weight.dtype).reshape(-1)
        if bias.shape[0] != cout:
            stats.unsupported_convtranspose.append(node.name or node.output[0])
            return None
        conv_bias_name = tensor_name(base, "_subpixel_conv_bias")
        new_initializers.append(
            numpy_helper.from_array(np.repeat(bias, kh * kw), conv_bias_name)
        )
        conv_inputs.append(conv_bias_name)
    elif len(node.input) >= 3:
        stats.unsupported_convtranspose.append(node.name or node.output[0])
        return None

    shape1_name = tensor_name(base, "_subpixel_shape1")
    shape2_name = tensor_name(base, "_subpixel_shape2")
    new_initializers.extend(
        [
            make_int64_initializer(shape1_name, [batch, cout, kh, kw, height, width]),
            make_int64_initializer(shape2_name, [batch, cout, out_height, out_width]),
        ]
    )

    conv_output = tensor_name(base, "_subpixel_conv_output")
    reshape1_output = tensor_name(base, "_subpixel_reshape1_output")
    transpose_output = tensor_name(base, "_subpixel_transpose_output")

    stats.convtranspose_to_subpixel += 1
    return [
        helper.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=[conv_output],
            name=f"{base}/SubpixelConv",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        ),
        helper.make_node(
            "Reshape",
            inputs=[conv_output, shape1_name],
            outputs=[reshape1_output],
            name=f"{base}/SubpixelReshape1",
        ),
        helper.make_node(
            "Transpose",
            inputs=[reshape1_output],
            outputs=[transpose_output],
            name=f"{base}/SubpixelTranspose",
            perm=[0, 1, 4, 2, 5, 3],
        ),
        helper.make_node(
            "Reshape",
            inputs=[transpose_output, shape2_name],
            outputs=list(node.output),
            name=f"{base}/SubpixelReshape2",
        ),
    ]


def rewrite_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, RewriteStats]:
    constants = collect_constant_arrays(model)
    shapes = collect_shapes(model)
    stats = RewriteStats()
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []

    for node in model.graph.node:
        if node.op_type == "MaxPool":
            dilations = list(get_attr(node, "dilations", []))
            if dilations in ([1, 1], [1]):
                new_nodes.append(make_node_like_without_attr(node, {"dilations"}))
                stats.maxpool_dilations_removed += 1
                continue

        if node.op_type == "Split":
            replacement = split_to_slices(node, constants, new_initializers, stats)
            if replacement is not None:
                new_nodes.extend(replacement)
                continue

        if node.op_type == "ConvTranspose":
            replacement = convtranspose_to_subpixel(
                node, constants, shapes, new_initializers, stats
            )
            if replacement is not None:
                new_nodes.extend(replacement)
                continue

        new_nodes.append(node)

    used_inputs = {
        input_name
        for node in new_nodes
        for input_name in node.input
        if input_name
    }
    graph_outputs = {output.name for output in model.graph.output}
    new_nodes = [
        node
        for node in new_nodes
        if node.op_type != "Constant"
        or any(output_name in used_inputs or output_name in graph_outputs for output_name in node.output)
    ]

    used_inputs = {
        input_name
        for node in new_nodes
        for input_name in node.input
        if input_name
    }
    initializers = [
        initializer
        for initializer in list(model.graph.initializer) + new_initializers
        if initializer.name in used_inputs
    ]

    rewritten = onnx.ModelProto()
    rewritten.CopyFrom(model)
    del rewritten.graph.node[:]
    del rewritten.graph.initializer[:]
    rewritten.graph.node.extend(new_nodes)
    rewritten.graph.initializer.extend(initializers)
    return rewritten, stats


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = (
        args.report.resolve() if args.report else output_path.with_suffix(".rewrite.json")
    )

    model = onnx.load(input_path)
    rewritten, stats = rewrite_model(model)
    onnx.checker.check_model(rewritten)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewritten, output_path)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "maxpool_dilations_removed": stats.maxpool_dilations_removed,
        "split_to_slice": stats.split_to_slice,
        "convtranspose_to_subpixel": stats.convtranspose_to_subpixel,
        "unsupported_convtranspose": stats.unsupported_convtranspose,
        "unsupported_split": stats.unsupported_split,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
