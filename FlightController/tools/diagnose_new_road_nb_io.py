"""Diagnose float input/output handling for the new road NBG on STM32MP25.

This tool only loads one still image and runs ``stai_mpu``.  It never opens
the flight controller, radar, or camera devices.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image")
    parser.add_argument("--tensor", help="Text tensor with one float value per line")
    parser.add_argument("--input-dtype", choices=("float32", "float16"), required=True)
    parser.add_argument("--color-order", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument("--input-scale", choices=("unit", "byte"), default="unit")
    parser.add_argument("--flatten", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from stai_mpu import stai_mpu_network

    network = stai_mpu_network(model_path=args.model, use_hw_acceleration=True)
    input_info = network.get_input_infos()[0]
    output_info = network.get_output_infos()[0]
    print(
        "input_info:",
        "name=", input_info.get_name(),
        "shape=", list(input_info.get_shape()),
        "dtype=", repr(input_info.get_dtype()),
        "qtype=", repr(input_info.get_qtype()),
    )
    print(
        "output_info:",
        "name=", output_info.get_name(),
        "shape=", list(output_info.get_shape()),
        "dtype=", repr(output_info.get_dtype()),
        "qtype=", repr(output_info.get_qtype()),
    )

    input_shape = tuple(int(value) for value in input_info.get_shape())
    if args.tensor:
        blob = np.loadtxt(args.tensor, dtype=np.float32).reshape(input_shape)
    else:
        if not args.image:
            raise ValueError("one of --image or --tensor is required")
        frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(args.image)
        height, width = input_shape[2:4]
        image = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
        if args.color_order == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = image.astype(np.float32)
        if args.input_scale == "unit":
            blob /= 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
    blob = blob.astype(np.dtype(args.input_dtype))
    if args.flatten:
        blob = blob.reshape(-1)
    else:
        blob = np.ascontiguousarray(blob)
    print(
        "input:", blob.shape, blob.dtype, "c_contiguous=", blob.flags.c_contiguous,
        "min=", float(blob.min()), "mean=", float(blob.mean()), "max=", float(blob.max()),
    )

    network.set_input(0, blob)
    started = time.perf_counter()
    network.run()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    output = np.asarray(network.get_output(0)).copy()
    print(
        "output:", output.shape, output.dtype,
        "finite=", bool(np.isfinite(output).all()),
        "min=", float(output.min()), "mean=", float(output.mean()), "max=", float(output.max()),
    )
    print("first_values:", output.reshape(-1)[:16].tolist())
    if output.ndim == 4 and output.shape[1] == 2:
        class_map = np.argmax(output.astype(np.float32), axis=1)
        print("road_fraction:", float(np.mean(class_map == 1)))
        for class_id in range(2):
            channel = output[0, class_id].astype(np.float32)
            print(
                f"channel_{class_id}:",
                "min=", float(channel.min()),
                "mean=", float(channel.mean()),
                "max=", float(channel.max()),
            )
    print("latency_ms:", elapsed_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
