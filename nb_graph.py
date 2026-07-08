"""NPU Network Binary Graph (.nb) loader for STM32MP2 VIP9000.

Wraps ST's ``stai_mpu.Network`` API with an interface that mirrors
``onnxruntime.InferenceSession`` so that the existing preprocessing /
postprocessing pipeline in ``road_perception.py`` works with minimal changes.

Usage (manual)::

    from nb_graph import NBGraphSession

    sess = NBGraphSession("FlightController/Solutions/model/road_yolo11n_seg_1.nb")
    print(sess.get_inputs()[0].name, sess.get_inputs()[0].shape)
    outputs = sess.run(None, {"images": input_blob})

Backends (tried in order):
1. ``stai_mpu.Network`` — ST's official Python API (requires ``python3-stai-mpu``)
2. *Future*: OpenVX ctypes fallback for boards without stai_mpu installed
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# ── sentinel objects for float32 fallback ──────────────────────────
_FLOAT32_MAX_INT8 = 127  # symmetric int8 max
_FLOAT32_SCALE_1_255 = np.float32(1.0 / 255.0)
_INT8_ZERO_POINT = np.int8(-128)


class _InputMeta:
    """Minimal mirror of ``onnxruntime.NodeArg`` for input metadata."""

    def __init__(self, name: str, shape: list, dtype: str = "tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = dtype


class _OutputMeta:
    """Minimal mirror of ``onnxruntime.NodeArg`` for output metadata."""

    def __init__(self, name: str, shape: list, dtype: str = "tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = dtype


# ────────────────────────────────────────────────────────────────────
# stai_mpu backend
# ────────────────────────────────────────────────────────────────────

def _load_via_stai_mpu(model_path: str) -> tuple[Any, list[_InputMeta], list[_OutputMeta]]:
    """Try to load the .nb file through ST's ``stai_mpu`` package.

    Returns ``(network, inputs_meta, outputs_meta)``.
    The returned *network* object must respond to:

    * ``network.run()`` — execute one inference
    * ``network.inputs[name].data = ndarray`` — feed input
    * ``network.outputs[name].data`` — read output
    """
    try:
        import stai_mpu  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "stai_mpu is not installed on this board.\n"
            "Install it via APT:  apt install python3-stai-mpu\n"
            "Or use an ONNX model with VSINPUExecutionProvider instead of a .nb file."
        )

    if not hasattr(stai_mpu, "Network"):
        raise RuntimeError(
            "stai_mpu imported but stai_mpu.Network is missing.\n"
            "Please update stai_mpu to a newer version."
        )

    network = stai_mpu.Network(model_path)

    # ── discover inputs ──
    inputs_meta: list[_InputMeta] = []
    input_names: list[str] = []
    if hasattr(network, "inputs") and network.inputs is not None:
        # stai_mpu v1.x: network.inputs is dict-like
        for name in network.inputs:
            tensor = network.inputs[name]
            shape = list(tensor.shape) if hasattr(tensor, "shape") else []
            dtype = _stai_dtype_to_ort(tensor)
            inputs_meta.append(_InputMeta(name=name, shape=shape, dtype=dtype))
            input_names.append(name)
    elif hasattr(network, "input_names"):
        # stai_mpu v0.x fallback
        for name in network.input_names:
            inputs_meta.append(_InputMeta(name=name, shape=[], dtype="tensor(float)"))
            input_names.append(name)
    else:
        raise RuntimeError("Cannot introspect .nb model inputs via stai_mpu.")

    # ── discover outputs ──
    outputs_meta: list[_OutputMeta] = []
    if hasattr(network, "outputs") and network.outputs is not None:
        for name in network.outputs:
            tensor = network.outputs[name]
            shape = list(tensor.shape) if hasattr(tensor, "shape") else []
            dtype = _stai_dtype_to_ort(tensor)
            outputs_meta.append(_OutputMeta(name=name, shape=shape, dtype=dtype))
    elif hasattr(network, "output_names"):
        for name in network.output_names:
            outputs_meta.append(_OutputMeta(name=name, shape=[], dtype="tensor(float)"))

    return network, inputs_meta, outputs_meta


def _stai_dtype_to_ort(tensor: Any) -> str:
    """Best-effort dtype label from a stai_mpu tensor object."""
    try:
        dt = tensor.dtype
    except AttributeError:
        return "tensor(float)"
    dt_str = str(dt).lower()
    if "float32" in dt_str or "float" in dt_str:
        return "tensor(float)"
    if "int8" in dt_str:
        return "tensor(int8)"
    if "uint8" in dt_str:
        return "tensor(uint8)"
    return "tensor(float)"


# ────────────────────────────────────────────────────────────────────
# Public session class
# ────────────────────────────────────────────────────────────────────

class NBGraphSession:
    """Load a compiled .nb network graph and expose an ONNX-Runtime-like API.

    ``providers`` is accepted for compatibility with the ``road_perception.py``
    call-site but is ignored — the .nb format is already compiled for the NPU.
    """

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"NB graph file not found: {model_path}")

        self._model_path = model_path
        self._backend = "unknown"
        self._internal: Any = None
        self._inputs_meta: list[_InputMeta] = []
        self._outputs_meta: list[_OutputMeta] = []
        self._input_name: str = ""
        self._output_names: list[str] = []

        # ── try backends ──
        last_err: Exception | None = None

        # 1. stai_mpu (preferred)
        try:
            self._internal, self._inputs_meta, self._outputs_meta = _load_via_stai_mpu(model_path)
            self._backend = "stai_mpu"
        except Exception as exc:
            last_err = exc

        if self._internal is None:
            msg = f"Failed to load .nb model '{model_path}' with any backend."
            if last_err is not None:
                msg += f"\n  stai_mpu: {last_err}"
            raise RuntimeError(msg)

        self._input_name = self._inputs_meta[0].name if self._inputs_meta else ""
        self._output_names = [o.name for o in self._outputs_meta]

    # ── public properties ────────────────────────────────────────

    @property
    def backend(self) -> str:
        return self._backend

    def get_providers(self) -> list[str]:
        return ["NPU_NBGraph"]  # custom label for logging

    def get_inputs(self) -> list[_InputMeta]:
        return self._inputs_meta  # type: ignore[return-value]

    def get_outputs(self) -> list[_OutputMeta]:
        return self._outputs_meta  # type: ignore[return-value]

    # ── inference ─────────────────────────────────────────────────

    def run(self, output_names: Optional[List[str]], feed_dict: Dict[str, np.ndarray]) -> list[np.ndarray]:
        """Execute one inference pass.

        Args:
            output_names: Pass ``None`` to retrieve all outputs.
            feed_dict: ``{input_name: numpy_blob}``.

        Returns:
            List of output numpy arrays in the same order as *output_names*
            (or in model-defined order when *output_names* is ``None``).
        """
        if self._backend == "stai_mpu":
            return self._run_stai_mpu(output_names, feed_dict)
        raise RuntimeError(f"Unknown backend: {self._backend}")

    def _run_stai_mpu(self, output_names: Optional[List[str]], feed_dict: Dict[str, np.ndarray]) -> list[np.ndarray]:
        network = self._internal

        # Feed inputs — handle dtype conversion if model expects int8
        for feed_name, blob in feed_dict.items():
            tensor = network.inputs[feed_name]
            expected_dtype = _stai_numpy_dtype(tensor)

            if expected_dtype is not None and blob.dtype != expected_dtype:
                blob = _convert_blob_dtype(blob, expected_dtype)

            tensor.data = blob

        # Run
        network.run()

        # Collect outputs
        names = output_names if output_names is not None else self._output_names
        results: list[np.ndarray] = []
        for name in names:
            out = network.outputs[name].data
            # Ensure float32 for downstream postprocessing
            results.append(_ensure_float32(np.asarray(out)))

        return results


# ────────────────────────────────────────────────────────────────────
# dtype utilities
# ────────────────────────────────────────────────────────────────────

def _stai_numpy_dtype(tensor: Any):
    """Map stai_mpu tensor dtype to numpy dtype, or None if unknown."""
    try:
        dt = str(tensor.dtype).lower()
    except AttributeError:
        return None
    if "float32" in dt:
        return np.float32
    if "int8" in dt:
        return np.int8
    if "uint8" in dt:
        return np.uint8
    if "float" in dt:
        return np.float32
    return None


def _convert_blob_dtype(blob: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Convert a float32 [0,1] NCHW blob to the dtype expected by the NPU."""
    if blob.dtype == target_dtype:
        return blob

    # float32 [0, 1] → int8 (symmetric quantization)
    if target_dtype == np.int8:
        # scale = 1/255 ≈ 0.00392157, zero_point = -128
        # value = (float_val / scale) + zero_point
        #       = float_val * 255 + (-128)
        # Since float_val ∈ [0, 1], result ∈ [-128, 127]
        scaled = (blob * 255.0).round().astype(np.int32)
        scaled += np.int32(-128)
        return scaled.astype(np.int8)

    # float32 → uint8
    if target_dtype == np.uint8:
        return (blob * 255.0).round().clip(0, 255).astype(np.uint8)

    # Generic fallback
    return blob.astype(target_dtype)


def _ensure_float32(arr: np.ndarray) -> np.ndarray:
    """Convert int8/uint8 output back to float32 if needed."""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr.astype(np.float32, copy=False)
    if arr.dtype == np.int8:
        # symmetric int8 → float32
        # float_val = (int_val - zero_point) * scale
        #           = (int_val + 128) / 255
        return (arr.astype(np.float32) + 128.0) * _FLOAT32_SCALE_1_255
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * _FLOAT32_SCALE_1_255
    return arr.astype(np.float32)


# ────────────────────────────────────────────────────────────────────
# file detection helper
# ────────────────────────────────────────────────────────────────────

def is_nb_model(model_path: str) -> bool:
    """Return True if *model_path* refers to a compiled .nb network binary."""
    if model_path.endswith(".nb"):
        return True
    # Check magic bytes for VPMN (VeriSilicon VIP Model Network)
    if os.path.isfile(model_path):
        try:
            with open(model_path, "rb") as fh:
                magic = fh.read(4)
            if magic == b"VPMN":
                return True
        except OSError:
            pass
    return False
