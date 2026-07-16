"""NPU Network Binary Graph (.nb) loader for STM32MP2 VIP9000.

Wraps ST's ``stai_mpu.stai_mpu_network`` API with an interface that
mirrors ``onnxruntime.InferenceSession`` so that the existing preprocessing /
postprocessing pipeline in ``road_perception.py`` works with minimal changes.

Usage::

    from nb_graph import NBGraphSession

    sess = NBGraphSession("model.nb")
    print(sess.get_inputs()[0].name, sess.get_inputs()[0].shape)
    outputs = sess.run(None, {"images": input_blob})

Requires: ``apt install python3-libstai-mpu``  (imports as ``stai_mpu``)
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np


# ── sentinel constants ────────────────────────────────────────────
_FLOAT32_SCALE_1_255 = np.float32(1.0 / 255.0)


# ── metadata mirrors of onnxruntime.NodeArg ──────────────────────

class _InputMeta:
    __slots__ = ("name", "shape", "type")

    def __init__(self, name: str, shape: List[int], dtype: str = "tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = dtype

    def __repr__(self) -> str:
        return f"_InputMeta(name={self.name!r}, shape={self.shape}, type={self.type!r})"


class _OutputMeta:
    __slots__ = ("name", "shape", "type")

    def __init__(self, name: str, shape: List[int], dtype: str = "tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = dtype

    def __repr__(self) -> str:
        return f"_OutputMeta(name={self.name!r}, shape={self.shape}, type={self.type!r})"


# ──────────────────────────────────────────────────────────────────
# stai_mpu backend  (real API: stai_mpu.stai_mpu_network)
# ──────────────────────────────────────────────────────────────────

def _load_via_stai_mpu(model_path: str) -> tuple[Any, List[_InputMeta], List[_OutputMeta], dict]:
    """Load .nb via ``stai_mpu.stai_mpu_network``.

    Returns ``(network, inputs_meta, outputs_meta, quant_info)`` where
    *quant_info* maps input/output names to ``(scale, zero_point)`` for
    staticAffine quantized tensors, or ``None`` for float tensors.
    """
    try:
        from stai_mpu import stai_mpu_network  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "stai_mpu is not installed.\n"
            "  Install:  apt install python3-libstai-mpu\n"
            "  Verify:   python3 -c \"from stai_mpu import stai_mpu_network; print('OK')\""
        )

    network = stai_mpu_network(model_path=model_path, use_hw_acceleration=True)

    inputs_meta: List[_InputMeta] = []
    outputs_meta: List[_OutputMeta] = []
    quant_info: dict[str, tuple[float, int] | None] = {}

    # ── inputs ──
    num_in = network.get_num_inputs()
    for i in range(num_in):
        info = network.get_input_infos()[i]
        shape = list(info.get_shape())
        name = info.get_name() or f"input_{i}"
        np_dtype = info.get_dtype()
        ort_dtype = _np_dtype_to_ort_type(np_dtype)
        inputs_meta.append(_InputMeta(name=name, shape=shape, dtype=ort_dtype))

        qtype = info.get_qtype()
        if qtype == "staticAffine":
            quant_info[name] = (float(info.get_scale()), int(info.get_zero_point()))
        else:
            quant_info[name] = None

    # ── outputs ──
    num_out = network.get_num_outputs()
    for i in range(num_out):
        info = network.get_output_infos()[i]
        shape = list(info.get_shape())
        name = info.get_name() or f"output_{i}"
        np_dtype = info.get_dtype()
        ort_dtype = _np_dtype_to_ort_type(np_dtype)
        outputs_meta.append(_OutputMeta(name=name, shape=shape, dtype=ort_dtype))

        qtype = info.get_qtype()
        if qtype == "staticAffine":
            quant_info[name] = (float(info.get_scale()), int(info.get_zero_point()))
        else:
            quant_info[name] = None

    return network, inputs_meta, outputs_meta, quant_info


def _np_dtype_to_ort_type(np_dtype) -> str:
    """Map numpy dtype to onnxruntime-style type string."""
    if np_dtype is None:
        return "tensor(float)"
    name = str(np_dtype).lower()
    if "float32" in name:
        return "tensor(float)"
    if "float16" in name:
        return "tensor(float16)"
    if "int8" in name:
        return "tensor(int8)"
    if "uint8" in name:
        return "tensor(uint8)"
    if "int32" in name:
        return "tensor(int32)"
    return "tensor(float)"


# ──────────────────────────────────────────────────────────────────
# Public session class
# ──────────────────────────────────────────────────────────────────

class NBGraphSession:
    """Load a compiled .nb network and expose an ONNX-Runtime-like API.

    ``providers`` is accepted for call-site compatibility but is
    ignored — the .nb format is already compiled for the NPU.
    """

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"NB graph file not found: {model_path}")

        self._model_path = model_path
        self._internal: Any
        self._inputs_meta: List[_InputMeta]
        self._outputs_meta: List[_OutputMeta]
        self._quant: dict[str, tuple[float, int] | None]

        (
            self._internal,
            self._inputs_meta,
            self._outputs_meta,
            self._quant,
        ) = _load_via_stai_mpu(model_path)

        self._backend = "stai_mpu"
        self._input_index: dict[str, int] = {
            meta.name: idx for idx, meta in enumerate(self._inputs_meta)
        }
        self._output_index: dict[str, int] = {
            meta.name: idx for idx, meta in enumerate(self._outputs_meta)
        }

    # ── properties ────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return self._backend

    def get_providers(self) -> List[str]:
        return ["NPU_NBGraph"]

    def get_inputs(self) -> List[_InputMeta]:
        return list(self._inputs_meta)

    def get_outputs(self) -> List[_OutputMeta]:
        return list(self._outputs_meta)

    # ── inference ─────────────────────────────────────────────────

    def run(
        self,
        output_names: Optional[List[str]],
        feed_dict: dict[str, np.ndarray],
    ) -> List[np.ndarray]:
        """Execute one inference.

        Args:
            output_names: Pass ``None`` to retrieve all outputs.
            feed_dict: ``{input_name: numpy_blob}``.
        """
        network = self._internal

        for feed_name, blob in feed_dict.items():
            idx = self._input_index[feed_name]
            target_dtype = self._infer_target_dtype(feed_name, blob)
            blob = _convert_input_blob(blob, target_dtype, feed_name, self._quant)
            network.set_input(idx, blob)

        network.run()

        names = output_names if output_names is not None else [o.name for o in self._outputs_meta]
        results: List[np.ndarray] = []
        for name in names:
            idx = self._output_index[name]
            raw = np.asarray(network.get_output(idx))
            results.append(_dequantize_output(raw, name, self._quant))

        return results

    def _infer_target_dtype(self, name: str, blob: np.ndarray) -> np.dtype:
        """Guess the dtype the NPU expects for *name*."""
        qi = self._quant.get(name)
        if qi is not None:
            # staticAffine quantization → NPU expects int8/uint8
            for meta in self._inputs_meta:
                if meta.name == name:
                    if "int8" in meta.type:
                        return np.dtype(np.int8)
                    if "uint8" in meta.type:
                        return np.dtype(np.uint8)
            return np.dtype(np.int8)

        return blob.dtype


# ──────────────────────────────────────────────────────────────────
# dtype / quantization utilities
# ──────────────────────────────────────────────────────────────────

def _convert_input_blob(
    blob: np.ndarray,
    target_dtype: np.dtype,
    name: str,
    quant: dict[str, tuple[float, int] | None],
) -> np.ndarray:
    """Convert float32 [0,1] NCHW blob to the NPU's expected dtype.

    Uses the model's own quantization parameters when available so that
    the conversion matches what the ONNX calibration pipeline expects.
    """
    qi = quant.get(name)
    if qi is not None:
        scale, zp = qi
    else:
        # The stai_mpu Python binding accepts float32 application buffers for
        # float graphs even when optimized NBG metadata exposes float16 I/O.
        return blob.astype(np.float32, copy=False)

    if blob.dtype == target_dtype:
        return blob

    if target_dtype in (np.int8, np.uint8):
        # int_val = round(float_val / scale) + zero_point
        # For typical scale=1/255, zp=-128: int_val = round(f*255) - 128
        scaled = np.round(blob.astype(np.float64) / float(scale)).astype(np.int32)
        scaled += np.int32(zp)
        info = np.iinfo(target_dtype)
        return np.clip(scaled, info.min, info.max).astype(target_dtype)

    return blob.astype(target_dtype)


def _dequantize_output(
    arr: np.ndarray,
    name: str,
    quant: dict[str, tuple[float, int] | None],
) -> np.ndarray:
    """Convert NPU output (int8/uint8/float16) back to float32."""
    if arr.dtype == np.float32:
        return arr.astype(np.float32, copy=False)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)

    qi = quant.get(name)
    if qi is not None and arr.dtype in (np.int8, np.uint8):
        scale, zp = qi
        # float_val = (int_val - zero_point) * scale
        return (arr.astype(np.float32) - float(zp)) * np.float32(scale)

    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * _FLOAT32_SCALE_1_255
    if arr.dtype == np.int8:
        return (arr.astype(np.float32) + 128.0) * _FLOAT32_SCALE_1_255
    if arr.dtype == np.float16:
        return arr.astype(np.float32)

    return arr.astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# file detection
# ──────────────────────────────────────────────────────────────────

def is_nb_model(model_path: str) -> bool:
    """Return True if *model_path* is a compiled .nb network binary."""
    if model_path.endswith(".nb"):
        return True
    if os.path.isfile(model_path):
        try:
            with open(model_path, "rb") as fh:
                magic = fh.read(4)
            if magic == b"VPMN":
                return True
        except OSError:
            pass
    return False
