"""Road perception module for segmentation-based visual line following.

The only required public entry point is ``get_road_perception(frame, ...)``.
This module does not implement flight control, route selection, MAVLink, or
ArduPilot logic. It only converts one BGR image into structured road geometry.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import cv2
import numpy as np

from nb_graph import NBGraphSession, is_nb_model


@dataclass
class RoadBranch:
    pixel_error: float
    centerline_angle: float
    path_width_px: float = 0.0
    score: float = 0.0
    confidence: float = 0.0
    label: str = "unknown"
    branch_id: int = 0
    points: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.confidence <= 0.0 and self.score > 0.0:
            self.confidence = float(self.score)


@dataclass
class RoadPerceptionResult:
    is_road_found: bool

    # single / lost
    road_state: str

    # Selected path result.
    pixel_error: float
    centerline_angle: float
    path_width_px: float
    confidence: float

    # V1 keeps this equal to pixel_error.
    corrected_pixel_error: float

    # Primary centerline. Branch fields remain as an empty compatibility shim
    # for older log/overlay consumers; branch detection is intentionally disabled.
    centerline_points: List[Tuple[float, float]] = field(default_factory=list)
    branches: List[RoadBranch] = field(default_factory=list)
    selected_branch: RoadBranch | None = None
    branch_decision: str = "disabled"

    debug_msg: str = ""
    debug_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    centerline_bottom_ratio: float = 0.0
    centerline_span_ratio: float = 0.0
    centerline_residual_p90_px: float = 0.0
    centerline_max_step_px: float = 0.0
    centerline_inlier_ratio: float = 0.0
    centerline_straightened: bool = False


@dataclass
class CameraOffsetCompensationConfig:
    """Signed road-camera body-frame offset used by yaw compensation.

    ``cam_forward_offset_m`` follows the body frame: +X is forward and -X is
    rearward.  The downward-facing road camera is mounted at X=-0.0787m,
    Y=0m.  The current compensation model only needs X because Y is zero.
    """

    enabled: bool = False
    cam_forward_offset_m: float = -0.0787
    meters_per_pixel_x: float | None = None
    correction_sign: float = 1.0
    max_correction_px: float = 120.0
    pipeline_latency_s: float = 0.0


@dataclass
class CameraGeometry:
    """摄像头几何标定参数。

    可通过标尺法实测反推，无需厂商文档。
    现有默认值来自功能互换前的路径识别摄像头（/dev/video9）在飞行高度
    17cm 时的标定数据。当前路径识别摄像头为垂直向下的 /dev/video7，
    旧的倾斜投影模型不再适用；在重新标定前，不应对它启用偏移补偿。
    """

    height_m: float = 0.17
    """摄像头离地高度 (m)"""
    alpha_deg: float = 30.27
    """光轴倾角 (度), 水平=0°, 垂直向下=90°"""
    beta_deg: float = 27.54
    """半垂直视场角 VFOV/2 (度)"""
    hfov_deg: float = 68.0
    """水平视场角 (度)"""
    img_w: int = 640
    img_h: int = 480


@dataclass
class CameraWhiteBalanceConfig:
    """软件白平衡修正配置。

    BGR 三通道系数，均以 G 通道为基准。
    例如 /dev/video9 实测偏青 (R/G=0.36, B/G=0.79)，系数为
    R×2.78、B×1.26、G×1.00。该相机现用于障碍物识别；/dev/video7
    用于道路识别时应先单独测量其白平衡系数。
    """

    enabled: bool = False
    r_gain: float = 1.0
    """红色通道增益 (乘性)，>1 增强红光"""
    g_gain: float = 1.0
    """绿色通道增益 (乘性)，通常保持 1.0"""
    b_gain: float = 1.0
    """蓝色通道增益 (乘性)，<1 减弱蓝光"""


def _apply_white_balance(
    frame: np.ndarray, cfg: CameraWhiteBalanceConfig
) -> np.ndarray:
    """对 BGR 帧逐通道应用增益系数。"""
    if not cfg.enabled:
        return frame
    corrected = frame.astype(np.float32, copy=True)
    corrected[:, :, 0] = np.clip(corrected[:, :, 0] * cfg.b_gain, 0.0, 255.0)
    corrected[:, :, 1] = np.clip(corrected[:, :, 1] * cfg.g_gain, 0.0, 255.0)
    corrected[:, :, 2] = np.clip(corrected[:, :, 2] * cfg.r_gain, 0.0, 255.0)
    return corrected.astype(np.uint8)


def compute_meters_per_pixel(
    row_from_bottom: int,
    geom: CameraGeometry | None = None,
    height_m: float | None = None,
) -> float:
    """逐行计算水平方向每像素对应的地面距离 (m/px)。

    Args:
        row_from_bottom: 像素行号, 0 = 画面最下行, img_h-1 = 画面最上行
        geom: 摄像头几何标定参数, 默认使用实测值
        height_m: 飞行高度 (m), 覆盖 geom.height_m。用于飞行时高度与标定高度不同时的自适应计算

    Returns:
        meters_per_pixel_x: 该行的水平 m/px 值

    Reference:
        HARDWARE_INTERFACE.md §5.4 — 摄像头几何标定

        θ(row) = α + β × (1 − 2 × row / (H_px − 1))
        D_ground = H / tan(θ)
        m/px = 2 × D_ground × tan(HFOV/2) / img_w
    """
    if geom is None:
        geom = CameraGeometry()
    import math

    h = height_m if height_m is not None else geom.height_m
    alpha = math.radians(geom.alpha_deg)
    beta = math.radians(geom.beta_deg)
    hfov_half = math.radians(geom.hfov_deg / 2.0)
    t = 1.0 - 2.0 * float(row_from_bottom) / float(geom.img_h - 1)
    theta = alpha + beta * t
    d_ground = h / math.tan(theta)
    return 2.0 * d_ground * math.tan(hfov_half) / float(geom.img_w)


@dataclass
class RoadInstance:
    mask: np.ndarray
    score: float
    box_xywh: Tuple[float, float, float, float]
    area: int
    bottom_touch_px: int
    bottom_cx: float
    centerline_points: List[CenterPoint] = field(default_factory=list)


# Existing lightweight YOLO segmentation model (CPU / XNNPACK fallback).
MODEL_PATH = "FlightController/Solutions/model/road_yolo11n_seg_128.onnx"
# New semantic road segmentation model compiled for the STM32MP257 VIP9000.
# Contract: RGB float32 [0, 1], NCHW [1, 3, 256, 256] -> logits
# [1, 2, 256, 256], where class 0 is background and class 1 is road.
MODEL_PATH_NPU = "FlightController/Solutions/model/new_road_seg_v3_final_fp32.nb"

# Prefer the NPU semantic model by default.  Set this to False (or call
# configure_model(..., backend="cpu")) to retain the existing CPU YOLO path.
_AUTO_USE_NPU = True

# Force CPU-only inference.  The lightweight 128×128 model contains ops
# (ConvTranspose, dilated MaxPool) that are known to crash VSINPU EP on
# STM32MP257.  When True, _select_providers() skips VSINPU / CUDA entirely.
_CPU_ONLY = True

POSTPROCESS_FAST_MAIN = "fast-main"
POSTPROCESS_FULL = "full"
POSTPROCESS_MODES = {POSTPROCESS_FAST_MAIN, POSTPROCESS_FULL}
# Both modes are single-road only. ``fast-main`` uses a sparse low-resolution
# semantic mask; ``full`` keeps full-resolution mask geometry without fork
# detection for compatibility and offline diagnostics.
_POSTPROCESS_MODE = POSTPROCESS_FAST_MAIN

FAST_MASK_WIDTH = 192
FAST_MASK_HEIGHT = 144
FAST_CENTERLINE_ROW_STEP = 2

INP_SIZE = 320
CONF_THRESH = 0.4
IOU_THRESH = 0.45
MASK_THRESH = 0.5

MIN_AREA_RATIO = 0.02
MIN_ROAD_PX_PER_ROW = 12
BOTTOM_RATIO = 0.10
BOTTOM_IGNORE_RATIO = 0.03
BOTTOM_ERROR_Y_MIN_RATIO = 0.82
BOTTOM_ERROR_Y_MAX_RATIO = 0.96
ANGLE_Y_MIN_RATIO = 0.60
CONTROL_ANGLE_Y_MIN_RATIO = 0.72
CONTROL_ANGLE_Y_MAX_RATIO = 0.98
CENTERLINE_SCAN_Y_MAX_RATIO = 0.97
MIN_FIT_PTS = 5

# A usable control path must reach the near field and cover enough image rows.
# The experiment road has no forks, so a complete but jagged path can be
# replaced by a robust straight centerline instead of following mask lobes.
MIN_CONTROL_CENTERLINE_POINTS = 18
MIN_CONTROL_BOTTOM_Y_RATIO = 0.82
MIN_CONTROL_SPAN_RATIO = 0.22
MAX_CENTERLINE_RESIDUAL_P90_RATIO = 0.08
MAX_CENTERLINE_STEP_RATIO = 0.10
ROBUST_CENTERLINE_INLIER_RATIO = 0.05
MIN_ROBUST_CENTERLINE_INLIERS = 0.55

CENTER_SMOOTH_ALPHA = 0.65


_SESSION: Any | None = None
_INPUT_NAME: str | None = None
_MODEL_INPUT_SIZE: int | None = None
_SESSION_PROVIDER: str = "unknown"
_MODEL_KIND: str = "unknown"

MODEL_KIND_YOLO = "yolo_instance_seg"
MODEL_KIND_SEMANTIC = "semantic_seg"


CenterPoint = Tuple[float, int, float]  # center_x, y, width
Interval = Tuple[int, int]
RowIntervals = List[Tuple[int, List[Interval]]]


@dataclass(frozen=True)
class _CenterlineQuality:
    usable: bool
    rough: bool
    reason: str
    bottom_ratio: float
    span_ratio: float
    residual_p90_px: float
    max_step_px: float
    robust_inlier_ratio: float
    robust_slope: float
    robust_intercept: float


def _lost_result(reason: str) -> RoadPerceptionResult:
    return RoadPerceptionResult(
        is_road_found=False,
        road_state="lost",
        pixel_error=0.0,
        centerline_angle=90.0,
        path_width_px=0.0,
        confidence=0.0,
        corrected_pixel_error=0.0,
        centerline_points=[],
        branches=[],
        selected_branch=None,
        branch_decision="disabled",
        debug_msg=reason,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def normalize_angle_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def apply_camera_offset_compensation(
    *,
    pixel_error: float,
    centerline_angle_deg: float,
    image_width: int,
    config: CameraOffsetCompensationConfig,
    yaw_rate_deg_s: float = 0.0,
) -> tuple[float, float]:
    """Return (corrected_pixel_error, correction_px).

    ``cam_forward_offset_m`` is signed in the body frame, so the configured
    rearward camera position (``-0.0787m``) naturally reverses the correction
    compared with a forward-mounted camera.
    """
    _ = image_width
    if not config.enabled:
        return float(pixel_error), 0.0
    if config.meters_per_pixel_x is None or config.meters_per_pixel_x <= 0.0:
        return float(pixel_error), 0.0

    predicted_angle_deg = float(centerline_angle_deg) + float(yaw_rate_deg_s) * config.pipeline_latency_s
    heading_error_rad = math.radians(predicted_angle_deg - 90.0)
    heading_error_rad = _clamp(
        heading_error_rad,
        -math.radians(60.0),
        math.radians(60.0),
    )
    correction_px = (
        config.correction_sign
        * (config.cam_forward_offset_m / config.meters_per_pixel_x)
        * math.tan(heading_error_rad)
    )
    correction_px = _clamp(correction_px, -config.max_correction_px, config.max_correction_px)
    corrected = float(pixel_error) - correction_px
    return corrected, correction_px


def _resolve_model_path() -> tuple[str, bool]:
    """Return ``(model_path, is_nb)``.

    Resolution order:
    1. If *MODEL_PATH* itself is a .nb file → use directly (--model-npu path).
    2. Absolute ONNX path + NPU auto-detect sideline check.
    3. Relative paths: .nb preferred when _AUTO_USE_NPU is set.
    """
    # 1. MODEL_PATH already points to a .nb file (e.g. --model-npu)
    if is_nb_model(MODEL_PATH):
        return MODEL_PATH, True

    # 2. Absolute ONNX path
    if os.path.isabs(MODEL_PATH):
        nb_candidate = MODEL_PATH_NPU
        if _AUTO_USE_NPU and os.path.isfile(nb_candidate):
            return nb_candidate, True
        return MODEL_PATH, False

    module_dir = os.path.dirname(os.path.abspath(__file__))

    # 3. NB model takes priority when auto-detect is on
    nb_candidate = os.path.join(module_dir, MODEL_PATH_NPU)
    if _AUTO_USE_NPU and os.path.isfile(nb_candidate):
        return nb_candidate, True

    # 4. ONNX model relative to module
    module_relative = os.path.join(module_dir, MODEL_PATH)
    if os.path.isfile(module_relative):
        return module_relative, False

    # 5. Fallback: bare paths
    return MODEL_PATH, False


def configure_model(
    *,
    backend: str = "npu",
    cpu_model_path: str | None = None,
    npu_model_path: str | None = None,
    postprocess_mode: str = POSTPROCESS_FAST_MAIN,
) -> None:
    """Configure the road model and reset the cached inference session.

    ``backend="npu"`` selects the compiled semantic ``.nb`` model, while
    ``backend="cpu"`` selects the existing lightweight YOLO ONNX model.  The
    reset matters for tests and long-running launchers that reconfigure the
    module before starting their inference thread.
    """
    normalized = str(backend).strip().lower()
    if normalized not in {"npu", "cpu"}:
        raise ValueError(f"unsupported road inference backend: {backend!r}")
    normalized_postprocess = str(postprocess_mode).strip().lower()
    if normalized_postprocess not in POSTPROCESS_MODES:
        raise ValueError(
            f"unsupported road postprocess mode: {postprocess_mode!r}; "
            f"expected one of {sorted(POSTPROCESS_MODES)}"
        )

    global MODEL_PATH, MODEL_PATH_NPU, _AUTO_USE_NPU, _CPU_ONLY
    global _SESSION, _INPUT_NAME, _MODEL_INPUT_SIZE, _SESSION_PROVIDER
    global _MODEL_KIND, _USE_CROP_PREPROCESS, _POSTPROCESS_MODE

    if cpu_model_path is not None:
        MODEL_PATH = str(cpu_model_path)
    if npu_model_path is not None:
        MODEL_PATH_NPU = str(npu_model_path)

    _AUTO_USE_NPU = normalized == "npu"
    # ONNX inference is deliberately CPU-only.  The legacy YOLO graph contains
    # operators that are known to crash the board's VSINPU execution provider.
    _CPU_ONLY = True
    _POSTPROCESS_MODE = normalized_postprocess

    _SESSION = None
    _INPUT_NAME = None
    _MODEL_INPUT_SIZE = None
    _SESSION_PROVIDER = "unknown"
    _MODEL_KIND = "unknown"
    _USE_CROP_PREPROCESS = False


def _select_providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())

    # Low-resolution CPU-only models contain ops (ConvTranspose, dilated
    # MaxPool) that crash VSINPU EP on STM32MP257.  Skip directly to
    # XNNPACK / CPU when _CPU_ONLY is set.
    if _CPU_ONLY:
        if "XnnpackExecutionProvider" in available:
            return ["XnnpackExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    for candidate in ("VSINPUExecutionProvider", "CUDAExecutionProvider"):
        if candidate in available:
            return [candidate, "CPUExecutionProvider"]

    if "XnnpackExecutionProvider" in available:
        return ["XnnpackExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def _make_session(model_path: str, is_nb: bool = False):
    if is_nb or is_nb_model(model_path):
        try:
            sess = NBGraphSession(model_path)
            return sess, "NPU_NBGraph"
        except Exception as exc:
            # NB load failed — cannot fall back to CPU for a compiled binary
            raise RuntimeError(
                f"Failed to load .nb NPU model: {model_path}\n"
                f"  Error: {exc}\n"
                f"  Verify that stai_mpu is installed:\n"
                f"    apt install python3-stai-mpu\n"
                f"  Or switch to the ONNX model by setting _AUTO_USE_NPU = False"
            ) from exc

    import onnxruntime as ort

    providers = _select_providers()
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        providers = ["CPUExecutionProvider"]

    return sess, providers[0]


def _get_session():
    global _SESSION, _INPUT_NAME, _MODEL_INPUT_SIZE, _SESSION_PROVIDER
    global _MODEL_KIND, _USE_CROP_PREPROCESS

    if _SESSION is None:
        model_path, is_nb = _resolve_model_path()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        _SESSION, provider = _make_session(model_path, is_nb=is_nb)
        _SESSION_PROVIDER = provider
        input_meta = _SESSION.get_inputs()[0]
        _INPUT_NAME = input_meta.name
        _MODEL_INPUT_SIZE = _input_size_from_meta(input_meta.shape)
        _MODEL_KIND = _model_kind_from_output_meta(_SESSION.get_outputs())

        if not is_nb:
            _log_low_resolution_warning(model_path, _MODEL_INPUT_SIZE)

        # At low resolutions letterbox wastes too much canvas on gray padding
        # for the legacy YOLO model.  The semantic model was trained by direct
        # full-frame resize and must never use the crop path.
        resolved_size = _MODEL_INPUT_SIZE or INP_SIZE
        if _MODEL_KIND == MODEL_KIND_YOLO and resolved_size <= 160:
            _USE_CROP_PREPROCESS = True

        if is_nb:
            _log_npu_info(model_path)

    return _SESSION, _INPUT_NAME


def _log_low_resolution_warning(model_path: str, model_input_size: int) -> None:
    """Warn when a low-resolution model is loaded (accuracy may degrade)."""
    if model_input_size is not None and model_input_size < 160:
        try:
            from loguru import logger
            logger.warning(
                "Low-resolution model loaded: {}  |  input_size={}px  "
                "|  road segmentation accuracy may be reduced.  "
                "Consider 160×160 if road detection is unreliable.",
                os.path.basename(model_path),
                model_input_size,
            )
        except ImportError:
            pass


def _log_npu_info(model_path: str) -> None:
    """Emit a one-line diagnostic when the NPU .nb model is active."""
    try:
        from loguru import logger
        logger.info(
            "NPU model loaded: {}  |  backend={}  |  kind={}  |  input={} shape=[{}]",
            os.path.basename(model_path),
            _SESSION_PROVIDER,
            _MODEL_KIND,
            _INPUT_NAME,
            "x".join(str(d) for d in (_SESSION.get_inputs()[0].shape or [])),
        )
    except ImportError:
        pass


def _input_size_from_meta(shape: Any) -> int:
    if shape and len(shape) >= 4:
        h_value = shape[2]
        w_value = shape[3]
        if isinstance(h_value, (int, np.integer)) and isinstance(w_value, (int, np.integer)):
            if h_value > 0 and h_value == w_value:
                return int(h_value)
    return INP_SIZE


def _model_kind_from_output_meta(outputs: Any) -> str:
    """Identify the decoder from the model's output tensor contract."""
    output_list = list(outputs or [])
    if len(output_list) == 1:
        shape = list(getattr(output_list[0], "shape", []) or [])
        if (
            len(shape) == 4
            and isinstance(shape[1], (int, np.integer))
            and int(shape[1]) == 2
        ):
            return MODEL_KIND_SEMANTIC
        raise ValueError(
            "unsupported single-output road model; expected logits "
            f"[1, 2, H, W], got shape={shape}"
        )
    if len(output_list) >= 2:
        return MODEL_KIND_YOLO
    raise ValueError("road model exposes no output tensors")


def _letterbox(frame: np.ndarray, new_size: int = INP_SIZE):
    """Resize a BGR frame with unchanged aspect ratio and gray padding."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("frame has invalid shape")

    scale = min(new_size / float(w), new_size / float(h))
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    pad_x = (new_size - resized_w) / 2.0
    pad_y = (new_size - resized_h) / 2.0

    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    img = np.full((new_size, new_size, 3), 114, dtype=np.uint8)

    left = int(round(pad_x))
    top = int(round(pad_y))
    img[top : top + resized_h, left : left + resized_w] = resized

    return img, scale, pad_x, pad_y


def _preprocess(frame: np.ndarray, input_size: int):
    img, scale, pad_x, pad_y = _letterbox(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img, scale, pad_x, pad_y


def _preprocess_semantic(frame: np.ndarray, input_size: int) -> np.ndarray:
    """Match the V3 training contract: full-frame resize, RGB, float [0, 1]."""
    resized = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = image_rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    # stai_mpu consumes the application buffer as contiguous memory and does
    # not honor NumPy strides from transpose().  ORT accepts the strided view,
    # which can hide this bug during CPU validation.
    return np.ascontiguousarray(np.expand_dims(blob, axis=0), dtype=np.float32)


def _preprocess_crop(frame: np.ndarray, input_size: int):
    """Preprocess via center-crop + resize (no letterbox / gray padding).

    At low resolutions (≤160 px) letterbox wastes too much canvas area on
    gray padding, starving the model of road features.  Center-crop keeps
    the entire spatial budget on content.

    Also stores crop metadata in ``_CROP_META`` for downstream mask
    geometry correction in ``get_road_perception()``.

    Returns ``(blob, 0.0, 0.0, 0.0)`` — *scale*, *pad_x*, *pad_y* are
    always zero here; masks are fixed by ``_fix_mask_for_crop()`` after
    the decode step.
    """
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("frame has invalid shape")

    crop_side = min(h, w)
    crop_y = (h - crop_side) // 2
    crop_x = (w - crop_side) // 2
    cropped = frame[crop_y : crop_y + crop_side, crop_x : crop_x + crop_side]

    # Persist crop geometry for mask correction.
    _CROP_META["crop_side"] = crop_side
    _CROP_META["crop_x"] = crop_x
    _CROP_META["crop_y"] = crop_y
    _CROP_META["frame_w"] = w
    _CROP_META["frame_h"] = h

    resized = cv2.resize(cropped, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img, 0.0, 0.0, 0.0


# Metadata written by _preprocess_crop on every call; consumed by
# _fix_mask_for_crop after the ONNX decode step.
_CROP_META: dict[str, int] = {}


def _fix_mask_for_crop(mask: np.ndarray) -> np.ndarray:
    """Correct a mask that was decoded assuming letterbox when crop was used.

    The mask is currently stretched to ``(frame_w, frame_h)``.  We resize it
    back to the square crop region and embed it at the correct position.
    """
    if mask is None or mask.size == 0:
        return np.zeros(
            (_CROP_META["frame_h"], _CROP_META["frame_w"]), dtype=np.uint8
        )

    crop_side: int = _CROP_META["crop_side"]
    crop_x: int = _CROP_META["crop_x"]
    crop_y: int = _CROP_META["crop_y"]
    frame_w: int = _CROP_META["frame_w"]
    frame_h: int = _CROP_META["frame_h"]

    corrected = cv2.resize(mask, (crop_side, crop_side), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((frame_h, frame_w), dtype=mask.dtype)
    canvas[crop_y : crop_y + crop_side, crop_x : crop_x + crop_side] = corrected
    return canvas


# Global flag: when True, use center-crop instead of letterbox at inference
# time.  Automatically enabled for models with input ≤ 160 px.
_USE_CROP_PREPROCESS: bool = False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _prepare_yolo_seg_tensors(output0: np.ndarray, output1: np.ndarray):
    if output0.ndim != 3:
        raise ValueError(f"output0 must be [1, C, N], got shape={output0.shape}")

    if output1.ndim == 4:
        proto = output1[0]
    elif output1.ndim == 3:
        proto = output1
    else:
        raise ValueError(f"output1 must be [1, M, H, W], got shape={output1.shape}")

    if proto.ndim != 3:
        raise ValueError(f"proto must be [M, H, W], got shape={proto.shape}")

    # Preferred YOLO segmentation export: output0[0] is [channels, candidates].
    # Transposing makes candidate count fully dynamic: [num_candidates, channels].
    raw = output0[0]
    preferred = raw.T
    alternatives = [preferred, raw]
    num_masks = int(proto.shape[0])

    for preds in alternatives:
        if preds.ndim != 2:
            continue
        channels = int(preds.shape[1])
        num_classes = channels - 4 - num_masks
        if 1 <= num_classes <= 100:
            return preds.astype(np.float32, copy=False), proto.astype(np.float32, copy=False)

    channels = int(preferred.shape[1]) if preferred.ndim == 2 else -1
    raise ValueError(
        f"invalid YOLO segmentation channels={channels}, num_masks={num_masks}"
    )


def _nms_indices(boxes_xywh: List[List[float]], scores: List[float]) -> List[int]:
    if not boxes_xywh:
        return []

    dnn = getattr(cv2, "dnn", None)
    if dnn is not None and hasattr(dnn, "NMSBoxes"):
        indices = dnn.NMSBoxes(boxes_xywh, scores, CONF_THRESH, IOU_THRESH)
        if indices is None or len(indices) == 0:
            return []
        return [int(i) for i in np.array(indices).reshape(-1)]

    return _nms_indices_numpy(boxes_xywh, scores)


def _nms_indices_numpy(
    boxes_xywh: List[List[float]], scores: List[float]
) -> List[int]:
    """OpenCV-DNN-independent NMS for minimal OpenSTLinux builds."""
    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] != 4:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    widths = np.maximum(0.0, boxes[:, 2])
    heights = np.maximum(0.0, boxes[:, 3])
    x2 = x1 + widths
    y2 = y1 + heights
    areas = widths * heights
    order = np.argsort(score_array)[::-1]
    keep: List[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        inter_x1 = np.maximum(x1[current], x1[rest])
        inter_y1 = np.maximum(y1[current], y1[rest])
        inter_x2 = np.minimum(x2[current], x2[rest])
        inter_y2 = np.minimum(y2[current], y2[rest])
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        order = rest[iou <= IOU_THRESH]

    return keep


def _decode_yolo_segmentation(
    outputs: List[np.ndarray],
    orig_w: int,
    orig_h: int,
    input_size: int,
    pad_x: float,
    pad_y: float,
) -> Tuple[np.ndarray | None, List[RoadInstance], float, str]:
    if len(outputs) < 2:
        return None, [], 0.0, f"expected at least 2 ONNX outputs, got {len(outputs)}"

    output0 = np.asarray(outputs[0])
    output1 = np.asarray(outputs[1])
    preds, proto = _prepare_yolo_seg_tensors(output0, output1)

    num_masks = int(proto.shape[0])
    channels = int(preds.shape[1])
    num_classes = channels - 4 - num_masks
    if num_classes <= 0:
        return None, [], 0.0, (
            f"invalid output layout: channels={channels}, num_masks={num_masks}"
        )

    boxes_xywh: List[List[float]] = []
    scores: List[float] = []
    coeffs: List[np.ndarray] = []

    for pred in preds:
        box_xywh = pred[0:4]
        class_scores = pred[4 : 4 + num_classes]
        if class_scores.size == 0:
            continue

        score = float(np.max(class_scores))
        cls_id = int(np.argmax(class_scores))
        _ = cls_id  # Kept for future multi-class handling.
        if not math.isfinite(score) or score < CONF_THRESH:
            continue

        mask_coeff = pred[4 + num_classes :]
        if mask_coeff.shape[0] != num_masks:
            continue

        cx, cy, w, h = [float(v) for v in box_xywh]
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            continue
        if w <= 1.0 or h <= 1.0:
            continue

        boxes_xywh.append([cx - w / 2.0, cy - h / 2.0, w, h])
        scores.append(score)
        coeffs.append(mask_coeff.astype(np.float32, copy=False))

    if not boxes_xywh:
        return None, [], 0.0, "no road detection above confidence threshold"

    keep = _nms_indices(boxes_xywh, scores)
    if not keep:
        return None, [], 0.0, "no road detection remained after NMS"

    proto_h, proto_w = int(proto.shape[1]), int(proto.shape[2])
    proto_flat = proto.reshape(num_masks, -1)
    merged_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    instances: List[RoadInstance] = []
    selected_scores: List[float] = []

    crop_x1 = max(0, min(input_size, int(round(pad_x))))
    crop_y1 = max(0, min(input_size, int(round(pad_y))))
    crop_x2 = max(crop_x1, min(input_size, int(round(input_size - pad_x))))
    crop_y2 = max(crop_y1, min(input_size, int(round(input_size - pad_y))))

    for idx in keep:
        raw_mask = coeffs[idx] @ proto_flat
        raw_mask = _sigmoid(raw_mask).reshape(proto_h, proto_w)

        mask_input = cv2.resize(
            raw_mask,
            (input_size, input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_input = (mask_input > MASK_THRESH).astype(np.uint8) * 255

        box_x, box_y, box_w, box_h = boxes_xywh[idx]
        box_x1 = max(0, min(input_size, int(math.floor(box_x))))
        box_y1 = max(0, min(input_size, int(math.floor(box_y))))
        box_x2 = max(box_x1, min(input_size, int(math.ceil(box_x + box_w))))
        box_y2 = max(box_y1, min(input_size, int(math.ceil(box_y + box_h))))
        clipped_mask = np.zeros_like(mask_input)
        if box_x2 > box_x1 and box_y2 > box_y1:
            clipped_mask[box_y1:box_y2, box_x1:box_x2] = mask_input[
                box_y1:box_y2,
                box_x1:box_x2,
            ]
        mask_input = clipped_mask

        mask_crop = mask_input[crop_y1:crop_y2, crop_x1:crop_x2]
        if mask_crop.size == 0:
            continue

        final_mask = cv2.resize(
            mask_crop,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        final_mask = (final_mask > 0).astype(np.uint8) * 255

        area = int(np.count_nonzero(final_mask))
        if area <= 0:
            continue

        bottom_y_start = int(orig_h * 0.85)
        bottom_region = final_mask[bottom_y_start:, :]
        bottom_cols = np.where(bottom_region > 0)[1]
        bottom_touch_px = int(len(bottom_cols))
        if bottom_touch_px > 0:
            bottom_cx = float(np.mean(bottom_cols))
        else:
            content_w = max(1.0, float(crop_x2 - crop_x1))
            box_center_x = box_x + box_w / 2.0
            bottom_cx = (box_center_x - crop_x1) / content_w * orig_w
            bottom_cx = float(max(0.0, min(float(orig_w - 1), bottom_cx)))

        content_w = max(1.0, float(crop_x2 - crop_x1))
        content_h = max(1.0, float(crop_y2 - crop_y1))
        orig_box_x = (box_x - crop_x1) / content_w * orig_w
        orig_box_y = (box_y - crop_y1) / content_h * orig_h
        orig_box_w = box_w / content_w * orig_w
        orig_box_h = box_h / content_h * orig_h

        instance = RoadInstance(
            mask=final_mask,
            score=float(scores[idx]),
            box_xywh=(
                float(orig_box_x),
                float(orig_box_y),
                float(orig_box_w),
                float(orig_box_h),
            ),
            area=area,
            bottom_touch_px=bottom_touch_px,
            bottom_cx=bottom_cx,
        )

        instances.append(instance)
        merged_mask = cv2.bitwise_or(merged_mask, final_mask)
        selected_scores.append(float(scores[idx]))

    if not instances or not selected_scores or np.count_nonzero(merged_mask) == 0:
        return None, [], 0.0, "decoded road instances are empty"

    return merged_mask, instances, float(np.mean(selected_scores)), "ok"


def _decode_semantic_segmentation(
    outputs: List[np.ndarray],
    orig_w: int,
    orig_h: int,
) -> Tuple[np.ndarray | None, List[RoadInstance], float, str]:
    """Decode background/road logits from the NPU semantic model."""
    if len(outputs) != 1:
        return None, [], 0.0, f"expected 1 semantic output, got {len(outputs)}"

    logits = np.asarray(outputs[0], dtype=np.float32)
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
        return None, [], 0.0, (
            "semantic logits must be [1, 2, H, W], "
            f"got shape={logits.shape}"
        )
    if not np.all(np.isfinite(logits)):
        return None, [], 0.0, "semantic logits contain non-finite values"

    # For a two-class softmax, P(road) = sigmoid(road_logit-bg_logit).  A
    # strict positive delta matches argmax semantics, including background on
    # exact ties.
    delta = logits[0, 1] - logits[0, 0]
    road_small = delta > 0.0
    if not np.any(road_small):
        return None, [], 0.0, "semantic model predicted no road pixels"

    road_probability = _sigmoid(delta)
    confidence = float(np.mean(road_probability[road_small]))
    mask_small = road_small.astype(np.uint8) * 255
    final_mask = cv2.resize(
        mask_small,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    final_mask = (final_mask > 0).astype(np.uint8) * 255
    # Semantic models can emit small disconnected false-positive islands.
    # Keep the plausible bottom-connected road component while preserving any
    # connected fork/intersection geometry inside it.
    final_mask = _clean_mask(final_mask)
    area = int(np.count_nonzero(final_mask))
    if area <= 0:
        return None, [], 0.0, "semantic road mask became empty after resize"

    x, y, width, height = cv2.boundingRect(final_mask)
    bottom_y_start = int(orig_h * 0.85)
    bottom_cols = np.where(final_mask[bottom_y_start:, :] > 0)[1]
    bottom_touch_px = int(len(bottom_cols))
    bottom_cx = (
        float(np.mean(bottom_cols))
        if bottom_touch_px > 0
        else float(x + width / 2.0)
    )
    instance = RoadInstance(
        mask=final_mask.copy(),
        score=confidence,
        box_xywh=(float(x), float(y), float(width), float(height)),
        area=area,
        bottom_touch_px=bottom_touch_px,
        bottom_cx=bottom_cx,
    )
    return final_mask, [instance], confidence, "ok"


def _decode_semantic_fast_main(
    outputs: List[np.ndarray],
) -> Tuple[np.ndarray | None, float, str]:
    """Decode the semantic output directly into the small fast working mask."""
    if len(outputs) != 1:
        return None, 0.0, f"expected 1 semantic output, got {len(outputs)}"

    logits = np.asarray(outputs[0], dtype=np.float32)
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
        return None, 0.0, (
            "semantic logits must be [1, 2, H, W], "
            f"got shape={logits.shape}"
        )
    if not np.all(np.isfinite(logits)):
        return None, 0.0, "semantic logits contain non-finite values"

    delta = logits[0, 1] - logits[0, 0]
    road_small = delta > 0.0
    if not np.any(road_small):
        return None, 0.0, "semantic model predicted no road pixels"

    # Confidence is diagnostic/control metadata; evaluate sigmoid only on the
    # predicted road pixels instead of materialising a second full logits map.
    confidence = float(np.mean(_sigmoid(delta[road_small])))
    mask = cv2.resize(
        road_small.astype(np.uint8) * 255,
        (FAST_MASK_WIDTH, FAST_MASK_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = _clean_fast_main_mask(mask)
    if mask.size == 0 or np.count_nonzero(mask) == 0:
        return None, 0.0, "semantic road mask became empty after fast cleanup"
    return mask, confidence, "ok"


def _clean_fast_main_mask(mask: np.ndarray) -> np.ndarray:
    """One low-resolution cleanup pass for the main-road-only profile."""
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if num_labels <= 1:
        return np.zeros_like(mask)

    h, w = mask.shape[:2]
    min_area = max(1, int(h * w * MIN_AREA_RATIO))
    bottom_labels = set(int(v) for v in np.unique(labels[int(h * 0.90) :, :]))
    bottom_labels.discard(0)
    candidates: List[Tuple[int, int, bool]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            candidates.append((label_id, area, label_id in bottom_labels))
    if not candidates:
        return np.zeros_like(mask)

    bottom_candidates = [candidate for candidate in candidates if candidate[2]]
    selected = max(bottom_candidates or candidates, key=lambda item: item[1])[0]
    return np.where(labels == selected, 255, 0).astype(np.uint8)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Fill holes and keep the most plausible road component."""
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if num_labels <= 1:
        return np.zeros_like(mask)

    h, w = mask.shape[:2]
    min_area = int(h * w * MIN_AREA_RATIO)
    bottom_y_start = int(h * 0.90)
    bottom_labels = set(int(v) for v in np.unique(labels[bottom_y_start:, :]))
    bottom_labels.discard(0)

    candidates: List[Tuple[int, int, bool]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        is_bottom_connected = label_id in bottom_labels
        candidates.append((label_id, area, is_bottom_connected))

    if not candidates:
        return np.zeros_like(mask)

    bottom_candidates = [c for c in candidates if c[2]]
    if bottom_candidates:
        best_label = max(bottom_candidates, key=lambda item: item[1])[0]
    else:
        best_label = max(candidates, key=lambda item: item[1])[0]

    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _clean_mask_keep_single_instance(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _refresh_instance_geometry(inst: RoadInstance, w: int, h: int) -> None:
    inst.area = int(np.count_nonzero(inst.mask))

    bottom_y_start = int(h * 0.85)
    bottom_region = inst.mask[bottom_y_start:, :]
    bottom_cols = np.where(bottom_region > 0)[1]
    inst.bottom_touch_px = int(len(bottom_cols))
    if inst.bottom_touch_px > 0:
        inst.bottom_cx = float(np.mean(bottom_cols))
        return

    x, _, box_w, _ = inst.box_xywh
    inst.bottom_cx = float(max(0.0, min(float(w - 1), x + box_w / 2.0)))


def _find_intervals(
    row_mask: np.ndarray,
    min_width: int = MIN_ROAD_PX_PER_ROW,
) -> List[Interval]:
    cols = np.where(row_mask > 0)[0]
    if len(cols) == 0:
        return []

    # Vectorized run-boundary extraction.  The previous per-pixel Python loop
    # dominated the complete perception time on Cortex-A35 even though NPU
    # inference itself was fast.
    gap_indices = np.flatnonzero(np.diff(cols) > 1)
    starts = np.concatenate((cols[:1], cols[gap_indices + 1]))
    ends = np.concatenate((cols[gap_indices], cols[-1:]))
    widths = ends - starts + 1
    valid = widths >= int(min_width)
    return [
        (int(start), int(end))
        for start, end in zip(starts[valid], ends[valid])
    ]


def _extract_centerline_and_intervals(clean_mask: np.ndarray) -> Tuple[List[CenterPoint], RowIntervals]:
    h, w = clean_mask.shape[:2]
    points: List[CenterPoint] = []

    last_cx = w / 2.0
    bottom_y = min(h - 1, int(h * CENTERLINE_SCAN_Y_MAX_RATIO))

    for y in range(bottom_y, h // 2, -1):
        intervals = _find_intervals(clean_mask[y, :], MIN_ROAD_PX_PER_ROW)

        if not intervals:
            continue

        best = min(
            intervals,
            key=lambda interval: abs(((interval[0] + interval[1]) / 2.0) - last_cx),
        )
        left, right = best
        width = float(right - left + 1)
        interval_mid = (left + right) / 2.0

        cx = (
            CENTER_SMOOTH_ALPHA * interval_mid
            + (1.0 - CENTER_SMOOTH_ALPHA) * last_cx
        )

        points.append((float(cx), int(y), width))
        last_cx = cx

    # Keep the historical tuple return shape while avoiding the row-interval
    # accumulation that was only consumed by fork/intersection detection.
    return _trim_bottom_centerline_outliers(points, w), []


def _extract_fast_main_centerline(
    clean_mask: np.ndarray,
    orig_w: int,
    orig_h: int,
) -> List[CenterPoint]:
    """Extract a sparse main-road centerline and return original-image pixels."""
    work_h, work_w = clean_mask.shape[:2]
    if work_h <= 0 or work_w <= 0 or orig_w <= 0 or orig_h <= 0:
        return []

    scale_x = float(orig_w) / float(work_w)
    scale_y = float(orig_h) / float(work_h)
    min_width = max(2, int(round(MIN_ROAD_PX_PER_ROW / scale_x)))
    last_cx = work_w / 2.0
    points: List[CenterPoint] = []
    bottom_y = min(work_h - 1, int(work_h * CENTERLINE_SCAN_Y_MAX_RATIO))

    for y in range(bottom_y, work_h // 2, -FAST_CENTERLINE_ROW_STEP):
        intervals = _find_intervals(clean_mask[y, :], min_width)
        if not intervals:
            continue
        left, right = min(
            intervals,
            key=lambda interval: abs(((interval[0] + interval[1]) / 2.0) - last_cx),
        )
        width = float(right - left + 1)
        interval_mid = (left + right) / 2.0
        center_x = (
            CENTER_SMOOTH_ALPHA * interval_mid
            + (1.0 - CENTER_SMOOTH_ALPHA) * last_cx
        )
        points.append(
            (
                float(center_x * scale_x),
                int(round(float(y) * scale_y)),
                float(width * scale_x),
            )
        )
        last_cx = center_x

    return _trim_bottom_centerline_outliers(points, orig_w)


def _trim_bottom_centerline_outliers(
    points: List[CenterPoint],
    w: int,
) -> List[CenterPoint]:
    if len(points) < MIN_FIT_PTS + 2:
        return points

    trimmed = list(points)
    while len(trimmed) >= MIN_FIT_PTS + 2:
        lookahead = trimmed[1 : min(len(trimmed), 18)]
        if len(lookahead) < MIN_FIT_PTS:
            break

        ref_x = float(np.median([p[0] for p in lookahead]))
        ref_width = float(np.median([p[2] for p in lookahead]))
        max_jump = max(30.0, min(w * 0.08, ref_width * 0.35))

        if abs(trimmed[0][0] - ref_x) > max_jump:
            trimmed.pop(0)
            continue
        break

    return trimmed


def _centerline_quality(
    points: List[CenterPoint],
    w: int,
    h: int,
) -> _CenterlineQuality:
    if not points or w <= 0 or h <= 0:
        return _CenterlineQuality(
            usable=False,
            rough=False,
            reason="empty",
            bottom_ratio=0.0,
            span_ratio=0.0,
            residual_p90_px=0.0,
            max_step_px=0.0,
            robust_inlier_ratio=0.0,
            robust_slope=0.0,
            robust_intercept=w / 2.0,
        )

    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    bottom_ratio = float(np.max(y) / h)
    span_ratio = float((np.max(y) - np.min(y)) / h)

    if len(points) >= 2:
        line = np.polyfit(y, x, deg=1)
        residuals = np.abs(x - np.polyval(line, y))
        residual_p90_px = float(np.percentile(residuals, 90))
        max_step_px = float(np.max(np.abs(np.diff(x))))

        slopes: List[float] = []
        for first in range(len(points) - 1):
            delta_y = y[first + 1 :] - y[first]
            valid = delta_y != 0.0
            if np.any(valid):
                delta_x = x[first + 1 :] - x[first]
                slopes.extend((delta_x[valid] / delta_y[valid]).tolist())
        robust_slope = float(np.median(slopes)) if slopes else 0.0
        robust_intercept = float(np.median(x - robust_slope * y))
        robust_residuals = np.abs(x - (robust_slope * y + robust_intercept))
        inlier_threshold_px = max(24.0, float(w) * ROBUST_CENTERLINE_INLIER_RATIO)
        robust_inlier_ratio = float(np.mean(robust_residuals <= inlier_threshold_px))
    else:
        residual_p90_px = 0.0
        max_step_px = 0.0
        robust_slope = 0.0
        robust_intercept = float(x[0])
        robust_inlier_ratio = 1.0

    rough = bool(
        residual_p90_px > float(w) * MAX_CENTERLINE_RESIDUAL_P90_RATIO
        or max_step_px > float(w) * MAX_CENTERLINE_STEP_RATIO
    )
    if len(points) < MIN_CONTROL_CENTERLINE_POINTS:
        usable, reason = False, "too_few_points"
    elif bottom_ratio < MIN_CONTROL_BOTTOM_Y_RATIO:
        usable, reason = False, "near_field_missing"
    elif span_ratio < MIN_CONTROL_SPAN_RATIO:
        usable, reason = False, "vertical_span_short"
    elif rough and robust_inlier_ratio < MIN_ROBUST_CENTERLINE_INLIERS:
        usable, reason = False, "no_straight_consensus"
    elif rough:
        usable, reason = True, "rough_straightened"
    else:
        usable, reason = True, "ok"

    return _CenterlineQuality(
        usable=usable,
        rough=rough,
        reason=reason,
        bottom_ratio=bottom_ratio,
        span_ratio=span_ratio,
        residual_p90_px=residual_p90_px,
        max_step_px=max_step_px,
        robust_inlier_ratio=robust_inlier_ratio,
        robust_slope=robust_slope,
        robust_intercept=robust_intercept,
    )


def _straighten_centerline(
    points: List[CenterPoint],
    quality: _CenterlineQuality,
    w: int,
) -> List[CenterPoint]:
    return [
        (
            float(
                np.clip(
                    quality.robust_slope * float(y) + quality.robust_intercept,
                    0.0,
                    max(0.0, float(w - 1)),
                )
            ),
            int(y),
            float(width),
        )
        for _, y, width in points
    ]


def _bottom_points(points: List[CenterPoint], h: int) -> List[CenterPoint]:
    bottom_y_min = int(h * BOTTOM_ERROR_Y_MIN_RATIO)
    bottom_y_max = int(h * BOTTOM_ERROR_Y_MAX_RATIO)
    bottom_pts = [p for p in points if bottom_y_min <= p[1] <= bottom_y_max]
    if bottom_pts:
        return bottom_pts
    bottom_y_min = int(h * (1.0 - BOTTOM_RATIO - BOTTOM_IGNORE_RATIO))
    bottom_y_max = int(h * (1.0 - BOTTOM_IGNORE_RATIO))
    bottom_pts = [p for p in points if bottom_y_min <= p[1] <= bottom_y_max]
    if bottom_pts:
        return bottom_pts
    return points[: min(5, len(points))]


def _compute_pixel_error(points: List[CenterPoint], w: int, h: int) -> Tuple[float, float]:
    pts = _bottom_points(points, h)
    if not pts:
        return 0.0, w / 2.0

    cx_bottom = float(np.median([p[0] for p in pts]))
    return cx_bottom - w / 2.0, cx_bottom


def _compute_path_width(points: List[CenterPoint], h: int) -> float:
    pts = [
        p
        for p in points
        if h * 0.55 <= p[1] <= h * 0.90
    ]
    if len(pts) < MIN_FIT_PTS:
        pts = _bottom_points(points, h)
    if not pts:
        return 0.0

    return float(np.median([p[2] for p in pts]))


def _compute_centerline_angle(points: List[CenterPoint], h: int) -> float:
    fit_pts = [
        (p[0], p[1])
        for p in points
        if h * CONTROL_ANGLE_Y_MIN_RATIO <= p[1] <= h * CONTROL_ANGLE_Y_MAX_RATIO
    ]
    if len(fit_pts) < MIN_FIT_PTS:
        fit_pts = [
            (p[0], p[1])
            for p in points
            if h * ANGLE_Y_MIN_RATIO <= p[1] <= h
        ]

    if len(fit_pts) < MIN_FIT_PTS:
        return 90.0

    try:
        arr = np.array(fit_pts, dtype=np.float32)
        coeffs = np.polyfit(arr[:, 1], arr[:, 0], deg=1)
        a = float(coeffs[0])

        angle_deg = math.degrees(math.atan2(1.0, -a))
        if angle_deg < 0:
            angle_deg += 360.0
        return float(angle_deg)
    except Exception:
        return 90.0


def _select_current_instance(
    instances: List[RoadInstance],
    w: int,
    h: int,
) -> RoadInstance | None:
    if not instances:
        return None

    min_area = int(w * h * MIN_AREA_RATIO * 0.3)
    candidates = [inst for inst in instances if inst.area >= min_area]
    if not candidates:
        return None

    bottom_candidates = [
        inst
        for inst in candidates
        if inst.bottom_touch_px > MIN_ROAD_PX_PER_ROW
    ]

    if not bottom_candidates:
        return max(
            candidates,
            key=lambda inst: (
                -abs(inst.bottom_cx - w / 2.0),
                inst.score,
                inst.area,
            ),
        )

    def score_instance(inst: RoadInstance) -> float:
        center_penalty = abs(inst.bottom_cx - w / 2.0) / max(1.0, w / 2.0)
        bottom_score = min(1.0, inst.bottom_touch_px / max(1.0, w * 0.2))
        area_score = min(1.0, inst.area / max(1.0, w * h * 0.25))
        return (
            3.0 * bottom_score
            + 2.0 * inst.score
            + 1.0 * area_score
            - 2.0 * center_penalty
        )

    return max(bottom_candidates, key=score_instance)


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)

    frame_float = np.nan_to_num(frame.astype(np.float32, copy=False), nan=0.0)
    max_value = float(np.max(frame_float)) if frame_float.size else 0.0
    if max_value <= 1.0:
        frame_float = frame_float * 255.0

    return np.ascontiguousarray(np.clip(frame_float, 0.0, 255.0).astype(np.uint8))


def _draw_polyline(
    img: np.ndarray,
    points: List[Tuple[float, float]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return

    pts = np.array(
        [[int(round(x)), int(round(y))] for x, y in points],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)


def _save_debug_image(
    frame: np.ndarray,
    mask: np.ndarray | None,
    result: RoadPerceptionResult,
    debug_save_path: str,
) -> None:
    try:
        debug_img = frame.copy()
        h, w = debug_img.shape[:2]

        if mask is not None and mask.size != 0:
            overlay = np.zeros_like(debug_img)
            overlay[mask > 0] = (255, 0, 0)
            debug_img = cv2.addWeighted(debug_img, 1.0, overlay, 0.35, 0.0)

        cv2.line(
            debug_img,
            (int(round(w / 2.0)), 0),
            (int(round(w / 2.0)), h - 1),
            (0, 255, 0),
            1,
        )

        _draw_polyline(debug_img, result.centerline_points, (0, 0, 255), 4)

        cx_bottom = int(round(w / 2.0 + result.pixel_error))
        cx_bottom = max(0, min(w - 1, cx_bottom))
        bottom_y = h - 12
        cv2.circle(debug_img, (cx_bottom, bottom_y), 6, (0, 255, 255), -1)

        arrow_len = max(35, int(min(w, h) * 0.16))
        angle_rad = math.radians(result.centerline_angle)
        end_x = int(round(cx_bottom + arrow_len * math.cos(angle_rad)))
        end_y = int(round(bottom_y - arrow_len * math.sin(angle_rad)))
        cv2.arrowedLine(
            debug_img,
            (cx_bottom, bottom_y),
            (end_x, end_y),
            (255, 255, 255),
            2,
            tipLength=0.25,
        )

        text_lines = [
            f"state={result.road_state} found={result.is_road_found}",
            f"error={result.pixel_error:.1f}px corrected={result.corrected_pixel_error:.1f}px",
            f"angle={result.centerline_angle:.1f}deg width={result.path_width_px:.1f}px",
            f"conf={result.confidence:.2f} mode=single-road",
        ]
        if result.debug_msg:
            text_lines.append(result.debug_msg[:72])
        x0, y0 = 10, 24
        for i, line in enumerate(text_lines):
            y = y0 + i * 22
            cv2.putText(
                debug_img,
                line,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_img,
                line,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        debug_dir = os.path.dirname(debug_save_path)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(debug_save_path, debug_img)
    except Exception:
        # Debug output must never break the perception call.
        return


def _validate_frame(frame: np.ndarray) -> str | None:
    if frame is None:
        return "frame is None"
    if not isinstance(frame, np.ndarray):
        return f"frame must be np.ndarray, got {type(frame).__name__}"
    if frame.ndim != 3:
        return f"frame must be a 3-channel BGR image, got ndim={frame.ndim}"
    if frame.shape[2] != 3:
        return f"frame must have 3 channels, got shape={frame.shape}"
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        return f"frame has invalid shape={frame.shape}"
    return None


def _build_fast_main_result(
    outputs: List[np.ndarray],
    *,
    orig_w: int,
    orig_h: int,
    yaw_rate_deg_s: float,
    cam_offset_m: float,
    offset_comp_config: CameraOffsetCompensationConfig | None,
) -> RoadPerceptionResult:
    work_mask, confidence, message = _decode_semantic_fast_main(outputs)
    debug_mask = None
    if work_mask is not None:
        debug_mask = cv2.resize(
            work_mask,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
    if work_mask is None:
        result = _lost_result(message)
        result.debug_mask = debug_mask
        return result

    points = _extract_fast_main_centerline(work_mask, orig_w, orig_h)
    if len(points) < MIN_FIT_PTS:
        result = _lost_result("fast main road has too few centerline points")
        result.debug_mask = debug_mask
        return result

    quality = _centerline_quality(points, orig_w, orig_h)
    if not quality.usable:
        result = _lost_result(
            "centerline_quality=reject "
            f"reason={quality.reason}, points={len(points)}, "
            f"bottom={quality.bottom_ratio:.2f}, span={quality.span_ratio:.2f}, "
            f"resid_p90={quality.residual_p90_px:.1f}, "
            f"step_max={quality.max_step_px:.1f}, "
            f"inliers={quality.robust_inlier_ratio:.2f}"
        )
        result.debug_mask = debug_mask
        result.centerline_bottom_ratio = quality.bottom_ratio
        result.centerline_span_ratio = quality.span_ratio
        result.centerline_residual_p90_px = quality.residual_p90_px
        result.centerline_max_step_px = quality.max_step_px
        result.centerline_inlier_ratio = quality.robust_inlier_ratio
        return result
    if quality.rough:
        points = _straighten_centerline(points, quality, orig_w)

    pixel_error, _ = _compute_pixel_error(points, orig_w, orig_h)
    centerline_angle = _compute_centerline_angle(points, orig_h)
    path_width_px = _compute_path_width(points, orig_h)

    config = offset_comp_config or CameraOffsetCompensationConfig(
        enabled=False,
        cam_forward_offset_m=cam_offset_m,
    )
    corrected_error, correction_px = apply_camera_offset_compensation(
        pixel_error=float(pixel_error),
        centerline_angle_deg=float(centerline_angle),
        image_width=orig_w,
        config=config,
        yaw_rate_deg_s=yaw_rate_deg_s,
    )
    return RoadPerceptionResult(
        is_road_found=True,
        road_state="single_rough" if quality.rough else "single",
        pixel_error=float(pixel_error),
        centerline_angle=float(centerline_angle),
        path_width_px=float(path_width_px),
        confidence=float(confidence),
        corrected_pixel_error=float(corrected_error),
        centerline_points=[(float(point[0]), float(point[1])) for point in points],
        branches=[],
        selected_branch=None,
        branch_decision="disabled",
        debug_msg=(
            f"postprocess={POSTPROCESS_FAST_MAIN}, points={len(points)}, "
            f"work_mask={FAST_MASK_WIDTH}x{FAST_MASK_HEIGHT}, "
            "branch_detection=disabled, "
            f"quality={quality.reason}, bottom={quality.bottom_ratio:.2f}, "
            f"span={quality.span_ratio:.2f}, resid_p90={quality.residual_p90_px:.1f}, "
            f"step_max={quality.max_step_px:.1f}, "
            f"inliers={quality.robust_inlier_ratio:.2f}, "
            f"offset_corr_px={correction_px:.1f}, corrected_error={corrected_error:.1f}"
        ),
        debug_mask=debug_mask,
        centerline_bottom_ratio=quality.bottom_ratio,
        centerline_span_ratio=quality.span_ratio,
        centerline_residual_p90_px=quality.residual_p90_px,
        centerline_max_step_px=quality.max_step_px,
        centerline_inlier_ratio=quality.robust_inlier_ratio,
        centerline_straightened=quality.rough,
    )


def get_model_io_info() -> dict[str, Any]:
    """Optional helper for inspecting ONNX model input/output metadata."""
    session, _ = _get_session()
    return {
        "provider": _SESSION_PROVIDER,
        "model_kind": _MODEL_KIND,
        "postprocess_mode": _POSTPROCESS_MODE,
        "input_size": _MODEL_INPUT_SIZE or INP_SIZE,
        "inputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_outputs()
        ],
    }


def get_road_perception(
    frame: np.ndarray,
    yaw_rate_deg_s: float = 0.0,
    cam_offset_m: float = -0.0787,
    flight_height_m: float = 1.0,
    debug_save_path: str | None = None,
    offset_comp_config: CameraOffsetCompensationConfig | None = None,
    branch_preference: str = "auto",
    previous_branch_label: str | None = None,
    wb_config: CameraWhiteBalanceConfig | None = None,
) -> RoadPerceptionResult:
    """Run one-frame, single-road perception on an OpenCV BGR frame.

    ``branch_preference`` and ``previous_branch_label`` are retained only so
    older callers do not fail; the single-road pipeline intentionally ignores
    them.
    """
    _ = branch_preference, previous_branch_label
    frame_error = _validate_frame(frame)
    if frame_error is not None:
        return _lost_result(frame_error)

    frame_bgr = _normalize_frame(frame)
    if wb_config is not None:
        frame_bgr = _apply_white_balance(frame_bgr, wb_config)
    debug_mask: np.ndarray | None = None

    try:
        orig_h, orig_w = frame_bgr.shape[:2]

        session, input_name = _get_session()
        input_size = _MODEL_INPUT_SIZE or INP_SIZE
        if _MODEL_KIND == MODEL_KIND_SEMANTIC:
            blob = _preprocess_semantic(frame_bgr, input_size)
            pad_x = 0.0
            pad_y = 0.0
        elif _USE_CROP_PREPROCESS:
            blob, _scale, pad_x, pad_y = _preprocess_crop(frame_bgr, input_size)
        else:
            blob, _scale, pad_x, pad_y = _preprocess(frame_bgr, input_size)
        outputs = session.run(None, {input_name: blob})

        if (
            _MODEL_KIND == MODEL_KIND_SEMANTIC
            and _POSTPROCESS_MODE == POSTPROCESS_FAST_MAIN
        ):
            result = _build_fast_main_result(
                outputs,
                orig_w=orig_w,
                orig_h=orig_h,
                yaw_rate_deg_s=yaw_rate_deg_s,
                cam_offset_m=cam_offset_m,
                offset_comp_config=offset_comp_config,
            )
            if debug_save_path:
                _save_debug_image(frame_bgr, result.debug_mask, result, debug_save_path)
            return result

        if _MODEL_KIND == MODEL_KIND_SEMANTIC:
            merged_mask, instances, confidence, decode_msg = _decode_semantic_segmentation(
                outputs,
                orig_w=orig_w,
                orig_h=orig_h,
            )
        else:
            merged_mask, instances, confidence, decode_msg = _decode_yolo_segmentation(
                outputs,
                orig_w=orig_w,
                orig_h=orig_h,
                input_size=input_size,
                pad_x=pad_x,
                pad_y=pad_y,
            )

        # Crop-mode preprocessing produces masks that are horizontally
        # stretched because the decode step resizes from a square crop to a
        # non-square frame.  Fix the geometry here so that downstream
        # centerline extraction sees the correct pixel coordinates.
        if _MODEL_KIND == MODEL_KIND_YOLO and _USE_CROP_PREPROCESS:
            if merged_mask is not None:
                merged_mask = _fix_mask_for_crop(merged_mask)
            for inst in instances:
                inst.mask = _fix_mask_for_crop(inst.mask)

        debug_mask = merged_mask
        if merged_mask is None or not instances or np.count_nonzero(merged_mask) == 0:
            result = _lost_result(decode_msg)
            result.debug_mask = merged_mask
            if debug_save_path:
                _save_debug_image(frame_bgr, merged_mask, result, debug_save_path)
            return result

        valid_instances: List[RoadInstance] = []
        for inst in instances:
            inst_clean = _clean_mask_keep_single_instance(inst.mask)
            if inst_clean.size == 0 or np.count_nonzero(inst_clean) == 0:
                continue

            inst.mask = inst_clean
            _refresh_instance_geometry(inst, orig_w, orig_h)
            valid_instances.append(inst)

        if not valid_instances:
            result = _lost_result("no valid road instances after cleanup")
            result.debug_mask = merged_mask
            if debug_save_path:
                _save_debug_image(frame_bgr, merged_mask, result, debug_save_path)
            return result

        current = _select_current_instance(valid_instances, orig_w, orig_h)
        if current is None:
            result = _lost_result("no current road instance selected")
            result.debug_mask = merged_mask
            if debug_save_path:
                _save_debug_image(frame_bgr, merged_mask, result, debug_save_path)
            return result

        points, _ = _extract_centerline_and_intervals(current.mask)
        if len(points) < MIN_FIT_PTS:
            result = _lost_result("selected road has too few centerline points")
            result.debug_mask = merged_mask
            if debug_save_path:
                _save_debug_image(frame_bgr, merged_mask, result, debug_save_path)
            return result
        current.centerline_points = points

        pixel_error, _ = _compute_pixel_error(
            points,
            orig_w,
            orig_h,
        )
        centerline_angle = _compute_centerline_angle(points, orig_h)
        path_width_px = _compute_path_width(points, orig_h)

        cfg = offset_comp_config or CameraOffsetCompensationConfig(
            enabled=False,
            cam_forward_offset_m=cam_offset_m,
        )
        _ = flight_height_m
        corrected_pixel_error, correction_px = apply_camera_offset_compensation(
            pixel_error=float(pixel_error),
            centerline_angle_deg=float(centerline_angle),
            image_width=orig_w,
            config=cfg,
            yaw_rate_deg_s=yaw_rate_deg_s,
        )

        result = RoadPerceptionResult(
            is_road_found=True,
            road_state="single",
            pixel_error=float(pixel_error),
            centerline_angle=float(centerline_angle),
            path_width_px=float(path_width_px),
            confidence=float(confidence),
            corrected_pixel_error=float(corrected_pixel_error),
            centerline_points=[(float(p[0]), float(p[1])) for p in points],
            branches=[],
            selected_branch=None,
            branch_decision="disabled",
            debug_msg=(
                f"instances={len(valid_instances)}, "
                "branch_detection=disabled, "
                f"offset_corr_px={correction_px:.1f}, "
                f"corrected_error={corrected_pixel_error:.1f}"
            ),
            debug_mask=merged_mask,
        )

        if debug_save_path:
            _save_debug_image(frame_bgr, merged_mask, result, debug_save_path)

        return result
    except Exception as exc:
        result = _lost_result(f"{type(exc).__name__}: {exc}")
        result.debug_mask = debug_mask
        if debug_save_path:
            _save_debug_image(frame_bgr, debug_mask, result, debug_save_path)
        return result


def _parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run road perception on one image")
    parser.add_argument("--image", required=True, help="Input BGR/RGB image path")
    parser.add_argument("--model", default=None, help="ONNX model path")
    parser.add_argument("--model-npu", default=None, help=".nb NPU compiled model path")
    parser.add_argument(
        "--road-postprocess-mode",
        choices=sorted(POSTPROCESS_MODES),
        default=POSTPROCESS_FAST_MAIN,
    )
    parser.add_argument("--debug-out", default=None, help="Optional debug image output path")
    parser.add_argument("--enable-offset-comp", action="store_true")
    parser.add_argument("--cam-forward-offset-m", type=float, default=-0.0787)
    parser.add_argument("--meters-per-pixel-x", type=float, default=None)
    parser.add_argument("--offset-correction-sign", type=float, default=1.0)
    parser.add_argument("--offset-max-correction-px", type=float, default=120.0)
    parser.add_argument("--pipeline-latency-s", type=float, default=0.0)
    parser.add_argument("--yaw-rate-deg-s", type=float, default=0.0)
    return parser.parse_args()


def _main_cli() -> int:
    args = _parse_cli_args()
    if args.model_npu:
        configure_model(
            backend="npu",
            npu_model_path=args.model_npu,
            postprocess_mode=args.road_postprocess_mode,
        )
    elif args.model:
        configure_model(
            backend="cpu",
            cpu_model_path=args.model,
            postprocess_mode=args.road_postprocess_mode,
        )
    else:
        configure_model(
            backend="npu",
            postprocess_mode=args.road_postprocess_mode,
        )

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"failed to read image: {args.image}")
        return 2

    offset_cfg = CameraOffsetCompensationConfig(
        enabled=bool(args.enable_offset_comp),
        cam_forward_offset_m=args.cam_forward_offset_m,
        meters_per_pixel_x=args.meters_per_pixel_x,
        correction_sign=args.offset_correction_sign,
        max_correction_px=args.offset_max_correction_px,
        pipeline_latency_s=args.pipeline_latency_s,
    )
    result = get_road_perception(
        frame,
        yaw_rate_deg_s=args.yaw_rate_deg_s,
        debug_save_path=args.debug_out,
        offset_comp_config=offset_cfg,
    )
    print(
        "road_state={} found={} conf={:.3f} error={:.1f} corrected={:.1f} "
        "angle={:.1f} debug={}".format(
            result.road_state,
            result.is_road_found,
            result.confidence,
            result.pixel_error,
            result.corrected_pixel_error,
            result.centerline_angle,
            result.debug_msg,
        )
    )
    return 0


__all__ = [
    "CameraGeometry",
    "CameraOffsetCompensationConfig",
    "RoadBranch",
    "RoadInstance",
    "RoadPerceptionResult",
    "apply_camera_offset_compensation",
    "compute_meters_per_pixel",
    "configure_model",
    "get_model_io_info",
    "get_road_perception",
    "normalize_angle_deg",
]


if __name__ == "__main__":
    raise SystemExit(_main_cli())
