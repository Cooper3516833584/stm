from __future__ import annotations

import numpy as np

from perception_pipeline import SharedLatest, YOLOInferenceThread
from nb_graph import NBGraphSession, _InputMeta, _convert_input_blob
import road_perception as road


class _Meta:
    def __init__(self, shape):
        self.shape = shape


def test_semantic_model_contract_is_detected() -> None:
    assert road._model_kind_from_output_meta([_Meta([1, 2, 256, 256])]) == road.MODEL_KIND_SEMANTIC
    assert road._model_kind_from_output_meta([_Meta([1, np.int32(2), 256, 256])]) == road.MODEL_KIND_SEMANTIC
    assert road._model_kind_from_output_meta([_Meta([1, 37, 336]), _Meta([1, 32, 32, 32])]) == road.MODEL_KIND_YOLO


def test_nb_float_graph_keeps_stai_application_buffer_float32() -> None:
    session = object.__new__(NBGraphSession)
    session._inputs_meta = [
        _InputMeta("input_0", [1, 3, 256, 256], "tensor(float16)")
    ]
    session._quant = {"input_0": None}
    blob = np.ones((1, 3, 2, 2), dtype=np.float32)

    target_dtype = session._infer_target_dtype("input_0", blob)
    converted = _convert_input_blob(blob, target_dtype, "input_0", session._quant)

    assert target_dtype == np.dtype(np.float32)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, np.ones_like(converted))


def test_backend_switch_keeps_legacy_cpu_model(monkeypatch) -> None:
    for name in (
        "MODEL_PATH",
        "MODEL_PATH_NPU",
        "_AUTO_USE_NPU",
        "_CPU_ONLY",
        "_SESSION",
        "_INPUT_NAME",
        "_MODEL_INPUT_SIZE",
        "_SESSION_PROVIDER",
        "_MODEL_KIND",
        "_USE_CROP_PREPROCESS",
        "_POSTPROCESS_MODE",
    ):
        monkeypatch.setattr(road, name, getattr(road, name))

    road.configure_model(backend="cpu", postprocess_mode="fast-main")
    cpu_path, is_nb = road._resolve_model_path()
    assert not is_nb
    assert cpu_path.endswith("road_yolo11n_seg_128.onnx")
    assert road._POSTPROCESS_MODE == road.POSTPROCESS_FAST_MAIN

    road.configure_model(backend="npu", postprocess_mode="full")
    npu_path, is_nb = road._resolve_model_path()
    assert is_nb
    assert npu_path.endswith("new_road_seg_v4_final_fp32.nb")
    assert road._POSTPROCESS_MODE == road.POSTPROCESS_FULL


def test_inference_thread_configures_selected_backend(monkeypatch) -> None:
    calls = []

    def fake_configure_model(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(road, "configure_model", fake_configure_model)
    monkeypatch.setattr(
        road,
        "get_model_io_info",
        lambda: {
            "provider": "NPU_NBGraph",
            "model_kind": road.MODEL_KIND_SEMANTIC,
            "postprocess_mode": road.POSTPROCESS_FAST_MAIN,
        },
    )

    camera = type("Camera", (), {"frame_buffer": SharedLatest()})()
    worker = YOLOInferenceThread(
        camera_thread=camera,
        model_path="legacy.onnx",
        npu_model_path="new.nb",
        inference_backend="npu",
        postprocess_mode="fast-main",
    )
    worker.start()
    worker.stop()

    assert calls == [
        {
            "backend": "npu",
            "cpu_model_path": "legacy.onnx",
            "npu_model_path": "new.nb",
            "postprocess_mode": "fast-main",
        }
    ]


def test_semantic_preprocess_matches_training_contract() -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)  # OpenCV BGR

    blob = road._preprocess_semantic(frame, 256)

    assert blob.shape == (1, 3, 256, 256)
    assert blob.dtype == np.float32
    assert blob.flags.c_contiguous
    np.testing.assert_allclose(blob[0, :, 0, 0], [30 / 255.0, 20 / 255.0, 10 / 255.0])


def test_semantic_logits_decode_to_road_instance() -> None:
    logits = np.zeros((1, 2, 256, 256), dtype=np.float32)
    logits[:, 0, :, :] = 2.0
    logits[:, 1, 80:256, 64:192] = 5.0

    mask, instances, confidence, message = road._decode_semantic_segmentation(
        [logits],
        orig_w=640,
        orig_h=480,
    )

    assert message == "ok"
    assert mask is not None and mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    assert len(instances) == 1
    assert instances[0].area == int(np.count_nonzero(mask))
    assert instances[0].bottom_touch_px > 0
    assert confidence > 0.9


def test_numpy_nms_fallback_suppresses_overlapping_boxes() -> None:
    boxes = [
        [10.0, 10.0, 100.0, 100.0],
        [12.0, 12.0, 100.0, 100.0],
        [200.0, 200.0, 30.0, 30.0],
    ]
    scores = [0.9, 0.8, 0.7]
    assert road._nms_indices_numpy(boxes, scores) == [0, 2]


def test_vectorized_interval_extraction_preserves_runs() -> None:
    row = np.zeros(40, dtype=np.uint8)
    row[2:8] = 255
    row[12:25] = 255
    row[30:40] = 255
    assert road._find_intervals(row, min_width=10) == [(12, 24), (30, 39)]


def test_semantic_decoder_rejects_invalid_or_empty_output() -> None:
    invalid = np.zeros((1, 1, 256, 256), dtype=np.float32)
    mask, instances, confidence, message = road._decode_semantic_segmentation(
        [invalid], orig_w=640, orig_h=480
    )
    assert mask is None
    assert instances == []
    assert confidence == 0.0
    assert "[1, 2, H, W]" in message

    background = np.zeros((1, 2, 256, 256), dtype=np.float32)
    background[:, 0] = 1.0
    mask, instances, confidence, message = road._decode_semantic_segmentation(
        [background], orig_w=640, orig_h=480
    )
    assert mask is None
    assert instances == []
    assert confidence == 0.0
    assert "no road" in message

    mask, confidence, message = road._decode_semantic_fast_main([background])
    assert mask is None
    assert confidence == 0.0
    assert "no road" in message


def test_fast_main_centerline_restores_original_pixel_scale() -> None:
    mask = np.zeros((road.FAST_MASK_HEIGHT, road.FAST_MASK_WIDTH), dtype=np.uint8)
    mask[road.FAST_MASK_HEIGHT // 2 :, 72:120] = 255

    points = road._extract_fast_main_centerline(mask, 640, 480)
    pixel_error, _ = road._compute_pixel_error(points, 640, 480)
    width = road._compute_path_width(points, 480)

    assert road.MIN_FIT_PTS <= len(points) <= 36
    assert abs(pixel_error) < 5.0
    assert abs(width - 160.0) < 5.0
    assert all(0 <= point[1] < 480 for point in points)


def test_fast_main_extrapolates_remote_only_centerline_to_near_field() -> None:
    logits = np.zeros((1, 2, 256, 256), dtype=np.float32)
    logits[:, 0] = 1.0
    logits[0, 1, 100:170, 70:150] = 4.0

    result = road._build_fast_main_result(
        [logits],
        orig_w=640,
        orig_h=480,
        yaw_rate_deg_s=0.0,
        cam_offset_m=-0.0787,
        offset_comp_config=None,
    )

    assert result.is_road_found
    assert result.road_state == "single_extrapolated"
    assert "quality=remote_extrapolated" in result.debug_msg
    assert result.centerline_bottom_ratio < road.MIN_CONTROL_BOTTOM_Y_RATIO
    assert result.centerline_extrapolated
    assert max(point[1] for point in result.centerline_points) >= 0.95 * 480


def test_rough_complete_centerline_is_straightened_with_robust_consensus() -> None:
    y_values = np.linspace(463, 243, 34).astype(int)
    points = [(330.0, int(y), 250.0) for y in y_values]
    points[10:20] = [(430.0, int(y), 250.0) for y in y_values[10:20]]

    quality = road._centerline_quality(points, 640, 480)
    straightened = road._fit_control_centerline(points, quality, 640, 480)

    assert quality.usable
    assert quality.rough
    assert not quality.extrapolate
    assert quality.reason == "rough_straightened"
    assert quality.robust_inlier_ratio >= road.MIN_ROBUST_CENTERLINE_INLIERS
    assert max(abs(point[0] - 330.0) for point in straightened) < 1.0
    assert max(abs(a[0] - b[0]) for a, b in zip(straightened, straightened[1:])) < 1.0


def test_centerline_fit_rejects_tiny_unsupported_patch() -> None:
    points = [(320.0, y, 100.0) for y in range(280, 245, -7)]

    quality = road._centerline_quality(points, 640, 480)

    assert not quality.usable
    assert quality.reason == "too_few_points"


def test_fast_main_geometry_stays_close_to_full_geometry() -> None:
    for curved in (False, True):
        logits = np.zeros((1, 2, 256, 256), dtype=np.float32)
        logits[:, 0] = 1.0
        for y in range(80, 256):
            center = 128 + (int((y - 168) * 0.12) if curved else 0)
            logits[0, 1, y, max(0, center - 42) : min(256, center + 42)] = 4.0

        full_mask, _, _, message = road._decode_semantic_segmentation(
            [logits], orig_w=640, orig_h=480
        )
        assert message == "ok" and full_mask is not None
        full_points, _ = road._extract_centerline_and_intervals(
            road._clean_mask_keep_single_instance(full_mask)
        )

        fast_mask, _, message = road._decode_semantic_fast_main([logits])
        assert message == "ok" and fast_mask is not None
        fast_points = road._extract_fast_main_centerline(fast_mask, 640, 480)

        full_error, _ = road._compute_pixel_error(full_points, 640, 480)
        fast_error, _ = road._compute_pixel_error(fast_points, 640, 480)
        full_angle = road._compute_centerline_angle(full_points, 480)
        fast_angle = road._compute_centerline_angle(fast_points, 480)
        full_width = road._compute_path_width(full_points, 480)
        fast_width = road._compute_path_width(fast_points, 480)

        assert abs(fast_error - full_error) <= 20.0
        assert abs(fast_angle - full_angle) <= 5.0
        assert abs(fast_width - full_width) / max(full_width, 1.0) <= 0.15


def test_semantic_mask_flows_through_existing_geometry(monkeypatch) -> None:
    class _Input:
        name = "images"
        shape = [1, 3, 256, 256]
        type = "tensor(float)"

    class _Output:
        name = "logits"
        shape = [1, 2, 256, 256]
        type = "tensor(float)"

    class _Session:
        def get_inputs(self):
            return [_Input()]

        def get_outputs(self):
            return [_Output()]

        def run(self, _names, feed):
            assert feed["images"].shape == (1, 3, 256, 256)
            logits = np.zeros((1, 2, 256, 256), dtype=np.float32)
            logits[:, 0] = 1.0
            # A centered road corridor covering the lower half of the frame.
            logits[:, 1, 100:256, 72:184] = 4.0
            return [logits]

    monkeypatch.setattr(road, "_SESSION", _Session())
    monkeypatch.setattr(road, "_INPUT_NAME", "images")
    monkeypatch.setattr(road, "_MODEL_INPUT_SIZE", 256)
    monkeypatch.setattr(road, "_SESSION_PROVIDER", "NPU_NBGraph")
    monkeypatch.setattr(road, "_MODEL_KIND", road.MODEL_KIND_SEMANTIC)
    monkeypatch.setattr(road, "_USE_CROP_PREPROCESS", False)
    monkeypatch.setattr(road, "_POSTPROCESS_MODE", road.POSTPROCESS_FAST_MAIN)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = road.get_road_perception(frame)

    assert result.is_road_found
    assert result.road_state == "single"
    assert result.branch_decision == "disabled"
    assert result.branches == []
    assert result.selected_branch is None
    assert len(result.centerline_points) >= road.MIN_FIT_PTS
    assert abs(result.pixel_error) < 5.0
    assert result.confidence > 0.9
    assert result.debug_mask is not None
    assert result.debug_mask.shape == (480, 640)

    monkeypatch.setattr(road, "_POSTPROCESS_MODE", road.POSTPROCESS_FULL)
    full_result = road.get_road_perception(frame, branch_preference="left")
    assert full_result.is_road_found
    assert full_result.branch_decision == "disabled"
    assert full_result.branches == []
    assert full_result.selected_branch is None
    assert len(full_result.centerline_points) >= road.MIN_FIT_PTS
    assert full_result.debug_mask is not None
    assert full_result.debug_mask.shape == (480, 640)
