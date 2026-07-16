from __future__ import annotations

import numpy as np

from perception_pipeline import SharedLatest, YOLOInferenceThread
import road_perception as road


class _Meta:
    def __init__(self, shape):
        self.shape = shape


def test_semantic_model_contract_is_detected() -> None:
    assert road._model_kind_from_output_meta([_Meta([1, 2, 256, 256])]) == road.MODEL_KIND_SEMANTIC
    assert road._model_kind_from_output_meta([_Meta([1, np.int32(2), 256, 256])]) == road.MODEL_KIND_SEMANTIC
    assert road._model_kind_from_output_meta([_Meta([1, 37, 336]), _Meta([1, 32, 32, 32])]) == road.MODEL_KIND_YOLO


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
    ):
        monkeypatch.setattr(road, name, getattr(road, name))

    road.configure_model(backend="cpu")
    cpu_path, is_nb = road._resolve_model_path()
    assert not is_nb
    assert cpu_path.endswith("road_yolo11n_seg_128.onnx")

    road.configure_model(backend="npu")
    npu_path, is_nb = road._resolve_model_path()
    assert is_nb
    assert npu_path.endswith("new_road_seg_v3_final_fp32.nb")


def test_inference_thread_configures_selected_backend(monkeypatch) -> None:
    calls = []

    def fake_configure_model(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(road, "configure_model", fake_configure_model)
    monkeypatch.setattr(
        road,
        "get_model_io_info",
        lambda: {"provider": "NPU_NBGraph", "model_kind": road.MODEL_KIND_SEMANTIC},
    )

    camera = type("Camera", (), {"frame_buffer": SharedLatest()})()
    worker = YOLOInferenceThread(
        camera_thread=camera,
        model_path="legacy.onnx",
        npu_model_path="new.nb",
        inference_backend="npu",
    )
    worker.start()
    worker.stop()

    assert calls == [
        {
            "backend": "npu",
            "cpu_model_path": "legacy.onnx",
            "npu_model_path": "new.nb",
        }
    ]


def test_semantic_preprocess_matches_training_contract() -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)  # OpenCV BGR

    blob = road._preprocess_semantic(frame, 256)

    assert blob.shape == (1, 3, 256, 256)
    assert blob.dtype == np.float32
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

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = road.get_road_perception(frame)

    assert result.is_road_found
    assert result.road_state == "single"
    assert abs(result.pixel_error) < 5.0
    assert result.confidence > 0.9
    assert result.debug_mask is not None
