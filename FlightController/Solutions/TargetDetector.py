from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    center: tuple[float, float]
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None


class TargetDetector:
    def __init__(
        self,
        model: str = "fastestdet_onnx",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        draw_output: bool = False,
    ):
        # Future model files should live in FlightController/models.
        # Legacy Vision_Net.py still resolves models from Solutions/models; resource
        # path cleanup can be done in a later migration step.
        import importlib

        Vision_Net = importlib.import_module(".Vision_Net", __package__)

        self.model = model
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.draw_output = draw_output

        if model == "fastestdet_onnx":
            self._detector = Vision_Net.FastestDetOnnx(
                confThreshold=conf_threshold,
                nmsThreshold=nms_threshold,
                drawOutput=draw_output,
            )
        elif model == "damoyolo":
            self._detector = Vision_Net.DAMO_YOLO(
                confThreshold=conf_threshold,
                nmsThreshold=nms_threshold,
                drawOutput=draw_output,
            )
        else:
            raise ValueError(f"Unsupported target detector model: {model}")

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        raw_results = self._detector.detect(frame)
        return [self._to_detection_result(result) for result in raw_results]

    def detect_best(self, frame: np.ndarray, class_name: str | None = None) -> DetectionResult | None:
        results = self.detect(frame)
        if class_name is not None:
            results = [result for result in results if result.class_name == class_name]
        if not results:
            return None
        return max(results, key=lambda result: result.confidence)

    def _to_detection_result(self, result) -> DetectionResult:
        if len(result) < 3:
            center, confidence = result
            class_name = ""
            bbox = None
        else:
            center, class_name, confidence = result[:3]
            bbox = result[3] if len(result) > 3 else None
        return DetectionResult(
            center=(float(center[0]), float(center[1])),
            class_name=str(class_name),
            confidence=float(confidence),
            bbox=tuple(float(value) for value in bbox) if bbox is not None else None,
        )


__all__ = ["DetectionResult", "TargetDetector"]
