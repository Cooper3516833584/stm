from types import SimpleNamespace

import perception_pipeline


class _FakeCapture:
    def __init__(self):
        self.settings = []

    def set(self, property_id, value):
        self.settings.append((property_id, value))
        return True


def test_open_camera_uses_v4l2_mjpg_and_bounded_buffer():
    capture = _FakeCapture()
    calls = []
    cv2 = SimpleNamespace(
        CAP_V4L2=200,
        CAP_PROP_FOURCC=6,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        CAP_PROP_BUFFERSIZE=38,
        VideoCapture=lambda index, backend: calls.append((index, backend)) or capture,
        VideoWriter_fourcc=lambda *chars: 1196444237,
    )

    opened = perception_pipeline._open_camera_capture(cv2, 7, 640, 480, 15)

    assert opened is capture
    assert calls == [(7, cv2.CAP_V4L2)]
    assert capture.settings == [
        (cv2.CAP_PROP_FOURCC, 1196444237),
        (cv2.CAP_PROP_FRAME_WIDTH, 640),
        (cv2.CAP_PROP_FRAME_HEIGHT, 480),
        (cv2.CAP_PROP_FPS, 15),
        (cv2.CAP_PROP_BUFFERSIZE, 1),
    ]
