"""HC-14 device discovery and startup-gate tests."""

import threading
from types import SimpleNamespace

import pytest

from fleet_bus import air_node
from fleet_bus import hc14_transport


def _port(device, *, vid=None, pid=None, hwid="", description=""):
    return SimpleNamespace(
        device=device,
        vid=vid,
        pid=pid,
        hwid=hwid,
        description=description,
    )


def test_discovery_selects_ch340_and_not_cp210x(monkeypatch):
    ports = [
        _port("/dev/ttyUSB0", vid=0x10C4, pid=0xEA60),
        _port("/dev/ttyUSB1", vid=0x1A86, pid=0x7523),
    ]
    monkeypatch.setattr(hc14_transport, "_list_serial_ports", lambda: ports)
    monkeypatch.setattr(
        hc14_transport,
        "_stable_serial_path",
        lambda device: (
            "/dev/serial/by-path/hc14"
            if device == "/dev/ttyUSB1"
            else device
        ),
    )

    port, baudrate = hc14_transport.resolve_hc14_settings()

    assert port == "/dev/serial/by-path/hc14"
    assert baudrate == 115200


def test_explicit_port_and_environment_override_auto_detection(monkeypatch):
    monkeypatch.setenv("D_TASK_HC14_PORT", "/dev/from-environment")
    monkeypatch.setenv("D_TASK_HC14_BAUDRATE", "57600")

    assert hc14_transport.resolve_hc14_settings() == (
        "/dev/from-environment",
        57600,
    )
    assert hc14_transport.resolve_hc14_settings("/dev/explicit", 9600) == (
        "/dev/explicit",
        9600,
    )


def test_enumerated_ch340_without_tty_reports_missing_kernel_driver(monkeypatch):
    monkeypatch.setattr(hc14_transport, "_list_serial_ports", lambda: [])
    monkeypatch.setattr(hc14_transport, "_usb_device_present", lambda *_: True)

    with pytest.raises(RuntimeError, match="CONFIG_USB_SERIAL_CH341"):
        hc14_transport.resolve_hc14_settings()


def test_multiple_ch340_devices_require_an_explicit_selection(monkeypatch):
    ports = [
        _port("/dev/ttyUSB1", vid=0x1A86, pid=0x7523),
        _port("/dev/ttyUSB2", hwid="USB VID:PID=1A86:7523"),
    ]
    monkeypatch.setattr(hc14_transport, "_list_serial_ports", lambda: ports)

    with pytest.raises(RuntimeError, match="Multiple CH340 devices"):
        hc14_transport.resolve_hc14_settings()


class _FakeTransport:
    instances = []
    should_connect = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_error = OSError("test port unavailable")
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        return None

    def stop(self):
        self.stopped = True

    def wait_connected(self, timeout_s):
        self.timeout_s = timeout_s
        return self.should_connect

    def write(self, data):
        return None


def _attach_fake_transport(monkeypatch, *, should_connect):
    _FakeTransport.instances.clear()
    _FakeTransport.should_connect = should_connect
    monkeypatch.setattr(
        hc14_transport,
        "resolve_hc14_settings",
        lambda port, baudrate: (port or "/dev/test-hc14", baudrate or 115200),
    )
    return air_node.attach_air_fleet_node(
        None,
        None,
        threading.Event(),
        state_provider=lambda: SimpleNamespace(),
        hc14_port="/dev/test-hc14",
        connect_timeout_s=0.25,
        transport_factory=_FakeTransport,
    )


def test_attach_refuses_to_return_until_hc14_is_connected(monkeypatch):
    with pytest.raises(RuntimeError, match="did not connect"):
        _attach_fake_transport(monkeypatch, should_connect=False)

    transport = _FakeTransport.instances[-1]
    assert transport.timeout_s == 0.25
    assert transport.stopped


def test_attach_returns_after_hc14_connection(monkeypatch):
    node = _attach_fake_transport(monkeypatch, should_connect=True)
    try:
        assert _FakeTransport.instances[-1].timeout_s == 0.25
    finally:
        node.close()


def test_attach_rejects_negative_connection_timeout_before_starting_transport():
    _FakeTransport.instances.clear()

    with pytest.raises(ValueError, match="must not be negative"):
        air_node.attach_air_fleet_node(
            None,
            None,
            threading.Event(),
            state_provider=lambda: SimpleNamespace(),
            hc14_port="/dev/test-hc14",
            connect_timeout_s=-0.1,
            transport_factory=_FakeTransport,
        )

    assert not _FakeTransport.instances
