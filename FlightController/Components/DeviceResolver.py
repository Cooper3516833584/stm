import glob
import os
from dataclasses import dataclass
from typing import List, Optional

from serial.tools.list_ports import comports

FC_VID_PID = "66CC:2233"
D500_VID_PID = "10C4:EA60"


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str
    hwid: str


def list_serial_ports() -> List[SerialPortInfo]:
    ports: List[SerialPortInfo] = []
    for port in sorted(comports(), key=lambda item: getattr(item, "device", "")):
        ports.append(
            SerialPortInfo(
                device=getattr(port, "device", "") or "",
                description=getattr(port, "description", "") or "",
                hwid=getattr(port, "hwid", "") or "",
            )
        )
    return ports


def find_serial_port(
    *,
    explicit: Optional[str] = None,
    env_var: Optional[str] = None,
    vid_pid: Optional[str] = None,
    by_id_keyword: Optional[str] = None,
    required: bool = True,
) -> Optional[str]:
    if explicit:
        return explicit

    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value

    if by_id_keyword:
        by_id_match = _find_by_id_keyword(by_id_keyword)
        if by_id_match:
            return by_id_match

    ports = list_serial_ports()
    if vid_pid:
        matches = _ports_matching_vid_pid(vid_pid, ports)
        if matches:
            return _stable_device_path(matches[0].device)
    elif ports:
        return _stable_device_path(ports[0].device)

    if required:
        raise RuntimeError(_format_not_found_error(vid_pid=vid_pid, by_id_keyword=by_id_keyword))
    return None


def resolve_fc_port(explicit: Optional[str] = None, required: bool = True) -> Optional[str]:
    return find_serial_port(
        explicit=explicit,
        env_var="FC_PORT",
        vid_pid=FC_VID_PID,
        by_id_keyword=FC_VID_PID,
        required=required,
    )


def resolve_radar_port(index: int = 0, explicit: Optional[str] = None, required: bool = True) -> Optional[str]:
    if index < 0:
        raise ValueError("Radar index must be >= 0")

    if explicit:
        return explicit

    env_value = os.environ.get(f"RADAR{index}_PORT")
    if env_value:
        return env_value

    matches = _ports_matching_vid_pid(D500_VID_PID, list_serial_ports())
    if index < len(matches):
        return _stable_device_path(matches[index].device)

    if required:
        raise RuntimeError(
            _format_not_found_error(
                vid_pid=D500_VID_PID,
                by_id_keyword=None,
                extra=f"radar index={index}",
            )
        )
    return None


class DeviceResolver:
    list_serial_ports = staticmethod(list_serial_ports)
    find_serial_port = staticmethod(find_serial_port)
    resolve_fc_port = staticmethod(resolve_fc_port)
    resolve_radar_port = staticmethod(resolve_radar_port)


def _ports_matching_vid_pid(vid_pid: str, ports: List[SerialPortInfo]) -> List[SerialPortInfo]:
    return sorted((port for port in ports if vid_pid in port.hwid), key=lambda port: port.device)


def _by_id_links() -> List[str]:
    return sorted(glob.glob("/dev/serial/by-id/*"))


def _find_by_id_keyword(keyword: str) -> Optional[str]:
    keyword_lower = keyword.lower()
    for link in _by_id_links():
        real_path = os.path.realpath(link)
        if keyword_lower in link.lower() or keyword_lower in real_path.lower():
            return link
    return None


def _stable_device_path(device: str) -> str:
    by_id = _by_id_link_for_device(device)
    return by_id or device


def _by_id_link_for_device(device: str) -> Optional[str]:
    real_device = os.path.realpath(device)
    for link in _by_id_links():
        if os.path.realpath(link) == real_device:
            return link
    return None


def _format_not_found_error(
    *,
    vid_pid: Optional[str],
    by_id_keyword: Optional[str],
    extra: Optional[str] = None,
) -> str:
    ports = list_serial_ports()
    port_text = ", ".join(
        f"{port.device} ({port.description}; {port.hwid})"
        for port in ports
    ) or "none"
    by_id_text = ", ".join(
        f"{link} -> {os.path.realpath(link)}"
        for link in _by_id_links()
    ) or "none"
    parts = ["Serial port not found"]
    if vid_pid:
        parts.append(f"vid_pid={vid_pid}")
    if by_id_keyword:
        parts.append(f"by_id_keyword={by_id_keyword}")
    if extra:
        parts.append(extra)
    parts.append(f"available ports: {port_text}")
    parts.append(f"/dev/serial/by-id: {by_id_text}")
    return "; ".join(parts)


__all__ = [
    "D500_VID_PID",
    "DeviceResolver",
    "FC_VID_PID",
    "SerialPortInfo",
    "find_serial_port",
    "list_serial_ports",
    "resolve_fc_port",
    "resolve_radar_port",
]
