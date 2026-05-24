from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from FlightController import FC_Controller


@dataclass
class FCConnectConfig:
    port: Optional[str] = None
    mode: int | None = 2
    timeout_s: float = 10.0
    open_timeout_s: float | None = 10.0
    switch_mode: bool = True


def connect_fc(config: FCConnectConfig | None = None) -> FC_Controller:
    cfg = config or FCConnectConfig()
    fc = FC_Controller()
    try:
        fc.start_listen_serial(
            serial_dev=cfg.port,
            block_until_connected=True,
            open_timeout_s=cfg.open_timeout_s,
        )
    except TimeoutError as exc:
        try:
            fc.close(joined=False)
        finally:
            raise RuntimeError("FC serial open timeout") from exc
    except Exception:
        try:
            fc.close(joined=False)
        finally:
            raise

    ok = fc.wait_for_connection(timeout_s=cfg.timeout_s)
    if ok is False:
        try:
            fc.close()
        finally:
            raise RuntimeError("FC connection timeout")

    if cfg.switch_mode and cfg.mode is not None:
        fc.set_flight_mode(cfg.mode)
        done = fc.wait_for_last_command_done(timeout_s=cfg.timeout_s)
        if done is False:
            try:
                fc.close()
            finally:
                raise RuntimeError(f"FC set_flight_mode({cfg.mode}) timeout")

    return fc


__all__ = ["FCConnectConfig", "connect_fc"]
