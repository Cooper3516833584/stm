"""Shared command types for top-level autonomy programs.

These classes are intentionally small. Mission modules create desired
commands, and the safety arbiter converts them into commands that may be sent
to the flight controller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityCommand:
    """Body-frame velocity command.

    Units:
        vx/vy/vz: cm/s
        yaw_rate: deg/s, positive convention follows the existing FC wrapper.
    """

    vx_cm_s: float = 0.0
    vy_cm_s: float = 0.0
    vz_cm_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    reason: str = "unset"

    @classmethod
    def zero(cls, reason: str = "zero") -> "VelocityCommand":
        return cls(0.0, 0.0, 0.0, 0.0, reason)

    def with_reason(self, reason: str) -> "VelocityCommand":
        return VelocityCommand(
            self.vx_cm_s,
            self.vy_cm_s,
            self.vz_cm_s,
            self.yaw_rate_deg_s,
            reason,
        )

    def clamp(
        self,
        *,
        max_xy_speed_cm_s: float,
        max_z_speed_cm_s: float,
        max_yaw_rate_deg_s: float,
        reason_suffix: str = "clamped",
    ) -> "VelocityCommand":
        vx = _clip(self.vx_cm_s, -max_xy_speed_cm_s, max_xy_speed_cm_s)
        vy = _clip(self.vy_cm_s, -max_xy_speed_cm_s, max_xy_speed_cm_s)
        vz = _clip(self.vz_cm_s, -max_z_speed_cm_s, max_z_speed_cm_s)
        yaw = _clip(self.yaw_rate_deg_s, -max_yaw_rate_deg_s, max_yaw_rate_deg_s)
        reason = self.reason
        if (vx, vy, vz, yaw) != (
            self.vx_cm_s,
            self.vy_cm_s,
            self.vz_cm_s,
            self.yaw_rate_deg_s,
        ):
            reason = f"{reason}+{reason_suffix}"
        return VelocityCommand(vx, vy, vz, yaw, reason)

    def as_fc_tuple(self) -> tuple[int, int, int, int]:
        return (
            round(self.vx_cm_s),
            round(self.vy_cm_s),
            round(self.vz_cm_s),
            round(self.yaw_rate_deg_s),
        )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))

