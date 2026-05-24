"""Mission logic for body-frame goal navigation."""

from __future__ import annotations

from dataclasses import dataclass

from autonomy_command import VelocityCommand
from direction_planner import DirectionPlanner
from local_world_model import LocalWorldModel


@dataclass
class GoalNavConfig:
    goal_x_cm: float = 300.0
    goal_y_cm: float = 0.0
    forward_test: bool = False


class GoalNavMission:
    """Relative goal navigation placeholder.

    Until a pose layer is added, the goal is treated as a body-frame direction.
    It is suitable for early forward/avoidance tests, not full map navigation.
    """

    def __init__(
        self,
        config: GoalNavConfig | None = None,
        planner: DirectionPlanner | None = None,
    ):
        self.config = config or GoalNavConfig()
        self.planner = planner or DirectionPlanner()

    def update(self, world: LocalWorldModel) -> VelocityCommand:
        if self.config.forward_test:
            return self.planner.plan_forward_test()
        return self.planner.plan_to_body_goal(
            self.config.goal_x_cm,
            self.config.goal_y_cm,
            world,
        )

