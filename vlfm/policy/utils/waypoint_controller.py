# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Geometric waypoint follower for global navigation (Module 3, option B).

Turns the next waypoint into a discrete Habitat action (turn-in-place until
roughly facing it, then step forward). Deliberately has no learned component and
no dependency on the pointnav RNN -- the static cat_demo scene does not need
learned local obstacle avoidance, and the A* path already routes around the
agent-radius-dilated obstacles.

The action tensors below mirror ``TorchActionIDs`` in
``vlfm/policy/habitat_policies.py`` (STOP=0, MOVE_FORWARD=1, TURN_LEFT=2,
TURN_RIGHT=3). They are duplicated here (rather than imported) to avoid a
circular import: ``habitat_policies`` imports ``base_objectnav_policy``, which
imports this module. Shape ``[[n]]`` / dtype long / CPU matches ``_stop_action``
so the downstream ``.detach().cpu().numpy()[0]`` consumer is unaffected.
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from vlfm.utils.geometry_utils import rho_theta

MOVE_FORWARD: Tensor = torch.tensor([[1]], dtype=torch.long)
TURN_LEFT: Tensor = torch.tensor([[2]], dtype=torch.long)
TURN_RIGHT: Tensor = torch.tensor([[3]], dtype=torch.long)


def step_towards(
    waypoint_xy: np.ndarray,
    robot_xy: np.ndarray,
    heading: float,
    turn_angle_rad: float,
    arrive_radius: float,
) -> Optional[Tensor]:
    """Return the discrete action that moves the robot toward ``waypoint_xy``.

    Args:
        waypoint_xy: Target ``(x, y)`` in meters (episodic frame).
        robot_xy: Current robot ``(x, y)`` in meters.
        heading: Current robot heading in radians (CCW-from-above, as used by
            ``rho_theta``).
        turn_angle_rad: The simulator's per-action turn step in radians. We turn
            whenever the heading error exceeds half of it, so a single turn does
            not overshoot the facing zone.
        arrive_radius: Distance at which the waypoint counts as reached.

    Returns:
        ``MOVE_FORWARD`` / ``TURN_LEFT`` / ``TURN_RIGHT`` tensor, or ``None`` if
        already within ``arrive_radius`` (caller should advance to the next
        waypoint or stop).
    """
    rho, theta = rho_theta(np.asarray(robot_xy, dtype=np.float64), heading, np.asarray(waypoint_xy, dtype=np.float64))
    if rho < arrive_radius:
        return None
    if abs(theta) > turn_angle_rad / 2.0:
        # theta > 0 means the goal is to the left (CCW), so turn left.
        return TURN_LEFT if theta > 0 else TURN_RIGHT
    return MOVE_FORWARD
