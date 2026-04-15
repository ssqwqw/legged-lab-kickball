from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_ball(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    forward_range: tuple = (5.0, 5.0),
    lateral_range: tuple = (0.0, 0.0),
    ball_radius: float = 0.11,
):
    """Reset ball to a randomised position in front of the robot at ground level.

    The ball is placed between *forward_range[0]* and *forward_range[1]* metres ahead
    of the robot (along its yaw direction) and between *lateral_range[0]* and
    *lateral_range[1]* metres to the left (+y) of that forward axis.
    z is set so the ball rests on the ground. Velocity is zeroed out.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos_w = robot.data.root_pos_w[env_ids]  # (N, 3)
    root_quat_w = robot.data.root_quat_w[env_ids]  # (N, 4)

    from isaaclab.utils.math import quat_apply, yaw_quat

    yaw_q = yaw_quat(root_quat_w)

    n = len(env_ids)
    # sample forward and lateral offsets
    fwd_lo, fwd_hi = forward_range
    lat_lo, lat_hi = lateral_range
    forward_dist = torch.rand(n, device=env.device) * (fwd_hi - fwd_lo) + fwd_lo  # (N,)
    lateral_dist = torch.rand(n, device=env.device) * (lat_hi - lat_lo) + lat_lo  # (N,)

    forward_w = quat_apply(yaw_q, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(n, -1))  # (N, 3)
    left_w = quat_apply(yaw_q, torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(n, -1))    # (N, 3)

    ball_pos = root_pos_w + forward_w * forward_dist.unsqueeze(-1) + left_w * lateral_dist.unsqueeze(-1)
    ball_pos[:, 2] = ball_radius + env.scene.env_origins[env_ids, 2]

    # identity quaternion
    ball_quat = torch.zeros(n, 4, device=env.device)
    ball_quat[:, 0] = 1.0  # w=1

    ball_pose = torch.cat([ball_pos, ball_quat], dim=-1)
    ball_vel = torch.zeros(n, 6, device=env.device)

    ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(ball_vel, env_ids=env_ids)
