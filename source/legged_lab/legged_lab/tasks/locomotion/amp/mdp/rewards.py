from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_orientation_l2(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)

    num_feet = len(sensor_cfg.body_ids)

    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]  # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)

    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )


def approach_ball_exp(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 2.0,
) -> torch.Tensor:
    """Exponential reward for reducing horizontal (XY) distance to the ball.

    r = exp(-||ball_xy - robot_xy|| / std)
    Always positive; pulls the robot toward the ball.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    dist = torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)  # (N,)
    return torch.exp(-dist / std)


def foot_to_ball_proximity(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.25,
) -> torch.Tensor:
    """Exponential reward for the closest foot approaching the ball.

    r = exp(-min_foot_dist / std)
    Activates sharply as any ankle enters kicking range (~0.3 m).
    requires asset_cfg to resolve body_ids for the feet.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]

    ball_pos_w = ball.data.root_pos_w  # (N, 3)
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, num_feet, 3)

    diff = foot_pos_w - ball_pos_w.unsqueeze(1)  # (N, num_feet, 3)
    min_dist = torch.norm(diff, dim=-1).min(dim=-1).values  # (N,)
    return torch.exp(-min_dist / std)


def ball_speed_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    max_speed: float = 5.0,
) -> torch.Tensor:
    """Tanh-scaled reward for ball speed after being kicked.

    r = tanh(||v_ball|| / max_speed)
    Zero when ball is still; saturates at 1.0 for a well-kicked ball.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    speed = torch.norm(ball.data.root_lin_vel_w, dim=-1)  # (N,)
    return torch.tanh(speed / max_speed)


def ball_kicked_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_separation: float = 0.5,
    max_speed: float = 5.0,
) -> torch.Tensor:
    """Ball-speed reward gated by separation distance — prevents dribble exploitation.

    r = tanh(||v_ball|| / max_speed)  *  𝟙(dist_xy > min_separation)

    Only non-zero when the ball has actually left the robot's feet.
    Dribbling (ball stays near feet) yields exactly 0.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    dist = torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    speed = torch.norm(ball.data.root_lin_vel_w, dim=-1)

    return torch.tanh(speed / max_speed) * (dist > min_separation).float()


def ball_linger_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
) -> torch.Tensor:
    """Penalty when ball lingers near the robot — discourages dribbling.

    r = clamp(1 - dist_xy / threshold, 0, 1)

    Returns 1.0 when ball is on top of the robot, linearly falls to 0 at *threshold* m.
    Use with a negative weight to penalise.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    dist = torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    return torch.clamp(1.0 - dist / threshold, min=0.0)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)
