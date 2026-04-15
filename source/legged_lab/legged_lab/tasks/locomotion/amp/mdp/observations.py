from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def ball_pos_b(
    env: ManagerBasedEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball position in the robot body frame. Returns shape (num_envs, 3)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ball_pos_w = ball.data.root_pos_w  # (num_envs, 3)
    root_pos_w = robot.data.root_pos_w  # (num_envs, 3)
    root_quat_w = robot.data.root_quat_w  # (num_envs, 4)

    ball_pos_rel = ball_pos_w - root_pos_w  # (num_envs, 3)
    ball_pos_body = math_utils.quat_apply_inverse(root_quat_w, ball_pos_rel)  # (num_envs, 3)
    return ball_pos_body


def ball_vel_b(
    env: ManagerBasedEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball linear velocity in the robot body frame. Returns shape (num_envs, 3)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ball_lin_vel_w = ball.data.root_lin_vel_w  # (num_envs, 3)
    root_quat_w = robot.data.root_quat_w  # (num_envs, 4)

    ball_vel_body = math_utils.quat_apply_inverse(root_quat_w, ball_lin_vel_w)  # (num_envs, 3)
    return ball_vel_body


def root_local_rot_tan_norm(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)

    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)

    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def ref_root_local_rot_tan_norm(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_yaw_quat = math_utils.yaw_quat(ref_root_quat)
    ref_root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(ref_yaw_quat), ref_root_quat
    )  # shape: (num_envs, num_steps, 4)
    ref_root_rotm_local = math_utils.matrix_from_quat(ref_root_quat_local)  # shape: (num_envs, num_steps, 3, 3)

    tan_vec = ref_root_rotm_local[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = ref_root_rotm_local[:, :, :, 2]  # (num_envs, num_steps, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)

    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs
