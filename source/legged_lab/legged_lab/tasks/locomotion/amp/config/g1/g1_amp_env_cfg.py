import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]  # if changed here and symmetry is enabled, remember to update amp.mdp.symmetry.g1 as well!
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4
BALL_RADIUS = 0.11
BALL_MASS = 0.45


@configclass
class G1AmpRewards:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- stand-still stability: penalise joint deviation when command is near zero
    stand_still_penalty = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ------------------------------------------------------------------
    # Kickball task rewards
    # ------------------------------------------------------------------
    # Stage 1 weights (ball at fixed 5 m front).
    # Increase approach/kick weights in Stage 2-3 as the robot gains skill.

    # r1  Approach: always-on exponential pull toward the ball
    #     r = exp(-dist_xy / 2.0)   weight chosen so full reward ≈ 1.0 at d=0
    approach_ball = RewTerm(
        func=mdp.approach_ball_exp,
        weight=1.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "std": 2.0,
        },
    )

    # r2  Foot-contact: light guidance for initial approach only
    #     Lowered from 3.0 to prevent dribble-style reward exploitation
    foot_to_ball = RewTerm(
        func=mdp.foot_to_ball_proximity,
        weight=0.5,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "std": 0.25,
        },
    )

    # r3  Kick success: ball speed — ONLY counted when ball is >0.5 m from robot
    #     Dribbling at feet yields exactly 0.  Must kick ball away to score.
    ball_kicked = RewTerm(
        func=mdp.ball_kicked_reward,
        weight=4.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "min_separation": 0.5,
            "max_speed": 5.0,
        },
    )

    # r4  Linger penalty: ball stuck near feet → continuous negative reward
    ball_linger = RewTerm(
        func=mdp.ball_linger_penalty,
        weight=-3.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "threshold": 0.5,
        },
    )


@configclass
class G1AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the G1 AMP environment."""

    rewards: G1AmpRewards = G1AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------
        # ball
        # ------------------------------------------------------
        self.scene.ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=BALL_RADIUS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.9, 0.1),
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6,
                    dynamic_friction=0.4,
                    restitution=0.5,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.0, 0.0, BALL_RADIUS),
            ),
        )

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "fpstrans_resampled"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            # "B10_-__Walk_turn_left_45_stageii": 1.0,
            # "B11_-__Walk_turn_left_135_stageii": 1.0,
            # "B13_-__Walk_turn_right_90_stageii": 1.0,
            # "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
            # "B15_-__Walk_turn_around_stageii": 1.0,
            # "B22_-__side_step_left_stageii": 1.0,
            # "B23_-__side_step_right_stageii": 1.0,
            # "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
            # "B9_-__Walk_turn_left_90_stageii": 1.0,
            # "C11_-_run_turn_left_90_stageii": 1.0,
            # "C12_-_run_turn_left_45_stageii": 1.0,
            # "C13_-_run_turn_left_135_stageii": 1.0,
            # "C14_-_run_turn_right_90_stageii": 1.0,
            # "C15_-_run_turn_right_45_stageii": 1.0,
            # "C16_-_run_turn_right_135_stageii": 1.0,
            # "C17_-_run_change_direction_stageii": 1.0,


            # "C1_-_stand_to_run_stageii": 1.0,
            "C3_-_run_stageii": 1.0,
            "C1_-_stand_to_run_stageii": 0.6,
            "C8_-_run_backwards_to_stand_stageii": 0.4,
            # "B9_-__Walk_turn_left_90_stageii": 1.0,
            # "B11_-__Walk_turn_left_135_stageii": 1.0,
            # "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
            # "football": 0.8,
            # "myfootball": 0.8,


            # "C4_-_run_to_walk_a_stageii": 1.0,
            # "C5_-_walk_to_run_stageii": 1.0,
            # "C6_-_stand_to_run_backwards_stageii": 1.0,
            # "C8_-_run_backwards_to_stand_stageii": 0.4,
            # "C9_-_run_backwards_turn_run_forward_stageii": 1.0,
            # "Walk_B10_-_Walk_turn_left_45_stageii": 1.0,
            # "Walk_B13_-_Walk_turn_right_45_stageii": 1.0,
            # "Walk_B15_-_Walk_turn_around_stageii": 1.0,
            # "Walk_B16_-_Walk_turn_change_stageii": 1.0,
            # "Walk_B22_-_Side_step_left_stageii": 1.0,
            # "Walk_B23_-_Side_step_right_stageii": 1.0,
            # "Walk_B4_-_Stand_to_Walk_Back_stageii": 1.0,
        }

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        self.terminal_obs_groups = ("disc",)

        # policy observations

        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }

        # critic observations

        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }

        # discriminator observations

        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.history_length = AMP_NUM_STEPS

        # discriminator demonstration observations

        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_from_ref.params = {"animation": ANIMATION_TERM_NAME, "height_offset": 0.1}
        self.events.reset_ball = EventTerm(
            func=mdp.reset_ball,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                # Stage 1: ball fixed at 5 m directly in front, no lateral offset
                "forward_range": (5.0, 5.0),
                "lateral_range": (0.0, 0.0),
                "ball_radius": BALL_RADIUS,
            },
        )

        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        # start narrow; curriculum will expand these up to the limits defined in CurriculumCfg
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        self.terminations.base_contact = None


@configclass
class G1AmpEnvCfg_PLAY(G1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None


# ===========================================================================
# Curriculum Stage 2 — small ball-position randomisation
# Usage: resume Stage-1 checkpoint, train ~2000 more iterations
#   --task LeggedLab-Isaac-AMP-Kickball-S2-G1-v0  --resume ...
# AMP style_reward_scale should be lowered to ~4.0 in the runner cfg
# ===========================================================================
@configclass
class G1AmpEnvCfg_KickballS2(G1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Increase kickball reward weights slightly
        self.rewards.approach_ball.weight = 1.5
        self.rewards.foot_to_ball.weight = 0.5
        self.rewards.ball_kicked.weight = 5.0
        self.rewards.ball_linger.weight = -4.0

        # Expand ball reset range: ±0.5 m forward, ±0.2 m lateral
        self.events.reset_ball.params["forward_range"] = (4.5, 5.5)
        self.events.reset_ball.params["lateral_range"] = (-0.2, 0.2)


# ===========================================================================
# Curriculum Stage 3 — full randomisation + environment perturbation
# Usage: resume Stage-2 checkpoint, train 1500+ more iterations
#   --task LeggedLab-Isaac-AMP-Kickball-S3-G1-v0  --resume ...
# AMP style_reward_scale should be no lower than 3.0 in the runner cfg
# ===========================================================================
@configclass
class G1AmpEnvCfg_KickballS3(G1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Full-strength kickball rewards
        self.rewards.approach_ball.weight = 1.5
        self.rewards.foot_to_ball.weight = 0.5
        self.rewards.ball_kicked.weight = 6.0
        self.rewards.ball_linger.weight = -5.0

        # Full ball-position randomisation: 4-6 m fwd, ±0.5 m lateral
        self.events.reset_ball.params["forward_range"] = (4.0, 6.0)
        self.events.reset_ball.params["lateral_range"] = (-0.5, 0.5)

        # Randomise ground friction to improve generalisation
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.2)
