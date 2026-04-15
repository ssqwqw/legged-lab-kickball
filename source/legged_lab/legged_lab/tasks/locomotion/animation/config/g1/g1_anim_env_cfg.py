import os

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.tasks.locomotion.animation.animation_env_cfg import AnimationEnvCfg


@configclass
class G1AnimEnvCfg(AnimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot_anim = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_anim")
        self.scene.robot_anim.spawn.rigid_props.disable_gravity = True  # type: ignore
        self.scene.robot_anim.spawn.articulation_props.enabled_self_collisions = False  # type: ignore
        self.scene.robot_anim.spawn.activate_contact_sensors = False  # type: ignore
        self.scene.robot_anim.spawn.collision_props = sim_utils.CollisionPropertiesCfg(  # type: ignore
            collision_enabled=False
        )

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "fpstrans_resampled"
        )
        self.motion_data.motion_dataset.motion_data_weights = {


            # "football": 1.0,
            # "myfootball": 1.0,



            # "C1_-_stand_to_run_stageii": 1.0,
            # "C3_-_run_stageii": 1.0,
            # "C4_-_run_to_walk_a_stageii": 1.0,
            # "C5_-_walk_to_run_stageii": 1.0,
            # "C6_-_stand_to_run_backwards_stageii": 1.0,
            # "C8_-_run_backwards_to_stand_stageii": 1.0,
            # "C9_-_run_backwards_turn_run_forward_stageii": 1.0,
            # "C11_-_run_turn_left_90_stageii": 1.0,
            # "C12_-_run_turn_left_45_stageii": 1.0,
            # "C13_-_run_turn_left_135_stageii": 1.0,
            # "C14_-_run_turn_right_90_stageii": 1.0,
            # "C15_-_run_turn_right_45_stageii": 1.0,
            # "C16_-_run_turn_right_135_stageii": 1.0,
            # "C17_-_run_change_direction_stageii": 1.0,


            # "E5_-__hook_left_stageii": 1.0,
            # "G10-__roundhouse_leading_left_stageii": 1.0,
            # "G12-__cresent_left_stageii": 1.0,
            # "G13-__cresent_right_stageii": 1.0,
            # "G19-__reverse_spin_cresent_left_stageii": 1.0,
            # "G20_-__reverse_spin_cresent_right_stageii": 1.0,
            # "G20_-__reverse_spin_cresent_right_stageii": 1.0,


            # "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
            # "B9_-__Walk_turn_left_90_stageii": 1.0,
            # "B10_-__Walk_turn_left_45_stageii": 1.0,
            # "B11_-__Walk_turn_left_135_stageii": 1.0,
            # "B13_-__Walk_turn_right_90_stageii": 1.0,
            # "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
            # "B15_-__Walk_turn_around_stageii": 1.0,
            # "B22_-__side_step_left_stageii": 1.0,
            # "B23_-__side_step_right_stageii": 1.0,
        

        }
        # self.motion_data.motion_dataset.key_link_names = [
        #     "left_ankle_roll_link",
        #     "right_ankle_roll_link",
        #     "left_wrist_yaw_link",
        #     "right_wrist_yaw_link",
        # ]

        self.animation.animation.random_initialize = True
        self.animation.animation.num_steps_to_use = 10
