from legged_gym.envs.jandi_base.jandi_base_config import (
    JandiRobotBaseCfg,
    JandiRobotBaseCfgPPO,
)


class JandiThrowCfg(JandiRobotBaseCfg):
    class env(JandiRobotBaseCfg.env):
        num_actions = 3
        num_observations = 23
        num_privileged_obs = 23
        episode_length_s = 3.0

    class commands(JandiRobotBaseCfg.commands):
        num_commands = 3
        resampling_time = 9999.0
        heading_command = False

        class ranges(JandiRobotBaseCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class throw:
        arm_side = "left"
        controlled_dof_names = [
            "LH1_joint",
            "LH3_joint",
            "LH4_joint",
        ]
        hand_body_name = "LH4_link"
        action_scale = 1.0
        phase_boundaries = [0.25, 0.55, 0.80]
        phase_hand_targets = [
            [0.1108, 0.1275, -0.0103],
            [-0.0191, 0.1275, 0.1758],
            [0.1711, 0.1275, 0.1519],
            [0.1108, 0.1275, -0.0103],
        ]
        interpolate_phase_targets = True
        pose_tracking_sigma = 0.06
        pose_error_norm = 0.06
        pose_error_clip = 4.0
        throw_dir_base = [1.0, 0.0, 0.2]
        throw_speed_sigma = 1.0

    class init_state(JandiRobotBaseCfg.init_state):
        default_joint_angles = {
            **JandiRobotBaseCfg.init_state.default_joint_angles,
            "LH1_joint": 0.0,
            "LH2_joint": 0.0,
            "LH3_joint": 0.0,
            "LH4_joint": 0.0,
            "waist_joint": 0.0
        }

    class control(JandiRobotBaseCfg.control):
        stiffness = {
            **JandiRobotBaseCfg.control.stiffness,
            "LH1_joint": 2.0,
            "LH2_joint": 2.0,
            "LH3_joint": 2.5,
            "LH4_joint": 1.5,
            "waist_joint": 1.5,
            # "LH1_joint": 8.0,
            # "LH2_joint": 15.0,
            # "LH3_joint": 10.0,
            # "LH4_joint": 10,
            # "waist_joint": 1.5,           
        }
        damping = {
            **JandiRobotBaseCfg.control.damping,
            "LH1_joint": 0.50,
            "LH2_joint": 0.70,
            "LH3_joint": 1.00,
            "LH4_joint": 0.50,
            "waist_joint": 0.08,
            # "LH1_joint": 0.5,
            # "LH2_joint": 0.70,
            # "LH3_joint": 1.0,
            # "LH4_joint": 0.5,
            # "waist_joint": 0.08,
        }
        action_scale = 1.0
        action_scale_per_joint = JandiRobotBaseCfg.control.action_scale_per_joint

    class asset(JandiRobotBaseCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/jandi/jandi_left_arm.urdf"
        name = "jandi"
        disable_gravity = False
        foot_name = "6_link"
        collapse_fixed_joints = True
        # Phase 1 throw pretraining: keep the trunk fixed so the policy can
        # learn arm motion before balance.
        fix_base_link = True
        penalize_contacts_on = ["base_link", "3_link", "4_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0
        flip_visual_attachments = False


    class rewards(JandiRobotBaseCfg.rewards):
        only_positive_rewards = False

        class scales(JandiRobotBaseCfg.rewards.scales):
            pose_tracking = 1.0
            pose_error = -2.0
            throw_velocity = 0.0
            action_rate = -0.08
            dof_vel = -0.001

    class noise(JandiRobotBaseCfg.noise):
        add_noise = False


class JandiThrowCfgPPO(JandiRobotBaseCfgPPO):
    class runner(JandiRobotBaseCfgPPO.runner):
        experiment_name = "jandi_throw"
