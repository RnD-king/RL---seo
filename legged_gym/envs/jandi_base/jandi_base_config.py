from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class JandiRobotBaseCfg(LeggedRobotCfg):
    """Robot-level defaults shared by Jandi tasks.

    Task configs should override observations, commands, and reward scales.
    """

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.39]
        default_joint_angles = {
            # "RL1_joint": 0.0,
            # "RL2_joint": 0.0,
            # "RL3_joint": 0.96,
            # "RL4_joint": -0.40,
            # "RL5_joint": 0.90,
            # "RL6_joint": 0.0,
            # "LL1_joint": 0.0,
            # "LL2_joint": 0.0,
            # "LL3_joint": -0.96,
            # "LL4_joint": 0.40,
            # "LL5_joint": -0.90,
            # "LL6_joint": -0.0,
# default_angles: [0.0, 0.0, 0.96, -0.40, 0.93, 0.0,
#                  0.0, 0.0, -0.96, 0.40, -0.93, 0.0]

            "RL1_joint": 0.0,   # dof10
            "RL2_joint": 0.0,   # dof8
            "RL3_joint": 0.98,   # dof6
            "RL4_joint": -0.38,   # dof4
            "RL5_joint": 0.84,   # dof2
            "RL6_joint": 0.0,   # dof0

            "LL1_joint": 0.0,   # dof11
            "LL2_joint": 0.0,   # dof9
            "LL3_joint": -0.98,   # dof7
            "LL4_joint": 0.38,   # dof5
            "LL5_joint": -0.84,   # dof3
            "LL6_joint": -0.0,   # dof1

        }
        reset_joint_noise = {}

    class env(LeggedRobotCfg.env):
        num_actions = 12
        episode_length_s = 20

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 3
        resampling_time = 10.0
        heading_command = False

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.1, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 3.0]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "L1_joint": 7,
            "L2_joint": 8,
            "L3_joint": 30,
            "L4_joint": 35,
            "L5_joint": 35,
            "L6_joint": 8,
            # "L1_joint": 24.00,
            # "L2_joint": 30.00,
            # "L3_joint": 30.00,
            # "L4_joint": 24.00,
            # "L5_joint": 26.25,
            # "RL6_joint": 21.00,
            # "LL6_joint": 26.25,
        }
        damping = {
            # "L1_joint": 1.08,
            # "L2_joint": 1.44,
            # "L3_joint": 1.44,
            # "L4_joint": 1.44,
            # "L5_joint": 1.08,
            # "LL6_joint": 1.4,
            # "RL6_joint": 1.4,
            "L1_joint": 0.5,
            "L2_joint": 0.7,
            "L3_joint": 1.1,
            "L4_joint": 0.9,
            "L5_joint": 0.9,
            "L6_joint": 0.7,
        }
        action_scale = 0.04
        action_scale_per_joint = {
            "LL1_joint": 0.025,
            "RL1_joint": 0.025,
            "LL2_joint": 0.020,
            "RL2_joint": 0.020,
            "LL3_joint": 0.050,
            "RL3_joint": 0.050,
            "LL4_joint": 0.055,
            "RL4_joint": 0.055,
            "LL5_joint": 0.050,
            "RL5_joint": 0.050,
            "LL6_joint": 0.020,
            "RL6_joint": 0.020,
        }
        decimation = 2

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/jandi/jandi1.urdf"
        name = "jandi"
        disable_gravity = False
        foot_name = "6_link"
        collapse_fixed_joints = True
        fix_base_link = False
        penalize_contacts_on = ["base_link", "3_link", "4_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        soft_dof_pos_limit = 0.9
        base_height_target = 0.4
        tracking_sigma = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = 0.0
            stand_still = 0.0

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 2
        clip_observations = 100.0


class JandiRobotBaseCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.3
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = "elu"
        rnn_type = "lstm"
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.002

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ""
        experiment_name = "jandi"
