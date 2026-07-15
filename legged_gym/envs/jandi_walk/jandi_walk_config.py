from legged_gym.envs.jandi_base.jandi_base_config import (
    JandiRobotBaseCfg,
    JandiRobotBaseCfgPPO,
)


class JandiRobotWalkCfg(JandiRobotBaseCfg):
    class env(JandiRobotBaseCfg.env):
        num_observations = 50
        num_privileged_obs = 50
        num_actions = 12

    class commands(JandiRobotBaseCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 3
        resampling_time = 3.0
        heading_command = False

        class ranges(JandiRobotBaseCfg.commands.ranges):
            lin_vel_x = [0.02, 0.10]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.0, 0.0]

    class asset(JandiRobotBaseCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/jandi/jandi.urdf"
        name = "jandi"
        disable_gravity = False
        foot_name = "6_link"
        collapse_fixed_joints = True
        fix_base_link = False
        penalize_contacts_on = ["base_link", "3_link", "4_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0
        flip_visual_attachments = False


    class rewards(JandiRobotBaseCfg.rewards):
        tracking_sigma = 0.01
        heading_sigma = 0.25
        only_positive_rewards = True
        target_feet_lateral_dist = 0.18
        min_feet_lateral_dist = 0.16
        max_feet_lateral_dist = 0.19
        min_swing_forward = 0.005
        target_swing_forward = 0.035

        contact_force_threshold = 5.0
        contact_mismatch_penalty = -1.0
        target_feet_height = 0.05
        feet_clearance_sigma = 0.000225
        # Air-time is only paid on touchdown.  Start with a reachable duration;
        # the 0.18 s threshold stayed effectively inactive for 6k iterations.
        target_air_time = 0.12
        max_air_time_bonus = 0.20
        default_joint_pos_deadzone = 0.04
        default_joint_pos_norm = 0.30

        # Stop an unrecoverable fall before it dominates the rollout.  The
        # terminal cost below is applied after positive-reward clipping.
        termination_pitch = 0.65
        termination_roll = 0.70
        pitch_deadband = 0.08

        hip_roll_safe = 0.10
        hip_roll_limit = 0.15
        hip_roll_cost_cap = 2.0
        hip_roll_mirror_deadzone = 0.010
        hip_roll_mirror_norm = 0.20
        hip_roll_mirror_cost_cap = 1.0

        class scales(JandiRobotBaseCfg.rewards.scales):
            # Main command tracking
            tracking_lin_x_vel = 0.30
            tracking_lin_vel = 0.0
            tracking_lin_y_vel = 0.0
            tracking_ang_vel = 0.30
            heading = 0.0

            # Base stability
            lin_vel_y = -2.0
            lin_vel_z = -1.5
            ang_vel_xy = -0.10
            yaw_rate = -0.15
            orientation = -2.5
            pitch = -2.5
            base_height = -6.0
            base_roll = 0.0

            # Humanoid-gym style gait terms, adapted to Jandi's right-left phase. 
            feet_contact_number = 0.7
            feet_clearance = 0.6 # 발을 적당한 높이까지 들어올리는 reward
            foot_slip = -0.05
            feet_air_time = 0.40
            feet_distance = 0.3
            # feet_center_y = -0.02
            swing_contact = -0.4
            swing_clearance_hint = 0.15 # 발 높이가 맞으면 보너스
            swing_foot_forward = 0.3
            # swing_foot_forward_lane = 0.0

            default_joint_pos = -0.3
            # sym_roll_dof_pos = -0.5 # hip, ankle roll 대칭

            # Smoothness / energy
            action_rate = -0.006
            action_smoothness = -0.001

            dof_vel = -7.0e-4
            dof_acc = -1.0e-7
            torques = -8.0e-5
            dof_pos_limits = -0.5

            # Safety
            alive = 0.05
            collision = -0.30
            termination = -10.0

            # 2nd-stage shaping terms. Keep off until the baseline walks.
            contact = 0.0
            # Dense toe-off/height reward; unlike air-time it provides a signal
            # before the policy can already sustain a long swing.
            feet_swing_height = 0.30
            contact_no_vel = 0.0
            swing_phase_fail = 0.0
            stance_no_contact = -0.50
            low_speed = 0.0
            stall = 0.0
            torque_saturation = 0.0
            hip_pos = 0.0
            hip_yaw_fix = 0.0
            hip_roll_pos = 0.0
            hip_roll_mirror = 0.0
            roll_joint_pos = 0.0
            roll_joint_vel = 0.0
            roll = 0.0




class JandiRobotWalkCfgPPO(JandiRobotBaseCfgPPO):
    class policy:
        init_noise_std = 0.2
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = "elu"
        rnn_type = "lstm"
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(JandiRobotBaseCfgPPO.algorithm):
        # The failed run's std grew from 0.32 to 1.68 after the first policy
        # collapse.  Use gentler updates for the stabilization fine-tune.
        entropy_coef = 0.0005
        learning_rate = 3.0e-4

    class runner(JandiRobotBaseCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        # Resume the good policy weights, but start a fresh Adam state so the
        # stabilization learning rate is not overwritten by the old checkpoint.
        load_optimizer = False
        run_name = ""
        experiment_name = "jandi"
