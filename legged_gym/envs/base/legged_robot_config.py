from .base_config import BaseConfig

#공통 기본값 제공
class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 48
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        test = False

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}
        reset_joint_noise = {
            "joint_a": 0.0,
            "joint_b": 0.0,
        }

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # Optional per-joint action scales. Keys can be full names (e.g. "LL3_joint")
        # or partial tokens (e.g. "3_joint") that match dof names.
        # When empty, global action_scale is used for every joint.
        action_scale_per_joint = {}
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True  # 마찰 범위 랜덤화
        friction_range = [0.5, 1.25]
        randomize_base_mass = False # 질량/관성 랜덤화 : 실제 로봇과 urdf 로봇의 무게가 다를 가능성이 크기때문에 urdf 값에 랜덤화한 값을 오프셋으로 줌
        added_mass_range = [-1., 1.]
        push_robots = True # 외란 랜덤화
        push_interval_s = 15
        max_push_vel_xy = 1.

    # RL 학습에서 쓰는 보상 함수 설정
    class rewards:
        # 각 보상 항목 가중치
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0 # 선속도 명령 추종 보상
            tracking_ang_vel = 0.5 # yaw 각속도 명령 추종 보상
            lin_vel_z = -2.0 # z축 속도 페널티
            ang_vel_xy = -0.05 # roll, pitch 각속도 페널티
            orientation = -0. # 기울어짐 페널티
            torques = -0.00001 # 토크 크기 페널티
            dof_vel = -0. # 관절 속도 페널티
            dof_acc = -2.5e-7 # 관절 가속도 페널티
            base_height = -0.  # 목표 높이와 베이스 높이 오차 페널티
            feet_air_time =  1.0 # 발 공중 체공시간 관련 보상
            collision = -1. # 지정된 바디 충돌 페널티
            feet_stumble = -0.0 # 발이 수직 장애물에 걸리는 현상 페널티
            action_rate = -0.01 # 연속 스텝 간 액션 변화량 페널티 
            stand_still = -0. # 정지 명령일 때 불필요한 움직임 페널티

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        # tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma) 추종 보상 민감도 
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma) 추종 보상 민감도 
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized 
        soft_dof_vel_limit = 1. # 관절 속도 한계 근처 페널티 비율
        soft_torque_limit = 1. # 토크 한계 근처 페널티 비율
        base_height_target = 1. # 목표 베이스 높이
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        # 관측 각 항목을 정책에 넣기전에 곱하는 스케일
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        
        # cliping : 범위 밖의 값들을 경계값으로 바꾸는 것
        # cliping 하는 이유 : 튄 값이 학습 망가뜨리는걸 막음
        clip_observations = 100. # 범위 [-100, 100]
        clip_actions = 100.

    # 관측 항목별 노이즈 크기, 실시간 노이즈가 아닌 인위적인 노이즈
    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    # 시뮬레이터 카메라 설정
    class viewer:
        ref_env = 0 # 어떤 env를 기준으로 볼지
        pos = [10, 0, 6]  # [m] 카메라 위치
        lookat = [11., 5, 3.]  # [m] 카메라가 보는 점

    # 물리 시뮬레이션 전역 설정
    class sim: 
        dt =  0.005 # 물리 timestep
        substeps = 1 # 한 스텝당 내무 물리 서브 스텝 수
        gravity = [0., 0. ,-9.81]  # [m/s^2] # 중력
        up_axis = 1  # 0 is y, 1 is z , z축이 위축

        #physx 상세 파라미터
        class physx:
            num_threads = 10 # cpu 스레드 수
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4 # 접촉/관절 위치 제약 반복 횟수
            num_velocity_iterations = 0 # 속도 제약 반복 횟수
            contact_offset = 0.01  # [m] 접촉으로 판단 시작 거리
            rest_offset = 0.0   # [m] 휴지 상태 거리
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


# PPO 학습 파라미터 묶음
class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'  # 학습 실행기 클래스
    class policy:
        init_noise_std = 1.0 # 초기 정책 출력 노이즈 크기
        actor_hidden_dims = [512, 256, 128] # actor MLP 크기
        critic_hidden_dims = [512, 256, 128] # critic MLP 크기
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid 활성화 함수
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    
    class algorithm:
        # training params
        value_loss_coef = 1.0 # value loss 가중치
        use_clipped_value_loss = True # value clippping 사용 여부
        clip_param = 0.2 # PPO ratio clipping 범위
        entropy_coef = 0.01  # 엔트로피 보너스 가중치(탐색)
        num_learning_epochs = 5 # 수집 데이터 반복 학습 횟수
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4 학습률
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1. # gradient clipping 한계

    class runner:
        policy_class_name = 'ActorCritic' # 정책 클래스(MLP/RNN 등)
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations, checkpoint 저장 주기
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
