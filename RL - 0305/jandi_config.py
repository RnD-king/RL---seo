from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class JandiRobotRoughCfg( LeggedRobotCfg ):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.45]  # x,y,z [m] (fine-tuned for stable initial contact)
        default_joint_angles = {  # target angles [rad] when action = 0.0
            "RL_joint1": 0.0,   # dof10
            "RL_joint2": 0.0,   # dof8
            "RL_joint3": 0.5,   # dof6
            "RL_joint4": 0.2,   # dof4
            "RL_joint5": 0.72,   # dof2
            "RL_joint6": 0.0,   # dof0

            "LL_joint1": 0.0,   # dof11
            "LL_joint2": 0.0,   # dof9
            "LL_joint3": -0.5,   # dof7
            "LL_joint4": -0.2,   # dof5
            "LL_joint5": -0.72,   # dof3
            "LL_joint6": -0.0,   # dof1
        }

        # pos = [0.0, 0.0, 0.8] # x,y,z [m]
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #    'left_hip_yaw_joint' : 0. ,   
        #    'left_hip_roll_joint' : 0,               
        #    'left_hip_pitch_joint' : -0.1,         
        #    'left_knee_joint' : 0.3,       
        #    'left_ankle_pitch_joint' : -0.2,     
        #    'left_ankle_roll_joint' : 0,     
        #    'right_hip_yaw_joint' : 0., 
        #    'right_hip_roll_joint' : 0, 
        #    'right_hip_pitch_joint' : -0.1,                                       
        #    'right_knee_joint' : 0.3,                                             
        #    'right_ankle_pitch_joint': -0.2,                              
        #    'right_ankle_roll_joint' : 0,       
        #    'torso_joint' : 0.
        # }
    class env(LeggedRobotCfg.env):
        num_observations = 47 # base observation layout
        num_privileged_obs = 50
        num_actions = 12 # 정책이 출력하는 액션 차원


    # input 설정
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0

        #input 명령 차원 
        num_commands = 3          # [lin_vel_x, lin_vel_y, ang_vel_yaw]
        resampling_time = 5.0    # 몇 초마다 command 바꿀지 ex) lin_vel_x 을 범위 안의 값 중 하나를 선택
        heading_command = False   # true면 명령이 목표 heading(방향) 기준으로 해석, 내부에서 현재 yaw와의 오차를 계산하여 그에 맞춰 회전 명령 계산

        class ranges(LeggedRobotCfg.commands.ranges): # legged_robot_config에 있는 LeggedRobotCfg.commands.ranges의 모든 항목 유지, 아래에 적은것만 g1에게만 덮어쓰기
            # lin_vel_x = [0.0, 0.0]     # 초반 안정화를 위한 정지 명령
            # lin_vel_y = [0.0, 0.0]     # 옆 이동 금지
            # ang_vel_yaw = [0.0, 0.0]   # 직진 학습 단계
            lin_vel_x = [0.0, 2.0]     # 초반 안정화를 위한 정지 명령
            lin_vel_y = [0.0, 0.0]     # 옆 이동 금지
            ang_vel_yaw = [0.0, 0.0]   # 직진 학습 단계



    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.1, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 3.]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control(LeggedRobotCfg.control):
        control_type = "P"
        # stiffness : 목표 각도로 당기는 힘
        # stiffness 올리면
        # 장점: 목표 자세를 더 강하게 유지, 처짐 감소
        # 단점: 튐/진동 증가, 접촉 시 충격 커짐, 불안정해질 수 있음
        # stiffness 내리면
        # 장점: 부드러워짐, 충격 감소
        # 단점: 자세 유지 약해짐, 축 처짐/느린 응답
        stiffness = {
            "joint1": 20.0,
            "joint2": 20.0,
            "joint3": 20.0,
            "joint4": 15.0,
            "joint5": 10.0,
            "joint6": 10.0,
        }  # [N*m/rad]
        
        # damping : 움직임 속도를 브레이크 거는 힘
        # damping 올리면
        # 장점: 흔들림/오버슈트 감소, 바운스 억제
        # 단점: 둔해짐, 동작이 무거워지고 빠른 반응 저하
        # damping 내리면
        # 장점: 반응 빠름
        # 단점: 떨림/진동/바운스 증가
        # damping = {          
        #     "joint1": 8.0,
        #     "joint2": 8.0,
        #     "joint3": 8.0,
        #     "joint4": 13.0,
        #     "joint5": 7.0,
        #     "joint6": 6.5,
        # }  # [N*m*s/rad]
        damping = {
            "joint1": 0.5,
            "joint2": 0.5,
            "joint3": 0.5,
            "joint4": 0.5,
            "joint5": 0.3,
            "joint6": 0.3,
        }  # [N*m*s/rad]

        # stiffness = {'hip_yaw': 100,
        #              'hip_roll': 100,
        #              'hip_pitch': 100,
        #              'knee': 150,
        #              'ankle': 40,
        #              }  # [N*m/rad]
        # damping = {  'hip_yaw': 2,
        #              'hip_roll': 2,
        #              'hip_pitch': 2,
        #              'knee': 4,
        #              'ankle': 2,
        #              }  # [N*m/rad]  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 0.15
        action_scale = 0.18
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/my_robot/sample_assembly_URDF1.urdf"
        name = "jandi"
        disable_gravity = False
        foot_name = "link6"
        collapse_fixed_joints = True
        # penalize_contacts_on = ["base_link","link1", "link2", "link3", "link4", "link5"]
        penalize_contacts_on = ["base_link","link1"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0
        flip_visual_attachments = False # false로 할것

    # class asset( LeggedRobotCfg.asset ):
    #     file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
    #     # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/my_robot/sample_assembly_URDF1.urdf'
    #     name = "jandi"
    #     foot_name = "ankle_roll"
    #     penalize_contacts_on = ["hip", "knee"]
    #     # penalize_contacts_on = ["link1", "link2", "link3", "link4", "link5"]
    #     terminate_after_contacts_on = ["pelvis"]
    #     # terminate_after_contacts_on = []
    #     self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    #     flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.38
        only_positive_rewards = True
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 2.5
            tracking_ang_vel = 0.3
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -5.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.003
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -10.0
            contact = 0.18
            torques = -1.0e-5
            sym_dof_pos = -0.005
            sym_dof_vel = -0.001
            forward_progress = 1.5


    class normalization(LeggedRobotCfg.normalization):
        # clip_actions = 1.0
        clip_actions = 100.0
        clip_observations = 100.0

# class JandiRobotRoughCfgPPO( LeggedRobotCfgPPO ):
#     class policy:
#         init_noise_std = 0.05
#         actor_hidden_dims = [256, 128]
#         critic_hidden_dims = [256, 128]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         learning_rate = 3.e-4
#         entropy_coef = 0.001
#         max_grad_norm = 0.5
#     class runner( LeggedRobotCfgPPO.runner ):
#         policy_class_name = "ActorCritic"
#         max_iterations = 6000
#         run_name = ''
#         experiment_name = 'jandi'

  
class JandiRobotRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.3
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.003
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'jandi'