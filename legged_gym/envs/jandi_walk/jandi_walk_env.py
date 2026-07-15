from legged_gym.envs.jandi_base.jandi_base_env import JandiRobotBase

from isaacgym.torch_utils import *
import numpy as np
import torch

class JandiWalkEnv(JandiRobotBase):
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions
        noise_vec[12+3*self.num_actions:12+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        stance_ratio = 0.55
        self.phase = (self.episode_length_buf * self.dt) % period / period
        # Keep legacy left/right phase for existing rewards.
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        # Explicit right->left phase used by step-order reward.
        self.phase_right_first = self.phase
        self.phase_left_second = (self.phase + offset) % 1
        self.leg_phase_right_left = torch.cat(
            [self.phase_right_first.unsqueeze(1), self.phase_left_second.unsqueeze(1)],
            dim=-1,
        )
        self.stance_ratio = stance_ratio
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        # def _chk(name, x):
        #     if not torch.isfinite(x).all():
        #         print(f"[NAN] {name}: nan={torch.isnan(x).sum().item()}, inf={torch.isinf(x).sum().item()}, shape={tuple(x.shape)}", flush=True)

        # # g1_env.py 안에 실제로 존재하는 변수들만 체크하세요.
        # # 아래는 보통 legged-gym 계열에서 쓰는 대표 항목들입니다.
        # # (에러 나면 그 변수는 g1_env.py에 없는 것이니 지우면 됩니다.)
        # _chk("base_lin_vel", self.base_lin_vel)
        # _chk("base_ang_vel", self.base_ang_vel)
        # _chk("projected_gravity", self.projected_gravity)
        # _chk("commands", self.commands)
        # _chk("dof_pos", self.dof_pos)
        # _chk("default_dof_pos", self.default_dof_pos)
        # _chk("dof_vel", self.dof_vel)
        # _chk("actions", self.actions)

        #clock input 부분 : 보행은 주기적인 행동 패턴이 필요 -> clock input이 현재 보행 주기 패턴 중 어디에 해당하는지 간단하게 알수있게 해줘 학습이 쉬워짐, 없으면 관측 상태만으로 상태 추론해야함
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1) 
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        
    
        # _chk("obs_buf", self.obs_buf)

        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    # reward 함수들

    def _reward_feet_contact_number(self):
        if self.feet_num < 2:
            return self._empty_reward()

        stance_mask = self._get_stance_mask()
        contact = self._get_ordered_foot_contact()
        contact_match = contact == stance_mask
        mismatch_penalty = getattr(self.cfg.rewards, "contact_mismatch_penalty", -1.0)

        per_foot_reward = torch.where(
            contact_match,
            torch.ones_like(contact, dtype=torch.float),
            mismatch_penalty * torch.ones_like(contact, dtype=torch.float),
        )
        return torch.mean(per_foot_reward, dim=1)

    def _reward_feet_clearance(self):
        if self.feet_num < 2:
            return self._empty_reward()

        swing_mask = self._get_swing_mask()
        contact = self._get_ordered_foot_contact()
        foot_height = self._get_ordered_foot_height()

        target_height = getattr(self.cfg.rewards, "target_feet_height", 0.02)
        sigma = getattr(self.cfg.rewards, "feet_clearance_sigma", 0.0001)

        clearance_error = torch.square(foot_height - target_height)
        clearance_score = torch.exp(-clearance_error / sigma)
        valid_swing = swing_mask & (~contact)

        return torch.sum(clearance_score * valid_swing.float(), dim=1)

    def _reward_foot_slip(self):
        if self.feet_num < 2:
            return self._empty_reward()

        contact = self._get_ordered_foot_contact()
        foot_xy_vel = self._get_ordered_foot_xy_vel()
        slip_speed = torch.norm(foot_xy_vel, dim=2)
        return torch.sum(slip_speed * contact.float(), dim=1)

    def _reward_action_smoothness(self):
        action_diff = self.actions - self.last_actions
        return torch.sum(torch.square(action_diff), dim=1)

    # 스탠스/스윙 위상과 실제 접촉 일치도 보상 (양발 모두 맞아야 보상)
    def _reward_contact(self):
        if self.feet_num < 2:
            return self._empty_reward()

        match = self._get_ordered_foot_contact() == self._get_stance_mask()
        return torch.prod(match.float(), dim=1)
    
    # 관절의 과도한 움직임 억제
    def _reward_action_smoothness_1(self):
        current_target = self.default_dof_pos + self.actions * self.action_scale_vec.unsqueeze(0)
        last_target = self.default_dof_pos + self.last_actions * self.action_scale_vec.unsqueeze(0)

        return torch.sum(torch.square(current_target - last_target), dim=1)

    def _reward_heading(self):
        yaw = self.rpy[:, 2]

        # 직진 목표 heading
        target_yaw = torch.zeros_like(yaw)

        yaw_error = torch.atan2(
            torch.sin(yaw - target_yaw),
            torch.cos(yaw - target_yaw)
        )

        return torch.exp(-torch.square(yaw_error) / self.cfg.rewards.heading_sigma)


    def _reward_hip_yaw_fix(self):
        hip_yaw_names = ["LL1_joint", "RL1_joint"]
        indices = []

        for name in hip_yaw_names:
            if name in self.dof_names:
                indices.append(self.dof_names.index(name))

        if len(indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        q = self.dof_pos[:, indices]
        q_default = self.default_dof_pos[:, indices]

        err = torch.abs(q - q_default)

        deadzone = 0.05
        violation = torch.clamp(err - deadzone, min=0.0)

        return torch.sum(torch.square(violation), dim=1)
    
    def _reward_low_speed(self):
        """
        너무 느리게 가거나, 명령과 반대 방향으로 움직이는 것을 막는 reward.

        목적:
        - 명령 방향과 반대로 움직이면 큰 penalty
        - 목표 속도의 절반보다 느리면 penalty
        - 목표 속도 범위 안이면 reward
        - 너무 빠르면 reward 없음
        """

        vx = self.base_lin_vel[:, 0]
        cmd_x = self.commands[:, 0]

        abs_vx = torch.abs(vx)
        abs_cmd = torch.abs(cmd_x)

        active_cmd = abs_cmd > 0.02

        # 명령 방향과 실제 이동 방향이 반대인지 확인
        mismatch = (vx * cmd_x) < 0.0

        too_slow = abs_vx < 0.5 * abs_cmd
        too_fast = abs_vx > 1.2 * abs_cmd

        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        reward = torch.where(
            mismatch & active_cmd,
            -2.0 * torch.ones_like(reward),
            reward,
        )

        reward = torch.where(
            (~mismatch) & too_slow & active_cmd,
            -1.0 * torch.ones_like(reward),
            reward,
        )

        reward = torch.where(
            (~mismatch) & (~too_slow) & (~too_fast) & active_cmd,
            1.0 * torch.ones_like(reward),
            reward,
        )

        # 너무 빠른 경우는 0점
        return reward
    
    def _reward_tracking_lin_x_vel(self):
        vx_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-vx_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_y_vel(self):
        vy_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        # print("lin_error",lin_vel_error)
        return torch.exp(-vy_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_feet_width_constraint(self):
        """
        양발 y방향 간격이 목표 범위 안에 들어오도록 하는 constraint cost.

        목적:
        - 발이 너무 모이는 것 방지
        - 발이 너무 넓게 벌어져 림보 자세가 되는 것 방지
        """

        if self.feet_num < 2:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        right_idx = int(self.feet_order_right_left[0].item())
        left_idx = int(self.feet_order_right_left[1].item())

        right_pos_world = self.feet_pos[:, right_idx, :]
        left_pos_world = self.feet_pos[:, left_idx, :]
        base_pos_world = self.root_states[:, 0:3]

        right_rel = right_pos_world - base_pos_world
        left_rel = left_pos_world - base_pos_world

        yaw = self.rpy[:, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        right_y = -sin_yaw * right_rel[:, 0] + cos_yaw * right_rel[:, 1]
        left_y = -sin_yaw * left_rel[:, 0] + cos_yaw * left_rel[:, 1]

        width = torch.abs(right_y - left_y)

        # 약간 넓은 안정 보행 기준
        min_width = 0.13
        max_width = 0.17

        norm_width = 0.03

        too_narrow = torch.clamp(min_width - width, min=0.0) / norm_width
        too_wide = torch.clamp(width - max_width, min=0.0) / norm_width

        return torch.square(too_narrow) + torch.square(too_wide)
    
    def _reward_feet_center_y(self):
        """ 
        양발의 y방향 중심이 base 중심선 근처에 오도록 하는 constraint cost.

        목적:
        - 발 간격만 맞추고 양발 전체가 한쪽으로 치우치는 현상 억제
        - hip roll 자체를 직접 억제하지 않고, 결과적인 발 위치 대칭성을 유도
        """

        if self.feet_num < 2:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        right_idx = int(self.feet_order_right_left[0].item())
        left_idx = int(self.feet_order_right_left[1].item())

        right_pos_world = self.feet_pos[:, right_idx, :]
        left_pos_world = self.feet_pos[:, left_idx, :]
        base_pos_world = self.root_states[:, 0:3]

        right_rel = right_pos_world - base_pos_world
        left_rel = left_pos_world - base_pos_world

        yaw = self.rpy[:, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        right_y = -sin_yaw * right_rel[:, 0] + cos_yaw * right_rel[:, 1]
        left_y = -sin_yaw * left_rel[:, 0] + cos_yaw * left_rel[:, 1]

        foot_center_y = 0.5 * (right_y + left_y)

        # 양발 중심이 몸통 중심에서 2cm 벗어나면 cost 1
        norm_center = 0.02

        return torch.square(foot_center_y / norm_center)

    def _reward_swing_phase_fail(self):
        """
        각 발이 swing phase인데도 계속 접촉하고 있으면 penalty.
        정상적인 single support 자체는 벌점 주지 않음.
        """

        if self.feet_num < 2:
            return self._empty_reward()

        ordered_contact = self._get_ordered_foot_contact()

        # swing phase 초반은 toe-off 지연이 있을 수 있으므로 조금 봐줌
        swing_phase_margin = 0.08
        should_swing = self.leg_phase_right_left >= (self.stance_ratio + swing_phase_margin)

        # swing해야 하는데 아직 접촉 중이면 실패
        swing_fail = should_swing & ordered_contact

        # 양발 평균 penalty
        return torch.mean(swing_fail.float(), dim=1)
    
    # 로봇이 살아있으면 +1 고정 보상
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    # 접촉 중 발 속도 크면 페널티(미끄럼 억제)
    def _reward_contact_no_vel(self):
        contact = self._get_ordered_foot_contact()
        foot_xy_vel = self._get_ordered_foot_xy_vel()
        return torch.sum(torch.square(foot_xy_vel) * contact.unsqueeze(-1).float(), dim=(1, 2))

    # 스윙 위상인데도 발이 땅에 닿아 있으면 패널티(발 끌기 억제)
    def _reward_swing_contact(self):
        if self.feet_num < 2:
            return self._empty_reward()

        ordered_contact = self._get_ordered_foot_contact()
        is_swing = self._get_swing_mask()
        penalty = ordered_contact * is_swing
        return torch.sum(penalty.float(), dim=1)
    
    def _reward_torque_saturation(self):
        torque_abs = torch.abs(self.torques)
        torque_limit = self.torque_limits.unsqueeze(0)

        threshold = 0.85 * torque_limit
        excess = torch.clamp(torque_abs - threshold, min=0.0)
        normalized_excess = excess / torque_limit

        return torch.sum(torch.square(normalized_excess), dim=1)
    
    def _reward_base_roll(self):
        roll = self.rpy[:, 0]

        deadband = 0.06  # 약 3.4도까지는 허용
        excess = torch.clamp(torch.abs(roll) - deadband, min=0.0)

        return torch.square(excess)
    
    def _reward_swing_over_height(self):
        if self.feet_num < 2:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        right_idx = int(self.feet_order_right_left[0].item())
        left_idx = int(self.feet_order_right_left[1].item())

        ordered_height = torch.stack(
            [self.feet_pos[:, right_idx, 2], self.feet_pos[:, left_idx, 2]],
            dim=1,
        )

        is_swing = self._get_swing_mask()

        max_height = 0.055  # 5.5cm 이상 들면 벌점
        over_height = torch.clamp(ordered_height - max_height, min=0.0)

        return torch.sum(torch.square(over_height) * is_swing.float(), dim=1)
        
    def _reward_feet_distance(self):
        """
        양발의 좌우 간격을 일정 범위 안에 유지하는 reward.

        기존 humanoid-gym의 feet_distance reward와 같은 방식:
        - min_dist보다 좁으면 reward 감소
        - max_dist보다 넓으면 reward 감소
        - 범위 안이면 reward가 최대
        """

        if self.feet_num < 2:
            return self._empty_reward()

        right_idx = int(self.feet_order_right_left[0].item())
        left_idx = int(self.feet_order_right_left[1].item())

        right_pos_world = self.feet_pos[:, right_idx, :]
        left_pos_world = self.feet_pos[:, left_idx, :]
        base_pos_world = self.root_states[:, 0:3]

        right_rel = right_pos_world - base_pos_world
        left_rel = left_pos_world - base_pos_world

        # base yaw 기준 world -> local 변환
        yaw = self.rpy[:, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        right_y = -sin_yaw * right_rel[:, 0] + cos_yaw * right_rel[:, 1]
        left_y = -sin_yaw * left_rel[:, 0] + cos_yaw * left_rel[:, 1]

        # 양발의 좌우 간격
        lateral_dist = torch.abs(right_y - left_y)

        fd = getattr(self.cfg.rewards, "min_feet_lateral_dist", 0.16)
        max_df = getattr(self.cfg.rewards, "max_feet_lateral_dist", 0.19)

        d_min = torch.clamp(lateral_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(lateral_dist - max_df, 0.0, 0.5)

        return (
            torch.exp(-torch.abs(d_min) * 100.0)
            + torch.exp(-torch.abs(d_max) * 100.0)
        ) / 2.0
    
    def _reward_swing_foot_lateral_position(self):
        """
        스윙 발이 앞으로 나갈 때 안쪽/대각선으로 말려 들어가는 것을 막는 reward.

        목적:
        - 오른발은 base 기준 +y 쪽 유지
        - 왼발은 base 기준 -y 쪽 유지
        - 스윙 중 발이 몸 중앙으로 들어오며 옆걸음이 되는 현상 억제
        """

        if self.feet_num < 2:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        right_idx = int(self.feet_order_right_left[0].item())
        left_idx = int(self.feet_order_right_left[1].item())

        right_pos_world = self.feet_pos[:, right_idx, :]
        left_pos_world = self.feet_pos[:, left_idx, :]
        base_pos_world = self.root_states[:, 0:3]

        right_rel_world = right_pos_world - base_pos_world
        left_rel_world = left_pos_world - base_pos_world

        yaw = self.rpy[:, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        right_y_local = -sin_yaw * right_rel_world[:, 0] + cos_yaw * right_rel_world[:, 1]
        left_y_local = -sin_yaw * left_rel_world[:, 0] + cos_yaw * left_rel_world[:, 1]

        # 기본 자세에서 확인한 발 위치
        # right_y_local ≈ +0.069
        # left_y_local  ≈ -0.076
        right_target_y = 0.070
        left_target_y = -0.075

        # 약간의 허용 오차
        tolerance = 0.015

        right_error = torch.clamp(torch.abs(right_y_local - right_target_y) - tolerance, min=0.0)
        left_error = torch.clamp(torch.abs(left_y_local - left_target_y) - tolerance, min=0.0)

        # 접촉 힘 기준으로 swing foot만 선택
        right_fz = self.contact_forces[:, self.feet_indices[right_idx], 2]
        left_fz = self.contact_forces[:, self.feet_indices[left_idx], 2]

        contact_threshold = 5.0

        right_is_swing = right_fz < contact_threshold
        left_is_swing = left_fz < contact_threshold

        penalty = right_error * right_is_swing.float() + left_error * left_is_swing.float()

        return penalty

    def _reward_forward_direction(self):
        vx = self.base_lin_vel[:, 0]
        vy = self.base_lin_vel[:, 1]

        speed = torch.sqrt(vx * vx + vy * vy + 1e-6)
        forward_ratio = vx / speed

        return torch.clamp(forward_ratio, min=0.0, max=1.0)

    # 오른발 -> 왼발 순서(교대 보행) 접촉 위상 일치 보상
    def _reward_step_order(self):   
        if self.feet_num < 2:
            return self._empty_reward()

        match = self._get_ordered_foot_contact() == self._get_stance_mask()
        return torch.sum(match.float(), dim=1)
    
    # 2번, 6번 조인트가 빠르게 흔들리는 걸 막는 reward
    def _reward_roll_joint_vel(self):
        roll_joint_names = [
            "LL2_joint", "RL2_joint",
            "LL6_joint", "RL6_joint",
        ]

        penalty = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        for name in roll_joint_names:
            if name in self.dof_names:
                idx = self.dof_names.index(name)
                penalty += torch.square(self.dof_vel[:, idx])

        return penalty
    
    # roll 조인트가 너무 많이 꺾일 때만 벌점
    def _reward_roll_joint_pos(self):
        roll_joint_names = ["LL2_joint", "RL2_joint", "LL6_joint", "RL6_joint"]
        indices = []

        for name in roll_joint_names:
            if name in self.dof_names:
                indices.append(self.dof_names.index(name))

        if len(indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        roll_pos = self.dof_pos[:, indices]
        return torch.sum(torch.square(roll_pos), dim=1)
        
        # 전진 관련 보상
    def _reward_forward_progress(self):
        # x축 전진 속도만 보상 (후진은 0)
        return torch.clamp(self.base_lin_vel[:, 0], min=0.0)


    def _reward_stall(self):
        moving_cmd = self.commands[:, 0] > 0.02 # lin_vel_x 명령이 0.04보다 크게 들어왔나
        too_slow = torch.abs(self.base_lin_vel[:, 0]) < 0.015 #실제 몸체 속도가 0.03보다 작으면 too slow
        return (moving_cmd & too_slow).float()

    # 스윙 중 발 높이가 목표와 다르면 오차 페널티, 오른발/왼발 swing 보상 분리
    def _reward_feet_swing_height(self):
        if self.feet_num < 2:
            return self._empty_reward()

        ordered_height = self._get_ordered_foot_height()
        ordered_contact = self._get_ordered_foot_contact()
        is_swing = self._get_swing_mask()
        min_height = 0.005
        target_height = getattr(self.cfg.rewards, "target_feet_height", 0.02)

        height_reward = torch.clamp(
            (ordered_height - min_height) / (target_height - min_height),
            min=0.0,
            max=1.0,
        )

        valid_swing = is_swing & (~ordered_contact)

        return torch.sum(height_reward * valid_swing.float(), dim=1)
    
    def _quat_to_yaw(self, quat):
        """
        quat: [num_envs, 4]
        Isaac Gym quaternion 순서가 xyzw인지 wxyz인지 반드시 확인 필요.
        legged_gym에서는 보통 xyzw를 씁니다.
        """

        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]
        w = quat[:, 3]

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return yaw


    def _reward_yaw_rate(self):
        return torch.square(self.base_ang_vel[:, 2])    
    
    # def _reward_lin_vel_y(self):
    #     return torch.square(self.base_lin_vel[:, 1])

    def _reward_lin_vel_y(self):
        """
        body frame lateral velocity constraint cost.

        목적:
        - commands[:, 1] == 0인 직진 학습에서 게걸음 억제
        - 작은 좌우 흔들림은 허용
        - 일정 lateral velocity 이상은 constraint violation처럼 강하게 penalty

        단위:
        - self.base_lin_vel[:, 1]: m/s
        - return: (m/s)^2 형태의 cost
        """

        vy = self.base_lin_vel[:, 1]

        # y 방향 명령이 거의 없을 때만 적용
        # 현재 Jandi config에서는 commands[:, 1]이 항상 0이므로 항상 적용됨
        no_side_cmd = torch.abs(self.commands[:, 1]) < 0.02

        # 허용 lateral velocity
        soft_limit = 0.04   # m/s, 작은 좌우 흔들림 허용
        hard_limit = 0.12   # m/s, 이 이상은 게걸음으로 보고 강하게 제한

        soft_violation = torch.clamp(torch.abs(vy) - soft_limit, min=0.0)
        hard_violation = torch.clamp(torch.abs(vy) - hard_limit, min=0.0)

        lateral_cost = (
            torch.square(soft_violation)
            + 5.0 * torch.square(hard_violation)
        )

        return torch.where(
            no_side_cmd,
            lateral_cost,
            torch.zeros_like(lateral_cost),
        )


    def _reward_swing_clearance_hint(self):
        if self.feet_num < 2:
            return self._empty_reward()

        ordered_height = self._get_ordered_foot_height()
        is_swing = self._get_swing_mask()
        min_height = 0.003
        target_height = getattr(self.cfg.rewards, "target_feet_height", 0.02)

        height_score = torch.clamp(
            (ordered_height - min_height) / (target_height - min_height),
            min=0.0,
            max=1.0,
        )

        return torch.sum(height_score * is_swing.float(), dim=1)
    
    def _reward_feet_air_time(self):
        """
        발이 일정 시간 이상 공중에 떠 있다가 착지하면 보상.

        목적:
        - 양발을 질질 끄는 보행 방지
        - 발을 번갈아 드는 step 유도
        - 너무 짧은 발 들림은 보상하지 않음
        """
        if self.feet_num < 2:
            return self._empty_reward()

        contact = self._get_ordered_foot_contact()

        # PhysX contact noise 완화.  Do not OR this with the desired stance
        # mask: a desired stance is not an actual touchdown.  Doing so grants
        # fake air-time rewards and resets the timer while the foot is airborne.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        # 발이 공중에 있다가 처음 다시 닿은 순간
        first_contact = (self.feet_air_time > 0.0) & contact_filt

        # air time 누적
        self.feet_air_time += self.dt

        # 최소 공중 시간 기준
        target_air_time = getattr(self.cfg.rewards, "target_air_time", 0.18)

        # 너무 오래 드는 꼼수 방지용 cap
        max_air_bonus = getattr(self.cfg.rewards, "max_air_time_bonus", 0.25)

        air_time_bonus = torch.clamp(
            self.feet_air_time - target_air_time,
            min=0.0,
            max=max_air_bonus,
        )

        rew_air_time = torch.sum(
            air_time_bonus * first_contact.float(),
            dim=1,
        )

        # 전진 명령이 있을 때만 보상
        moving_cmd = torch.abs(self.commands[:, 0]) > 0.03
        rew_air_time *= moving_cmd.float()

        # 접촉 중인 발은 air time reset
        self.feet_air_time *= ~contact_filt

        return rew_air_time

    
    # 발을 안 내리면 벌점, 들고 조금 지나도 안 내리면 벌점
    def _reward_stance_no_contact(self):
        if self.feet_num < 2:
            return self._empty_reward()

        ordered_contact = self._get_ordered_foot_contact()
        phase = self.leg_phase_right_left

        # stance 전체가 아니라, stance 중간 이후부터 접촉 요구
        touchdown_required = (phase > 0.10) & (phase < self.stance_ratio)

        no_contact_during_required_stance = touchdown_required & (~ordered_contact)

        return torch.sum(no_contact_during_required_stance.float(), dim=1)
    

    def _reward_hip_roll_pos(self):
        """
        LL2/RL2 hip_roll 관절이 limit 근처에 붙는 것을 막기 위한 angle-only soft cost.

        목적:
        - 관절 position을 hard clamp하지 않는다.
        - default 자세 주변의 작은 균형 보정은 허용한다.
        - |q - q_default|가 safe 각도보다 커질 때만 cost를 준다.
        - cost 폭발을 막기 위해 normalized value와 최종 cost에 cap을 둔다.

        config 기본값:
        - hip_roll_safe = 0.10 rad
        - hip_roll_limit = 0.15 rad
        - hip_roll_cost_cap = 2.0

        return:
        - shape: [num_envs]
        - 값이 클수록 나쁜 자세. config scale은 음수로 사용한다.
        """

        hip_roll_names = ["LL2_joint", "RL2_joint"]

        indices = []
        for name in hip_roll_names:
            if name in self.dof_names:
                indices.append(self.dof_names.index(name))

        if len(indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        q = self.dof_pos[:, indices]
        q_default = self.default_dof_pos[:, indices]

        err = torch.abs(q - q_default)

        safe = getattr(self.cfg.rewards, "hip_roll_safe", 0.10)
        limit = getattr(self.cfg.rewards, "hip_roll_limit", 0.15)
        cost_cap = getattr(self.cfg.rewards, "hip_roll_cost_cap", 2.0)

        # safe 안쪽은 cost 없음.
        # safe ~ limit 사이에서는 0~1로 증가.
        # limit 이상은 1로 cap.
        margin = max(limit - safe, 1e-6)
        excess = torch.clamp(err - safe, min=0.0)
        normalized_excess = torch.clamp(excess / margin, min=0.0, max=1.0)

        cost = torch.sum(torch.square(normalized_excess), dim=1)
        cost = torch.clamp(cost, max=cost_cap)

        return cost
        
    def _reward_sym_roll_dof_pos(self):
        """
        좌우 roll 계열 관절만 mirror symmetry로 묶는 penalty.

        목적:
        - 오른쪽 다리만 안쪽으로 꺾이는 비대칭 억제
        - sagittal 보행 관절(hip_pitch, knee, ankle_pitch)은 자유롭게 둠

        return은 cost이므로 scale은 음수로 사용.
        """

        if self.num_dof < 12:
            return self._empty_reward()

        # DOF index 기준:
        # left  : LL1~LL6 -> 0~5
        # right : RL1~RL6 -> 6~11
        ll_hip_roll = self.dof_pos[:, 1]   # LL2
        rl_hip_roll = self.dof_pos[:, 7]   # RL2

        ll_ankle_roll = self.dof_pos[:, 5] # LL6
        rl_ankle_roll = self.dof_pos[:, 11] # RL6

        hip_roll_err = ll_hip_roll + rl_hip_roll
        ankle_roll_err = ll_ankle_roll + rl_ankle_roll

        return (
            torch.square(hip_roll_err)
            + 0.5 * torch.square(ankle_roll_err)
        )
    
    def _reward_default_joint_pos(self):
        """
        Jandi DOF 순서 기준 roll 계열 default pose penalty.

        목적:
        - 시작하자마자 다리가 안쪽으로 말리는 것 방지
        - 발 간격이 좁아지는 자세 억제
        - hip/knee/ankle pitch는 보행에 필요하므로 묶지 않음

        return은 cost/penalty.
        따라서 scale은 음수로 사용.
        """

        joint_diff = self.dof_pos - self.default_dof_pos

        # 실제 DOF order:
        # 0 LL1_joint
        # 1 LL2_joint  <- hip roll
        # 2 LL3_joint
        # 3 LL4_joint
        # 4 LL5_joint
        # 5 LL6_joint  <- ankle roll
        # 6 RL1_joint
        # 7 RL2_joint  <- hip roll
        # 8 RL3_joint
        # 9 RL4_joint
        # 10 RL5_joint
        # 11 RL6_joint <- ankle roll

        roll_diff = torch.stack(
            [
                joint_diff[:, 1],   # LL2_joint
                joint_diff[:, 5],   # LL6_joint
                joint_diff[:, 7],   # RL2_joint
                joint_diff[:, 11],  # RL6_joint
            ],
            dim=1,
        )

        # 작은 흔들림은 허용
        deadband = getattr(self.cfg.rewards, "default_roll_deadband", 0.03)

        roll_err = torch.clamp(torch.abs(roll_diff) - deadband, min=0.0)

        # hip roll을 ankle roll보다 조금 더 강하게 봄
        weights = torch.tensor(
            [
                1.0,  # LL2 hip roll
                0.5,  # LL6 ankle roll
                1.0,  # RL2 hip roll
                0.5,  # RL6 ankle roll
            ],
            dtype=torch.float,
            device=self.device,
        )

        return torch.sum(weights * torch.square(roll_err), dim=1)
        
    
    def _reward_swing_foot_forward(self):
        """
        swing 발이 반대발보다 base local x 방향으로 앞으로 나가면 보상.

        조건:
        - phase상 swing 발이어야 함
        - 실제 contact가 없어야 함
        - 반대발은 contact 중이어야 함
        - 전진 명령이 있을 때만 보상
        """

        if self.feet_num < 2:
            return self._empty_reward()

        right_idx = int(self.feet_order_right_left[0].item())
        left_idx = int(self.feet_order_right_left[1].item())

        right_pos_world = self.feet_pos[:, right_idx, :]
        left_pos_world = self.feet_pos[:, left_idx, :]
        base_pos_world = self.root_states[:, 0:3]

        right_rel = right_pos_world - base_pos_world
        left_rel = left_pos_world - base_pos_world

        # base yaw 기준 local x 변환
        yaw = self.rpy[:, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        right_x = cos_yaw * right_rel[:, 0] + sin_yaw * right_rel[:, 1]
        left_x = cos_yaw * left_rel[:, 0] + sin_yaw * left_rel[:, 1]

        # phase 기준 swing/stance
        is_swing = self._get_swing_mask()

        # 실제 접촉 상태
        ordered_contact = self._get_ordered_foot_contact()
        right_contact = ordered_contact[:, 0]
        left_contact = ordered_contact[:, 1]

        # 오른발 swing 조건: 오른발은 떠 있고, 왼발은 지지 중
        right_actual_swing = is_swing[:, 0] & (~right_contact) & left_contact

        # 왼발 swing 조건: 왼발은 떠 있고, 오른발은 지지 중
        left_actual_swing = is_swing[:, 1] & (~left_contact) & right_contact

        # swing 발이 반대발보다 앞으로 나간 정도
        right_forward = right_x - left_x
        left_forward = left_x - right_x

        # 너무 작은 전진은 보상하지 않음
        min_forward = getattr(self.cfg.rewards, "min_swing_forward", 0.005)

        # 이 정도 앞으로 나가면 최대 보상
        target_forward = getattr(self.cfg.rewards, "target_swing_forward", 0.05)

        right_score = torch.clamp(
            (right_forward - min_forward) / (target_forward - min_forward),
            min=0.0,
            max=1.0,
        )

        left_score = torch.clamp(
            (left_forward - min_forward) / (target_forward - min_forward),
            min=0.0,
            max=1.0,
        )

        reward = (
            right_score * right_actual_swing.float()
            + left_score * left_actual_swing.float()
        )

        # 전진 명령이 있을 때만 활성화
        moving_cmd = self.commands[:, 0] > 0.02
        reward *= moving_cmd.float()

        return reward
    
    def _reward_swing_foot_forward_lane(self):
        """
        목적:
        1. 실제로 발이 떠 있을 때만 보상
        2. 반대발이 지지 중일 때만 보상
        3. swing 발이 base local x 방향으로 앞으로 나가면 보상
        4. swing 발이 자기 y lane에서 벗어나면 보상 감소
        5. foot yaw가 base yaw와 너무 다르면 보상 감소

        return 값은 positive reward.
        scale은 양수로 사용.
        """

        if self.feet_num < 2:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        right_idx = int(self.feet_order_right_left[0].item())
        left_idx = int(self.feet_order_right_left[1].item())

        # --------------------------------------------------
        # 1. foot position
        # --------------------------------------------------
        right_pos_world = self.feet_pos[:, right_idx, :]
        left_pos_world = self.feet_pos[:, left_idx, :]
        base_pos_world = self.root_states[:, 0:3]

        right_rel = right_pos_world - base_pos_world
        left_rel = left_pos_world - base_pos_world

        yaw = self.rpy[:, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # base yaw 기준 local x, y
        right_x = cos_yaw * right_rel[:, 0] + sin_yaw * right_rel[:, 1]
        left_x = cos_yaw * left_rel[:, 0] + sin_yaw * left_rel[:, 1]

        right_y = -sin_yaw * right_rel[:, 0] + cos_yaw * right_rel[:, 1]
        left_y = -sin_yaw * left_rel[:, 0] + cos_yaw * left_rel[:, 1]

        # --------------------------------------------------
        # 2. actual swing 조건
        # --------------------------------------------------
        ordered_contact = self._get_ordered_foot_contact()
        right_contact = ordered_contact[:, 0]
        left_contact = ordered_contact[:, 1]

        is_swing = self._get_swing_mask()

        right_phase_swing = is_swing[:, 0]
        left_phase_swing = is_swing[:, 1]

        # phase상 swing + 실제로 contact 없음 + 반대발은 지지 중
        right_actual_swing = right_phase_swing & (~right_contact) & left_contact
        left_actual_swing = left_phase_swing & (~left_contact) & right_contact

        # --------------------------------------------------
        # 3. forward reward
        # --------------------------------------------------
        # swing 발이 반대발보다 앞으로 나가면 보상
        # 너무 멀리 앞으로 보내는 꼼수 방지를 위해 max 제한
        max_forward = 0.08

        right_forward = torch.clamp(right_x - left_x, min=0.0, max=max_forward)
        left_forward = torch.clamp(left_x - right_x, min=0.0, max=max_forward)

        # 0~1 정규화
        right_forward = right_forward / max_forward
        left_forward = left_forward / max_forward

        # --------------------------------------------------
        # 4. lateral lane gate
        # --------------------------------------------------
        # 오른발/왼발 목표 y lane.
        # 실제 초기 자세에서 발 중심의 local y 값을 보고 조정 가능.
        target_right_y = 0.07
        target_left_y = -0.07

        lane_sigma = 0.0025  # 0.05m 오차에서 exp(-1) 정도

        right_lat_err = torch.square(right_y - target_right_y)
        left_lat_err = torch.square(left_y - target_left_y)

        right_lane_gate = torch.exp(-right_lat_err / lane_sigma)
        left_lane_gate = torch.exp(-left_lat_err / lane_sigma)

        # --------------------------------------------------
        # 5. foot yaw align gate
        # --------------------------------------------------
        # rigid_body_states: [pos(0:3), quat(3:7), lin_vel(7:10), ang_vel(10:13)]
        num_bodies = self.rigid_body_states.shape[0] // self.num_envs
        rigid_body_states = self.rigid_body_states.view(self.num_envs, num_bodies, 13)

        right_quat = rigid_body_states[:, self.feet_indices[right_idx], 3:7]
        left_quat = rigid_body_states[:, self.feet_indices[left_idx], 3:7]

        right_foot_yaw = self._quat_to_yaw(right_quat)
        left_foot_yaw = self._quat_to_yaw(left_quat)

        right_yaw_err = torch.atan2(
            torch.sin(right_foot_yaw - yaw),
            torch.cos(right_foot_yaw - yaw)
        )
        left_yaw_err = torch.atan2(
            torch.sin(left_foot_yaw - yaw),
            torch.cos(left_foot_yaw - yaw)
        )

        yaw_sigma = 0.25

        right_yaw_gate = torch.exp(-torch.square(right_yaw_err) / yaw_sigma)
        left_yaw_gate = torch.exp(-torch.square(left_yaw_err) / yaw_sigma)

        # --------------------------------------------------
        # final reward
        # --------------------------------------------------
        right_reward = (
            right_forward
            * right_actual_swing.float()
            * right_lane_gate
            * right_yaw_gate
        )

        left_reward = (
            left_forward
            * left_actual_swing.float()
            * left_lane_gate
            * left_yaw_gate
        )

        return right_reward + left_reward
        
    def _reward_pitch(self):
        # Explicitly suppress persistent torso pitch drift (backward/forward
        # lean), while allowing the small lean needed for normal walking.
        deadband = getattr(self.cfg.rewards, "pitch_deadband", 0.08)
        pitch_error = torch.clamp(torch.abs(self.rpy[:, 1]) - deadband, min=0.0)
        return torch.square(pitch_error)
    
    # 특정 hip 관절 각도 제곱합 페널티
    def _reward_hip_pos(self):
        """
        hip roll + hip yaw가 초기 자세에서 과하게 벗어나는 것을 막는 penalty.
        작은 균형 보정은 허용하고, 과한 대각선 보행/골반 비틀림만 억제한다.
        """

        hip_names = [
            "LL1_joint", "RL1_joint",  # hip yaw 후보
            "LL2_joint", "RL2_joint",  # hip roll 후보
        ]

        hip_indices = []

        for name in hip_names:
            if name in self.dof_names:
                hip_indices.append(self.dof_names.index(name))

        if len(hip_indices) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        hip_pos = self.dof_pos[:, hip_indices]
        hip_default = self.default_dof_pos[:, hip_indices]

        error = torch.abs(hip_pos - hip_default)

        # 약 0.08 rad = 4.6도까지 허용
        deadzone = 0.08

        violation = torch.clamp(error - deadzone, min=0.0)

        return torch.sum(torch.square(violation), dim=1)
    
    def _reward_yaw_rate_straight(self):
        yaw_cmd = torch.abs(self.commands[:, 2])
        straight_cmd = yaw_cmd < 0.05

        penalty = torch.square(self.base_ang_vel[:, 2])

        return torch.where(
            straight_cmd,
            penalty,
            torch.zeros_like(penalty),
        )
    

    def _reward_lin_vel_y_straight(self):
        """
        옆이동 명령이 없을 때만 base y방향 속도를 억제하는 penalty.

        commands[:, 1] = lin_vel_y 명령
        base_lin_vel[:, 1] = 실제 base y velocity
        """
        y_cmd = torch.abs(self.commands[:, 1])
        no_side_cmd = y_cmd < 0.02

        penalty = torch.square(self.base_lin_vel[:, 1])

        return torch.where(
            no_side_cmd,
            penalty,
            torch.zeros_like(penalty),
        )



    # # 지금 대칭 보상 함수는 그전 보폭을 반영 X / 그래서 왼발 오른발 한 스텝당 보폭이 달라짐 -> 나중에 그전 시점 반영 추가
    def _reward_sym_dof_pos(self):
        left = self.dof_pos[:, :6]
        right = self.dof_pos[:, 6:12]
        # DOF order: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
        # mirror sign: [ -, +, +, -, -, + ]
        # 앞 뒤로 움직이는 건 부호가 같아 -, 그 외는 부호가 달라 +
        err = torch.stack([
            left[:, 0] + right[:, 0],  # hip_pitch
            left[:, 1] + right[:, 1],  # hip_roll
            left[:, 2] + right[:, 2],  # hip_yaw
            left[:, 3] + right[:, 3],  # knee
            left[:, 4] + right[:, 4],  # ankle_pitch
            left[:, 5] + right[:, 5],  # ankle_roll
        ], dim=1)
        return torch.sum(torch.square(err), dim=1)

    def _reward_sym_dof_vel(self):
        left = self.dof_vel[:, :6]
        right = self.dof_vel[:, 6:12]
        err = torch.stack([
            left[:, 0] + right[:, 0],  # hip_pitch
            left[:, 1] + right[:, 1],  # hip_roll
            left[:, 2] + right[:, 2],  # hip_yaw
            left[:, 3] + right[:, 3],  # knee
            left[:, 4] + right[:, 4],  # ankle_pitch
            left[:, 5] + right[:, 5],  # ankle_roll
        ], dim=1)
        return torch.sum(torch.square(err), dim=1)

    def _reward_roll(self):
        return torch.square(self.rpy[:, 0])

    def _reward_right_hip_roll_positive(self):
        """
        오른쪽 hip roll(RL2_joint)이 + 방향으로 말리는 것만 막는 penalty.

        Jandi DOF order:
        7 = RL2_joint

        return은 cost이므로 scale은 음수로 사용.
        """

        rl2 = self.dof_pos[:, 7]

        # RL2가 0보다 커지면 안쪽으로 말리는 것으로 봄
        limit = 0.0

        # RL2가 +0.05 rad 정도 되면 penalty가 1 정도
        norm = 0.05

        positive_inward = torch.clamp(rl2 - limit, min=0.0)

        return torch.clamp(torch.square(positive_inward / norm), max=4.0)
