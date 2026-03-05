# from legged_gym.envs.base.legged_robot import LeggedRobot

# from isaacgym.torch_utils import torch_rand_float
# from isaacgym import gymtorch
# import torch

# class JandiRobot(LeggedRobot):
#     def _reset_dofs(self, env_ids):
#         """Reset joints to default pose without random scaling."""
#         self.dof_pos[env_ids] = self.default_dof_pos
#         self.dof_vel[env_ids] = 0.0
#         self._update_dof_limit_violation_stats(env_ids)

#         env_ids_int32 = env_ids.to(dtype=torch.int32)
#         self.gym.set_dof_state_tensor_indexed(
#             self.sim,
#             gymtorch.unwrap_tensor(self.dof_state),
#             gymtorch.unwrap_tensor(env_ids_int32),
#             len(env_ids_int32),
#         )

#     def _reset_root_states(self, env_ids):
#         """Reset root pose and keep base linear/angular velocity zero."""
#         if self.custom_origins:
#             self.root_states[env_ids] = self.base_init_state
#             self.root_states[env_ids, :3] += self.env_origins[env_ids]
#             self.root_states[env_ids, :2] += torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
#         else:
#             self.root_states[env_ids] = self.base_init_state
#             self.root_states[env_ids, :3] += self.env_origins[env_ids]

#         self.root_states[env_ids, 7:13] = 0.0

#         env_ids_int32 = env_ids.to(dtype=torch.int32)
#         self.gym.set_actor_root_state_tensor_indexed(
#             self.sim,
#             gymtorch.unwrap_tensor(self.root_states),
#             gymtorch.unwrap_tensor(env_ids_int32),
#             len(env_ids_int32),
#         )

# ---------------------------------------------------------------------------
# Legacy implementation (commented out)
# ---------------------------------------------------------------------------
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class JandiRobot(LeggedRobot):
    def _reset_dofs(self, env_ids):
        """Reset joints to the configured default pose (no random scaling)."""
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """Reset root pose and zero base linear/angular velocity for stability."""
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # Keep reset velocity at zero to avoid immediate tumble/explosions.
        self.root_states[env_ids, 7:13] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    
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
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        # Debug: print current base height statistics periodically.
        if self.common_step_counter % 200 == 0:
            base_z = self.root_states[:, 2]
            print(
                f"[jandi] step={int(self.common_step_counter)} "
                f"base_z mean={base_z.mean().item():.4f} "
                f"min={base_z.min().item():.4f} "
                f"max={base_z.max().item():.4f}",
                flush=True,
            )
        
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
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
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
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        self.privileged_obs_buf = torch.nan_to_num(self.privileged_obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        
    
        # _chk("obs_buf", self.obs_buf)

        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    # reward 함수들

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.05) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    # 전진 관련 보상
    def _reward_forward_progress(self):
        # x축 전진 속도만 보상 (후진은 0)
        return torch.clamp(self.base_lin_vel[:, 0], min=0.0)

    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[0,1,6,7]]), dim=1) # 각각 hip roll, yaw
    
    # # 지금 대칭 보상 함수는 그전 보폭을 반영 X / 그래서 왼발 오른발 한 스텝당 보폭이 달라짐 -> 나중에 그전 시점 반영 추가

    def _reward_sym_dof_pos(self):
        left = self.dof_pos[:, :6]
        right = self.dof_pos[:, 6:12]
        # DOF order: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
        err = torch.stack([
            left[:, 0] + right[:, 0],  # hip_yaw
            left[:, 1] - right[:, 1],  # hip_roll
            left[:, 2] + right[:, 2],  # hip_pitch
            left[:, 3] + right[:, 3],  # knee
            left[:, 4] + right[:, 4],  # ankle_pitch
            left[:, 5] + right[:, 5],  # ankle_roll
        ], dim=1)
        return torch.sum(torch.square(err), dim=1)

    def _reward_sym_dof_vel(self):
        left = self.dof_vel[:, :6]
        right = self.dof_vel[:, 6:12]
        err = torch.stack([
            left[:, 0] + right[:, 0],  # hip_yaw
            left[:, 1] - right[:, 1],  # hip_roll
            left[:, 2] + right[:, 2],  # hip_pitch
            left[:, 3] + right[:, 3],  # knee
            left[:, 4] + right[:, 4],  # ankle_pitch
            left[:, 5] + right[:, 5],  # ankle_roll
        ], dim=1)
        return torch.sum(torch.square(err), dim=1)
