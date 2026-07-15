import os
import time

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.jandi_base.jandi_base_env import JandiRobotBase
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor


class JandiThrowEnv(JandiRobotBase):
    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self._validate_throw_names()

        print("DOF order:")
        for i, name in enumerate(self.dof_names):
            print(i, name)

        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.num_actors_per_env = 1
        self.envs = []
        self.actor_handles = []
        self.robot_actor_handles = []
        self.robot_actor_indices = []

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.base_init_state[:3].clone()
            pos += self.env_origins[i]
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, robot_handle, body_props, recomputeInertia=True
            )

            self.envs.append(env_handle)
            self.actor_handles.append(robot_handle)
            self.robot_actor_handles.append(robot_handle)
            self.robot_actor_indices.append(
                self.gym.get_actor_index(env_handle, robot_handle, gymapi.DOMAIN_SIM)
            )

        self.robot_actor_indices = torch.tensor(self.robot_actor_indices, dtype=torch.int32, device=self.device)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(feet_names):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.robot_actor_handles[0], name
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, name in enumerate(penalized_contact_names):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.robot_actor_handles[0], name
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, name in enumerate(termination_contact_names):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.robot_actor_handles[0], name
            )

        self.hand_body_idx = self.body_names.index(self.cfg.throw.hand_body_name)

    def _validate_throw_names(self):
        missing_dofs = [name for name in self.cfg.throw.controlled_dof_names if name not in self.dof_names]
        if missing_dofs:
            raise ValueError(
                "[ERROR] controlled dof name not found: {}\nAvailable dof names: {}".format(
                    missing_dofs, self.dof_names
                )
            )
        if self.cfg.throw.hand_body_name not in self.body_names:
            raise ValueError(
                "[ERROR] hand body name not found: {}\nAvailable body names: {}".format(
                    self.cfg.throw.hand_body_name, self.body_names
                )
            )
        if self.cfg.env.num_actions != len(self.cfg.throw.controlled_dof_names):
            raise ValueError(
                "[ERROR] cfg.env.num_actions must match len(cfg.throw.controlled_dof_names): "
                f"{self.cfg.env.num_actions} vs {len(self.cfg.throw.controlled_dof_names)}"
            )
        expected_obs = 3 * len(self.cfg.throw.controlled_dof_names) + 14
        if self.cfg.env.num_observations != expected_obs:
            raise ValueError(
                "[ERROR] cfg.env.num_observations must be 3 * num_actions + 14: "
                f"{self.cfg.env.num_observations} vs {expected_obs}"
            )

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.all_root_states_flat = gymtorch.wrap_tensor(actor_root_state)
        self.all_root_states = self.all_root_states_flat.view(self.num_envs, self.num_actors_per_env, 13)
        self.robot_root_states = self.all_root_states[:, 0, :]
        self.root_states = self.robot_root_states

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.hand_state = self.rigid_body_states_view[:, self.hand_body_idx, :]
        self.hand_pos_world = self.hand_state[:, 0:3]
        self.hand_vel_world = self.hand_state[:, 7:10]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))

        self.controlled_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.cfg.throw.controlled_dof_names],
            dtype=torch.long,
            device=self.device,
        )
        self.hand_pos_base = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.hand_vel_base = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.phase_boundaries = to_torch(
            self.cfg.throw.phase_boundaries,
            dtype=torch.float,
            device=self.device,
        )
        self.phase_hand_targets = to_torch(
            self.cfg.throw.phase_hand_targets,
            dtype=torch.float,
            device=self.device,
        )
        throw_dir = to_torch(self.cfg.throw.throw_dir_base, dtype=torch.float, device=self.device)
        self.throw_dir_base = throw_dir / torch.clamp(torch.norm(throw_dir), min=1e-6)
        self.phase_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.throw_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reward_pose_tracking = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reward_pose_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reward_throw_velocity = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reward_action_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reward_dof_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.action_scale_vec = torch.full(
            (len(self.controlled_dof_indices),),
            float(self.cfg.throw.action_scale),
            dtype=torch.float,
            device=self.device,
        )
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.reset_dof_noise = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        for i, name in enumerate(self.dof_names):
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            for dof_name, noise_scale in self.cfg.init_state.reset_joint_noise.items():
                if dof_name in name:
                    self.reset_dof_noise[i] = noise_scale
                    break
            if name in self.cfg.control.stiffness:
                self.p_gains[i] = self.cfg.control.stiffness[name]
                self.d_gains[i] = self.cfg.control.damping[name]
                found = True
            else:
                found = False
                for dof_name in self.cfg.control.stiffness.keys():
                    if dof_name in name:
                        self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                        self.d_gains[i] = self.cfg.control.damping[dof_name]
                        found = True
                        break
            if not found and self.cfg.control.control_type in ["P", "V"]:
                print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.dof_limit_violation_lower_total = torch.zeros(self.num_dof, dtype=torch.long, device=self.device)
        self.dof_limit_violation_upper_total = torch.zeros(self.num_dof, dtype=torch.long, device=self.device)
        self.dof_limit_violation_total = torch.zeros(self.num_dof, dtype=torch.long, device=self.device)
        self.dof_limit_violation_count_total = 0
        self.dof_limit_violation_batch_count = 0
        self.dof_limit_violation_batch_env_count = 0
        self.dof_limit_violation_reset_calls = 0

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = self.cfg.noise.add_noise
        return torch.zeros_like(self.obs_buf[0])

    def _prepare_reward_function(self):
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        self.reward_functions = []
        self.reward_names = []
        for name in self.reward_scales.keys():
            if name == "termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))

        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        full_target = self.default_dof_pos.repeat(self.num_envs, 1)
        full_target[:, self.controlled_dof_indices] += actions * self.action_scale_vec.unsqueeze(0)

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (full_target - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            full_vel_target = torch.zeros_like(self.dof_vel)
            full_vel_target[:, self.controlled_dof_indices] = actions * self.action_scale_vec.unsqueeze(0)
            torques = self.p_gains * (full_vel_target - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = torch.zeros_like(self.dof_vel)
            torques[:, self.controlled_dof_indices] = actions * self.action_scale_vec.unsqueeze(0)
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _post_physics_step_callback(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_throw_kinematics()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_throw_kinematics()

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self._update_throw_extras()
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        env_actor_ids = self.robot_actor_indices[env_ids]
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_actor_ids),
            len(env_actor_ids),
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0
        actor_ids = self.robot_actor_indices[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.all_root_states_flat),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def _update_throw_kinematics(self):
        base_pos = self.root_states[:, 0:3]
        base_quat = self.root_states[:, 3:7]
        base_lin_vel_world = self.root_states[:, 7:10]
        self.hand_state = self.rigid_body_states_view[:, self.hand_body_idx, :]
        self.hand_pos_world = self.hand_state[:, 0:3]
        self.hand_vel_world = self.hand_state[:, 7:10]
        self.hand_pos_base = quat_rotate_inverse(base_quat, self.hand_pos_world - base_pos)
        self.hand_vel_base = quat_rotate_inverse(base_quat, self.hand_vel_world - base_lin_vel_world)

    def _update_throw_phase(self):
        self.throw_phase = torch.clamp(
            self.episode_length_buf.float() * self.dt / self.max_episode_length_s,
            0.0,
            1.0,
        )

        self.phase_index[:] = 0
        self.phase_index += (self.throw_phase >= self.phase_boundaries[0]).long()
        self.phase_index += (self.throw_phase >= self.phase_boundaries[1]).long()
        self.phase_index += (self.throw_phase >= self.phase_boundaries[2]).long()

    def _get_phase_hand_target(self):
        if not self.cfg.throw.interpolate_phase_targets:
            return self.phase_hand_targets[self.phase_index]

        phase = self.throw_phase
        b0 = self.phase_boundaries[0]
        b1 = self.phase_boundaries[1]
        b2 = self.phase_boundaries[2]
        p0 = self.phase_hand_targets[0]
        p1 = self.phase_hand_targets[1]
        p2 = self.phase_hand_targets[2]
        p3 = self.phase_hand_targets[3]

        target = p3.unsqueeze(0).repeat(self.num_envs, 1)

        first = phase < b0
        alpha0 = torch.clamp(phase / torch.clamp(b0, min=1e-6), 0.0, 1.0).unsqueeze(1)
        target_first = p0 + alpha0 * (p1 - p0)
        target = torch.where(first.unsqueeze(1), target_first, target)

        second = (phase >= b0) & (phase < b1)
        alpha1 = torch.clamp((phase - b0) / torch.clamp(b1 - b0, min=1e-6), 0.0, 1.0).unsqueeze(1)
        target_second = p1 + alpha1 * (p2 - p1)
        target = torch.where(second.unsqueeze(1), target_second, target)

        third = (phase >= b1) & (phase < b2)
        alpha2 = torch.clamp((phase - b1) / torch.clamp(b2 - b1, min=1e-6), 0.0, 1.0).unsqueeze(1)
        target_third = p2 + alpha2 * (p3 - p2)
        target = torch.where(third.unsqueeze(1), target_third, target)

        return target

    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def compute_reward(self):
        self._update_throw_kinematics()
        self._update_throw_phase()

        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        self._update_throw_extras()

    def _reward_pose_tracking(self):
        """Reward hand position tracking for the current throw phase."""
        target_pos = self._get_phase_hand_target()
        hand_pos_error = self.hand_pos_base - target_pos
        hand_pos_dist_sq = torch.sum(torch.square(hand_pos_error), dim=1)
        pose_sigma = max(float(self.cfg.throw.pose_tracking_sigma), 1e-6)
        self.reward_pose_tracking = torch.exp(-hand_pos_dist_sq / (pose_sigma * pose_sigma))
        return self.reward_pose_tracking

    def _reward_pose_error(self):
        """Penalty for missing the current phase hand target."""
        target_pos = self._get_phase_hand_target()
        hand_pos_error = self.hand_pos_base - target_pos
        hand_pos_dist_sq = torch.sum(torch.square(hand_pos_error), dim=1)
        pose_error_norm = max(float(self.cfg.throw.pose_error_norm), 1e-6)
        pose_error = hand_pos_dist_sq / (pose_error_norm * pose_error_norm)
        self.reward_pose_error = torch.clamp(
            pose_error,
            max=float(self.cfg.throw.pose_error_clip),
        )
        return self.reward_pose_error

    def _reward_throw_velocity(self):
        """Reward hand velocity aligned with the throw direction during release."""
        hand_speed = torch.norm(self.hand_vel_base, dim=1)
        hand_vel_dir = self.hand_vel_base / torch.clamp(hand_speed.unsqueeze(1), min=1e-6)
        vel_alignment = torch.sum(hand_vel_dir * self.throw_dir_base.unsqueeze(0), dim=1)
        vel_alignment = torch.clamp(vel_alignment, min=0.0, max=1.0)
        speed_sigma = max(float(self.cfg.throw.throw_speed_sigma), 1e-6)
        speed_reward = 1.0 - torch.exp(-torch.square(hand_speed) / (speed_sigma * speed_sigma))
        throw_phase = self.phase_index == 2
        self.reward_throw_velocity = vel_alignment * speed_reward * throw_phase.float()
        return self.reward_throw_velocity

    def _reward_action_rate(self):
        """Penalty for rapid action changes."""
        self.reward_action_rate = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return self.reward_action_rate

    def _reward_dof_vel(self):
        """Penalty for controlled arm joint velocity."""
        self.reward_dof_vel = torch.sum(torch.square(self.dof_vel[:, self.controlled_dof_indices]), dim=1)
        return self.reward_dof_vel

    def compute_observations(self):
        self._update_throw_kinematics()
        controlled_pos_error = (
            self.dof_pos[:, self.controlled_dof_indices]
            - self.default_dof_pos[:, self.controlled_dof_indices]
        ) * self.obs_scales.dof_pos
        controlled_vel = self.dof_vel[:, self.controlled_dof_indices] * self.obs_scales.dof_vel
        sin_phase = torch.sin(2.0 * np.pi * self.episode_length_buf.float() * self.dt / self.max_episode_length_s).unsqueeze(1)
        cos_phase = torch.cos(2.0 * np.pi * self.episode_length_buf.float() * self.dt / self.max_episode_length_s).unsqueeze(1)

        self.obs_buf = torch.cat(
            (
                controlled_pos_error,
                controlled_vel,
                self.last_actions,
                self.hand_pos_base,
                self.hand_vel_base,
                self.projected_gravity,
                self.base_ang_vel * self.obs_scales.ang_vel,
                sin_phase,
                cos_phase,
            ),
            dim=-1,
        )
        self.privileged_obs_buf = self.obs_buf

    def _update_throw_extras(self):
        self.extras["hand_height"] = torch.mean(self.hand_pos_base[:, 2])
        self.extras["hand_speed"] = torch.mean(torch.norm(self.hand_vel_base, dim=1))
        self.extras["throw_phase"] = torch.mean(self.phase_index.float())
        self.extras["rew_pose_tracking"] = torch.mean(self.reward_pose_tracking)
        self.extras["penalty_pose_error"] = torch.mean(self.reward_pose_error)
        self.extras["rew_throw_velocity"] = torch.mean(self.reward_throw_velocity)
        self.extras["penalty_action_rate"] = torch.mean(self.reward_action_rate)
        self.extras["penalty_dof_vel"] = torch.mean(self.reward_dof_vel)
