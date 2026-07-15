from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym import gymtorch
import torch


class JandiRobotBase(LeggedRobot):
    """Common Jandi robot helpers shared by task environments."""

    def check_termination(self):
        """Keep Jandi-specific contact and attitude termination limits."""
        max_pitch = getattr(self.cfg.rewards, "termination_pitch", 1.0)
        max_roll = getattr(self.cfg.rewards, "termination_roll", 0.8)
        contact_norm = torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        )
        self.reset_buf = torch.any(contact_norm > 5.0, dim=1)
        self.reset_buf |= torch.logical_or(
            torch.abs(self.rpy[:, 1]) > max_pitch,
            torch.abs(self.rpy[:, 0]) > max_roll,
        )
        if not hasattr(self, "fold_dof_indices"):
            fold_idx = [
                i
                for i, name in enumerate(self.dof_names)
                if ("4_joint" in name or "5_joint" in name or "6_joint" in name)
            ]
            self.fold_dof_indices = torch.tensor(
                fold_idx, dtype=torch.long, device=self.device
            )
        if self.fold_dof_indices.numel() > 0:
            folded = torch.any(
                torch.abs(
                    self.dof_pos[:, self.fold_dof_indices]
                    - self.default_dof_pos[:, self.fold_dof_indices]
                )
                > 0.75,
                dim=1,
            )
            self.reset_buf |= folded
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _empty_reward(self):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

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
        self._init_foot_order()

    def _init_foot_order(self):
        """Set local foot index order as [right, left]."""
        self.feet_num = len(self.feet_indices)

        if self.feet_num < 2:
            self.right_foot_local_idx = 0
            self.left_foot_local_idx = 0
            self.feet_order_right_left = torch.tensor(
                [0, 0],
                dtype=torch.long,
                device=self.device,
            )
            return

        self.right_foot_local_idx = 0
        self.left_foot_local_idx = 1

        self.feet_order_right_left = torch.tensor(
            [self.right_foot_local_idx, self.left_foot_local_idx],
            dtype=torch.long,
            device=self.device,
        )

        print("[DEBUG] feet_indices:", self.feet_indices, flush=True)
        print("[DEBUG] right_foot_local_idx:", self.right_foot_local_idx, flush=True)
        print("[DEBUG] left_foot_local_idx:", self.left_foot_local_idx, flush=True)
        print(
            "[DEBUG] feet_order_right_left:",
            self.feet_order_right_left.detach().cpu().numpy(),
            flush=True,
        )

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _get_stance_mask(self):
        if self.feet_num < 2:
            return torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        right_stance = self.leg_phase_right_left[:, 0] < self.stance_ratio
        left_stance = self.leg_phase_right_left[:, 1] < self.stance_ratio
        return torch.stack([right_stance, left_stance], dim=1)

    def _get_swing_mask(self):
        return ~self._get_stance_mask()

    def _get_ordered_foot_contact(self, threshold=None):
        if self.feet_num < 2:
            return torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        if threshold is None:
            threshold = getattr(self.cfg.rewards, "contact_force_threshold", 5.0)

        contact = self.contact_forces[:, self.feet_indices, 2] > threshold
        right_idx = int(self.feet_order_right_left[0].item())
        left_idx = int(self.feet_order_right_left[1].item())
        return torch.stack([contact[:, right_idx], contact[:, left_idx]], dim=1)

    def _get_ordered_foot_height(self):
        if self.feet_num < 2:
            return torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)

        right_idx = int(self.feet_order_right_left[0].item())
        left_idx = int(self.feet_order_right_left[1].item())
        return torch.stack(
            [self.feet_pos[:, right_idx, 2], self.feet_pos[:, left_idx, 2]],
            dim=1,
        )

    def _get_ordered_foot_xy_vel(self):
        if self.feet_num < 2:
            return torch.zeros(
                self.num_envs, 2, 2, dtype=torch.float, device=self.device
            )

        right_idx = int(self.feet_order_right_left[0].item())
        left_idx = int(self.feet_order_right_left[1].item())
        return torch.stack(
            [self.feet_vel[:, right_idx, :2], self.feet_vel[:, left_idx, :2]],
            dim=1,
        )
