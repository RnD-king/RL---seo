import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Isaac Gym must be imported before torch and legged_gym.
import isaacgym  # noqa: F401, E402
import torch  # noqa: E402
from isaacgym import gymtorch  # noqa: E402

from legged_gym.envs import *  # noqa: F401, F403, E402
from legged_gym.utils import get_args, task_registry  # noqa: E402


# Controlled arm pose targets in radians.
# Order does not matter; names are matched against env.cfg.throw.controlled_dof_names.
homing_joint_angle_1 = {
    "LH1_joint": 0.0,
    # "LH2_joint": 0.0,
    "LH3_joint": 0.0,
    "LH4_joint": 0.0,
}

homing_joint_angle_2 = {

    # "LH1_joint": 2.8,
    # # "LH2_joint": 0.0,
    # "LH3_joint": 0.0,
    # "LH4_joint": 0.0,
    "LH1_joint": 2.1,
    # "LH2_joint": 0.0,
    
    "LH3_joint": -1.6,
    "LH4_joint": 0.5,
}

# Motion timing in seconds.
START_HOLD_DURATION = 2.0
HOMING_DURATION = 2.0
FINAL_HOLD_DURATION = 3.0

# True: set dof_pos directly like a visualizer, ignoring action_scale/PD/torque.
# False: drive the same poses through env actions like a policy would.
DIRECT_POSE_MODE = True

# Log every N control steps. Smaller values print more hand_pos_base/hand_vel_base.
LOG_INTERVAL = 20


def controlled_joint_dict_to_tensor(env, joint_angles):
    """Return controlled joint targets in env.cfg.throw.controlled_dof_names order."""
    controlled_names = list(env.cfg.throw.controlled_dof_names)
    missing = set(controlled_names) - set(joint_angles)
    extra = set(joint_angles) - set(controlled_names)
    if missing or extra:
        raise ValueError(
            f"Controlled joint-name mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    return torch.tensor(
        [joint_angles[name] for name in controlled_names],
        dtype=torch.float,
        device=env.device,
    )


def smoothstep(progress):
    """Cubic interpolation with zero velocity at both ends."""
    return progress * progress * (3.0 - 2.0 * progress)


def target_positions_to_actions(env, controlled_target_positions):
    """Convert desired controlled q targets to actions for the environment's P controller."""
    default_controlled_pos = env.default_dof_pos[:, env.controlled_dof_indices]
    actions = (
        controlled_target_positions.unsqueeze(0) - default_controlled_pos
    ) / env.action_scale_vec.unsqueeze(0)
    return torch.clip(actions, -env.cfg.normalization.clip_actions, env.cfg.normalization.clip_actions)


def refresh_hand_kinematics(env):
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env._update_throw_kinematics()


def set_controlled_pose_direct(env, controlled_positions, zero_velocity=True):
    """Set controlled DOFs directly for visualizer-like pose probing."""
    env.render()
    controlled_positions = controlled_positions.view(1, -1).expand(env.num_envs, -1)
    env.dof_pos[:, env.controlled_dof_indices] = controlled_positions
    if zero_velocity:
        env.dof_vel[:, env.controlled_dof_indices] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.simulate(env.sim)
    if env.device == "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    refresh_hand_kinematics(env)


def log_hand_state(env, step, label):
    refresh_hand_kinematics(env)
    hand_pos = env.hand_pos_base[0].detach().cpu()
    hand_vel = env.hand_vel_base[0].detach().cpu()
    dof_pos = env.dof_pos[0, env.controlled_dof_indices].detach().cpu()
    speed = torch.linalg.norm(hand_vel).item()

    print(
        f"[homing] {label} step={step:05d} | "
        f"hand_pos_base=(x={hand_pos[0]:+.4f}, y={hand_pos[1]:+.4f}, z={hand_pos[2]:+.4f}) | "
        f"hand_vel_base=(x={hand_vel[0]:+.4f}, y={hand_vel[1]:+.4f}, z={hand_vel[2]:+.4f}) | "
        f"speed={speed:.4f} | "
        f"q=({', '.join(f'{q.item():+.3f}' for q in dof_pos)})"
    )


def move_to(env, target_positions, duration, label):
    """Interpolate from measured controlled joint positions to a target pose."""
    start_positions = env.dof_pos[:, env.controlled_dof_indices].detach().clone()
    target_positions = target_positions.unsqueeze(0).expand_as(start_positions)
    control_dt = env.dt
    num_steps = max(1, int(round(duration / control_dt)))

    print(f"[homing] {label}: {duration:.2f} s ({num_steps} control steps)")
    for step in range(num_steps):
        progress = torch.tensor(
            (step + 1) / num_steps,
            dtype=torch.float,
            device=env.device,
        )
        desired_positions = torch.lerp(start_positions, target_positions, smoothstep(progress))
        if DIRECT_POSE_MODE:
            set_controlled_pose_direct(env, desired_positions[0])
        else:
            actions = target_positions_to_actions(env, desired_positions[0])
            env.step(actions)

        if step % LOG_INTERVAL == 0 or step == num_steps - 1:
            log_hand_state(env, step, label)


def hold_position(env, target_positions, duration, label):
    num_steps = max(1, int(round(duration / env.dt)))
    actions = target_positions_to_actions(env, target_positions)
    print(f"[homing] {label}: holding for {duration:.2f} s ({num_steps} control steps)")
    for step in range(num_steps):
        if DIRECT_POSE_MODE:
            set_controlled_pose_direct(env, target_positions)
        else:
            env.step(actions)

        if step % LOG_INTERVAL == 0 or step == num_steps - 1:
            log_hand_state(env, step, label)


def homing(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.fix_base_link = True
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.viewer.pos = [2.0, -2.0, 1.2]
    env_cfg.viewer.lookat = [0.0, 0.0, 0.6]

    # Start directly from homing pose 1.
    env_cfg.init_state.default_joint_angles = {
        **env_cfg.init_state.default_joint_angles,
        **homing_joint_angle_1,
    }

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    pose_1 = controlled_joint_dict_to_tensor(env, homing_joint_angle_1)
    pose_2 = controlled_joint_dict_to_tensor(env, homing_joint_angle_2)

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    env.reset_idx(env_ids)
    env.reset_buf[:] = 0
    env.gym.refresh_dof_state_tensor(env.sim)
    refresh_hand_kinematics(env)
    env.render()

    print("[homing] controlled DOF order:")
    for index, name in enumerate(env.cfg.throw.controlled_dof_names):
        dof_index = int(env.controlled_dof_indices[index].item())
        print(
            f"  a{index}: {name}, "
            f"dof_index={dof_index}, "
            f"initial={env.dof_pos[0, dof_index].item():+.3f} rad"
        )
    print(f"[homing] DIRECT_POSE_MODE={DIRECT_POSE_MODE}")

    hold_position(env, pose_1, START_HOLD_DURATION, "pose_1")
    move_to(env, pose_2, HOMING_DURATION, "pose_1 -> pose_2")
    hold_position(env, pose_2, FINAL_HOLD_DURATION, "pose_2")
    print("[homing] complete")


if __name__ == "__main__":
    homing(get_args())
