#!/usr/bin/env python3

import os
import csv
import time
import math

from isaacgym import gymapi, gymtorch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch

REAL_TORQUE_NM = {
    "RL1_joint":  0.070,
    "RL2_joint": -0.033,
    "RL3_joint": -0.047,   # 실제 기준에서 부호 반전
    "RL4_joint":  0.490,
    "RL5_joint":  0.142,
    "RL6_joint": -0.071,

    "LL1_joint":  0.015,
    "LL2_joint":  0.079,
    "LL3_joint":  0.122,   # 실제 기준에서 부호 반전
    "LL4_joint": -0.528,
    "LL5_joint": -0.244,
    "LL6_joint":  0.062,
}


def _rand_uniform(device, shape, low, high):
    return torch.empty(*shape, device=device).uniform_(low, high)


def _quat_to_rpy_xyzw(q):
    """Convert quaternion [x, y, z, w] to roll/pitch/yaw in rad."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _reset_preview_state(env, env_ids=None, joint_noise=0.0, root_vel_noise=0.0):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif len(env_ids) == 0:
        return

    env.root_states[env_ids] = env.base_init_state
    env.root_states[env_ids, :3] += env.env_origins[env_ids]
    env.root_states[env_ids, 7:13] = 0.0
    if root_vel_noise > 0.0:
        env.root_states[env_ids, 7:13] = _rand_uniform(
            env.device, (len(env_ids), 6), -root_vel_noise, root_vel_noise
        )

    q = env.default_dof_pos.repeat(len(env_ids), 1)
    if joint_noise > 0.0:
        q = q + _rand_uniform(
            env.device, (len(env_ids), env.num_dof), -joint_noise, joint_noise
        )
    lower = env.dof_pos_limits_urdf[:, 0].unsqueeze(0)
    upper = env.dof_pos_limits_urdf[:, 1].unsqueeze(0)
    margin = 0.02
    env.dof_pos[env_ids] = torch.clamp(q, lower + margin, upper - margin)
    env.dof_vel[env_ids] = 0.0

    env.actions[env_ids] = 0.0
    env.last_actions[env_ids] = 0.0
    env.last_dof_vel[env_ids] = 0.0
    env.last_root_vel[env_ids] = 0.0
    env.episode_length_buf[env_ids] = 0
    env.reset_buf[env_ids] = 0

    env_ids_int32 = env_ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )


def _read_base_contact_state(env, t):
    """Read base pose and foot contact force for env 0."""
    root_state = env.root_states[0].detach().cpu()
    base_x = float(root_state[0])
    base_y = float(root_state[1])
    base_z = float(root_state[2])

    # Isaac Gym root quaternion is [x, y, z, w]
    base_quat = root_state[3:7]
    base_roll, base_pitch, base_yaw = _quat_to_rpy_xyzw(base_quat)

    left_fz = 0.0
    right_fz = 0.0
    total_fz = 0.0
    num_feet = 0

    if hasattr(env, "feet_indices"):
        foot_fz = env.contact_forces[0, env.feet_indices, 2].detach().cpu()
        num_feet = int(len(foot_fz))
        if len(foot_fz) >= 2:
            # legged_gym의 feet_indices 순서가 보통 config/body order를 따릅니다.
            # 여기서는 first=left, second=right로 기록하고, 실제 이름 순서는 출력으로 확인하세요.
            left_fz = float(foot_fz[0])
            right_fz = float(foot_fz[1])
            total_fz = left_fz + right_fz
        elif len(foot_fz) == 1:
            left_fz = float(foot_fz[0])
            right_fz = 0.0
            total_fz = left_fz

    return {
        "time_sec": t,
        "base_x": base_x,
        "base_y": base_y,
        "base_z": base_z,
        "base_roll_rad": base_roll,
        "base_pitch_rad": base_pitch,
        "base_yaw_rad": base_yaw,
        "base_roll_deg": base_roll * 180.0 / math.pi,
        "base_pitch_deg": base_pitch * 180.0 / math.pi,
        "base_yaw_deg": base_yaw * 180.0 / math.pi,
        "num_feet_indices": num_feet,
        "left_fz": left_fz,
        "right_fz": right_fz,
        "total_fz": total_fz,
        "fz_diff_left_minus_right": left_fz - right_fz,
        "abs_fz_diff_left_right": abs(left_fz - right_fz),
    }


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def _abs_mean(rows, key):
    return sum(abs(r[key]) for r in rows) / len(rows)


def main():
    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.fix_base_link = False
    env_cfg.commands.curriculum = False

    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.viewer.pos = [2.0, -2.0, 1.2]
    env_cfg.viewer.lookat = [0.0, 0.0, 0.6]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    _reset_preview_state(
        env,
        joint_noise=args.reset_joint_noise,
        root_vel_noise=args.reset_root_vel_noise,
    )

    device = env.device
    actions = torch.zeros(env.num_envs, env.num_actions, device=device)

    # action=0이면 default_joint_angles가 목표각입니다.
    # standing pose에서 안정화
    warmup_steps = int(3.0 / env.dt)
    log_steps = int(10.0 / env.dt)

    print("[INFO] DOF order")
    for i, name in enumerate(env.dof_names):
        print(f"{i:2d}: {name}")

    if hasattr(env, "feet_indices"):
        print(f"[INFO] feet_indices: {env.feet_indices}")
    else:
        print("[WARN] env.feet_indices not found. Foot contact force will be 0.")

    print("\n[INFO] Warmup standing pose...")
    for _ in range(warmup_steps):
        _, _, _, dones, _ = env.step(actions)
        if torch.any(dones):
            done_ids = dones.nonzero(as_tuple=False).flatten()
            _reset_preview_state(
                env,
                env_ids=done_ids,
                joint_noise=args.reset_joint_noise,
                root_vel_noise=args.reset_root_vel_noise,
            )

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "jandi_standing_torque")
    os.makedirs(log_dir, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"sim_standing_torque_{stamp}.csv")
    state_csv_path = os.path.join(log_dir, f"sim_standing_state_{stamp}.csv")

    rows = []
    state_rows = []

    print("[INFO] Logging sim torque...")
    skipped_reset_steps = 0
    for step in range(log_steps):
        _, _, _, dones, _ = env.step(actions)
        if torch.any(dones):
            skipped_reset_steps += 1
            done_ids = dones.nonzero(as_tuple=False).flatten()
            _reset_preview_state(
                env,
                env_ids=done_ids,
                joint_noise=args.reset_joint_noise,
                root_vel_noise=args.reset_root_vel_noise,
            )
            continue

        torques = env.torques[0].detach().cpu()
        dof_pos = env.dof_pos[0].detach().cpu()
        dof_vel = env.dof_vel[0].detach().cpu()
        default_pos = env.default_dof_pos[0].detach().cpu()

        t = step * env.dt

        # Base / contact state는 joint row와 별도 CSV로 저장
        state_rows.append(_read_base_contact_state(env, t))

        for i, name in enumerate(env.dof_names):
            goal_rad = float(default_pos[i])
            present_rad = float(dof_pos[i])
            error_rad = goal_rad - present_rad
            sim_torque = float(torques[i])
            real_torque = REAL_TORQUE_NM.get(name, None)

            rows.append({
                "time_sec": t,
                "joint_name": name,
                "dof_index": i,
                "goal_rad": goal_rad,
                "present_rad": present_rad,
                "error_rad": error_rad,
                "error_deg": error_rad * 180.0 / math.pi,
                "present_vel_rad_s": float(dof_vel[i]),
                "sim_torque_Nm": sim_torque,
                "real_est_torque_Nm": real_torque,
                "torque_error_Nm": None if real_torque is None else sim_torque - real_torque,
                "abs_sim_torque_Nm": abs(sim_torque),
                "abs_real_est_torque_Nm": None if real_torque is None else abs(real_torque),
            })

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "time_sec",
            "joint_name",
            "dof_index",
            "goal_rad",
            "present_rad",
            "error_rad",
            "error_deg",
            "present_vel_rad_s",
            "sim_torque_Nm",
            "real_est_torque_Nm",
            "torque_error_Nm",
            "abs_sim_torque_Nm",
            "abs_real_est_torque_Nm",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(state_csv_path, "w", newline="") as f:
        fieldnames = [
            "time_sec",
            "base_x",
            "base_y",
            "base_z",
            "base_roll_rad",
            "base_pitch_rad",
            "base_yaw_rad",
            "base_roll_deg",
            "base_pitch_deg",
            "base_yaw_deg",
            "num_feet_indices",
            "left_fz",
            "right_fz",
            "total_fz",
            "fz_diff_left_minus_right",
            "abs_fz_diff_left_right",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(state_rows)

    print(f"\n[INFO] Saved joint CSV: {csv_path}")
    print(f"[INFO] Saved state CSV: {state_csv_path}")
    print(f"[INFO] Skipped reset/fall steps: {skipped_reset_steps}")

    print("\n[INFO] Mean torque + error summary")
    print("joint        sim_mean    real_est      diff   err_mean(rad)  abs_err_mean(rad)  err_mean(deg)  abs_err_mean(deg)")

    for i, name in enumerate(env.dof_names):
        joint_rows = [r for r in rows if r["joint_name"] == name]

        if len(joint_rows) == 0:
            print(f"{name:10s}      None      None      None        None             None           None             None")
            continue

        joint_torque_vals = [r["sim_torque_Nm"] for r in joint_rows]
        joint_error_vals = [r["error_rad"] for r in joint_rows]

        sim_mean = sum(joint_torque_vals) / len(joint_torque_vals)
        err_mean = sum(joint_error_vals) / len(joint_error_vals)
        abs_err_mean = sum(abs(e) for e in joint_error_vals) / len(joint_error_vals)

        err_mean_deg = err_mean * 180.0 / math.pi
        abs_err_mean_deg = abs_err_mean * 180.0 / math.pi

        real = REAL_TORQUE_NM.get(name, None)
        diff = None if real is None else sim_mean - real

        if real is None:
            print(
                f"{name:10s} "
                f"{sim_mean:+9.4f}      None      None "
                f"{err_mean:+14.6f} {abs_err_mean:18.6f} "
                f"{err_mean_deg:+14.4f} {abs_err_mean_deg:18.4f}"
            )
        else:
            print(
                f"{name:10s} "
                f"{sim_mean:+9.4f} {real:+9.4f} {diff:+9.4f} "
                f"{err_mean:+14.6f} {abs_err_mean:18.6f} "
                f"{err_mean_deg:+14.4f} {abs_err_mean_deg:18.4f}"
            )

    if len(state_rows) > 0:
        print("\n[INFO] Base / contact summary")
        print(f"base_z_mean              = {_mean(state_rows, 'base_z'):.4f} m")
        print(f"base_roll_mean_deg       = {_mean(state_rows, 'base_roll_deg'):+.4f} deg")
        print(f"base_pitch_mean_deg      = {_mean(state_rows, 'base_pitch_deg'):+.4f} deg")
        print(f"base_roll_abs_mean_deg   = {_abs_mean(state_rows, 'base_roll_deg'):.4f} deg")
        print(f"base_pitch_abs_mean_deg  = {_abs_mean(state_rows, 'base_pitch_deg'):.4f} deg")
        print(f"left_fz_mean             = {_mean(state_rows, 'left_fz'):+.4f} N")
        print(f"right_fz_mean            = {_mean(state_rows, 'right_fz'):+.4f} N")
        print(f"total_fz_mean            = {_mean(state_rows, 'total_fz'):+.4f} N")
        print(f"fz_diff_L_minus_R_mean   = {_mean(state_rows, 'fz_diff_left_minus_right'):+.4f} N")
        print(f"fz_diff_L_minus_R_abs    = {_abs_mean(state_rows, 'fz_diff_left_minus_right'):.4f} N")

    print("\n[INFO] Focus check: LL4/RL4/LL5/RL5")
    for name in ["LL4_joint", "RL4_joint", "LL5_joint", "RL5_joint"]:
        joint_rows = [r for r in rows if r["joint_name"] == name]
        if len(joint_rows) == 0:
            print(f"{name}: no data")
            continue
        torque_mean = _mean(joint_rows, "sim_torque_Nm")
        err_mean = _mean(joint_rows, "error_rad")
        abs_err_mean = _abs_mean(joint_rows, "error_rad")
        print(
            f"{name:10s} "
            f"torque_mean={torque_mean:+.4f} Nm, "
            f"err_mean={err_mean:+.6f} rad ({err_mean * 180.0 / math.pi:+.4f} deg), "
            f"abs_err_mean={abs_err_mean:.6f} rad ({abs_err_mean * 180.0 / math.pi:.4f} deg)"
        )


if __name__ == "__main__":
    main()