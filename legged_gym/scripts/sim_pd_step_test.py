#!/usr/bin/env python3
"""
Simulation PD step-response tester (single joint), mirroring dxl_pd_step_test flow.

Use this to match sim Kp/Kd to real motor response metrics:
- rise time
- settling time
- overshoot
- steady-state error
- vel/acc/jerk RMS
"""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import isaacgym  # noqa: F401
from isaacgym import gymutil
from isaacgym import gymtorch
import torch

from legged_gym.envs import *  # noqa: F403,F401
from legged_gym.utils import task_registry


def rms(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def count_error_sign_flips(err: List[float], eps: float = 1e-5) -> int:
    flips = 0
    prev_sign = 0
    for e in err:
        if abs(e) <= eps:
            continue
        sign = 1 if e > 0 else -1
        if prev_sign != 0 and sign != prev_sign:
            flips += 1
        prev_sign = sign
    return flips


def estimate_step_metrics(
    t: List[float], y: List[float], y0: float, y_target: float, tol_ratio: float
) -> Dict[str, float]:
    if len(t) < 5:
        return {
            "rise_time_s": float("nan"),
            "settling_time_s": float("nan"),
            "overshoot_pct": float("nan"),
            "steady_state_err_rad": float("nan"),
            "peak_rad": float("nan"),
        }

    step = y_target - y0
    amp = abs(step)
    if amp < 1e-8:
        return {
            "rise_time_s": 0.0,
            "settling_time_s": 0.0,
            "overshoot_pct": 0.0,
            "steady_state_err_rad": 0.0,
            "peak_rad": y[-1],
        }

    low = y0 + 0.1 * step
    high = y0 + 0.9 * step
    t10, t90 = None, None
    for ti, yi in zip(t, y):
        if t10 is None and ((step > 0 and yi >= low) or (step < 0 and yi <= low)):
            t10 = ti
        if t90 is None and ((step > 0 and yi >= high) or (step < 0 and yi <= high)):
            t90 = ti
            break
    rise_time = (t90 - t10) if (t10 is not None and t90 is not None) else float("nan")

    peak = max(y) if step > 0 else min(y)
    if step > 0:
        overshoot = max(0.0, ((peak - y_target) / amp) * 100.0)
    else:
        overshoot = max(0.0, ((y_target - peak) / amp) * 100.0)

    band = tol_ratio * amp
    settle_idx = None
    for i in range(len(y)):
        if all(abs(yy - y_target) <= band for yy in y[i:]):
            settle_idx = i
            break
    settling = t[settle_idx] if settle_idx is not None else float("nan")

    tail = y[max(0, int(0.8 * len(y))):]
    steady = sum(tail) / len(tail) if tail else y[-1]
    ss_err = steady - y_target

    return {
        "rise_time_s": rise_time,
        "settling_time_s": settling,
        "overshoot_pct": overshoot,
        "steady_state_err_rad": ss_err,
        "peak_rad": peak,
    }


def build_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "jandi"},
        {"name": "--resume", "action": "store_true", "default": False},
        {"name": "--experiment_name", "type": str},
        {"name": "--run_name", "type": str},
        {"name": "--load_run", "type": str},
        {"name": "--checkpoint", "type": int},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--horovod", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--num_envs", "type": int, "default": 1},
        {"name": "--seed", "type": int},
        {"name": "--max_iterations", "type": int},
        {"name": "--joint-name", "type": str, "default": "RL4_joint"},
        {"name": "--step-deg", "type": float, "default": 5.0},
        {"name": "--repeats", "type": int, "default": 5},
        {"name": "--warmup-s", "type": float, "default": 0.8},
        {"name": "--hold-s", "type": float, "default": 0.8},
        {"name": "--return-s", "type": float, "default": 0.8},
        {"name": "--sample-hz", "type": float, "default": 200.0},
        {"name": "--tol-ratio", "type": float, "default": 0.05},
        {"name": "--kp", "type": float},
        {"name": "--kd", "type": float},
        # Optional gains for all non-tested joints. This is useful for isolating one joint.
        {"name": "--lock-other-kp", "type": float},
        {"name": "--lock-other-kd", "type": float},
        {"name": "--fix-base", "action": "store_true", "default": False},
        {"name": "--disable-termination", "action": "store_true", "default": False},
        {"name": "--zero-gravity", "action": "store_true", "default": False},
        {"name": "--air-z", "type": float},
        {"name": "--log-dir", "type": str, "default": "/tmp"},
        {"name": "--tag", "type": str, "default": "sim_pd_step"},
    ]
    args = gymutil.parse_arguments(description="Sim PD step tester", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def main():
    args = build_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Deterministic single-env test setup
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.curriculum = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    if args.fix_base:
        env_cfg.asset.fix_base_link = True
    if args.zero_gravity:
        env_cfg.sim.gravity = [0.0, 0.0, 0.0]
        env_cfg.asset.disable_gravity = True
    if args.air_z is not None:
        env_cfg.init_state.pos[2] = float(args.air_z)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if args.disable_termination:
        def _no_termination():
            env.reset_buf[:] = 0
            env.time_out_buf[:] = 0
        env.check_termination = _no_termination

    env.reset()

    if args.air_z is not None:
        target_z = float(args.air_z)
        env.root_states[:, 2] = target_z
        env.root_states[:, 7:13] = 0.0
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))

    def hard_reset_to_nominal():
        # Reset dof states to default and zero velocity every repeat for comparable trials.
        env.dof_pos[:] = env.default_dof_pos
        env.dof_vel[:] = 0.0
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        if args.air_z is not None:
            env.root_states[:] = env.base_init_state
            env.root_states[:, :3] += env.env_origins
            env.root_states[:, 2] = float(args.air_z)
            env.root_states[:, 7:13] = 0.0
            env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        env.reset_buf[:] = 0
        env.time_out_buf[:] = 0
        env.episode_length_buf[:] = 0

    dof_id = {name: i for i, name in enumerate(env.dof_names)}
    if args.joint_name not in dof_id:
        raise ValueError(f"Unknown joint '{args.joint_name}'. Available: {env.dof_names}")
    j = dof_id[args.joint_name]

    # Tested joint gains.
    if args.kp is not None:
        env.p_gains[j] = float(args.kp)
    if args.kd is not None:
        env.d_gains[j] = float(args.kd)

    # Optional isolation: make all other joints hold their default pose more strongly.
    if args.lock_other_kp is not None or args.lock_other_kd is not None:
        for idx in range(env.num_dof):
            if idx == j:
                continue
            if args.lock_other_kp is not None:
                env.p_gains[idx] = float(args.lock_other_kp)
            if args.lock_other_kd is not None:
                env.d_gains[idx] = float(args.lock_other_kd)

    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.log_dir, f"{args.tag}_{ts}.csv")
    summary_path = os.path.join(args.log_dir, f"{args.tag}_{ts}_summary.csv")

    sim_hz = 1.0 / float(env.dt)
    sample_period = 1.0 / max(1e-6, float(args.sample_hz))
    step_rad = math.radians(args.step_deg)

    action_scale_vec = env.action_scale_vec.to(env.device)
    if action_scale_vec.dim() == 1:
        action_scale_vec = action_scale_vec.unsqueeze(0)
    if torch.any(torch.abs(action_scale_vec) < 1e-8):
        bad = torch.nonzero(torch.abs(action_scale_vec[0]) < 1e-8).flatten().tolist()
        raise ValueError(f"Action scale contains zero for DOF indices: {bad}")

    clip_actions = float(env.cfg.normalization.clip_actions)

    def target_to_action(target_pos: torch.Tensor) -> torch.Tensor:
        """Convert explicit joint position target to LeggedGym action.

        LeggedGym position PD normally uses:
            target_pos = default_dof_pos + action * action_scale
        So this function lets us command a desired target position directly while still
        using the environment's built-in PD torque computation.
        """
        action = (target_pos - env.default_dof_pos) / action_scale_vec
        return torch.clamp(action, -clip_actions, clip_actions)

    print(f"[info] joint={args.joint_name} idx={j}")
    print(
        f"[info] env.dt={env.dt:.6f}s sim_hz={sim_hz:.2f} "
        f"sample_hz_req={args.sample_hz:.2f} sample_period={sample_period:.4f}s"
    )
    print(f"[info] tested_joint_kp={float(env.p_gains[j]):.4f} tested_joint_kd={float(env.d_gains[j]):.4f}")
    print(f"[info] step_deg={args.step_deg:.3f} step_rad={step_rad:.6f}")
    print(f"[info] action_scale[{args.joint_name}]={float(action_scale_vec[0, j].item()):.6f}")
    if args.lock_other_kp is not None or args.lock_other_kd is not None:
        print(f"[info] other joints locked with kp={args.lock_other_kp} kd={args.lock_other_kd}")
    print(f"[info] csv={csv_path}")
    print(f"[info] summary={summary_path}")

    default_target = env.default_dof_pos.clone()

    def step_with_target(target_pos: torch.Tensor):
        actions = target_to_action(target_pos)
        obs, _, _, done, _ = env.step(actions)
        return obs, done, actions

    def run_phase(
        phase: str,
        repeat_idx: int,
        duration_s: float,
        target_pos: torch.Tensor,
        goal_pos_rad: float,
        seg_t: List[float],
        seg_y: List[float],
        err_buf: List[float],
        vel_buf: List[float],
        acc_buf: List[float],
        jerk_buf: List[float],
        writer,
    ) -> None:
        phase_steps = max(1, int(round(duration_s / env.dt)))
        prev_vel: Optional[float] = None
        prev_acc: Optional[float] = None
        next_sample_t = sample_period

        for k in range(phase_steps):
            obs, done, actions = step_with_target(target_pos)
            _ = obs

            if int(done[0].item()) == 1:
                print(f"[warn] env reset detected during phase={phase}, repeat={repeat_idx}")

            t = (k + 1) * env.dt
            if t + 1e-12 < next_sample_t:
                continue
            while next_sample_t <= t + 1e-12:
                next_sample_t += sample_period

            pos = float(env.dof_pos[0, j].item())
            vel = float(env.dof_vel[0, j].item())
            target_j = float(target_pos[0, j].item())
            action_j = float(actions[0, j].item())
            err = goal_pos_rad - pos
            acc = 0.0 if prev_vel is None else (vel - prev_vel) / env.dt
            jerk = 0.0 if prev_acc is None else (acc - prev_acc) / env.dt

            seg_t.append(t)
            seg_y.append(pos)
            err_buf.append(err)
            vel_buf.append(vel)
            if prev_vel is not None:
                acc_buf.append(acc)
            if prev_acc is not None:
                jerk_buf.append(jerk)

            writer.writerow(
                [
                    phase,
                    repeat_idx,
                    f"{t:.6f}",
                    args.joint_name,
                    f"{action_j:.6f}",
                    f"{target_j:.6f}",
                    f"{math.degrees(target_j):.6f}",
                    f"{pos:.6f}",
                    f"{math.degrees(pos):.6f}",
                    f"{vel:.6f}",
                    f"{math.degrees(vel):.6f}",
                    f"{acc:.6f}",
                    f"{math.degrees(acc):.6f}",
                    f"{jerk:.6f}",
                    f"{math.degrees(jerk):.6f}",
                    f"{err:.6f}",
                    f"{math.degrees(err):.6f}",
                ]
            )
            prev_vel = vel
            prev_acc = acc

    with open(csv_path, "w", newline="") as f, open(summary_path, "w", newline="") as sf:
        writer = csv.writer(f)
        swriter = csv.writer(sf)
        writer.writerow(
            [
                "phase",
                "repeat",
                "t",
                "joint",
                "action_cmd",
                "target_pos_rad",
                "target_pos_deg",
                "pos_rad",
                "pos_deg",
                "vel_rad_s",
                "vel_deg_s",
                "acc_rad_s2",
                "acc_deg_s2",
                "jerk_rad_s3",
                "jerk_deg_s3",
                "err_rad",
                "err_deg",
            ]
        )
        swriter.writerow(
            [
                "repeat",
                "phase",
                "joint",
                "rise_time_s",
                "settling_time_s",
                "overshoot_pct",
                "steady_state_err_rad",
                "steady_state_err_deg",
                "vel_rms_rad_s",
                "vel_rms_deg_s",
                "acc_rms_rad_s2",
                "acc_rms_deg_s2",
                "jerk_rms_rad_s3",
                "jerk_rms_deg_s3",
                "peak_abs_vel_rad_s",
                "peak_abs_vel_deg_s",
                "peak_abs_acc_rad_s2",
                "peak_abs_acc_deg_s2",
                "error_sign_flips",
            ]
        )

        # Warmup: hold every joint exactly at default target.
        warmup_steps = max(1, int(round(args.warmup_s / env.dt)))
        for _ in range(warmup_steps):
            step_with_target(default_target)

        for rep in range(args.repeats):
            hard_reset_to_nominal()
            for _ in range(2):
                step_with_target(default_target)

            # Use the actual current position after reset/settling as the center.
            # This mirrors the real motor script, which reads the present position as center.
            center_pos = float(env.dof_pos[0, j].item())
            goal_up = center_pos + step_rad
            goal_dn = center_pos

            target_up = env.default_dof_pos.clone()
            target_dn = env.default_dof_pos.clone()
            target_up[0, j] = goal_up
            target_dn[0, j] = goal_dn

            action_up = target_to_action(target_up)
            action_dn = target_to_action(target_dn)
            if abs(float(action_up[0, j].item())) >= clip_actions - 1e-6:
                print(f"[warn] step_up action for {args.joint_name} is clipped: {float(action_up[0, j].item()):.6f}")
            if abs(float(action_dn[0, j].item())) >= clip_actions - 1e-6:
                print(f"[warn] return action for {args.joint_name} is clipped: {float(action_dn[0, j].item()):.6f}")

            print(
                f"[debug] repeat={rep} center={math.degrees(center_pos):+.3f}deg "
                f"goal_up={math.degrees(goal_up):+.3f}deg action_up={float(action_up[0, j].item()):+.6f}"
            )

            # step_up
            t_buf: List[float] = []
            y_buf: List[float] = []
            err_buf: List[float] = []
            vel_buf: List[float] = []
            acc_buf: List[float] = []
            jerk_buf: List[float] = []

            run_phase(
                "step_up",
                rep,
                args.hold_s,
                target_up,
                goal_up,
                t_buf,
                y_buf,
                err_buf,
                vel_buf,
                acc_buf,
                jerk_buf,
                writer,
            )
            y0 = center_pos
            m = estimate_step_metrics(t_buf, y_buf, y0, goal_up, args.tol_ratio)
            print(
                f"[metrics] repeat={rep} phase=step_up "
                f"rise={m['rise_time_s']:.4f}s settle={m['settling_time_s']:.4f}s "
                f"overshoot={m['overshoot_pct']:.2f}% ss_err={math.degrees(m['steady_state_err_rad']):+.3f}deg "
                f"vel_rms={rms(vel_buf):.4f}rad/s ({math.degrees(rms(vel_buf)):.2f}deg/s) "
                f"acc_rms={rms(acc_buf):.4f}rad/s^2 ({math.degrees(rms(acc_buf)):.2f}deg/s^2)"
            )
            swriter.writerow(
                [
                    rep,
                    "step_up",
                    args.joint_name,
                    f"{m['rise_time_s']:.6f}",
                    f"{m['settling_time_s']:.6f}",
                    f"{m['overshoot_pct']:.6f}",
                    f"{m['steady_state_err_rad']:.6f}",
                    f"{math.degrees(m['steady_state_err_rad']):.6f}",
                    f"{rms(vel_buf):.6f}",
                    f"{math.degrees(rms(vel_buf)):.6f}",
                    f"{rms(acc_buf):.6f}",
                    f"{math.degrees(rms(acc_buf)):.6f}",
                    f"{rms(jerk_buf):.6f}",
                    f"{math.degrees(rms(jerk_buf)):.6f}",
                    f"{max([abs(v) for v in vel_buf], default=0.0):.6f}",
                    f"{math.degrees(max([abs(v) for v in vel_buf], default=0.0)):.6f}",
                    f"{max([abs(a) for a in acc_buf], default=0.0):.6f}",
                    f"{math.degrees(max([abs(a) for a in acc_buf], default=0.0)):.6f}",
                    count_error_sign_flips(err_buf),
                ]
            )

            # return
            t_buf = []
            y_buf = []
            err_buf = []
            vel_buf = []
            acc_buf = []
            jerk_buf = []
            run_phase(
                "return",
                rep,
                args.return_s,
                target_dn,
                goal_dn,
                t_buf,
                y_buf,
                err_buf,
                vel_buf,
                acc_buf,
                jerk_buf,
                writer,
            )
            y0 = goal_up
            m = estimate_step_metrics(t_buf, y_buf, y0, goal_dn, args.tol_ratio)
            print(
                f"[metrics] repeat={rep} phase=return "
                f"rise={m['rise_time_s']:.4f}s settle={m['settling_time_s']:.4f}s "
                f"overshoot={m['overshoot_pct']:.2f}% ss_err={math.degrees(m['steady_state_err_rad']):+.3f}deg "
                f"vel_rms={rms(vel_buf):.4f}rad/s ({math.degrees(rms(vel_buf)):.2f}deg/s) "
                f"acc_rms={rms(acc_buf):.4f}rad/s^2 ({math.degrees(rms(acc_buf)):.2f}deg/s^2)"
            )
            swriter.writerow(
                [
                    rep,
                    "return",
                    args.joint_name,
                    f"{m['rise_time_s']:.6f}",
                    f"{m['settling_time_s']:.6f}",
                    f"{m['overshoot_pct']:.6f}",
                    f"{m['steady_state_err_rad']:.6f}",
                    f"{math.degrees(m['steady_state_err_rad']):.6f}",
                    f"{rms(vel_buf):.6f}",
                    f"{math.degrees(rms(vel_buf)):.6f}",
                    f"{rms(acc_buf):.6f}",
                    f"{math.degrees(rms(acc_buf)):.6f}",
                    f"{rms(jerk_buf):.6f}",
                    f"{math.degrees(rms(jerk_buf)):.6f}",
                    f"{max([abs(v) for v in vel_buf], default=0.0):.6f}",
                    f"{math.degrees(max([abs(v) for v in vel_buf], default=0.0)):.6f}",
                    f"{max([abs(a) for a in acc_buf], default=0.0):.6f}",
                    f"{math.degrees(max([abs(a) for a in acc_buf], default=0.0)):.6f}",
                    count_error_sign_flips(err_buf),
                ]
            )

    print(f"[done] csv: {csv_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()