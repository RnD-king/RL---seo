import isaacgym
import importlib
import numpy as np
import os
import torch
import time
import re
from collections import deque
from isaacgym import gymapi, gymtorch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

TORQUE_NEAR_LIMIT_RATIO = 0.95
VEL_ABS_SPIKE_THRESHOLD = 5.0
VEL_DELTA_SPIKE_THRESHOLD = 3.0
# Saturation probe: raise only selected joints to check if stiffening disappears.
TORQUE_LIMIT_OVERRIDES = {
#     "LL3_joint": 20.0,
#     "RL3_joint": 20.0,
#     "LL4_joint": 20.0,
#     "RL4_joint": 20.0,
#     "LL5_joint": 20.0,
#     "RL5_joint": 20.0,
#     "LL6_joint": 20.0,
#     "RL6_joint": 20.0,
}


def _zero_base_velocity(env):
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))


def _apply_pose_immediately(env):
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel[:] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


def _rand_uniform(device, shape, low, high):
    return torch.empty(*shape, device=device).uniform_(low, high)


def _log_hand_kinematics(env, step, env_id=0):
    if not all(
        hasattr(env, name)
        for name in (
            "hand_pos_base",
            "hand_vel_base",
            "hand_pos_world",
            "hand_vel_world",
        )
    ):
        print(f"[hand_axis] step={step:05d} hand state not available")
        return

    hand_pos_base = env.hand_pos_base[env_id].detach().cpu().numpy()
    hand_vel_base = env.hand_vel_base[env_id].detach().cpu().numpy()
    hand_pos_world = env.hand_pos_world[env_id].detach().cpu().numpy()
    hand_vel_world = env.hand_vel_world[env_id].detach().cpu().numpy()
    actions = env.actions[env_id].detach().cpu().numpy() if hasattr(env, "actions") else None
    speed = float(np.linalg.norm(hand_vel_base))

    print(
        f"[hand_axis] step={step:05d} env={env_id:03d} | "
        f"base_pos=(x={hand_pos_base[0]:+.4f}, y={hand_pos_base[1]:+.4f}, z={hand_pos_base[2]:+.4f}) | "
        f"base_vel=(x={hand_vel_base[0]:+.4f}, y={hand_vel_base[1]:+.4f}, z={hand_vel_base[2]:+.4f}) | "
        f"speed={speed:.4f}"
    )
    if actions is not None:
        action_str = ", ".join(f"a{i}={value:+.3f}" for i, value in enumerate(actions))
        print(f"[hand_axis] step={step:05d} env={env_id:03d} | action=({action_str})")
    print(
        f"[hand_axis] step={step:05d} env={env_id:03d} | "
        f"world_pos=(x={hand_pos_world[0]:+.4f}, y={hand_pos_world[1]:+.4f}, z={hand_pos_world[2]:+.4f}) | "
        f"world_vel=(x={hand_vel_world[0]:+.4f}, y={hand_vel_world[1]:+.4f}, z={hand_vel_world[2]:+.4f})"
    )


def _make_axis_probe_actions(env, step, amplitude=0.6, segment_steps=200):
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    active_action = int((step // segment_steps) % env.num_actions)
    local_phase = (step % segment_steps) / float(segment_steps)
    actions[:, active_action] = float(amplitude) * np.sin(2.0 * np.pi * local_phase)
    return actions, active_action


def _log_foot_lateral_distance(env, step):
    """
    오른발/왼발 foot link 사이의 좌우 거리 출력.

    출력값:
    - world_y_distance:
        월드 좌표계 y축 기준 발 사이 거리
    - base_local_y_distance:
        base yaw 기준, 로봇 몸통 좌우축 기준 발 사이 거리
    """

    if (
        not hasattr(env, "feet_indices")
        or not hasattr(env, "feet_pos")
        or len(env.feet_indices) < 2
    ):
        print(f"[foot_distance] step={step:05d} feet state not available")
        return

    # right/left 순서를 이미 env에 저장해둔 경우 사용
    if hasattr(env, "feet_order_right_left"):
        right_idx = int(env.feet_order_right_left[0].item())
        left_idx = int(env.feet_order_right_left[1].item())
    else:
        # fallback: raw order 사용
        right_idx = 0
        left_idx = 1

    right_pos_world = env.feet_pos[:, right_idx, :]  # [num_envs, 3]
    left_pos_world = env.feet_pos[:, left_idx, :]    # [num_envs, 3]

    # 1) 월드 y 기준 발 사이 거리
    world_y_distance = torch.abs(
        left_pos_world[:, 1] - right_pos_world[:, 1]
    )

    # 2) base 기준 local y 발 사이 거리
    base_pos_world = env.root_states[:, 0:3]

    right_rel_world = right_pos_world - base_pos_world
    left_rel_world = left_pos_world - base_pos_world

    yaw = env.rpy[:, 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    right_y_local = -sin_yaw * right_rel_world[:, 0] + cos_yaw * right_rel_world[:, 1]
    left_y_local = -sin_yaw * left_rel_world[:, 0] + cos_yaw * left_rel_world[:, 1]

    base_local_y_distance = torch.abs(left_y_local - right_y_local)

    print(
        f"[foot_distance] step={step:05d} | "
        f"world_y={world_y_distance.mean().item():.4f} m | "
        f"base_local_y={base_local_y_distance.mean().item():.4f} m | "
        f"right_y_local={right_y_local.mean().item():+.4f} m | "
        f"left_y_local={left_y_local.mean().item():+.4f} m"
    )


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


def _log_base_heights(env, step):
    base_heights = env.root_states[:env.num_envs, 2].detach().cpu().numpy()
    heights_str = ", ".join(f"{h:.4f}" for h in base_heights)
    if np.allclose(base_heights.min(), base_heights.max()):
        dominant_range = f"[{base_heights[0]:.4f}, {base_heights[0]:.4f}]"
        dominant_count = len(base_heights)
    else:
        hist, edges = np.histogram(base_heights, bins=min(10, len(base_heights)))
        dominant_idx = int(np.argmax(hist))
        dominant_range = f"[{edges[dominant_idx]:.4f}, {edges[dominant_idx + 1]:.4f}]"
        dominant_count = int(hist[dominant_idx])
    print(
        f"[base_height] step={step:05d} "
        f"min={base_heights.min():.4f} "
        f"max={base_heights.max():.4f} "
        f"dominant_range={dominant_range} "
        f"count={dominant_count}/{len(base_heights)}"
    )
    print(f"[base_height] env_z=[{heights_str}]")


def _log_torque_stats(env, step):
    abs_torque = torch.abs(env.torques)
    mean_abs = torch.mean(abs_torque, dim=0).detach().cpu().numpy()
    max_abs = torch.max(abs_torque, dim=0).values.detach().cpu().numpy()
    sat_ratio = torch.mean(
        (abs_torque >= (env.torque_limits.unsqueeze(0) * 0.98)).float(),
        dim=0,
    ).detach().cpu().numpy()

    mean_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(env.dof_names, mean_abs))
    max_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(env.dof_names, max_abs))
    sat_str = ", ".join(f"{name}={val:.2f}" for name, val in zip(env.dof_names, sat_ratio))
    print(f"[torque] step={step:05d} mean_abs=[{mean_str}]")
    print(f"[torque] step={step:05d} max_abs=[{max_str}]")
    print(f"[torque] step={step:05d} sat_ratio=[{sat_str}]")


def _make_ankle_monitor(dof_id, window_size=100):
    ankle_names = ["LL5_joint", "LL6_joint", "RL5_joint", "RL6_joint"]
    indices = {name: dof_id[name] for name in ankle_names if name in dof_id}
    history = {name: deque(maxlen=window_size) for name in indices}
    return indices, history


def _update_ankle_monitor(env, ankle_indices, ankle_history):
    if env.num_envs == 0:
        return
    for name, idx in ankle_indices.items():
        torque = float(env.torques[0, idx].item())
        vel = float(env.dof_vel[0, idx].item())
        ankle_history[name].append((torque, vel))


def _log_ankle_monitor(step, ankle_indices, ankle_history):
    if not ankle_indices:
        return

    print(f"[ankle_lr] step={step:05d}")
    for left_name, right_name in (("LL5_joint", "RL5_joint"), ("LL6_joint", "RL6_joint")):
        if left_name not in ankle_indices or right_name not in ankle_indices:
            continue
        left_hist = list(ankle_history[left_name])
        right_hist = list(ankle_history[right_name])
        if not left_hist or not right_hist:
            continue

        left_torque = np.array([x[0] for x in left_hist], dtype=np.float64)
        right_torque = np.array([x[0] for x in right_hist], dtype=np.float64)
        left_vel = np.array([x[1] for x in left_hist], dtype=np.float64)
        right_vel = np.array([x[1] for x in right_hist], dtype=np.float64)

        left_rms_tau = float(np.sqrt(np.mean(np.square(left_torque))))
        right_rms_tau = float(np.sqrt(np.mean(np.square(right_torque))))
        left_rms_vel = float(np.sqrt(np.mean(np.square(left_vel))))
        right_rms_vel = float(np.sqrt(np.mean(np.square(right_vel))))

        print(
            f"  {left_name[:-6]} vs {right_name[:-6]} | "
            f"tau_now=({left_torque[-1]:+6.3f}, {right_torque[-1]:+6.3f}) "
            f"tau_rms=({left_rms_tau:6.3f}, {right_rms_tau:6.3f}) "
            f"vel_now=({left_vel[-1]:+6.3f}, {right_vel[-1]:+6.3f}) "
            f"vel_rms=({left_rms_vel:6.3f}, {right_rms_vel:6.3f}) "
            f"rms_ratio={(left_rms_tau / max(right_rms_tau, 1e-6)):5.2f}"
        )


def _make_lr_joint_pairs(dof_names):
    left_map = {}
    right_map = {}
    for idx, name in enumerate(dof_names):
        m_left = re.match(r"^LL(\d+)_joint$", name)
        m_right = re.match(r"^RL(\d+)_joint$", name)
        if m_left:
            left_map[int(m_left.group(1))] = idx
        if m_right:
            right_map[int(m_right.group(1))] = idx

    pairs = []
    for joint_num in sorted(set(left_map.keys()) & set(right_map.keys())):
        pairs.append((joint_num, left_map[joint_num], right_map[joint_num]))
    return pairs


def _make_lr_monitor(dof_names, window_size=100):
    pairs = _make_lr_joint_pairs(dof_names)
    history = {}
    for _, left_idx, right_idx in pairs:
        history[left_idx] = deque(maxlen=window_size)
        history[right_idx] = deque(maxlen=window_size)
    return pairs, history


def _update_lr_monitor(env, lr_pairs, lr_history):
    if env.num_envs == 0:
        return
    for _, left_idx, right_idx in lr_pairs:
        left_torque = float(env.torques[0, left_idx].item())
        right_torque = float(env.torques[0, right_idx].item())
        left_vel = float(env.dof_vel[0, left_idx].item())
        right_vel = float(env.dof_vel[0, right_idx].item())
        lr_history[left_idx].append((left_torque, left_vel))
        lr_history[right_idx].append((right_torque, right_vel))


def _log_lr_monitor(step, env, lr_pairs, lr_history):
    if not lr_pairs:
        return
    print(f"[joint_lr] step={step:05d}")
    for joint_num, left_idx, right_idx in lr_pairs:
        left_hist = list(lr_history[left_idx])
        right_hist = list(lr_history[right_idx])
        if not left_hist or not right_hist:
            continue

        left_torque = np.array([x[0] for x in left_hist], dtype=np.float64)
        right_torque = np.array([x[0] for x in right_hist], dtype=np.float64)
        left_vel = np.array([x[1] for x in left_hist], dtype=np.float64)
        right_vel = np.array([x[1] for x in right_hist], dtype=np.float64)

        left_rms_tau = float(np.sqrt(np.mean(np.square(left_torque))))
        right_rms_tau = float(np.sqrt(np.mean(np.square(right_torque))))
        left_rms_vel = float(np.sqrt(np.mean(np.square(left_vel))))
        right_rms_vel = float(np.sqrt(np.mean(np.square(right_vel))))

        tau_ratio = left_rms_tau / max(right_rms_tau, 1e-6)
        vel_ratio = left_rms_vel / max(right_rms_vel, 1e-6)
        print(
            f"  J{joint_num} | "
            f"tau_rms=({left_rms_tau:6.3f}, {right_rms_tau:6.3f}) "
            f"tau_ratio={tau_ratio:5.2f} "
            f"vel_rms=({left_rms_vel:6.3f}, {right_rms_vel:6.3f}) "
            f"vel_ratio={vel_ratio:5.2f}"
        )


def _format_joint_events(env, indices, abs_torque, abs_vel, vel_delta):
    events = []
    for idx in indices.tolist():
        name = env.dof_names[idx]
        events.append(
            f"{name}(|tau|={abs_torque[idx]:.2f}, limit={float(env.torque_limits[idx].item()):.2f}, "
            f"|vel|={abs_vel[idx]:.2f}, dvel={vel_delta[idx]:+.2f})"
        )
    return events


def _log_instability_events(env, step, prev_dof_vel):
    dof_vel = env.dof_vel[0].detach().cpu()
    abs_vel = torch.abs(dof_vel)
    vel_delta = torch.abs(dof_vel - prev_dof_vel)

    torque = env.torques[0].detach().cpu()
    abs_torque = torch.abs(torque)
    torque_limits = env.torque_limits.detach().cpu()
    near_limit = abs_torque >= (torque_limits * TORQUE_NEAR_LIMIT_RATIO)
    vel_spike = (abs_vel >= VEL_ABS_SPIKE_THRESHOLD) | (vel_delta >= VEL_DELTA_SPIKE_THRESHOLD)

    raw_abs_torque = None
    if getattr(env.cfg.control, "control_type", "P") == "P":
        # preview_pose drives zero actions, so commanded position is default_dof_pos.
        raw_torque = env.p_gains.detach().cpu() * (
            env.default_dof_pos[0].detach().cpu() - env.dof_pos[0].detach().cpu()
        ) - env.d_gains.detach().cpu() * env.dof_vel[0].detach().cpu()
        raw_abs_torque = torch.abs(raw_torque)

    near_limit_idx = torch.nonzero(near_limit, as_tuple=False).flatten()
    vel_spike_idx = torch.nonzero(vel_spike, as_tuple=False).flatten()
    if near_limit_idx.numel() == 0 and vel_spike_idx.numel() == 0:
        return dof_vel

    print(f"[instability] step={step:05d}")
    if near_limit_idx.numel() > 0:
        near_events = _format_joint_events(env, near_limit_idx, abs_torque, abs_vel, vel_delta)
        print(f"  near_limit: {', '.join(near_events)}")
        if raw_abs_torque is not None:
            extra = []
            for idx in near_limit_idx.tolist():
                name = env.dof_names[idx]
                over = float(raw_abs_torque[idx].item() - torque_limits[idx].item())
                extra.append(
                    f"{name}(raw={float(raw_abs_torque[idx].item()):.2f}, over={over:+.2f})"
                )
            print(f"  raw_demand: {', '.join(extra)}")
    if vel_spike_idx.numel() > 0:
        spike_events = _format_joint_events(env, vel_spike_idx, abs_torque, abs_vel, vel_delta)
        print(f"  vel_spike : {', '.join(spike_events)}")
    return dof_vel


def _compute_raw_torque(env):
    if getattr(env.cfg.control, "control_type", "P") != "P":
        return None
    return env.p_gains.detach().cpu() * (
        env.default_dof_pos[0].detach().cpu() - env.dof_pos[0].detach().cpu()
    ) - env.d_gains.detach().cpu() * env.dof_vel[0].detach().cpu()


def _apply_torque_limit_overrides(env, overrides):
    if not overrides:
        return
    dof_id = {name: i for i, name in enumerate(env.dof_names)}
    applied = []
    for name, new_limit in overrides.items():
        if name not in dof_id:
            continue
        idx = dof_id[name]
        old_limit = float(env.torque_limits[idx].item())
        env.torque_limits[idx] = float(new_limit)
        applied.append(f"{name}:{old_limit:.2f}->{float(new_limit):.2f}")
    if applied:
        print(f"[torque_limit_override] {' | '.join(applied)}")


def _apply_pd_gains_from_dict(env, stiffness, damping):
    applied = []
    for i, name in enumerate(env.dof_names):
        matched_key = name if name in stiffness else None
        if matched_key is None:
            for dof_name in stiffness.keys():
                if dof_name in name:
                    matched_key = dof_name
                    break
        if matched_key is None:
            continue
        if matched_key not in damping:
            print(f"[pd_hot_reload] damping missing for {matched_key}; skipped {name}")
            continue

        old_kp = float(env.p_gains[i].item())
        old_kd = float(env.d_gains[i].item())
        new_kp = float(stiffness[matched_key])
        new_kd = float(damping[matched_key])
        env.p_gains[i] = new_kp
        env.d_gains[i] = new_kd
        if abs(old_kp - new_kp) > 1e-6 or abs(old_kd - new_kd) > 1e-6:
            applied.append(f"{name}:kp {old_kp:.3f}->{new_kp:.3f}, kd {old_kd:.3f}->{new_kd:.3f}")

    env.cfg.control.stiffness = dict(stiffness)
    env.cfg.control.damping = dict(damping)
    return applied


def _apply_throw_config_from_class(env, cfg_cls):
    if not hasattr(cfg_cls, "throw"):
        return []

    applied = []
    if hasattr(cfg_cls.throw, "action_scale") and hasattr(env, "action_scale_vec"):
        old_scale = env.action_scale_vec.clone()
        new_scale = torch.full_like(env.action_scale_vec, float(cfg_cls.throw.action_scale))
        env.action_scale_vec[:] = new_scale
        env.cfg.throw.action_scale = float(cfg_cls.throw.action_scale)
        if torch.max(torch.abs(old_scale - new_scale)).item() > 1e-6:
            applied.append(f"throw action_scale -> {float(cfg_cls.throw.action_scale):.4f}")

    return applied


def _make_pd_config_watcher(args):
    if args.task in ("jandi", "jandi_walk"):
        import legged_gym.envs.jandi_walk.jandi_walk_config as config_module

        cfg_class_name = "JandiRobotWalkCfg"
    elif args.task == "throw":
        import legged_gym.envs.jandi_throw.jandi_throw_config as config_module

        cfg_class_name = "JandiThrowCfg"
    else:
        return None

    path = os.path.abspath(config_module.__file__)
    print(f"[pd_hot_reload] watching {path}")
    return {
        "module": config_module,
        "cfg_class_name": cfg_class_name,
        "path": path,
        "mtime": os.path.getmtime(path),
    }


def _reload_pd_config_if_changed(env, watcher, joint_noise, root_vel_noise):
    if watcher is None:
        return False

    try:
        mtime = os.path.getmtime(watcher["path"])
    except OSError as exc:
        print(f"[pd_hot_reload] cannot stat config: {exc}")
        return False

    if mtime <= watcher["mtime"]:
        return False

    # Give the editor a brief moment to finish writing the file.
    time.sleep(0.05)
    watcher["mtime"] = mtime

    try:
        module = importlib.reload(watcher["module"])
        watcher["module"] = module
        cfg_cls = getattr(module, watcher["cfg_class_name"])
        stiffness = cfg_cls.control.stiffness
        damping = cfg_cls.control.damping
        applied = _apply_pd_gains_from_dict(env, stiffness, damping)
        applied.extend(_apply_throw_config_from_class(env, cfg_cls))
    except Exception as exc:
        print(f"[pd_hot_reload] reload failed: {exc}")
        return False

    print(f"[pd_hot_reload] {os.path.basename(watcher['path'])} changed; config reloaded")
    if applied:
        for line in applied:
            print(f"  {line}")
    else:
        print("  no PD gain value changed")
    _reset_preview_state(env, joint_noise=joint_noise, root_vel_noise=root_vel_noise)
    print("[pd_hot_reload] robot reset after PD reload")
    return True


def _make_joint_demand_monitor(dof_id, names, window_size=300):
    monitor = {}
    for name in names:
        if name not in dof_id:
            continue
        monitor[name] = {
            "idx": dof_id[name],
            "raw_abs": deque(maxlen=window_size),
            "cmd_abs": deque(maxlen=window_size),
        }
    return monitor


def _update_joint_demand_monitor(env, monitor):
    if not monitor:
        return
    raw_torque = _compute_raw_torque(env)
    cmd_torque = env.torques[0].detach().cpu()
    for item in monitor.values():
        idx = item["idx"]
        if raw_torque is None:
            item["raw_abs"].append(abs(float(cmd_torque[idx].item())))
        else:
            item["raw_abs"].append(abs(float(raw_torque[idx].item())))
        item["cmd_abs"].append(abs(float(cmd_torque[idx].item())))


def _log_joint_demand_monitor(step, env, monitor):
    if not monitor:
        return
    # print(f"[joint_demand] step={step:05d}")
    for name in sorted(monitor.keys()):
        item = monitor[name]
        raw_hist = np.array(item["raw_abs"], dtype=np.float64)
        cmd_hist = np.array(item["cmd_abs"], dtype=np.float64)
        if raw_hist.size == 0 or cmd_hist.size == 0:
            continue
        # print(
        #     f"  {name:>10} | "
        #     f"raw_now={raw_hist[-1]:6.2f} raw_max={raw_hist.max():6.2f} raw_p95={np.percentile(raw_hist, 95):6.2f} "
        #     f"cmd_now={cmd_hist[-1]:6.2f} cmd_max={cmd_hist.max():6.2f} limit={float(env.torque_limits[item['idx']].item()):5.2f}"
        # )


def _resolve_joint_ids(dof_id, side, joint_num):
    targets = []
    candidates = []

    if side in ("left", "both"):
        candidates.extend([
            f"LL{joint_num}_joint",
            f"LL_joint{joint_num}",
            f"LH{joint_num}_joint",
            f"LH_joint{joint_num}",
        ])
    if side in ("right", "both"):
        candidates.extend([
            f"RL{joint_num}_joint",
            f"RL_joint{joint_num}",
            f"RH{joint_num}_joint",
            f"RH_joint{joint_num}",
        ])

    seen = set()
    for name in candidates:
        if name in dof_id and name not in seen:
            targets.append(dof_id[name])
            seen.add(name)
    return targets


def _bar(value, max_value, width=24):
    ratio = 0.0 if max_value <= 0.0 else max(0.0, min(float(value) / float(max_value), 1.0))
    filled = int(round(ratio * width))
    return "#" * filled + "-" * (width - filled)


def _print_gain_status(env, dof_id, selected_side, selected_joint, selected_mode):
    targets = _resolve_joint_ids(dof_id, selected_side, selected_joint)
    # print("-" * 88)
    # print(f"[pd_tuner] joint={selected_joint} side={selected_side} mode={selected_mode}")
    if len(targets) == 0:
        # print("[pd_tuner] no matching joints")
        return

    max_kp = max(1.0, float(torch.max(env.p_gains).item()) * 1.2)
    max_kd = max(1.0, float(torch.max(env.d_gains).item()) * 1.2)
    for idx in targets:
        name = env.dof_names[idx]
        kp = float(env.p_gains[idx].item())
        kd = float(env.d_gains[idx].item())
        kp_bar = _bar(kp, max_kp)
        kd_bar = _bar(kd, max_kd)
        # print(f"{name:12s} | kp={kp:7.3f} [{kp_bar}]")
        # print(f"{name:12s} | kd={kd:7.3f} [{kd_bar}]")


def _subscribe_tuner_keys(env):
    if env.viewer is None:
        return

    gym = env.gym
    viewer = env.viewer
    for key, action in (
        (gymapi.KEY_1, "joint_1"),
        (gymapi.KEY_2, "joint_2"),
        (gymapi.KEY_3, "joint_3"),
        (gymapi.KEY_4, "joint_4"),
        (gymapi.KEY_5, "joint_5"),
        (gymapi.KEY_6, "joint_6"),
        (gymapi.KEY_B, "side_both"),
        (gymapi.KEY_L, "side_left"),
        (gymapi.KEY_R, "side_right"),
        (gymapi.KEY_K, "mode_kp"),
        (gymapi.KEY_D, "mode_kd"),
        (gymapi.KEY_UP, "inc_small"),
        (gymapi.KEY_DOWN, "dec_small"),
        (gymapi.KEY_RIGHT, "inc_big"),
        (gymapi.KEY_LEFT, "dec_big"),
        (gymapi.KEY_I, "print_status"),
        (gymapi.KEY_S, "snap_pose"),
        (gymapi.KEY_H, "print_help"),
    ):
        gym.subscribe_viewer_keyboard_event(viewer, key, action)


def _print_tuner_help():
    print("[pd_tuner] 1~6: leg joint select, 1~4: arm joint select | B/L/R: both/left/right")
    print("[pd_tuner] K/D: edit Kp/Kd")
    print("[pd_tuner] UP/DOWN: +/- small | LEFT/RIGHT: -/+ big")
    print("[pd_tuner] Kp small=0.5 big=2.0 | Kd small=0.02 big=0.1")
    print("[pd_tuner] I: print current status | S: snap pose | H: help")
    print("[pd_hot_reload] save the active task config while preview is running to reload PD and reset")


def _handle_tuner_events(env, dof_id, tuner_state):
    if env.viewer is None:
        return

    for evt in env.gym.query_viewer_action_events(env.viewer):
        if evt.value <= 0:
            continue

        action = evt.action
        if action.startswith("joint_"):
            tuner_state["joint"] = int(action.split("_")[1])
            _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
            continue
        if action == "side_both":
            tuner_state["side"] = "both"
            _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
            continue
        if action == "side_left":
            tuner_state["side"] = "left"
            _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
            continue
        if action == "side_right":
            tuner_state["side"] = "right"
            _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
            continue
        if action == "mode_kp":
            tuner_state["mode"] = "kp"
            _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
            continue
        if action == "mode_kd":
            tuner_state["mode"] = "kd"
            _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
            continue
        if action == "print_status":
            _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
            continue
        if action == "print_help":
            _print_tuner_help()
            continue
        if action == "snap_pose":
            _reset_preview_state(
                env,
                joint_noise=tuner_state["joint_noise"],
                root_vel_noise=tuner_state["root_vel_noise"],
            )
            _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
            continue

        if action not in ("inc_small", "dec_small", "inc_big", "dec_big"):
            continue

        targets = _resolve_joint_ids(dof_id, tuner_state["side"], tuner_state["joint"])
        if len(targets) == 0:
            print(f"[pd_tuner] joint not found: joint{tuner_state['joint']} side={tuner_state['side']}")
            continue

        if tuner_state["mode"] == "kp":
            delta = 0.5 if action in ("inc_small", "dec_small") else 2.0
        else:
            delta = 0.02 if action in ("inc_small", "dec_small") else 0.1
        if action in ("dec_small", "dec_big"):
            delta = -delta

        for idx in targets:
            if tuner_state["mode"] == "kp":
                env.p_gains[idx] = torch.clamp(env.p_gains[idx] + delta, min=0.0)
            else:
                env.d_gains[idx] = torch.clamp(env.d_gains[idx] + delta, min=0.0)

        _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
        _reset_preview_state(
            env,
            joint_noise=tuner_state["joint_noise"],
            root_vel_noise=tuner_state["root_vel_noise"],
        )
        print("[pd_tuner] robot reset after PD change")


def preview_pose(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1 if args.num_envs is None else args.num_envs
    env_cfg.env.test = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    # Keep the task config's base mode. Throw pretraining intentionally fixes
    # the base, and overriding it here makes pose/PD previews misleading.
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.viewer.pos = [2.0, -2.0, 1.2]   # 카메라 위치 x,y,z
    env_cfg.viewer.lookat = [0.0, 0.0, 0.6] # 바라볼 점 x,y,z


    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _apply_torque_limit_overrides(env, TORQUE_LIMIT_OVERRIDES)
    _subscribe_tuner_keys(env)
    pd_config_watcher = _make_pd_config_watcher(args)
    obs = env.get_observations()

    _reset_preview_state(
        env,
        joint_noise=args.reset_joint_noise,
        root_vel_noise=args.reset_root_vel_noise,
    )
    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    dof_id = {name: i for i, name in enumerate(env.dof_names)}
    ankle_indices, ankle_history = _make_ankle_monitor(dof_id)
    lr_pairs, lr_history = _make_lr_monitor(env.dof_names)
    joint_demand_monitor = _make_joint_demand_monitor(
        dof_id,
        ["LL3_joint", "RL3_joint", "LH1_joint", "LH2_joint", "LH3_joint", "LH4_joint"],
        window_size=400,
    )
    tuner_state = {
        "joint": 3,
        "side": "both",
        "mode": "kp",
        "joint_noise": args.reset_joint_noise,
        "root_vel_noise": args.reset_root_vel_noise,
    }

    # _print_tuner_help()
    # print(
    #     f"[preview_pose] reset_joint_noise={args.reset_joint_noise:.4f} rad, "
    #     f"reset_root_vel_noise={args.reset_root_vel_noise:.4f}"
    # )
    # _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
    settle_steps = 50
    eval_steps = 150
    log_interval = 50
    both_contact_count = 0
    left_contact_count = 0
    right_contact_count = 0
    playback_slowdown = 2.0
    extra_sleep = env.dt * (playback_slowdown - 1.0)
    prev_dof_vel = env.dof_vel[0].detach().cpu().clone()
    has_foot_state = hasattr(env, "feet_indices") and hasattr(env, "feet_pos") and len(env.feet_indices) >= 2
    axis_probe_enabled = bool(getattr(args, "axis_probe", False)) and args.task == "throw"
    if axis_probe_enabled:
        print(
            f"[axis_probe] enabled amp={float(args.axis_probe_amp):.3f}; "
            "actions are swept one at a time in controlled_dof_names order"
        )

    for i in range(4000):
        _reload_pd_config_if_changed(
            env,
            pd_config_watcher,
            joint_noise=args.reset_joint_noise,
            root_vel_noise=args.reset_root_vel_noise,
        )
        _handle_tuner_events(env, dof_id, tuner_state)
        actions = zero_actions
        if axis_probe_enabled:
            actions, _ = _make_axis_probe_actions(env, i, amplitude=args.axis_probe_amp)
        obs, _, _, dones, _ = env.step(actions)
        if i % log_interval == 0:
            _log_hand_kinematics(env, i)
        if has_foot_state and i % log_interval == 0:
            _log_foot_lateral_distance(env, i)
            
        if extra_sleep > 0.0:
            time.sleep(extra_sleep)
        # prev_dof_vel = _log_instability_events(env, i, prev_dof_vel)
        # _update_ankle_monitor(env, ankle_indices, ankle_history)
        # _update_lr_monitor(env, lr_pairs, lr_history)
        # _update_joint_demand_monitor(env, joint_demand_monitor)
        # if i % log_interval == 0:
        #     _log_ankle_monitor(i, ankle_indices, ankle_history)
        #     _log_lr_monitor(i, env, lr_pairs, lr_history)
        #     _log_joint_demand_monitor(i, env, joint_demand_monitor)
        # if i % log_interval == 0:
        #     _log_base_heights(env, i)
        #     _log_torque_stats(env, i)

        # # Print raw contact status for the first ~2 sec.
        # if i < 100 and i % 10 == 0:
        #     fz = env.contact_forces[0, env.feet_indices, 2]
        #     contact = torch.abs(fz) > 5.0
        #     fz_str = ", ".join([f"{v.item():.2f}" for v in fz])
        #     c_str = ", ".join(["1" if v.item() else "0" for v in contact])
        #     print(f"step={i:03d} | Fz=[{fz_str}] | contact=[{c_str}]")

        # Contact statistics after settling.
        if has_foot_state and settle_steps <= i < settle_steps + eval_steps:
            fz = env.contact_forces[0, env.feet_indices, 2]
            contact = torch.abs(fz) > 5.0
            if len(contact) >= 2:
                left_contact_count += int(contact[0].item())
                right_contact_count += int(contact[1].item())
                both_contact_count += int((contact[0] and contact[1]).item())
            # if i == settle_steps + eval_steps - 1:
            #     print("[preview_pose] Contact summary (after settling):")
            #     print(f"  left contact ratio : {left_contact_count / eval_steps:.2f}")
            #     print(f"  right contact ratio: {right_contact_count / eval_steps:.2f}")
            #     print(f"  both contact ratio : {both_contact_count / eval_steps:.2f}")

        if torch.any(dones):
            done_ids = dones.nonzero(as_tuple=False).flatten()
            _reset_preview_state(
                env,
                env_ids=done_ids,
                joint_noise=args.reset_joint_noise,
                root_vel_noise=args.reset_root_vel_noise,
            )


if __name__ == "__main__":
    args = get_args()
    preview_pose(args)












# import isaacgym
# import numpy as np
# import torch
# import time
# import re
# from collections import deque
# from isaacgym import gymapi, gymtorch

# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry

# TORQUE_NEAR_LIMIT_RATIO = 0.95
# VEL_ABS_SPIKE_THRESHOLD = 5.0
# VEL_DELTA_SPIKE_THRESHOLD = 3.0
# # Saturation probe: raise only selected joints to check if stiffening disappears.
# TORQUE_LIMIT_OVERRIDES = {
# #     "LL3_joint": 20.0,
# #     "RL3_joint": 20.0,
# #     "LL4_joint": 20.0,
# #     "RL4_joint": 20.0,
# #     "LL5_joint": 20.0,
# #     "RL5_joint": 20.0,
# #     "LL6_joint": 20.0,
# #     "RL6_joint": 20.0,
# }


# def _zero_base_velocity(env):
#     env.root_states[:, 7:13] = 0.0
#     env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))


# def _apply_pose_immediately(env):
#     env.dof_pos[:] = env.default_dof_pos
#     env.dof_vel[:] = 0.0
#     env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


# def _rand_uniform(device, shape, low, high):
#     return torch.empty(*shape, device=device).uniform_(low, high)


# def _reset_preview_state(env, env_ids=None, joint_noise=0.0, root_vel_noise=0.0):
#     if env_ids is None:
#         env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
#     elif len(env_ids) == 0:
#         return

#     env.root_states[env_ids] = env.base_init_state
#     env.root_states[env_ids, :3] += env.env_origins[env_ids]
#     env.root_states[env_ids, 7:13] = 0.0
#     if root_vel_noise > 0.0:
#         env.root_states[env_ids, 7:13] = _rand_uniform(
#             env.device, (len(env_ids), 6), -root_vel_noise, root_vel_noise
#         )

#     q = env.default_dof_pos.repeat(len(env_ids), 1)
#     if joint_noise > 0.0:
#         q = q + _rand_uniform(
#             env.device, (len(env_ids), env.num_dof), -joint_noise, joint_noise
#         )
#     lower = env.dof_pos_limits_urdf[:, 0].unsqueeze(0)
#     upper = env.dof_pos_limits_urdf[:, 1].unsqueeze(0)
#     margin = 0.02
#     env.dof_pos[env_ids] = torch.clamp(q, lower + margin, upper - margin)
#     env.dof_vel[env_ids] = 0.0

#     env.actions[env_ids] = 0.0
#     env.last_actions[env_ids] = 0.0
#     env.last_dof_vel[env_ids] = 0.0
#     env.last_root_vel[env_ids] = 0.0
#     env.episode_length_buf[env_ids] = 0
#     env.reset_buf[env_ids] = 0

#     env_ids_int32 = env_ids.to(dtype=torch.int32)
#     env.gym.set_actor_root_state_tensor_indexed(
#         env.sim,
#         gymtorch.unwrap_tensor(env.root_states),
#         gymtorch.unwrap_tensor(env_ids_int32),
#         len(env_ids_int32),
#     )
#     env.gym.set_dof_state_tensor_indexed(
#         env.sim,
#         gymtorch.unwrap_tensor(env.dof_state),
#         gymtorch.unwrap_tensor(env_ids_int32),
#         len(env_ids_int32),
#     )


# def _log_base_heights(env, step):
#     base_heights = env.root_states[:env.num_envs, 2].detach().cpu().numpy()
#     heights_str = ", ".join(f"{h:.4f}" for h in base_heights)
#     if np.allclose(base_heights.min(), base_heights.max()):
#         dominant_range = f"[{base_heights[0]:.4f}, {base_heights[0]:.4f}]"
#         dominant_count = len(base_heights)
#     else:
#         hist, edges = np.histogram(base_heights, bins=min(10, len(base_heights)))
#         dominant_idx = int(np.argmax(hist))
#         dominant_range = f"[{edges[dominant_idx]:.4f}, {edges[dominant_idx + 1]:.4f}]"
#         dominant_count = int(hist[dominant_idx])
#     print(
#         f"[base_height] step={step:05d} "
#         f"min={base_heights.min():.4f} "
#         f"max={base_heights.max():.4f} "
#         f"dominant_range={dominant_range} "
#         f"count={dominant_count}/{len(base_heights)}"
#     )
#     print(f"[base_height] env_z=[{heights_str}]")


# def _log_torque_stats(env, step):
#     abs_torque = torch.abs(env.torques)
#     mean_abs = torch.mean(abs_torque, dim=0).detach().cpu().numpy()
#     max_abs = torch.max(abs_torque, dim=0).values.detach().cpu().numpy()
#     sat_ratio = torch.mean(
#         (abs_torque >= (env.torque_limits.unsqueeze(0) * 0.98)).float(),
#         dim=0,
#     ).detach().cpu().numpy()

#     mean_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(env.dof_names, mean_abs))
#     max_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(env.dof_names, max_abs))
#     sat_str = ", ".join(f"{name}={val:.2f}" for name, val in zip(env.dof_names, sat_ratio))
#     print(f"[torque] step={step:05d} mean_abs=[{mean_str}]")
#     print(f"[torque] step={step:05d} max_abs=[{max_str}]")
#     print(f"[torque] step={step:05d} sat_ratio=[{sat_str}]")


# def _make_ankle_monitor(dof_id, window_size=100):
#     ankle_names = ["LL5_joint", "LL6_joint", "RL5_joint", "RL6_joint"]
#     indices = {name: dof_id[name] for name in ankle_names if name in dof_id}
#     history = {name: deque(maxlen=window_size) for name in indices}
#     return indices, history


# def _update_ankle_monitor(env, ankle_indices, ankle_history):
#     if env.num_envs == 0:
#         return
#     for name, idx in ankle_indices.items():
#         torque = float(env.torques[0, idx].item())
#         vel = float(env.dof_vel[0, idx].item())
#         ankle_history[name].append((torque, vel))


# def _log_ankle_monitor(step, ankle_indices, ankle_history):
#     if not ankle_indices:
#         return

#     print(f"[ankle_lr] step={step:05d}")
#     for left_name, right_name in (("LL5_joint", "RL5_joint"), ("LL6_joint", "RL6_joint")):
#         if left_name not in ankle_indices or right_name not in ankle_indices:
#             continue
#         left_hist = list(ankle_history[left_name])
#         right_hist = list(ankle_history[right_name])
#         if not left_hist or not right_hist:
#             continue

#         left_torque = np.array([x[0] for x in left_hist], dtype=np.float64)
#         right_torque = np.array([x[0] for x in right_hist], dtype=np.float64)
#         left_vel = np.array([x[1] for x in left_hist], dtype=np.float64)
#         right_vel = np.array([x[1] for x in right_hist], dtype=np.float64)

#         left_rms_tau = float(np.sqrt(np.mean(np.square(left_torque))))
#         right_rms_tau = float(np.sqrt(np.mean(np.square(right_torque))))
#         left_rms_vel = float(np.sqrt(np.mean(np.square(left_vel))))
#         right_rms_vel = float(np.sqrt(np.mean(np.square(right_vel))))

#         print(
#             f"  {left_name[:-6]} vs {right_name[:-6]} | "
#             f"tau_now=({left_torque[-1]:+6.3f}, {right_torque[-1]:+6.3f}) "
#             f"tau_rms=({left_rms_tau:6.3f}, {right_rms_tau:6.3f}) "
#             f"vel_now=({left_vel[-1]:+6.3f}, {right_vel[-1]:+6.3f}) "
#             f"vel_rms=({left_rms_vel:6.3f}, {right_rms_vel:6.3f}) "
#             f"rms_ratio={(left_rms_tau / max(right_rms_tau, 1e-6)):5.2f}"
#         )


# def _make_lr_joint_pairs(dof_names):
#     left_map = {}
#     right_map = {}
#     for idx, name in enumerate(dof_names):
#         m_left = re.match(r"^LL(\d+)_joint$", name)
#         m_right = re.match(r"^RL(\d+)_joint$", name)
#         if m_left:
#             left_map[int(m_left.group(1))] = idx
#         if m_right:
#             right_map[int(m_right.group(1))] = idx

#     pairs = []
#     for joint_num in sorted(set(left_map.keys()) & set(right_map.keys())):
#         pairs.append((joint_num, left_map[joint_num], right_map[joint_num]))
#     return pairs


# def _make_lr_monitor(dof_names, window_size=100):
#     pairs = _make_lr_joint_pairs(dof_names)
#     history = {}
#     for _, left_idx, right_idx in pairs:
#         history[left_idx] = deque(maxlen=window_size)
#         history[right_idx] = deque(maxlen=window_size)
#     return pairs, history


# def _update_lr_monitor(env, lr_pairs, lr_history):
#     if env.num_envs == 0:
#         return
#     for _, left_idx, right_idx in lr_pairs:
#         left_torque = float(env.torques[0, left_idx].item())
#         right_torque = float(env.torques[0, right_idx].item())
#         left_vel = float(env.dof_vel[0, left_idx].item())
#         right_vel = float(env.dof_vel[0, right_idx].item())
#         lr_history[left_idx].append((left_torque, left_vel))
#         lr_history[right_idx].append((right_torque, right_vel))


# def _log_lr_monitor(step, env, lr_pairs, lr_history):
#     if not lr_pairs:
#         return
#     print(f"[joint_lr] step={step:05d}")
#     for joint_num, left_idx, right_idx in lr_pairs:
#         left_hist = list(lr_history[left_idx])
#         right_hist = list(lr_history[right_idx])
#         if not left_hist or not right_hist:
#             continue

#         left_torque = np.array([x[0] for x in left_hist], dtype=np.float64)
#         right_torque = np.array([x[0] for x in right_hist], dtype=np.float64)
#         left_vel = np.array([x[1] for x in left_hist], dtype=np.float64)
#         right_vel = np.array([x[1] for x in right_hist], dtype=np.float64)

#         left_rms_tau = float(np.sqrt(np.mean(np.square(left_torque))))
#         right_rms_tau = float(np.sqrt(np.mean(np.square(right_torque))))
#         left_rms_vel = float(np.sqrt(np.mean(np.square(left_vel))))
#         right_rms_vel = float(np.sqrt(np.mean(np.square(right_vel))))

#         tau_ratio = left_rms_tau / max(right_rms_tau, 1e-6)
#         vel_ratio = left_rms_vel / max(right_rms_vel, 1e-6)
#         print(
#             f"  J{joint_num} | "
#             f"tau_rms=({left_rms_tau:6.3f}, {right_rms_tau:6.3f}) "
#             f"tau_ratio={tau_ratio:5.2f} "
#             f"vel_rms=({left_rms_vel:6.3f}, {right_rms_vel:6.3f}) "
#             f"vel_ratio={vel_ratio:5.2f}"
#         )


# def _format_joint_events(env, indices, abs_torque, abs_vel, vel_delta):
#     events = []
#     for idx in indices.tolist():
#         name = env.dof_names[idx]
#         events.append(
#             f"{name}(|tau|={abs_torque[idx]:.2f}, limit={float(env.torque_limits[idx].item()):.2f}, "
#             f"|vel|={abs_vel[idx]:.2f}, dvel={vel_delta[idx]:+.2f})"
#         )
#     return events


# def _log_instability_events(env, step, prev_dof_vel):
#     dof_vel = env.dof_vel[0].detach().cpu()
#     abs_vel = torch.abs(dof_vel)
#     vel_delta = torch.abs(dof_vel - prev_dof_vel)

#     torque = env.torques[0].detach().cpu()
#     abs_torque = torch.abs(torque)
#     torque_limits = env.torque_limits.detach().cpu()
#     near_limit = abs_torque >= (torque_limits * TORQUE_NEAR_LIMIT_RATIO)
#     vel_spike = (abs_vel >= VEL_ABS_SPIKE_THRESHOLD) | (vel_delta >= VEL_DELTA_SPIKE_THRESHOLD)

#     raw_abs_torque = None
#     if getattr(env.cfg.control, "control_type", "P") == "P":
#         # preview_pose drives zero actions, so commanded position is default_dof_pos.
#         raw_torque = env.p_gains.detach().cpu() * (
#             env.default_dof_pos[0].detach().cpu() - env.dof_pos[0].detach().cpu()
#         ) - env.d_gains.detach().cpu() * env.dof_vel[0].detach().cpu()
#         raw_abs_torque = torch.abs(raw_torque)

#     near_limit_idx = torch.nonzero(near_limit, as_tuple=False).flatten()
#     vel_spike_idx = torch.nonzero(vel_spike, as_tuple=False).flatten()
#     if near_limit_idx.numel() == 0 and vel_spike_idx.numel() == 0:
#         return dof_vel

#     print(f"[instability] step={step:05d}")
#     if near_limit_idx.numel() > 0:
#         near_events = _format_joint_events(env, near_limit_idx, abs_torque, abs_vel, vel_delta)
#         print(f"  near_limit: {', '.join(near_events)}")
#         if raw_abs_torque is not None:
#             extra = []
#             for idx in near_limit_idx.tolist():
#                 name = env.dof_names[idx]
#                 over = float(raw_abs_torque[idx].item() - torque_limits[idx].item())
#                 extra.append(
#                     f"{name}(raw={float(raw_abs_torque[idx].item()):.2f}, over={over:+.2f})"
#                 )
#             print(f"  raw_demand: {', '.join(extra)}")
#     if vel_spike_idx.numel() > 0:
#         spike_events = _format_joint_events(env, vel_spike_idx, abs_torque, abs_vel, vel_delta)
#         print(f"  vel_spike : {', '.join(spike_events)}")
#     return dof_vel


# def _compute_raw_torque(env):
#     if getattr(env.cfg.control, "control_type", "P") != "P":
#         return None
#     return env.p_gains.detach().cpu() * (
#         env.default_dof_pos[0].detach().cpu() - env.dof_pos[0].detach().cpu()
#     ) - env.d_gains.detach().cpu() * env.dof_vel[0].detach().cpu()


# def _apply_torque_limit_overrides(env, overrides):
#     if not overrides:
#         return
#     dof_id = {name: i for i, name in enumerate(env.dof_names)}
#     applied = []
#     for name, new_limit in overrides.items():
#         if name not in dof_id:
#             continue
#         idx = dof_id[name]
#         old_limit = float(env.torque_limits[idx].item())
#         env.torque_limits[idx] = float(new_limit)
#         applied.append(f"{name}:{old_limit:.2f}->{float(new_limit):.2f}")
#     if applied:
#         print(f"[torque_limit_override] {' | '.join(applied)}")

# def _setup_pd_ramp(env):
#     """Save nominal PD gains for ramp-up."""
#     env.p_gains_nominal = env.p_gains.clone()
#     env.d_gains_nominal = env.d_gains.clone()


# def _apply_pd_ramp(env, local_step, ramp_time_sec=0.5, start_ratio=0.3):
#     """
#     Gradually increase PD gains after reset.

#     local_step:
#         reset 이후 몇 번째 step인지
#     ramp_time_sec:
#         몇 초 동안 ramp-up 할지
#     start_ratio:
#         시작 PD 비율. 0.3이면 nominal gain의 30%부터 시작
#     """
#     ramp_steps = max(1, int(ramp_time_sec / env.dt))

#     if local_step >= ramp_steps:
#         ratio = 1.0
#     else:
#         alpha = local_step / ramp_steps

#         # cosine ramp: 처음과 끝이 부드럽게 변함
#         smooth = 0.5 * (1.0 - np.cos(np.pi * alpha))
#         ratio = start_ratio + (1.0 - start_ratio) * smooth

#     env.p_gains[:] = env.p_gains_nominal * ratio
#     env.d_gains[:] = env.d_gains_nominal * ratio

#     return ratio


# def _make_joint_demand_monitor(dof_id, names, window_size=300):
#     monitor = {}
#     for name in names:
#         if name not in dof_id:
#             continue
#         monitor[name] = {
#             "idx": dof_id[name],
#             "raw_abs": deque(maxlen=window_size),
#             "cmd_abs": deque(maxlen=window_size),
#         }
#     return monitor


# def _update_joint_demand_monitor(env, monitor):
#     if not monitor:
#         return
#     raw_torque = _compute_raw_torque(env)
#     cmd_torque = env.torques[0].detach().cpu()
#     for item in monitor.values():
#         idx = item["idx"]
#         if raw_torque is None:
#             item["raw_abs"].append(abs(float(cmd_torque[idx].item())))
#         else:
#             item["raw_abs"].append(abs(float(raw_torque[idx].item())))
#         item["cmd_abs"].append(abs(float(cmd_torque[idx].item())))


# def _log_joint_demand_monitor(step, env, monitor):
#     if not monitor:
#         return
#     print(f"[joint_demand] step={step:05d}")
#     for name in sorted(monitor.keys()):
#         item = monitor[name]
#         raw_hist = np.array(item["raw_abs"], dtype=np.float64)
#         cmd_hist = np.array(item["cmd_abs"], dtype=np.float64)
#         if raw_hist.size == 0 or cmd_hist.size == 0:
#             continue
#         print(
#             f"  {name:>10} | "
#             f"raw_now={raw_hist[-1]:6.2f} raw_max={raw_hist.max():6.2f} raw_p95={np.percentile(raw_hist, 95):6.2f} "
#             f"cmd_now={cmd_hist[-1]:6.2f} cmd_max={cmd_hist.max():6.2f} limit={float(env.torque_limits[item['idx']].item()):5.2f}"
#         )


# def _resolve_joint_ids(dof_id, side, joint_num):
#     targets = []
#     candidates = []

#     if side in ("left", "both"):
#         candidates.extend([
#             f"LL{joint_num}_joint",
#             f"LL_joint{joint_num}",
#         ])
#     if side in ("right", "both"):
#         candidates.extend([
#             f"RL{joint_num}_joint",
#             f"RL_joint{joint_num}",
#         ])

#     seen = set()
#     for name in candidates:
#         if name in dof_id and name not in seen:
#             targets.append(dof_id[name])
#             seen.add(name)
#     return targets


# def _bar(value, max_value, width=24):
#     ratio = 0.0 if max_value <= 0.0 else max(0.0, min(float(value) / float(max_value), 1.0))
#     filled = int(round(ratio * width))
#     return "#" * filled + "-" * (width - filled)


# def _print_gain_status(env, dof_id, selected_side, selected_joint, selected_mode):
#     targets = _resolve_joint_ids(dof_id, selected_side, selected_joint)
#     print("-" * 88)
#     print(f"[pd_tuner] joint={selected_joint} side={selected_side} mode={selected_mode}")
#     if len(targets) == 0:
#         print("[pd_tuner] no matching joints")
#         return

#     max_kp = max(1.0, float(torch.max(env.p_gains).item()) * 1.2)
#     max_kd = max(1.0, float(torch.max(env.d_gains).item()) * 1.2)
#     for idx in targets:
#         name = env.dof_names[idx]
#         kp = float(env.p_gains[idx].item())
#         kd = float(env.d_gains[idx].item())
#         kp_bar = _bar(kp, max_kp)
#         kd_bar = _bar(kd, max_kd)
#         print(f"{name:12s} | kp={kp:7.3f} [{kp_bar}]")
#         print(f"{name:12s} | kd={kd:7.3f} [{kd_bar}]")


# def _subscribe_tuner_keys(env):
#     if env.viewer is None:
#         return

#     gym = env.gym
#     viewer = env.viewer
#     for key, action in (
#         (gymapi.KEY_1, "joint_1"),
#         (gymapi.KEY_2, "joint_2"),
#         (gymapi.KEY_3, "joint_3"),
#         (gymapi.KEY_4, "joint_4"),
#         (gymapi.KEY_5, "joint_5"),
#         (gymapi.KEY_6, "joint_6"),
#         (gymapi.KEY_B, "side_both"),
#         (gymapi.KEY_L, "side_left"),
#         (gymapi.KEY_R, "side_right"),
#         (gymapi.KEY_K, "mode_kp"),
#         (gymapi.KEY_D, "mode_kd"),
#         (gymapi.KEY_UP, "inc_small"),
#         (gymapi.KEY_DOWN, "dec_small"),
#         (gymapi.KEY_RIGHT, "inc_big"),
#         (gymapi.KEY_LEFT, "dec_big"),
#         (gymapi.KEY_I, "print_status"),
#         (gymapi.KEY_S, "snap_pose"),
#         (gymapi.KEY_H, "print_help"),
#     ):
#         gym.subscribe_viewer_keyboard_event(viewer, key, action)


# def _print_tuner_help():
#     print("[pd_tuner] 1~6: joint select | B/L/R: both/left/right")
#     print("[pd_tuner] K/D: edit Kp/Kd")
#     print("[pd_tuner] UP/DOWN: +/- small | LEFT/RIGHT: -/+ big")
#     print("[pd_tuner] Kp small=0.5 big=2.0 | Kd small=0.02 big=0.1")
#     print("[pd_tuner] I: print current status | S: snap pose | H: help")


# def _handle_tuner_events(env, dof_id, tuner_state):
#     if env.viewer is None:
#         return

#     for evt in env.gym.query_viewer_action_events(env.viewer):
#         if evt.value <= 0:
#             continue

#         action = evt.action
#         if action.startswith("joint_"):
#             tuner_state["joint"] = int(action.split("_")[1])
#             _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
#             continue
#         if action == "side_both":
#             tuner_state["side"] = "both"
#             _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
#             continue
#         if action == "side_left":
#             tuner_state["side"] = "left"
#             _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
#             continue
#         if action == "side_right":
#             tuner_state["side"] = "right"
#             _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
#             continue
#         if action == "mode_kp":
#             tuner_state["mode"] = "kp"
#             _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
#             continue
#         if action == "mode_kd":
#             tuner_state["mode"] = "kd"
#             _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
#             continue
#         if action == "print_status":
#             _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
#             continue
#         if action == "print_help":
#             _print_tuner_help()
#             continue
#         if action == "snap_pose":
#             _reset_preview_state(
#                 env,
#                 joint_noise=tuner_state["joint_noise"],
#                 root_vel_noise=tuner_state["root_vel_noise"],
#             )
#             _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
#             continue

#         if action not in ("inc_small", "dec_small", "inc_big", "dec_big"):
#             continue

#         targets = _resolve_joint_ids(dof_id, tuner_state["side"], tuner_state["joint"])
#         if len(targets) == 0:
#             print(f"[pd_tuner] joint not found: joint{tuner_state['joint']} side={tuner_state['side']}")
#             continue

#         if tuner_state["mode"] == "kp":
#             delta = 0.5 if action in ("inc_small", "dec_small") else 2.0
#         else:
#             delta = 0.02 if action in ("inc_small", "dec_small") else 0.1
#         if action in ("dec_small", "dec_big"):
#             delta = -delta

#         for idx in targets:
#             if tuner_state["mode"] == "kp":
#                 env.p_gains[idx] = torch.clamp(env.p_gains[idx] + delta, min=0.0)
#             else:
#                 env.d_gains[idx] = torch.clamp(env.d_gains[idx] + delta, min=0.0)

#         _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])


# def preview_pose(args):
#     env_cfg, _ = task_registry.get_cfgs(name=args.task)
#     env_cfg.env.num_envs = 1 if args.num_envs is None else args.num_envs
#     env_cfg.env.test = True
#     env_cfg.noise.add_noise = False
#     env_cfg.domain_rand.randomize_friction = False
#     env_cfg.domain_rand.randomize_base_mass = False
#     env_cfg.domain_rand.push_robots = False
#     env_cfg.asset.fix_base_link = False
#     env_cfg.commands.curriculum = False
#     env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
#     env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
#     env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
#     env_cfg.viewer.pos = [2.0, -2.0, 1.2]   # 카메라 위치 x,y,z
#     env_cfg.viewer.lookat = [0.0, 0.0, 0.6] # 바라볼 점 x,y,z


#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#     _apply_torque_limit_overrides(env, TORQUE_LIMIT_OVERRIDES)

#     # PD ramp-up용 nominal gain 저장
#     _setup_pd_ramp(env)

#     _subscribe_tuner_keys(env)
#     obs = env.get_observations()

#     _reset_preview_state(
#         env,
#         joint_noise=args.reset_joint_noise,
#         root_vel_noise=args.reset_root_vel_noise,
#     )
#     zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
#     dof_id = {name: i for i, name in enumerate(env.dof_names)}
#     ankle_indices, ankle_history = _make_ankle_monitor(dof_id)
#     lr_pairs, lr_history = _make_lr_monitor(env.dof_names)
#     joint_demand_monitor = _make_joint_demand_monitor(
#         dof_id, ["LL3_joint", "RL3_joint"], window_size=400
#     )
#     tuner_state = {
#         "joint": 3,
#         "side": "both",
#         "mode": "kp",
#         "joint_noise": args.reset_joint_noise,
#         "root_vel_noise": args.reset_root_vel_noise,
#     }

#     _print_tuner_help()
#     print(
#         f"[preview_pose] reset_joint_noise={args.reset_joint_noise:.4f} rad, "
#         f"reset_root_vel_noise={args.reset_root_vel_noise:.4f}"
#     )
#     _print_gain_status(env, dof_id, tuner_state["side"], tuner_state["joint"], tuner_state["mode"])
#     settle_steps = 50
#     eval_steps = 150
#     log_interval = 50
#     both_contact_count = 0
#     left_contact_count = 0
#     right_contact_count = 0
#     playback_slowdown = 2.0
#     extra_sleep = env.dt * (playback_slowdown - 1.0)
#     prev_dof_vel = env.dof_vel[0].detach().cpu().clone()

#     ramp_start_step = 0

#     for i in range(4000):
#         _handle_tuner_events(env, dof_id, tuner_state)

#         # reset 이후 local step 기준으로 PD를 30% → 100% ramp-up
#         local_ramp_step = i - ramp_start_step
#         pd_ratio = _apply_pd_ramp(
#             env,
#             local_ramp_step,
#             ramp_time_sec=0.5,
#             start_ratio=0.3,
#         )

#         obs, _, _, dones, _ = env.step(zero_actions)

#         if i % 50 == 0:
#             print(f"[pd_ramp] step={i:05d} local={local_ramp_step:04d} ratio={pd_ratio:.3f}")
#         if extra_sleep > 0.0:
#             time.sleep(extra_sleep)
#         prev_dof_vel = _log_instability_events(env, i, prev_dof_vel)
#         _update_ankle_monitor(env, ankle_indices, ankle_history)
#         _update_lr_monitor(env, lr_pairs, lr_history)
#         _update_joint_demand_monitor(env, joint_demand_monitor)
#         if i % log_interval == 0:
#             _log_ankle_monitor(i, ankle_indices, ankle_history)
#             _log_lr_monitor(i, env, lr_pairs, lr_history)
#             _log_joint_demand_monitor(i, env, joint_demand_monitor)
#         # if i % log_interval == 0:
#         #     _log_base_heights(env, i)
#         #     _log_torque_stats(env, i)

#         # # Print raw contact status for the first ~2 sec.
#         # if i < 100 and i % 10 == 0:
#         #     fz = env.contact_forces[0, env.feet_indices, 2]
#         #     contact = torch.abs(fz) > 5.0
#         #     fz_str = ", ".join([f"{v.item():.2f}" for v in fz])
#         #     c_str = ", ".join(["1" if v.item() else "0" for v in contact])
#         #     print(f"step={i:03d} | Fz=[{fz_str}] | contact=[{c_str}]")

#         # Contact statistics after settling.
#         if settle_steps <= i < settle_steps + eval_steps:
#             fz = env.contact_forces[0, env.feet_indices, 2]
#             contact = torch.abs(fz) > 5.0
#             if len(contact) >= 2:
#                 left_contact_count += int(contact[0].item())
#                 right_contact_count += int(contact[1].item())
#                 both_contact_count += int((contact[0] and contact[1]).item())
#             # if i == settle_steps + eval_steps - 1:
#             #     print("[preview_pose] Contact summary (after settling):")
#             #     print(f"  left contact ratio : {left_contact_count / eval_steps:.2f}")
#             #     print(f"  right contact ratio: {right_contact_count / eval_steps:.2f}")
#             #     print(f"  both contact ratio : {both_contact_count / eval_steps:.2f}")

#         if torch.any(dones):
#             done_ids = dones.nonzero(as_tuple=False).flatten()
#             _reset_preview_state(
#                 env,
#                 env_ids=done_ids,
#                 joint_noise=args.reset_joint_noise,
#                 root_vel_noise=args.reset_root_vel_noise,
#             )

#             # reset 후 다시 30%부터 PD ramp-up 시작
#             ramp_start_step = i


# if __name__ == "__main__":
#     args = get_args()
#     preview_pose(args)
