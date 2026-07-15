import os
import sys

import isaacgym  # 반드시 torch, legged_gym보다 먼저

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger


EXPORT_POLICY = True
RECORD_FRAMES = False
MOVE_CAMERA = False

TORQUE_LOG_WINDOW = 100
TORQUE_LOG_INTERVAL = 100
ENABLE_PLAY_DEBUG_LOGS = True
FOOT_DISTANCE_LOG_INTERVAL = 100
JOINT_POS_LOG_EVERY_STEP = False

TORQUE_MODE_BIN = 0.1
TORQUE_EVENT_STEPS = 0

# effort limit이 5.5라면 90%, 98% 근처를 같이 봅니다.
SAT_RATIO_LOW = 0.90
SAT_RATIO_HIGH = 0.98


def summarize_play_state(env, actions, torque_samples, action_samples, dof_pos_samples, dof_vel_samples, step):
    if len(torque_samples) == 0:
        return

    torque_window = torch.cat(torque_samples, dim=0).abs()  # [window*num_envs, num_dof]
    action_window = torch.cat(action_samples, dim=0).abs()  # [window*num_envs, num_actions]
    dof_pos_window = torch.cat(dof_pos_samples, dim=0)
    dof_vel_window = torch.cat(dof_vel_samples, dim=0)

    torque_limits = env.torque_limits.detach().cpu()
    joint_names = env.dof_names
    default_dof_pos = env.default_dof_pos.detach().cpu().squeeze(0)

    max_abs = torch.max(torque_window, dim=0).values
    mean_abs = torch.mean(torque_window, dim=0)
    p95_abs = torch.quantile(torque_window, 0.95, dim=0)
    mean_pos = torch.mean(dof_pos_window, dim=0)
    mean_pos_err = mean_pos - default_dof_pos
    max_pos_err_abs = torch.max(torch.abs(dof_pos_window - default_dof_pos.unsqueeze(0)), dim=0).values
    mean_vel = torch.mean(dof_vel_window, dim=0)

    sat90_ratio = torch.mean(
        (torque_window >= torque_limits.unsqueeze(0) * SAT_RATIO_LOW).float(),
        dim=0,
    )
    sat98_ratio = torch.mean(
        (torque_window >= torque_limits.unsqueeze(0) * SAT_RATIO_HIGH).float(),
        dim=0,
    )

    mean_action_abs = torch.mean(action_window, dim=0)

    # ------------------------------------------------------------------
    # Action scale / target delta debug
    # ------------------------------------------------------------------
    raw_action_window = torch.cat(action_samples, dim=0)

    action_scale_vec = get_action_scale_vec_cpu(env)

    num_action_debug = min(
        raw_action_window.shape[1],
        action_scale_vec.numel(),
        len(joint_names),
        dof_pos_window.shape[1],
    )

    action_scale_vec = action_scale_vec[:num_action_debug]

    target_delta_window = (
        raw_action_window[:, :num_action_debug]
        * action_scale_vec.unsqueeze(0)
    )

    mean_target_delta_abs = torch.mean(torch.abs(target_delta_window), dim=0)
    max_target_delta_abs = torch.max(torch.abs(target_delta_window), dim=0).values

    target_pos_window = (
        default_dof_pos[:num_action_debug].unsqueeze(0)
        + target_delta_window
    )

    target_tracking_err_abs = torch.abs(
        target_pos_window - dof_pos_window[:, :num_action_debug]
    )

    mean_target_tracking_err_abs = torch.mean(target_tracking_err_abs, dim=0)

    action_sat_ratio = torch.mean(
        (torch.abs(raw_action_window[:, :num_action_debug]) > 0.90).float(),
        dim=0,
    )

    # base velocity / command 확인
    base_vx_mean = env.base_lin_vel[:, 0].mean().item()
    base_vy_mean = env.base_lin_vel[:, 1].mean().item()
    base_wz_mean = env.base_ang_vel[:, 2].mean().item()

    cmd_x_mean = env.commands[:, 0].mean().item()
    cmd_y_mean = env.commands[:, 1].mean().item()
    cmd_yaw_mean = env.commands[:, 2].mean().item()

    # base attitude
    if hasattr(env, "rpy"):
        roll_mean = env.rpy[:, 0].mean().item()
        pitch_mean = env.rpy[:, 1].mean().item()
        yaw_mean = env.rpy[:, 2].mean().item()
    else:
        roll_mean = 0.0
        pitch_mean = 0.0
        yaw_mean = 0.0

    # ------------------------------------------------------------------
    # Foot contact / foot order debug
    # ------------------------------------------------------------------
    foot0_fz_mean = 0.0
    foot1_fz_mean = 0.0
    total_fz_mean = 0.0
    contact_ratio = None

    ordered_right_contact = None
    ordered_left_contact = None
    ordered_right_z = None
    ordered_left_z = None
    ordered_right_fz = None
    ordered_left_fz = None
    right_idx = None
    left_idx = None

    if hasattr(env, "feet_indices"):
        # raw foot order 기준: env.feet_indices[0], env.feet_indices[1]
        foot_fz_gpu = env.contact_forces[:, env.feet_indices, 2]
        foot_fz = foot_fz_gpu.detach().cpu()

        if foot_fz.shape[1] >= 2:
            foot0_fz_mean = foot_fz[:, 0].mean().item()
            foot1_fz_mean = foot_fz[:, 1].mean().item()
            total_fz_mean = (foot_fz[:, 0] + foot_fz[:, 1]).mean().item()
            contact_ratio = torch.mean((foot_fz > 1.0).float(), dim=0)

        # right/left foot order 기준: env.feet_order_right_left
        if (
            hasattr(env, "feet_order_right_left")
            and hasattr(env, "feet_pos")
            and hasattr(env, "feet_num")
            and env.feet_num >= 2
        ):
            right_idx = int(env.feet_order_right_left[0].item())
            left_idx = int(env.feet_order_right_left[1].item())

            contact = foot_fz_gpu > 1.0

            ordered_right_contact = contact[:, right_idx].float().mean().item()
            ordered_left_contact = contact[:, left_idx].float().mean().item()

            ordered_right_z = env.feet_pos[:, right_idx, 2].mean().item()
            ordered_left_z = env.feet_pos[:, left_idx, 2].mean().item()

            ordered_right_fz = foot_fz_gpu[:, right_idx].mean().item()
            ordered_left_fz = foot_fz_gpu[:, left_idx].mean().item()

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"[play_summary] step={step}")
    print("-" * 90)
    print(
        f"cmd:  vx={cmd_x_mean:+.3f}, vy={cmd_y_mean:+.3f}, wz={cmd_yaw_mean:+.3f} | "
        f"base: vx={base_vx_mean:+.3f}, vy={base_vy_mean:+.3f}, wz={base_wz_mean:+.3f}"
    )
    print(
        f"base rpy mean: roll={roll_mean:+.4f} rad, "
        f"pitch={pitch_mean:+.4f} rad, yaw={yaw_mean:+.4f} rad"
    )
    print(
        f"foot Fz mean(raw order): foot0={foot0_fz_mean:+.2f} N, "
        f"foot1={foot1_fz_mean:+.2f} N, total={total_fz_mean:+.2f} N"
    )

    if contact_ratio is not None:
        contact_text = ", ".join(
            [
                f"foot{i}={contact_ratio[i].item() * 100:.1f}%"
                for i in range(contact_ratio.numel())
            ]
        )
        print(f"contact ratio(raw order): {contact_text}")

    if ordered_right_contact is not None:
        print(
            f"ordered feet(right/left): "
            f"right_idx={right_idx}, left_idx={left_idx}, "
            f"right_contact={ordered_right_contact * 100:.1f}%, "
            f"left_contact={ordered_left_contact * 100:.1f}%, "
            f"right_z={ordered_right_z:.4f} m, "
            f"left_z={ordered_left_z:.4f} m, "
            f"right_fz={ordered_right_fz:+.2f} N, "
            f"left_fz={ordered_left_fz:+.2f} N"
        )

    print("-" * 90)
    print(
        "joint             pos      def      err  max|err|  mean_vel  "
        "mean|tau|   max|tau|  sat90%  mean|act| act>90% "
        "mean|tgtΔ| max|tgtΔ| mean|tgt-pos|"
    )

    for idx, name in enumerate(joint_names):
        act_idx = idx if idx < mean_action_abs.numel() else None
        act_val = mean_action_abs[act_idx].item() if act_idx is not None else 0.0

        if idx < num_action_debug:
            act_sat = action_sat_ratio[idx].item() * 100.0
            mean_tgt_delta = mean_target_delta_abs[idx].item()
            max_tgt_delta = max_target_delta_abs[idx].item()
            mean_tgt_err = mean_target_tracking_err_abs[idx].item()
        else:
            act_sat = 0.0
            mean_tgt_delta = 0.0
            max_tgt_delta = 0.0
            mean_tgt_err = 0.0

        print(
            f"{name:12s} "
            f"{mean_pos[idx].item():+8.3f} "
            f"{default_dof_pos[idx].item():+8.3f} "
            f"{mean_pos_err[idx].item():+8.3f} "
            f"{max_pos_err_abs[idx].item():8.3f} "
            f"{mean_vel[idx].item():+9.3f} "
            f"{mean_abs[idx].item():9.3f} "
            f"{max_abs[idx].item():9.3f} "
            f"{sat90_ratio[idx].item() * 100:6.1f} "
            f"{act_val:9.3f}"
        )

    print("-" * 90)
    print("left/right joint symmetry (left + right should be near 0 for mirrored joints)")
    lr_pairs = [
        ("LL1_joint", "RL1_joint", +1.0),
        ("LL2_joint", "RL2_joint", +1.0),
        ("LL3_joint", "RL3_joint", +1.0),
        ("LL4_joint", "RL4_joint", +1.0),
        ("LL5_joint", "RL5_joint", +1.0),
        ("LL6_joint", "RL6_joint", -1.0),
    ]
    for left_name, right_name, right_sign in lr_pairs:
        if left_name not in joint_names or right_name not in joint_names:
            continue
        left_idx = joint_names.index(left_name)
        right_idx = joint_names.index(right_name)
        left_pos = mean_pos[left_idx].item()
        right_pos = mean_pos[right_idx].item()
        sym_err = left_pos + right_sign * right_pos
        print(
            f"{left_name}/{right_name}: "
            f"L={left_pos:+.3f}, R={right_pos:+.3f}, sym_err={sym_err:+.3f}"
        )

    print("=" * 90 + "\n")

def get_action_scale_vec_cpu(env):
    """
    현재 env에서 실제 action_scale_vec를 CPU tensor로 가져온다.

    목적:
    - action이 1.0까지 나와도 실제 목표각 변화량이 얼마나 되는지 확인
    - action_scale이 너무 작아서 발을 못 드는지 판단
    """

    if hasattr(env, "action_scale_vec"):
        return env.action_scale_vec.detach().cpu().view(-1)

    # fallback: action_scale_vec가 없는 경우 cfg에서 구성
    base_scale = getattr(env.cfg.control, "action_scale", 1.0)
    scale_vec = torch.ones(env.num_actions, dtype=torch.float) * float(base_scale)

    per_joint = getattr(env.cfg.control, "action_scale_per_joint", None)

    if isinstance(per_joint, dict):
        for joint_name, scale in per_joint.items():
            if joint_name in env.dof_names:
                idx = env.dof_names.index(joint_name)
                if idx < scale_vec.numel():
                    scale_vec[idx] = float(scale)

    return scale_vec

# def log_joint_positions(env, step):
#     dof_pos = env.dof_pos[0].detach().cpu()
#     joint_text = " ".join(
#         f"{name}={dof_pos[idx].item():+.3f}"
#         for idx, name in enumerate(env.dof_names)
#     )
#     print(f"[joint_pos] step={step:05d} {joint_text}")


# def log_torque_events(env, step, threshold_ratio=0.90):
#     abs_torque = torch.abs(env.torques.detach().cpu())
#     if abs_torque.numel() == 0:
#         return

#     torque_limits = env.torque_limits.detach().cpu()
#     threshold = torque_limits.unsqueeze(0) * threshold_ratio

#     over = abs_torque >= threshold
#     if not torch.any(over):
#         return

#     print(f"[torque_event] step={step:05d}, threshold={threshold_ratio * 100:.1f}% of limit")

#     for joint_idx in range(abs_torque.shape[1]):
#         env_ids = torch.nonzero(over[:, joint_idx], as_tuple=False).flatten()
#         if env_ids.numel() == 0:
#             continue

#         env_id = int(env_ids[0].item())
#         torque = float(env.torques[env_id, joint_idx].item())
#         pos = float(env.dof_pos[env_id, joint_idx].item())
#         vel = float(env.dof_vel[env_id, joint_idx].item())
#         limit = float(env.torque_limits[joint_idx].item())

#         print(
#             f"  {env.dof_names[joint_idx]:>12s} env={env_id:03d} "
#             f"tau={torque:+7.3f} |tau|={abs(torque):6.3f} "
#             f"limit={limit:5.2f} pos={pos:+7.3f} vel={vel:+7.3f}"
#         )


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # env_cfg.asset.file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/jandi/jandi.urdf"

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # 고정 명령으로 gait 확인
    # 제자리 발 들기/교대 발 들기 확인용으로 0 명령을 사용합니다.
    env_cfg.commands.resampling_time = 1e6
    env_cfg.commands.ranges.lin_vel_x = [0.02, 0.08]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.debug_action_scale = ENABLE_PLAY_DEBUG_LOGS
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to:", path)

    print("[INFO] DOF order")
    for i, name in enumerate(env.dof_names):
        print(f"{i:2d}: {name}, torque_limit={env.torque_limits[i].item():.3f} Nm")

    if hasattr(env, "feet_indices"):
        print(f"[INFO] feet_indices: {env.feet_indices}")

    if hasattr(env, "feet_order_right_left"):
        print(f"[INFO] feet_order_right_left: {env.feet_order_right_left}")

    torque_samples = []
    action_samples = []
    dof_pos_samples = []
    dof_vel_samples = []

    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if JOINT_POS_LOG_EVERY_STEP:
            log_joint_positions(env, i)

        if ENABLE_PLAY_DEBUG_LOGS:
            torque_samples.append(env.torques.detach().cpu())
            action_samples.append(actions.detach().cpu())
            dof_pos_samples.append(env.dof_pos.detach().cpu())
            dof_vel_samples.append(env.dof_vel.detach().cpu())

            if i < TORQUE_EVENT_STEPS:
                log_torque_events(env, i, threshold_ratio=SAT_RATIO_LOW)

            if len(torque_samples) > TORQUE_LOG_WINDOW:
                torque_samples.pop(0)

            if len(action_samples) > TORQUE_LOG_WINDOW:
                action_samples.pop(0)

            if len(dof_pos_samples) > TORQUE_LOG_WINDOW:
                dof_pos_samples.pop(0)

            if len(dof_vel_samples) > TORQUE_LOG_WINDOW:
                dof_vel_samples.pop(0)

            if (i + 1) % TORQUE_LOG_INTERVAL == 0:
                summarize_play_state(
                    env,
                    actions,
                    torque_samples,
                    action_samples,
                    dof_pos_samples,
                    dof_vel_samples,
                    step=i + 1,
                )

        if (
            FOOT_DISTANCE_LOG_INTERVAL > 0
            and (i + 1) % FOOT_DISTANCE_LOG_INTERVAL == 0
            and hasattr(env, "feet_order_right_left")
            and hasattr(env, "feet_pos")
            and hasattr(env, "feet_num")
            and env.feet_num >= 2
        ):
            right_idx = int(env.feet_order_right_left[0].item())
            left_idx = int(env.feet_order_right_left[1].item())

            right_y = env.feet_pos[:, right_idx, 1]
            left_y = env.feet_pos[:, left_idx, 1]

            foot_distance_y = torch.abs(left_y - right_y)

            print(
                f"foot lateral distance: "
                f"mean={foot_distance_y.mean().item():.4f} m, "
                f"min={foot_distance_y.min().item():.4f} m, "
                f"max={foot_distance_y.max().item():.4f} m"
            )


if __name__ == "__main__":
    args = get_args()
    play(args)
