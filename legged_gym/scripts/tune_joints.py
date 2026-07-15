import math

import isaacgym
import torch
from isaacgym import gymapi, gymtorch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def _zero_base_velocity(env):
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))


def _apply_pose_immediately(env):
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel[:] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


def _print_pose(env):
    print("-" * 72)
    for i, name in enumerate(env.dof_names):
        rad = float(env.default_dof_pos[0, i].item())
        deg = math.degrees(rad)
        print(f"{i:2d} {name:12s} : {rad:+.4f} rad ({deg:+7.2f} deg)")


def _print_selection_status(env, dof_id, side, joint_num):
    targets = _resolve_joint_ids(dof_id, side, joint_num)
    print("-" * 72)
    print(f"selected joint={joint_num}, side={side}")
    if len(targets) == 0:
        print("no matching joints")
        return
    for idx in targets:
        name = env.dof_names[idx]
        q = float(env.default_dof_pos[0, idx].item())
        q_deg = math.degrees(q)
        kp = float(env.p_gains[idx].item())
        kd = float(env.d_gains[idx].item())
        print(f"{name:12s} | q={q:+.4f} rad ({q_deg:+7.2f} deg) | kp={kp:+.4f} | kd={kd:+.4f}")


def _resolve_joint_ids(dof_id, side, joint_num):
    targets = []
    candidates = []

    if side in ("left", "both"):
        candidates.extend([
            f"LL{joint_num}_joint",
            f"LL_joint{joint_num}",
        ])
    if side in ("right", "both"):
        candidates.extend([
            f"RL{joint_num}_joint",
            f"RL_joint{joint_num}",
        ])

    seen = set()
    for name in candidates:
        if name in dof_id and name not in seen:
            targets.append(dof_id[name])
            seen.add(name)

    return targets


def _subscribe_keys(env):
    if env.viewer is None:
        return
    gym = env.gym
    viewer = env.viewer

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "joint_1")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "joint_2")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "joint_3")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_4, "joint_4")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_5, "joint_5")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_6, "joint_6")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_B, "side_both")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_L, "side_left")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "side_right")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "inc_small")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "dec_small")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "inc_big")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "dec_big")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "mode_angle")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_K, "mode_kp")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "mode_kd")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "print_pose")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_I, "print_status")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "snap_pose")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_F, "toggle_freeze")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "quit_tuner")


def tune_joints(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.fix_base_link = True
    env_cfg.asset.disable_gravity = True
    env_cfg.sim.gravity = [0.0, 0.0, 0.0]
    suspended_base_height = 1.4
    env_cfg.init_state.pos = [
        env_cfg.init_state.pos[0],
        env_cfg.init_state.pos[1],
        suspended_base_height,
    ]
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.viewer.pos = [2.0, -2.0, suspended_base_height + 0.3]
    env_cfg.viewer.lookat = [0.0, 0.0, suspended_base_height - 0.2]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _subscribe_keys(env)
    _zero_base_velocity(env)
    _apply_pose_immediately(env)

    dof_id = {name: i for i, name in enumerate(env.dof_names)}
    selected_joint = 3
    selected_side = "both"  # left | right | both
    step_small = 0.01
    step_big = 0.05
    edit_mode = "angle"  # angle | kp | kd
    freeze_pose = True

    print("[Joint Tuner]")
    print("1~6: joint select | B/L/R: both/left/right")
    print("A/K/D: edit angle/kp/kd")
    print("UP/DOWN: +/-%small | LEFT/RIGHT: -/+%big")
    print("  angle: small=0.01 rad, big=0.05 rad")
    print("  kp   : small=0.5,     big=2.0")
    print("  kd   : small=0.02,    big=0.1")
    print(f"base fixed in air at z={suspended_base_height:.2f} m")
    print("gravity disabled, pose freeze enabled")
    print("S: apply immediately | F: toggle freeze | I: print selected joint | P: print pose | Q: quit")
    _print_pose(env)
    _print_selection_status(env, dof_id, selected_side, selected_joint)

    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    running = True
    while running:
        if env.viewer is not None:
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.value <= 0:
                    continue
                action = evt.action

                if action.startswith("joint_"):
                    selected_joint = int(action.split("_")[1])
                    print(f"selected joint: {selected_joint}")
                    _print_selection_status(env, dof_id, selected_side, selected_joint)
                    continue
                if action == "side_both":
                    selected_side = "both"
                    print("selected side: both")
                    _print_selection_status(env, dof_id, selected_side, selected_joint)
                    continue
                if action == "side_left":
                    selected_side = "left"
                    print("selected side: left")
                    _print_selection_status(env, dof_id, selected_side, selected_joint)
                    continue
                if action == "side_right":
                    selected_side = "right"
                    print("selected side: right")
                    _print_selection_status(env, dof_id, selected_side, selected_joint)
                    continue
                if action == "mode_angle":
                    edit_mode = "angle"
                    print("edit mode: angle")
                    _print_selection_status(env, dof_id, selected_side, selected_joint)
                    continue
                if action == "mode_kp":
                    edit_mode = "kp"
                    print("edit mode: kp")
                    _print_selection_status(env, dof_id, selected_side, selected_joint)
                    continue
                if action == "mode_kd":
                    edit_mode = "kd"
                    print("edit mode: kd")
                    _print_selection_status(env, dof_id, selected_side, selected_joint)
                    continue
                if action == "print_pose":
                    _print_pose(env)
                    continue
                if action == "print_status":
                    _print_selection_status(env, dof_id, selected_side, selected_joint)
                    continue
                if action == "snap_pose":
                    _apply_pose_immediately(env)
                    _zero_base_velocity(env)
                    print("pose snapped")
                    _print_selection_status(env, dof_id, selected_side, selected_joint)
                    continue
                if action == "toggle_freeze":
                    freeze_pose = not freeze_pose
                    if freeze_pose:
                        _apply_pose_immediately(env)
                        _zero_base_velocity(env)
                    print(f"freeze_pose: {freeze_pose}")
                    continue
                if action == "quit_tuner":
                    running = False
                    break

                delta = 0.0
                if action == "inc_small":
                    delta = step_small
                elif action == "dec_small":
                    delta = -step_small
                elif action == "inc_big":
                    delta = step_big
                elif action == "dec_big":
                    delta = -step_big

                if delta == 0.0:
                    continue

                targets = _resolve_joint_ids(dof_id, selected_side, selected_joint)
                if len(targets) == 0:
                    print(f"joint not found: joint{selected_joint}, side={selected_side}")
                    continue

                if edit_mode == "kp":
                    delta = 0.5 if action in ("inc_small", "dec_small") else 2.0
                    if action in ("dec_small", "dec_big"):
                        delta = -delta
                elif edit_mode == "kd":
                    delta = 0.02 if action in ("inc_small", "dec_small") else 0.1
                    if action in ("dec_small", "dec_big"):
                        delta = -delta

                for idx in targets:
                    if edit_mode == "angle":
                        env.default_dof_pos[0, idx] += delta
                    elif edit_mode == "kp":
                        env.p_gains[idx] = torch.clamp(env.p_gains[idx] + delta, min=0.0)
                    elif edit_mode == "kd":
                        env.d_gains[idx] = torch.clamp(env.d_gains[idx] + delta, min=0.0)

                _apply_pose_immediately(env)
                _zero_base_velocity(env)
                _print_selection_status(env, dof_id, selected_side, selected_joint)

        _, _, _, dones, _ = env.step(zero_actions)
        if freeze_pose:
            _apply_pose_immediately(env)
            _zero_base_velocity(env)
        if torch.any(dones):
            _zero_base_velocity(env)
            _apply_pose_immediately(env)


if __name__ == "__main__":
    args = get_args()
    tune_joints(args)
