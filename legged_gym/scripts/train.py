import os
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def export_reward_plots_after_training(ppo_runner):
    log_dir = getattr(ppo_runner, "log_dir", None)
    if log_dir is None:
        return

    try:
        writer = getattr(ppo_runner, "writer", None)
        if writer is not None:
            writer.flush()
            writer.close()

        from legged_gym.scripts.export_reward_plots import export_run

        print(f"Exporting reward plots to: {os.path.join(log_dir, 'reward_plots')}")
        export_run(
            run_dir=Path(log_dir).resolve(),
            output_name="reward_plots",
            only_rewards=True,
            dpi=180,
            trim_to_latest_checkpoint=True,
        )
    except Exception as exc:
        print(f"Could not export reward plots: {exc}")


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    interrupted = False
    try:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted. Exporting reward plots before exit...")
    finally:
        export_reward_plots_after_training(ppo_runner)

    if interrupted:
        print("Training stopped by user.")

if __name__ == '__main__':
    args = get_args()
    train(args)
