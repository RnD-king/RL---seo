from __future__ import annotations

import argparse
import math
import os
import re
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_SUMMARY_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Policy/mean_noise_std",
    "Episode/rew_tracking_lin_vel",
    "Episode/rew_tracking_ang_vel",
    "Episode/rew_contact",
    "Episode/rew_feet_swing_height",
    "Episode/rew_action_rate",
    "Episode/rew_dof_vel",
    "Episode/rew_dof_acc",
    "Episode/rew_torques",
    "Episode/rew_lin_vel_y",
    "Episode/rew_yaw_rate",
)


def safe_name(tag: str) -> str:
    name = tag.replace("/", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def find_run_dirs(path: Path) -> list[Path]:
    if any(path.glob("events.out.tfevents*")):
        return [path]

    return sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and any(child.glob("events.out.tfevents*"))
    )


def checkpoint_steps(run_dir: Path) -> list[int]:
    steps = []
    for model_path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", model_path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def load_scalars(run_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    event_acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    event_acc.Reload()

    scalars = {}
    for tag in event_acc.Tags().get("scalars", []):
        events = event_acc.Scalars(tag)
        if events:
            scalars[tag] = ([event.step for event in events], [event.value for event in events])
    return scalars


def trim_scalars_to_step(
    scalars: dict[str, tuple[list[int], list[float]]], max_step: int
) -> dict[str, tuple[list[int], list[float]]]:
    trimmed = {}
    for tag, (steps, values) in scalars.items():
        pairs = [(step, value) for step, value in zip(steps, values) if step <= max_step]
        if pairs:
            trimmed[tag] = ([step for step, _ in pairs], [value for _, value in pairs])
    return trimmed


def draw_checkpoint_lines(steps: list[int]) -> None:
    for step in steps:
        plt.axvline(step, color="0.85", linewidth=0.7, zorder=0)


def apply_y_axis_padding(ax, values: list[float], pad_ratio: float) -> None:
    if pad_ratio <= 0 or not values:
        return

    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return

    ymin = min(finite_values)
    ymax = max(finite_values)
    span = ymax - ymin
    if span == 0:
        pad = max(abs(ymin) * pad_ratio, 1e-3)
    else:
        pad = span * pad_ratio

    ax.set_ylim(ymin - pad, ymax + pad)


def export_single_tag(
    tag: str,
    steps: list[int],
    values: list[float],
    output_dir: Path,
    model_steps: list[int],
    dpi: int,
    y_pad_ratio: float,
) -> Path:
    save_path = output_dir / f"{safe_name(tag)}.png"

    plt.figure(figsize=(10, 4.8))
    draw_checkpoint_lines(model_steps)
    plt.plot(steps, values, linewidth=1.8)
    apply_y_axis_padding(plt.gca(), values, y_pad_ratio)
    plt.title(tag)
    plt.xlabel("iteration")
    plt.ylabel(tag)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    return save_path


def export_summary(
    scalars: dict[str, tuple[list[int], list[float]]],
    output_dir: Path,
    model_steps: list[int],
    dpi: int,
    summary_tags: tuple[str, ...],
    y_pad_ratio: float,
) -> Path | None:
    tags = [tag for tag in summary_tags if tag in scalars]
    if not tags:
        return None

    cols = 2
    rows = math.ceil(len(tags) / cols)
    save_path = output_dir / "reward_summary.png"

    fig, axes = plt.subplots(rows, cols, figsize=(14, max(4, rows * 3.2)))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, tag in zip(axes, tags):
        steps, values = scalars[tag]
        for step in model_steps:
            ax.axvline(step, color="0.88", linewidth=0.6, zorder=0)
        ax.plot(steps, values, linewidth=1.5)
        apply_y_axis_padding(ax, values, y_pad_ratio)
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(tags) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


def export_run(
    run_dir: Path,
    output_name: str,
    only_rewards: bool,
    dpi: int,
    trim_to_latest_checkpoint: bool = False,
    y_pad_ratio: float = 0.4,
) -> int:
    scalars = load_scalars(run_dir)
    if not scalars:
        print(f"[SKIP] no scalar data: {run_dir}")
        return 0

    output_dir = run_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_steps = checkpoint_steps(run_dir)
    if trim_to_latest_checkpoint and model_steps:
        max_step = max(model_steps)
        scalars = trim_scalars_to_step(scalars, max_step)
        model_steps = [step for step in model_steps if step <= max_step]
        print(f"[INFO] trimming plots to latest checkpoint: model_{max_step}.pt")

    tags = sorted(scalars)
    if only_rewards:
        tags = [
            tag
            for tag in tags
            if tag.startswith("Episode/rew_")
            or tag in {"Train/mean_reward", "Train/mean_episode_length"}
        ]

    for tag in tags:
        steps, values = scalars[tag]
        export_single_tag(tag, steps, values, output_dir, model_steps, dpi, y_pad_ratio)

    summary_path = export_summary(
        scalars,
        output_dir,
        model_steps,
        dpi,
        DEFAULT_SUMMARY_TAGS,
        y_pad_ratio,
    )
    print(f"[DONE] {run_dir}: {len(tags)} plots -> {output_dir}")
    if summary_path:
        print(f"       summary -> {summary_path}")
    return len(tags)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TensorBoard reward/scalar graphs next to policy checkpoint files."
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("logs/jandi_foot_up"),
        help="Run directory or experiment log root. Default: logs/jandi_foot_up",
    )
    parser.add_argument(
        "--output-name",
        default="reward_plots",
        help="Folder name created inside each run directory.",
    )
    parser.add_argument(
        "--only-rewards",
        action="store_true",
        help="Export only reward and mean episode graphs.",
    )
    parser.add_argument(
        "--trim-to-latest-checkpoint",
        action="store_true",
        help="Plot only scalar data up to the latest saved model_N.pt checkpoint.",
    )
    parser.add_argument(
        "--y-pad-ratio",
        type=float,
        default=0.4,
        help="Extra y-axis range as a fraction of data span. Default: 0.4",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.path.expanduser())
    if not run_dirs:
        raise SystemExit(f"No TensorBoard event files found under: {args.path}")

    total = 0
    for run_dir in run_dirs:
        total += export_run(
            run_dir,
            args.output_name,
            args.only_rewards,
            args.dpi,
            args.trim_to_latest_checkpoint,
            args.y_pad_ratio,
        )

    print(f"[OK] exported {total} plots from {len(run_dirs)} run folder(s)")


if __name__ == "__main__":
    main()
