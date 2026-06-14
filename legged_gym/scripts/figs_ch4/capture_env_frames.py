#!/usr/bin/env python3
"""Run play.py with viewer frame capture enabled.

Do not pass --headless: Isaac Gym needs a viewer to write screenshots.
Example:
  conda run -n pointfoot_legged_gym python legged_gym/scripts/figs_ch4/capture_env_frames.py \
    --task wheelfoot_flat \
    --load_run exper_qs_resi_load_boost_3_seed_45_pemass \
    --checkpoint 11000 \
    --load_mass_min 2 --load_mass_max 4 \
    --load_hold --exit_after_save
"""

from __future__ import annotations

import os
from pathlib import Path

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts import play as play_mod
from legged_gym.utils import task_registry


def ensure_frame_dirs(args) -> None:
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    if getattr(args, "experiment_name", None):
        train_cfg.runner.experiment_name = args.experiment_name

    experiment = train_cfg.runner.experiment_name
    root = Path(LEGGED_GYM_ROOT_DIR)

    # play.py currently writes frames to logs/<experiment>/exported/frames.
    # The task-qualified path is created too for humans looking next to runs.
    for frame_dir in (
        root / "logs" / experiment / "exported" / "frames",
        root / "logs" / args.task / experiment / "exported" / "frames",
    ):
        frame_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = play_mod.get_args()
    if bool(getattr(args, "headless", False)):
        raise SystemExit("Frame capture needs the Isaac Gym viewer; remove --headless.")

    ensure_frame_dirs(args)
    play_mod.EXPORT_POLICY = True
    play_mod.RECORD_FRAMES = True
    play_mod.MOVE_CAMERA = True
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    play_mod.play(args)


if __name__ == "__main__":
    main()
