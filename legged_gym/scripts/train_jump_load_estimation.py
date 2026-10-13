#!/usr/bin/env python3
"""Resume a forward-jump policy with physical payload supervision enabled."""

import json
import os

import isaacgym  # noqa: F401

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def _apply_dict(cfg, values):
    for key, value in values.items():
        current = cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key, None)
        if isinstance(value, dict) and current is not None:
            _apply_dict(current, value)
        elif isinstance(cfg, dict):
            cfg[key] = value
        else:
            setattr(cfg, key, value)


def _restore_source_cfg(env_cfg, train_cfg, args):
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name
    )
    run_dir = os.path.join(log_root, str(args.load_run))
    for filename, cfg in (
        ("env_cfg.json", env_cfg),
        ("train_cfg.json", train_cfg),
    ):
        path = os.path.join(run_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required source config not found: {path}")
        with open(path) as stream:
            _apply_dict(cfg, json.load(stream))
    print(f"[jump-load-train] restored source config from {run_dir}")


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _restore_source_cfg(env_cfg, train_cfg, args)

    load_min = float(os.getenv("JUMP_LOAD_MIN_KG", "1.0"))
    load_max = float(os.getenv("JUMP_LOAD_MAX_KG", "6.0"))
    offset_xy = float(os.getenv("JUMP_LOAD_OFFSET_XY", "0.02"))
    load_mode = os.getenv("JUMP_LOAD_MODE", "attached")
    if load_mode not in ("attached", "free"):
        raise ValueError(f"Unsupported JUMP_LOAD_MODE={load_mode!r}")
    dr = env_cfg.domain_rand
    dr.add_random_load = load_mode == "free"
    dr.add_load_range = [load_min, load_max]
    dr.per_env_load_mass = load_mode == "free"
    dr.load_start_time_s = 0.5
    dr.load_duration_range_s = [3.0, 4.0]
    dr.load_interval_range_s = [4.5, 6.0]
    dr.load_contact_grace_s = 0.25
    dr.load_on_body_position_only = load_mode == "free"
    dr.load_drop_terminal = load_mode == "free"
    dr.load_offset_range_x = [-offset_xy, offset_xy]
    dr.load_offset_range_y = [-offset_xy, offset_xy]
    dr.load_offset_range_z = [0.10, 0.10]
    dr.attached_payload = load_mode == "attached"
    dr.attached_payload_mass_range = [load_min, load_max]
    dr.attached_payload_offset_range_x = [-offset_xy, offset_xy]
    dr.attached_payload_offset_range_y = [-offset_xy, offset_xy]
    dr.attached_payload_offset_range_z = [0.10, 0.10]

    train_cfg.runner.resume_noise_std = float(
        os.getenv("JUMPFWD_RESUME_NOISE_STD", "0.5")
    )
    train_cfg.runner.save_interval = int(os.getenv("JUMPFWD_SAVE_INTERVAL", "100"))

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    task_registry.save_cfgs(name=args.task)
    print(
        "[jump-load-train] payload "
        f"mode={load_mode}, {load_min:g}-{load_max:g} kg, "
        f"offset_xy=+/-{offset_xy:g} m, "
        f"resume_noise_std={train_cfg.runner.resume_noise_std:g}"
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    train(get_args())
