#!/usr/bin/env python3
"""Verify wheel_roll_sign: spin the wheels and check that v_roll (omega*r*sign) has the
SAME sign as the actual base forward velocity. If anti-correlated, wheel_roll_sign must flip."""
import os
import isaacgym  # noqa
from isaacgym.torch_utils import *  # noqa
from legged_gym.envs import *  # noqa
from legged_gym.utils import get_args, task_registry
import numpy as np
import torch


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 16
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.add_random_load = False
    env_cfg.domain_rand.push_robots = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, *_ = env.get_observations()
    n = env.num_envs
    act = torch.zeros(n, env.num_actions, device=env.device)
    # drive both wheels FORWARD-positive (wheel action indices 3 and 7), legs held at default (0)
    act[:, 3] = 3.0
    act[:, 7] = 3.0
    base_vx, v_roll = [], []
    for i in range(220):
        obs, *_ , = env.step(act)
        if i > 120:  # after it gets rolling
            base_vx.append(float(env.base_lin_vel[:, 0].mean()))
            v_roll.append(float(env.v_roll.mean()))
    bvx = np.mean(base_vx); vr = np.mean(v_roll)
    print(f"[sign] mean base_vx = {bvx:+.4f}   mean v_roll = {vr:+.4f}")
    print(f"[sign] same sign? {'YES -> wheel_roll_sign=+1.0 CORRECT' if bvx*vr > 0 else 'NO  -> FLIP wheel_roll_sign to -1.0'}")
    print(f"[sign] |v_roll|/|base_vx| ratio = {abs(vr)/max(abs(bvx),1e-6):.2f}  (should be ~1 if pure rolling)")


if __name__ == "__main__":
    main()
