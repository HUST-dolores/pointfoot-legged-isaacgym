"""Quick smoke test: load WF env, query each env's load actor mass via Isaac Gym.

If domain_rand.per_env_load_mass=True: 20 envs should show ~20 distinct masses in [2,4]
If False:                              all 20 envs show the same mass (legacy mode)
"""
import os
import sys

sys.path.insert(0, "/home/xu/limx_rl/pointfoot-legged-gym")

import isaacgym  # noqa
from isaacgym.torch_utils import *  # noqa
import numpy as np
import torch

from legged_gym.envs import *  # noqa  (registers tasks)
from legged_gym.utils import get_args, task_registry

if __name__ == "__main__":
    args = get_args()
    args.headless = True
    args.num_envs = 20

    env_cfg, _ = task_registry.get_cfgs(name="wheelfoot_flat")
    env_cfg.domain_rand.per_env_load_mass = True
    env_cfg.env.num_envs = 20
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.mesh_type = "plane"  # fast init

    env, _ = task_registry.make_env(name="wheelfoot_flat", args=args, env_cfg=env_cfg)

    # 1) Stored label list
    init_list = env.load_mass_per_env_init
    init_arr = np.array(init_list)
    print(f"\n========== load_mass_per_env_init (stored labels) ==========")
    print(f"  count   = {len(init_list)}")
    print(f"  min/max = {init_arr.min():.4f} / {init_arr.max():.4f}")
    print(f"  mean    = {init_arr.mean():.4f}")
    print(f"  std     = {init_arr.std():.4f}")
    print(f"  values  = {init_arr.tolist()}")

    # 2) Per-env load_mass_per_env tensor
    tensor_vals = env.load_mass_per_env.detach().cpu().numpy()
    print(f"\n========== self.load_mass_per_env (tensor used for truth) ==========")
    print(f"  min/max = {tensor_vals.min():.4f} / {tensor_vals.max():.4f}")
    print(f"  unique  = {len(np.unique(np.round(tensor_vals, 4)))}")

    # 3) Actual physical mass via Gym API (the moment of truth)
    print(f"\n========== Actual physical mass via gym.get_actor_rigid_body_properties ==========")
    physical_masses = []
    for i in range(env.num_envs):
        props = env.gym.get_actor_rigid_body_properties(env.envs[i], env.load_handles[i])
        physical_masses.append(props[0].mass)
    phys_arr = np.array(physical_masses)
    print(f"  count   = {len(physical_masses)}")
    print(f"  min/max = {phys_arr.min():.6f} / {phys_arr.max():.6f}")
    print(f"  mean    = {phys_arr.mean():.6f}")
    print(f"  std     = {phys_arr.std():.6f}")
    print(f"  unique  = {len(np.unique(np.round(phys_arr, 4)))}")
    print(f"  values  = {phys_arr.tolist()}")

    # 4) Diff
    print(f"\n========== Compare label vs physical ==========")
    diff = phys_arr - init_arr
    print(f"  max |label - physical| = {np.abs(diff).max():.6f}")
    if np.abs(diff).max() < 1e-3:
        print(f"  ✓ labels match physics — fix works")
    else:
        print(f"  ✗ labels DO NOT match physics — fix broken")
        print(f"  This is the bug: lstsq sees varying labels but R_L from uniform-mass physics")
