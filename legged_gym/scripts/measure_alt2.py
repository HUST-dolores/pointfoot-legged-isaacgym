#!/usr/bin/env python3
"""Alternation diagnostic — the HARD test the no_fly metric couldn't do: does the stance foot
actually take turns between LEFT and RIGHT (true alternating stepping) or does one foot dominate
(limp)? Loads a policy, rolls out with a stepping command, records per-foot ground contact, and
computes each foot's swing share + swing-foot switch count."""
import os
import isaacgym  # noqa
from isaacgym.torch_utils import *  # noqa
from legged_gym.envs import *  # noqa
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play import _apply_saved_env_cfg
import numpy as np
import torch


def main():
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name)
    _apply_saved_env_cfg(env_cfg, train_cfg, args, log_root)
    env_cfg.env.num_envs = 64
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.add_random_load = False
    env_cfg.domain_rand.push_robots = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history, commands, critic_obs = env.get_observations()
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    cmd_vx = float(getattr(args, "cmd_vx", 0.2) or 0.2)   # cmd_roll
    fi = env.feet_indices
    Lc, Rc = [], []
    for i in range(600):
        env.commands[:, 0] = cmd_vx
        if hasattr(env, "cmd_step"):
            env.cmd_step[:] = float(os.environ.get("CMD_STEP","0.4"))
        with torch.no_grad():
            est = encoder(obs_history, obs)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(actions.detach())
        if i > 150:
            Lc.append((env.contact_forces[:, fi[0], 2] > 1.0).cpu().numpy())
            Rc.append((env.contact_forces[:, fi[1], 2] > 1.0).cpu().numpy())
    Lc = np.array(Lc); Rc = np.array(Rc)     # [T,N]
    T, N = Lc.shape
    single_L = Lc & ~Rc      # left stance, RIGHT swinging
    single_R = Rc & ~Lc      # right stance, LEFT swinging
    swing = np.zeros_like(Lc, dtype=int)
    swing[single_L] = 1      # right foot in the air
    swing[single_R] = -1     # left foot in the air
    tot_single = tot_switch = rsw = lsw = 0
    for e in range(N):
        nz = swing[:, e][swing[:, e] != 0]
        tot_single += len(nz); rsw += int((nz == 1).sum()); lsw += int((nz == -1).sum())
        if len(nz) > 1:
            tot_switch += int((np.diff(nz) != 0).sum())
    lsh = lsw / max(tot_single, 1); rsh = rsw / max(tot_single, 1)
    print("[alt] ===================== ALTERNATION DIAGNOSTIC =====================")
    print(f"[alt] load_run={args.load_run} ckpt={args.checkpoint}  cmd_roll={cmd_vx} cmd_step=0.4  {N}env x {T}steps")
    print(f"[alt] stance fraction:  left={Lc.mean():.2f}  right={Rc.mean():.2f}")
    print(f"[alt] support mix:  single-L={single_L.mean():.2f}  single-R={single_R.mean():.2f}  double={(Lc&Rc).mean():.2f}  flight={(~Lc&~Rc).mean():.2f}")
    print(f"[alt] SWING SHARE (of single-support):  LEFT-foot-swings={lsh:.2f}   RIGHT-foot-swings={rsh:.2f}   (0.5/0.5=perfect alternation, 1/0=limp)")
    print(f"[alt] swing-foot SWITCHES per env = {tot_switch/N:.1f}  over {T} steps  (frequent=alternating)")
    alt = (min(lsh, rsh) > 0.25) and (tot_switch / N > 5)
    print(f"[alt] VERDICT: {'*** TRUE ALTERNATING STEPPING *** (both feet take turns swinging + frequent switches)' if alt else 'NOT true alternation (one foot dominates swing / few switches = limp/shuffle)'}")


if __name__ == "__main__":
    main()
