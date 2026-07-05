"""Collect a PHYSICALLY-VALID stepping reference by rolling out the v8 policy (headless).

Mirrors play.py's setup (make_env / make_alg_runner / get_inference_policy+encoder / the
4-tensor act loop) but runs headless and RECORDS per-step reference data to an npz, then exits.

The gait config (use_gait_phase/stepping, cmd_step, frequencies) is set by the wrapper
collect_v8_ref.sh via the SAME sed edits play_stepV8.sh uses — so the recorded gait is
byte-identical to the demo the user just confirmed. cmd_vx/steps/out come from env vars.

Records (env 0, after each step): 8 dof_pos, 8 dof_vel, base_quat(xyzw), projected_gravity,
base_lin_vel, base_ang_vel, base_height, gait clock (sin/cos + raw index), commands + dof_names.
Isaac-Gym dof order here = [abad_L,hip_L,knee_L,wheel_L,abad_R,hip_R,knee_R,wheel_R] — recorded
WITH names so the Isaac-Lab side maps by NAME (orders differ; axes mirror L<->R).
"""
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym  # noqa: F401  (must precede torch)
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
import numpy as np
import torch
from isaacgym.torch_utils import to_torch


def main():
    args = get_args()
    cmd_vx = float(os.environ.get("COLLECT_CMD_VX", "0.2"))
    n_steps = int(os.environ.get("COLLECT_STEPS", "600"))
    n_warmup = int(os.environ.get("COLLECT_WARMUP", "60"))
    out_path = os.environ.get("COLLECT_OUT", "/tmp/v8_ref.npz")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    # clean reference: kill pushes / obs noise if present (don't corrupt the recorded gait)
    for attr, val in [("push_robots", False), ("randomize_friction", False),
                      ("randomize_base_mass", False), ("randomize_base_com", False)]:
        if hasattr(env_cfg, "domain_rand") and hasattr(env_cfg.domain_rand, attr):
            setattr(env_cfg.domain_rand, attr, val)
    if hasattr(env_cfg, "noise") and hasattr(env_cfg.noise, "add_noise"):
        env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    commands_val = to_torch([cmd_vx, 0.0, 0.0], device=env.device)  # WF_TRON1A = [vx,vy,yaw]
    env.commands[:, :] = commands_val
    obs, obs_history, commands, critic_obs = env.get_observations()

    print(f"[COLLECT] dof_names = {list(env.dof_names)}")
    print(f"[COLLECT] cmd_vx={cmd_vx} warmup={n_warmup} record={n_steps} dt={env.dt} out={out_path}")

    log = {k: [] for k in ("dof_pos", "dof_vel", "base_quat", "proj_grav",
                           "base_lin_vel", "base_ang_vel", "base_height",
                           "clock_sin", "clock_cos", "gait_index", "commands")}
    dones_during = 0
    total = n_warmup + n_steps
    for i in range(total):
        with torch.no_grad():
            est = encoder(obs_history, obs)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        env.commands[:, :] = commands_val
        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(actions.detach())
        if i < n_warmup:
            continue
        if bool(dones[0].item()):
            dones_during += 1  # a reset happened mid-record; flag it (walker shouldn't fall)
        log["dof_pos"].append(env.dof_pos[0].detach().cpu().numpy())
        log["dof_vel"].append(env.dof_vel[0].detach().cpu().numpy())
        log["base_quat"].append(env.base_quat[0].detach().cpu().numpy())
        log["proj_grav"].append(env.projected_gravity[0].detach().cpu().numpy())
        log["base_lin_vel"].append(env.base_lin_vel[0].detach().cpu().numpy())
        log["base_ang_vel"].append(env.base_ang_vel[0].detach().cpu().numpy())
        log["base_height"].append(env.root_states[env.actor_indices, 2][0].item())
        log["clock_sin"].append(env.clock_inputs_sin.reshape(-1)[0].item())
        log["clock_cos"].append(env.clock_inputs_cos.reshape(-1)[0].item())
        log["gait_index"].append(env.gait_indices.reshape(-1)[0].item())
        log["commands"].append(env.commands[0].detach().cpu().numpy())

    for k in list(log):
        log[k] = np.asarray(log[k])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, dof_names=list(env.dof_names), dt=float(env.dt),
             cmd_vx=cmd_vx, dones_during=dones_during, **log)

    # quick summary
    dp = log["dof_pos"]
    print(f"[COLLECT] recorded {dp.shape[0]} steps, dones_during={dones_during}")
    print(f"[COLLECT] per-joint pk-pk (rad):")
    for j, nm in enumerate(env.dof_names):
        col = dp[:, j]
        print(f"    {nm:<16} min={col.min():+.3f} max={col.max():+.3f} pkpk={col.max()-col.min():.3f}")
    bh = log["base_height"]
    print(f"[COLLECT] base_height mean={bh.mean():.3f} range={bh.max()-bh.min():.3f}")
    print(f"[COLLECT] base_lin_vel_x mean={log['base_lin_vel'][:,0].mean():+.3f} (cmd {cmd_vx})")
    print(f"[COLLECT] saved {out_path}")


if __name__ == "__main__":
    main()
