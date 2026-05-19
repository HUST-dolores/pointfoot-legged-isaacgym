# Deterministic load-position scan for load-estimation calibration.
#
# Usage (run once per mass):
#   python legged_gym/scripts/collect_calibration_data.py \
#       --task=wheelfoot_flat --load_run=test32 --checkpoint=10000 \
#       --headless --mass=5.0
#
# Output: logs/calibration/<datetime>/calib_m<mass>.npz
# Combine all mass NPZs with legged_gym/scripts/fit_load_estimation.py.

import math
import os
import sys
from datetime import datetime

import isaacgym  # noqa: F401  (must import before torch)
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse, to_torch

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401, F403  (registers tasks)
from legged_gym.utils import get_args, task_registry


# -------- calibration-specific knobs (edit here if needed) --------
# Default matches training (1024). Pass --num_envs N on the CLI to scale up
# (4096 needs more PhysX/GPU memory and can crash on smaller GPUs).
NUM_ENVS = 1024
X_MIN, X_MAX, NX = -0.17, 0.21, 4
Y_MIN, Y_MAX, NY = -0.19, 0.19, 4
SETTLE_S = 2.0
COLLECT_START_S = 5.0
COLLECT_END_S = 15.0
EPISODE_LENGTH_S = 60.0
LOAD_Z_OFFSET = 0.10   # height of load CoM above body origin (matches load_estimation_load_z)
DELTA = 0.05144        # additive term in load-estimation denominator (fixed for now)
# ------------------------------------------------------------------


def pop_arg(name, default, cast=float):
    """Pop --name VALUE or --name=VALUE from sys.argv so isaacgym's parser does not see it."""
    val = default
    out, i = [], 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok == name and i + 1 < len(sys.argv):
            val = cast(sys.argv[i + 1])
            i += 2
            continue
        if tok.startswith(name + "="):
            val = cast(tok.split("=", 1)[1])
            i += 1
            continue
        out.append(tok)
        i += 1
    sys.argv = out
    return val




def build_grid():
    xs = np.linspace(X_MIN, X_MAX, NX)
    ys = np.linspace(Y_MIN, Y_MAX, NY)
    return [(float(x), float(y)) for x in xs for y in ys]


def _safe_set(cfg_obj, name, value):
    """Set cfg.<name> = value if the attribute already exists; silently skip otherwise."""
    if hasattr(cfg_obj, name):
        setattr(cfg_obj, name, value)


def configure_env(env_cfg, mass):
    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.env.episode_length_s = EPISODE_LENGTH_S

    # Flat terrain only.
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1

    # Kill randomization so each env only differs in the (x, y) we assign.
    env_cfg.noise.add_noise = False
    for flag in (
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_com",
        "randomize_base_mass",
        "randomize_inertia",
        "push_robots",
        "push_curriculum",
        "randomize_Kp",
        "randomize_Kd",
        "randomize_motor_torque",
        "randomize_default_dof_pos",
        "randomize_action_delay",
        "randomize_imu_offset",
    ):
        _safe_set(env_cfg.domain_rand, flag, False)

    # Load is required (infrastructure must run). Mass is fixed at `mass`.
    env_cfg.domain_rand.add_random_load = True
    env_cfg.domain_rand.add_load_range = [mass, mass]
    env_cfg.domain_rand.load_start_time_s = 0.0
    env_cfg.domain_rand.load_duration_s = EPISODE_LENGTH_S + 5.0
    env_cfg.domain_rand.load_interval_s = EPISODE_LENGTH_S + 5.0
    if hasattr(env_cfg.domain_rand, "load_duration_range_s"):
        env_cfg.domain_rand.load_duration_range_s = [EPISODE_LENGTH_S + 5.0, EPISODE_LENGTH_S + 5.0]
    if hasattr(env_cfg.domain_rand, "load_interval_range_s"):
        env_cfg.domain_rand.load_interval_range_s = [EPISODE_LENGTH_S + 5.0, EPISODE_LENGTH_S + 5.0]


def install_deterministic_spawner(env, env_x_true, env_y_true):
    """Replace env._maybe_spawn_loads with a one-shot deterministic placement."""
    state = {"done": False}

    def deterministic_spawn():
        if state["done"]:
            return
        t = env.episode_length_buf.float() * env.dt
        start_s = float(getattr(env.cfg.domain_rand, "load_start_time_s", 0.0))
        spawn_mask = (~env.has_load) & (t >= start_s)
        if not spawn_mask.any():
            return
        ids = spawn_mask.nonzero(as_tuple=False).flatten()
        n = ids.numel()

        r_local = torch.stack(
            (
                env_x_true[ids],
                env_y_true[ids],
                torch.full_like(env_x_true[ids], LOAD_Z_OFFSET),
            ),
            dim=-1,
        )
        base_pos = env.root_states[env.actor_indices[ids], 0:3]
        base_quat = env.root_states[env.actor_indices[ids], 3:7]
        pos_world = quat_rotate(base_quat, r_local) + base_pos
        orientation = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=env.device, dtype=env.root_states.dtype
        ).view(1, 4).repeat(n, 1)

        env.add_load(ids, pos_world, orientation)
        env.has_load[ids] = True
        if hasattr(env, "loads_spawned_count"):
            env.loads_spawned_count[ids] += 1
        # Push remove/respawn far enough that the load stays on board through the whole run.
        env.load_remove_time[ids] = t[ids] + EPISODE_LENGTH_S + 10.0
        env.next_load_spawn_time[ids] = t[ids] + EPISODE_LENGTH_S + 10.0

        # All envs spawned in one shot at t=0. Lock it.
        if int(spawn_mask.sum().item()) == env.num_envs:
            state["done"] = True

    env._maybe_spawn_loads = deterministic_spawn


def main():
    mass = pop_arg("--mass", 5.0, cast=float)
    output_dir = pop_arg(
        "--output_dir",
        os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            "calibration",
            datetime.now().strftime("%b%d_%H-%M-%S"),
        ),
        cast=str,
    )

    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_env(env_cfg, mass)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Assign deterministic (x, y) per env. We split env_id ranges across the grid.
    grid = build_grid()
    num_conditions = len(grid)
    envs_per_cond = env.num_envs // num_conditions
    assert envs_per_cond > 0, (
        f"num_envs ({env.num_envs}) must be >= num_conditions ({num_conditions})"
    )
    env_x_true = torch.zeros(env.num_envs, device=env.device, dtype=env.root_states.dtype)
    env_y_true = torch.zeros(env.num_envs, device=env.device, dtype=env.root_states.dtype)
    env_cond_id = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    for cid, (x, y) in enumerate(grid):
        lo = cid * envs_per_cond
        hi = (cid + 1) * envs_per_cond if cid + 1 < num_conditions else env.num_envs
        env_x_true[lo:hi] = x
        env_y_true[lo:hi] = y
        env_cond_id[lo:hi] = cid

    install_deterministic_spawner(env, env_x_true, env_y_true)

    # Load checkpoint policy + encoder.
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    obs, obs_history, commands, critic_obs = env.get_observations()
    # Zero commands so the robot just tries to stand.
    robot_type = os.getenv("ROBOT_TYPE", "WF_TRON1A")
    if robot_type == "WF_TRON1A":
        commands_val = to_torch([0.0, 0.0, 0.0], device=env.device)
    elif robot_type.startswith("PF"):
        commands_val = to_torch([0.0, 0.0, 0.0, 0.0], device=env.device)
    else:
        commands_val = to_torch([0.0, 0.0, 0.0, 0.0, 0.0], device=env.device)

    dt = env.dt
    settle_steps = int(round(SETTLE_S / dt))
    collect_start_step = int(round(COLLECT_START_S / dt))
    collect_end_step = int(round(COLLECT_END_S / dt))
    print(
        f"[calib] dt={dt:.4f}s  settle@step={settle_steps}  "
        f"collect=[{collect_start_step},{collect_end_step})  total_steps={collect_end_step}"
    )

    # DOF indices.
    def dof_idx(name):
        return env.dof_names.index(name)

    hip_L_idx = dof_idx("hip_L_Joint")
    hip_R_idx = dof_idx("hip_R_Joint")
    knee_L_idx = dof_idx("knee_L_Joint")
    knee_R_idx = dof_idx("knee_R_Joint")
    abad_L_idx = dof_idx("abad_L_Joint")
    abad_R_idx = dof_idx("abad_R_Joint")

    # Geometric constants from the existing load-estimation formula.
    cfg_asset = env.cfg.asset
    zero_thigh_angle = float(getattr(cfg_asset, "load_estimation_zero_thigh_angle", 2.0 * math.pi / 3.0))
    thigh_len = float(getattr(cfg_asset, "load_estimation_thigh_length", 0.3))
    robot_width = float(getattr(cfg_asset, "load_estimation_robot_width", 0.251))
    leg_eff_length = float(getattr(cfg_asset, "load_estimation_leg_eff_length", 0.55))
    abad_R_sign = float(getattr(cfg_asset, "load_estimation_abad_R_sign", -1.0))
    gravity_cfg = getattr(env.sim_params, "gravity", None)
    if gravity_cfg is None or getattr(gravity_cfg, "z", None) is None:
        gravity = 9.81
    else:
        gravity = float(abs(gravity_cfg.z))

    N = env.num_envs
    feature_names = [
        "R_L", "R_R",
        "cos_l", "cos_r", "cos_abad_L", "cos_abad_R",
        "y_foot_L", "y_foot_R",
        "tau_lhip", "tau_rhip", "tau_lknee", "tau_rknee",
        "T_L", "T_R",
        "pitch", "cos_pitch", "tan_pitch",
        "theta_lhip", "theta_rhip", "theta_labad", "theta_rabad",
        "x_rel_true", "y_rel_true", "z_rel_true",
    ]
    accum = {n: torch.zeros(N, device=env.device) for n in feature_names}
    count = 0

    eps = 1e-3

    def _safe_cos(x):
        sign = torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))
        return torch.where(torch.abs(x) < eps, sign * eps, x)

    for step in range(collect_end_step):
        with torch.no_grad():
            est = encoder(obs_history, obs)
            actions = policy(torch.cat((est, obs, commands), dim=-1))

        env.commands[:, :] = commands_val
        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(actions.detach())

        if step < collect_start_step:
            if step % 200 == 0:
                print(f"[calib] step={step}/{collect_end_step} (warmup)")
            continue

        tau_lhip = env.torques[:, hip_L_idx]
        tau_rhip = env.torques[:, hip_R_idx]
        tau_lknee = env.torques[:, knee_L_idx]
        tau_rknee = env.torques[:, knee_R_idx]

        theta_lhip = env.dof_pos[:, hip_L_idx]
        theta_rhip = env.dof_pos[:, hip_R_idx]
        theta_labad = env.dof_pos[:, abad_L_idx]
        theta_rabad = env.dof_pos[:, abad_R_idx]

        qx = env.base_quat[:, 0]
        qy = env.base_quat[:, 1]
        qz = env.base_quat[:, 2]
        qw = env.base_quat[:, 3]
        pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0 + 1e-6, 1.0 - 1e-6))
        cos_pitch = torch.cos(pitch)
        tan_pitch = torch.tan(pitch)

        lieangle_L = math.pi - (zero_thigh_angle - theta_lhip) - pitch
        lieangle_R = math.pi - (zero_thigh_angle + theta_rhip) - pitch
        cos_l = torch.cos(lieangle_L)
        cos_r = torch.cos(lieangle_R)
        cos_abad_L = torch.cos(theta_labad)
        cos_abad_R = torch.cos(theta_rabad)

        T_L = -tau_lknee - tau_lhip
        T_R = tau_rknee + tau_rhip
        denom_L = (thigh_len * _safe_cos(cos_l) + DELTA) * gravity * _safe_cos(cos_abad_L)
        denom_R = (thigh_len * _safe_cos(cos_r) + DELTA) * gravity * _safe_cos(cos_abad_R)
        R_L = T_L / denom_L
        R_R = T_R / denom_R

        y_foot_L = 0.5 * robot_width + leg_eff_length * torch.sin(theta_labad)
        y_foot_R = -0.5 * robot_width + abad_R_sign * leg_eff_length * torch.sin(theta_rabad)

        base_pos = env.root_states[env.actor_indices, 0:3]
        base_quat = env.root_states[env.actor_indices, 3:7]
        load_pos = env.root_states[env.load_indices, 0:3]
        rel_world = load_pos - base_pos
        rel_body = quat_rotate_inverse(base_quat, rel_world)

        for name, val in (
            ("R_L", R_L), ("R_R", R_R),
            ("cos_l", cos_l), ("cos_r", cos_r),
            ("cos_abad_L", cos_abad_L), ("cos_abad_R", cos_abad_R),
            ("y_foot_L", y_foot_L), ("y_foot_R", y_foot_R),
            ("tau_lhip", tau_lhip), ("tau_rhip", tau_rhip),
            ("tau_lknee", tau_lknee), ("tau_rknee", tau_rknee),
            ("T_L", T_L), ("T_R", T_R),
            ("pitch", pitch), ("cos_pitch", cos_pitch), ("tan_pitch", tan_pitch),
            ("theta_lhip", theta_lhip), ("theta_rhip", theta_rhip),
            ("theta_labad", theta_labad), ("theta_rabad", theta_rabad),
            ("x_rel_true", rel_body[:, 0]),
            ("y_rel_true", rel_body[:, 1]),
            ("z_rel_true", rel_body[:, 2]),
        ):
            accum[name] += val
        count += 1

        if step % 200 == 0:
            print(f"[calib] step={step}/{collect_end_step} (collecting, {count} samples in window)")

    if count == 0:
        raise RuntimeError("No samples collected. Check COLLECT_START_S/COLLECT_END_S vs episode length.")

    means = {k: (v / count).detach().cpu().numpy() for k, v in accum.items()}

    # Validity: load must still be roughly on the body (height near LOAD_Z_OFFSET, not fallen off).
    z_rel = means["z_rel_true"]
    valid = (z_rel > LOAD_Z_OFFSET - 0.30) & (z_rel < LOAD_Z_OFFSET + 0.30)
    print(f"[calib] valid envs: {int(valid.sum())} / {len(valid)} (mass={mass})")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"calib_m{mass:.3f}.npz")
    np.savez(
        output_path,
        mass_true=np.full(N, mass, dtype=np.float32),
        x_set=env_x_true.detach().cpu().numpy().astype(np.float32),
        y_set=env_y_true.detach().cpu().numpy().astype(np.float32),
        cond_id=env_cond_id.detach().cpu().numpy().astype(np.int32),
        valid=valid.astype(np.bool_),
        delta=np.float32(DELTA),
        thigh_len=np.float32(thigh_len),
        gravity=np.float32(gravity),
        robot_width=np.float32(robot_width),
        leg_eff_length=np.float32(leg_eff_length),
        abad_R_sign=np.float32(abad_R_sign),
        zero_thigh_angle=np.float32(zero_thigh_angle),
        **{k: v.astype(np.float32) for k, v in means.items()},
    )
    print(f"[calib] saved -> {output_path}")


if __name__ == "__main__":
    main()
