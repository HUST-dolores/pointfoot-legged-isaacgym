# SPDX-License-Identifier: BSD-3-Clause
#
# eval_motor_fk.py — f(k) fitness-landscape evaluation for the MOTOR-TORQUE
# co-design variable k = motor_torque_scale on the WF_TRON1A wheel-foot biped.
#
# Background:
#   k ∈ [0.6, 1.6] is a per-env scalar. When domain_rand.randomize_motor_design=True
#   each env samples its own k (seeded) at env creation and keeps it FIXED for the
#   whole rollout (NOT resampled on reset). k scales each joint's torque LIMIT
#   (legs 80→80k, wheels 40→40k) AND adds motor mass to each actuated link via the
#   ENCOS cost curve. Bigger k = stronger but heavier motor.
#
# This script is INFERENCE only (actor+encoder), but make_alg_runner builds the
# full net, so the env config must match the checkpoint's training cfg or
# runner.load() fails on a shape mismatch. We reuse play.py's loading +
# config-alignment pattern and additionally align the motor-design critic dim.
#
# Usage (run on the workstation):
#   export ROBOT_TYPE=WF_TRON1A
#   python legged_gym/scripts/eval_motor_fk.py --task=wheelfoot_flat --headless \
#       --num_envs 2048 --load_run Jun30_20-36-00_ --checkpoint 2000
#
#   # specialist run (motor-design OFF in its native cfg → all envs k=1):
#   python legged_gym/scripts/eval_motor_fk.py --task=wheelfoot_flat --headless \
#       --num_envs 2048 --load_run Jun30_21-02-41_ --checkpoint 2000

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import json

import isaacgym  # noqa: F401  (must precede torch)
from isaacgym.torch_utils import *  # noqa: F401,F403
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch

from legged_gym.scripts.play import _apply_saved_env_cfg


# -------- fixed eval command (clean forward walking) --------
# CMD_VX is set per-run from --cmd_vx (defaults to 0.5 if not given).
CMD_VY = 0.0
CMD_YAW = 0.0

# -------- rollout / binning config --------
N_STEPS = 1500          # ≈ 30 s at dt=0.02
N_BINS = 15             # k bins over [0.6, 1.6]
K_LO, K_HI = 0.6, 1.6

# Output dir. Override with env var FK_OUT_DIR (e.g. on the workstation, where
# /home/xu/Desktop does not exist). Plot + data are pulled back to the local
# Desktop via rsync afterwards.
OUT_DIR = os.environ.get(
    "FK_OUT_DIR",
    os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "codesign_fk"),
)


def _load_saved_env_dict(args, log_root):
    """Return the saved env_cfg.json dict for the loaded run (or {})."""
    load_run = getattr(args, "load_run", None) or ""
    if not load_run:
        return {}
    env_p = os.path.join(log_root, load_run, "env_cfg.json")
    if not os.path.isfile(env_p):
        return {}
    try:
        return json.load(open(env_p))
    except Exception:
        return {}


def _align_motor_design_cfg(env_cfg, saved_env):
    """Align motor-design fields that play.py's _apply_saved_env_cfg does NOT cover.

    Critical for clean checkpoint loading:
      - env.use_motor_design_in_critic  -> drives critic input width (+1 dim)
      - env.num_critic_observations     -> the critic's first layer size
      - domain_rand.randomize_motor_design / motor_torque_scale_range / motor_design_seed
        -> whether/how each env samples its own k.
    Without these, a generalist ckpt (critic built with the motor dim) fails to load.
    """
    se = saved_env.get("env", {}) if isinstance(saved_env.get("env"), dict) else {}
    sdr = saved_env.get("domain_rand", {}) if isinstance(saved_env.get("domain_rand"), dict) else {}

    applied = []

    def _ovr(obj, attr, new):
        old = getattr(obj, attr, None)
        if old != new:
            applied.append(f"{obj.__class__.__name__}.{attr}: {old} -> {new}")
            setattr(obj, attr, new)

    if se.get("use_motor_design_in_critic") is not None:
        _ovr(env_cfg.env, "use_motor_design_in_critic", bool(se["use_motor_design_in_critic"]))
    if se.get("num_critic_observations") is not None:
        _ovr(env_cfg.env, "num_critic_observations", int(se["num_critic_observations"]))
    if sdr.get("randomize_motor_design") is not None:
        _ovr(env_cfg.domain_rand, "randomize_motor_design", bool(sdr["randomize_motor_design"]))
    if sdr.get("motor_torque_scale_range") is not None:
        _ovr(env_cfg.domain_rand, "motor_torque_scale_range",
             [float(x) for x in sdr["motor_torque_scale_range"]])
    if sdr.get("motor_design_seed") is not None:
        _ovr(env_cfg.domain_rand, "motor_design_seed", int(sdr["motor_design_seed"]))

    if applied:
        print(f"[fk] applied {len(applied)} motor-design cfg override(s) from saved cfg:")
        for s in applied:
            print(f"     - {s}")
    else:
        print("[fk] no motor-design cfg overrides needed (in-tree already matches)")


def _make_eval_env_cfg(env_cfg, args):
    """Apply eval overrides (mirrors play.py's eval setup) plus demanding conditions.

    Demanding-condition CLI (mirrors play.py wiring):
      --slope_deg D       : flat terrain + gravity tilted D° (≡ walking up a D° slope).
      --load_mass_min/max : if both >0, a FIXED constant payload of that mass (kg).
                            Held the whole episode (load_hold semantics), centered.
      --cmd_vx V          : forward speed command (read in evaluate()).
    """
    env_cfg.env.episode_length_s = 60

    _slope_deg = float(getattr(args, "slope_deg", 0.0) or 0.0)
    _lm_min = float(getattr(args, "load_mass_min", -1.0) or -1.0)
    _lm_max = float(getattr(args, "load_mass_max", -1.0) or -1.0)
    _payload_on = (_lm_min >= 0.0 and _lm_max >= 0.0 and _lm_max > 0.0)
    _slope_on = abs(_slope_deg) > 1e-6

    # Level-0 terrain, mirror play.py trimesh layout (spacious, all lowest difficulty)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.curriculum = True
    if env_cfg.terrain.mesh_type == "trimesh":
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 10
        env_cfg.terrain.terrain_length = 20.0
        env_cfg.terrain.terrain_width = 10.0
        env_cfg.terrain.border_size = 20
        env_cfg.terrain.max_init_terrain_level = 0

    # Slope conditions force FLAT terrain (a tilted-gravity plane ≡ a slope), so the
    # only "incline" effect is the gravity tilt, not terrain roughness (matches play.py).
    if _slope_on:
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.measure_heights = False
        _b = np.radians(_slope_deg); _g = 9.81
        env_cfg.sim.gravity = [float(_g * np.sin(_b)), 0.0, float(-_g * np.cos(_b))]
        print(f"[fk] SLOPE {_slope_deg}deg: flat plane + sim.gravity={env_cfg.sim.gravity}")

    # Clean locomotion: no perturbations / DR (but KEEP randomize_motor_design as saved)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_curriculum = False
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False

    if _payload_on:
        # FIXED constant payload, held the whole episode, centered (load_hold semantics).
        env_cfg.domain_rand.add_random_load = True
        env_cfg.domain_rand.per_env_load_mass = False     # same mass for every env
        env_cfg.domain_rand.add_load_range = [_lm_min, _lm_max]
        env_cfg.domain_rand.load_start_time_s = 0.5
        env_cfg.domain_rand.load_duration_range_s = [1.0e6, 1.0e6]
        env_cfg.domain_rand.load_interval_range_s = [1.0e6, 1.0e6]
        print(f"[fk] PAYLOAD held: {_lm_min}-{_lm_max} kg, centered, 0.5s->end")
    else:
        # No payload — isolate the motor-design effect.
        env_cfg.domain_rand.add_random_load = False

    return {"slope_deg": _slope_deg, "slope_on": _slope_on,
            "payload_on": _payload_on, "load_mass": _lm_min if _payload_on else 0.0}


def evaluate(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name
    )

    # 1) play.py's standard alignment (obs dims, encoder dim, etc.)
    _apply_saved_env_cfg(env_cfg, train_cfg, args, log_root)
    # 2) extra motor-design alignment (critic dim + k-sampling), needed for clean load.
    saved_env = _load_saved_env_dict(args, log_root)
    _align_motor_design_cfg(env_cfg, saved_env)

    # 3) eval overrides (+ demanding conditions from CLI)
    cond = _make_eval_env_cfg(env_cfg, args)
    cmd_vx = float(getattr(args, "cmd_vx", 0.0) or 0.0) or 0.5  # default 0.5 if 0/unset

    randomize_md = bool(getattr(env_cfg.domain_rand, "randomize_motor_design", False))
    print(f"[fk] randomize_motor_design = {randomize_md} "
          f"(True=generalist k-sweep, False=specialist all k=1)")

    # ---- build env ----
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Slope: tilt the obs "down" direction to match the tilted physics gravity
    # (else projected_gravity disagrees with physics) — mirrors play.py.
    if cond["slope_on"]:
        _b = np.radians(cond["slope_deg"])
        env.gravity_vec = to_torch([float(np.sin(_b)), 0.0, float(-np.cos(_b))],
                                   device=env.device).repeat(env.num_envs, 1)
        print(f"[fk] SLOPE: env.gravity_vec tilted {cond['slope_deg']}deg")

    # Payload: force centered load offset (no eccentric tipping moment).
    if cond["payload_on"] and hasattr(env, "load_offset_range"):
        env.load_offset_range["x"] = (0.0, 0.0)
        env.load_offset_range["y"] = (0.0, 0.0)
        env.load_offset_range["z"] = (0.10, 0.10)
        print("[fk] PAYLOAD: load offset centered (x=0,y=0,z=0.10)")

    # fixed forward command
    robot_type = os.getenv("ROBOT_TYPE")
    if robot_type == "WF_TRON1A":
        commands_val = to_torch([cmd_vx, CMD_VY, CMD_YAW], device=env.device)
    elif robot_type.startswith("PF"):
        commands_val = to_torch([cmd_vx, CMD_VY, CMD_YAW, 0], device=env.device)
    else:
        commands_val = to_torch([cmd_vx, CMD_VY, CMD_YAW, 0.0, 0.0], device=env.device)
    print(f"[fk] command = vx={cmd_vx} vy={CMD_VY} yaw={CMD_YAW}; num_envs={env.num_envs}")

    obs, obs_history, commands, critic_obs = env.get_observations()

    # ---- load policy + encoder (full runner, inference views) ----
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    env.commands[:, :] = commands_val
    obs, obs_history, commands, critic_obs = env.get_observations()

    # ---- per-env k (FIXED for the run) ----
    N = env.num_envs
    if getattr(env, "env_motor_scale", None) is not None:
        k = env.env_motor_scale.detach().clone().float()           # [N]
    else:
        k = torch.ones(N, device=env.device)                       # specialist: all k=1
    print(f"[fk] k range over envs: [{float(k.min()):.3f}, {float(k.max()):.3f}], "
          f"mean={float(k.mean()):.3f}")

    # ---- per-env accumulators ----
    dev = env.device
    rew_sum = torch.zeros(N, device=dev)            # summed per-step reward
    rew_cnt = torch.zeros(N, device=dev)            # steps counted
    vxerr_sum = torch.zeros(N, device=dev)          # summed |vx - cmd_vx|
    alive_steps = torch.zeros(N, device=dev)        # steps survived without a done
    done_count = torch.zeros(N, device=dev)         # number of resets seen
    total_steps = torch.zeros(N, device=dev)        # total steps observed

    # ---- rollout ----
    for i in range(N_STEPS):
        with torch.no_grad():
            est = encoder(obs_history, obs)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        env.commands[:, :] = commands_val
        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(actions.detach())
        env.commands[:, :] = commands_val

        dones_f = dones.float().view(-1)
        total_steps += 1.0
        rew_sum += rews.view(-1)
        rew_cnt += 1.0
        # velocity-tracking error this step
        vxerr_sum += torch.abs(env.base_lin_vel[:, 0] - cmd_vx)
        # survival: count steps where the env did NOT reset this step
        alive_steps += (1.0 - dones_f)
        done_count += dones_f

        if (i % 200) == 0:
            mr = float((rew_sum / rew_cnt.clamp(min=1)).mean())
            md = float(done_count.mean())
            print(f"[fk] step {i}/{N_STEPS}  mean_step_rew={mr:.4f}  mean_dones={md:.3f}")

    # ---- per-env metrics ----
    mean_step_rew = (rew_sum / rew_cnt.clamp(min=1)).cpu().numpy()          # [N]
    mean_vxerr = (vxerr_sum / total_steps.clamp(min=1)).cpu().numpy()       # [N]
    survival = (alive_steps / total_steps.clamp(min=1)).cpu().numpy()       # [N] fraction not-terminated
    dones_per_env = done_count.cpu().numpy()                                # [N]
    k_np = k.cpu().numpy()                                                  # [N]

    return {
        "k": k_np,
        "mean_step_rew": mean_step_rew,
        "mean_vxerr": mean_vxerr,
        "survival": survival,
        "dones_per_env": dones_per_env,
        "randomize_motor_design": randomize_md,
        "load_run": str(getattr(args, "load_run", "")),
        "checkpoint": str(getattr(args, "checkpoint", "")),
        "n_steps": N_STEPS,
        "cmd_vx": cmd_vx,
        "num_envs": int(N),
        "slope_deg": float(cond["slope_deg"]),
        "load_mass": float(cond["load_mass"]),
        "cond_tag": os.environ.get("FK_COND_TAG", "cond"),
    }


def _bin_fk(k, vals, lo=K_LO, hi=K_HI, n_bins=N_BINS):
    """Return (bin_centers, bin_mean, bin_std, bin_count) for vals binned by k."""
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    idx = np.clip(np.digitize(k, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b] > 0:
            means[b] = float(np.mean(vals[m]))
            stds[b] = float(np.std(vals[m]))
    return centers, means, stds, counts


def _cond_label(data):
    """Human-readable condition string from data fields."""
    parts = []
    if data["slope_deg"] and abs(data["slope_deg"]) > 1e-6:
        parts.append(f"slope{data['slope_deg']:g}deg")
    if data["load_mass"] and data["load_mass"] > 0:
        parts.append(f"load{data['load_mass']:g}kg")
    parts.append(f"vx{data['cmd_vx']:g}")
    return "_".join(parts)


def main():
    args = get_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    data = evaluate(args)
    cond_tag = data["cond_tag"]
    role = "generalist" if data["randomize_motor_design"] else "specialist"

    # save raw per-env arrays (condition-specific filename)
    npz_path = os.path.join(OUT_DIR, f"fk_data_{cond_tag}_{role}.npz")
    np.savez(npz_path, **{k: v for k, v in data.items()
                          if isinstance(v, np.ndarray)},
             meta=json.dumps({k: v for k, v in data.items()
                              if not isinstance(v, np.ndarray)}))
    print(f"[fk] saved per-env data: {npz_path}")

    k = data["k"]
    bins_r = _bin_fk(k, data["mean_step_rew"])
    bins_s = _bin_fk(k, data["survival"])
    bins_v = _bin_fk(k, data["mean_vxerr"])
    centers, _, _, counts = bins_r

    # ---- text summary ----
    print("\n[fk] ===================== f(k) SUMMARY =====================")
    print(f"[fk]  cond={cond_tag}  [{_cond_label(data)}]  ({role})")
    print(f"[fk]  run={data['load_run']} ckpt={data['checkpoint']}  "
          f"num_envs={data['num_envs']} n_steps={data['n_steps']} cmd_vx={data['cmd_vx']}")
    print(f"[fk]  {'k_bin':>8} {'reward':>9} {'survival':>9} {'vx_err':>9} {'n':>6}")
    for b in range(len(centers)):
        print(f"[fk]  {centers[b]:8.3f} {bins_r[1][b]:9.4f} "
              f"{bins_s[1][b]:9.4f} {bins_v[1][b]:9.4f} {counts[b]:6d}")

    _plot_fk(data, bins_r, bins_s, bins_v, cond_tag)

    # ---- torque-floor / interior-optimum detection ----
    surv = bins_s[1]
    rew = bins_r[1]
    valid = ~np.isnan(surv)
    if valid.any():
        lo_surv = float(np.nanmean(surv[:5]))    # low-k third of [0.6,1.6]
        hi_surv = float(np.nanmean(surv[-5:]))   # high-k third
        rng_surv = float(np.nanmax(surv) - np.nanmin(surv))
        print(f"[fk]  survival: low-k(<~0.93)={lo_surv:.3f}  high-k(>~1.27)={hi_surv:.3f}  "
              f"range={rng_surv:.3f}")
        floor = (hi_surv - lo_surv) > 0.05  # low k clearly worse than high k
        all_collapse = float(np.nanmax(surv)) < 0.3
        if all_collapse:
            print("[fk]  *** ALL-COLLAPSE: survival < 0.3 even at best k -> TOO HARD, rerun milder")
        elif floor:
            # lightest k whose survival reaches >=90% of the max survival
            tgt = 0.9 * float(np.nanmax(surv))
            ok = np.where(valid & (surv >= tgt))[0]
            kstar = float(centers[ok[0]]) if len(ok) else float("nan")
            print(f"[fk]  *** TORQUE FLOOR detected: low-k survival collapses, recovers at higher k")
            print(f"[fk]  *** k* (lightest k reaching >=90% of max survival) ≈ {kstar:.3f}")
        else:
            print("[fk]  no clear survival torque-floor (survival ~flat in k)")
        # reward interior optimum
        best_b = int(np.nanargmax(rew))
        print(f"[fk]  reward-optimal k bin ≈ {centers[best_b]:.3f} (reward={rew[best_b]:.4f})")
    print("[fk] =========================================================\n")


def _plot_fk(data, bins_r, bins_s, bins_v, cond_tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (cs, mn, sd, ct), ylabel, title in [
        (axes[0], bins_r, "mean per-step reward", "Reward vs k"),
        (axes[1], bins_s, "survival (frac not terminated)", "Survival vs k"),
        (axes[2], bins_v, "mean |vx - cmd| (m/s)", "vx tracking error vs k"),
    ]:
        ax.errorbar(cs, mn, yerr=sd, marker="o", capsize=3, lw=1.5)
        ax.axvline(1.0, color="r", ls="--", lw=1.0, label="k=1 (nominal)")
        ax.set_xlabel("k = motor_torque_scale")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[1].set_ylim(-0.02, 1.05)  # survival is a fraction; fixed scale shows collapses

    fig.suptitle(
        f"f(k) — {cond_tag} [{_cond_label(data)}] — {data['load_run']} ckpt{data['checkpoint']}, "
        f"{data['n_steps']} steps, {data['num_envs']} envs",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, f"fk_{cond_tag}.png")
    fig.savefig(out, dpi=140)
    print(f"[fk] saved plot: {out}")


if __name__ == "__main__":
    main()
