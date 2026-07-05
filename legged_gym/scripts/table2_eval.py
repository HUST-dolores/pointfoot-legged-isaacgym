#!/usr/bin/env python3
"""Table-II evaluation (co-design Phase 1): held-out morphology generalization at FIXED difficulty.

Protocol (KNOWLEDGE_NOTES §12, aligned with Chen et al. arXiv:2407.06770 Table II):
  * HELD-OUT morphologies: (thigh,shank) ~ U[0.8,1.2]^2 sampled with morphology_seed=777.
    Training used seed 0 / 64 buckets -> a DIFFERENT set. Rationale: the BO outer loop will
    propose continuous xi values that are almost never equal to a training bucket, so the
    generalist must work on unseen sizes; also keeps the B-vs-NoDR comparison fair (neither
    policy has "seen" the eval robots) and checks the policy didn't memorize the 64 buckets.
    URDFs are written with prefix 'morphev' so training morph_*.urdf files are not clobbered.
  * 4 scenarios (obstacle/slope/load/accel) at FIXED difficulty: scenario.curriculum=False and
    scenario_level = EVAL_DIFFICULTY for every env. Curriculum must NOT run at eval, or weak
    morphologies hide behind easier conditions (KNOWLEDGE_NOTES §12 trap).
  * The same script evaluates ANY policy checkpoint under IDENTICAL conditions: same eval
    morph seed + same --seed -> identical robots, bucket assignment, scenario assignment.

Metrics per completed episode: episodic reward, length, fell (done before timeout).
Aggregation: per-scenario means + mean-of-bucket-means (the Table-II headline number).

Recording protocol: FIRST completed episode per env only (equal weight per env/morphology).
Recording every episode over a fixed window would weight metrics by each policy's own fall
frequency (fragile envs log many short fall episodes, robust envs log one timeout) and censor
the tail toward fast falls — audit-confirmed metric-corrupting bias.

Usage — ALWAYS pass --headless (the workstation has no X display; without --headless BaseTask
creates a viewer and core-dumps. This was the whole mystery behind measure_alt2 crashing there).
NEVER use --seed 1: helpers.set_seed treats seed==1 as "randomize" (helpers.py:74), which would
give each eval run different bucket/scenario/DR assignments; the script forces 12345 if you do.
  EVAL_BUCKETS=100 EVAL_DIFFICULTY=0.5 python legged_gym/scripts/table2_eval.py \
      --task=wheelfoot_flat --load_run <run> --checkpoint <iter> --num_envs 512 --seed 12345 --headless
Env-var knobs (get_args is gymutil-based, so knobs go via env vars like measure_alt2):
  EVAL_BUCKETS(100)  EVAL_DIFFICULTY(0.5)  EVAL_STEPS(0=auto: max_ep+250)  EVAL_OUT(auto)
  EVAL_NOMINAL=1 -> single nominal morphology (1.0,1.0) sanity row (prefix 'morphnom').
Run policies SEQUENTIALLY (regeneration writes shared morphev_*.urdf files).
Compare two saved runs:  python legged_gym/scripts/table2_compare.py a.npz b.npz
"""
import os
import json
import isaacgym  # noqa  (must import before torch)
from isaacgym.torch_utils import *  # noqa
from legged_gym.envs import *  # noqa
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play import _apply_saved_env_cfg
import numpy as np
import torch

EVAL_MORPH_SEED = 777
SCEN_NAMES = ["obstacle", "slope", "load", "accel"]


def main():
    args = get_args()
    assert args.load_run and args.checkpoint is not None, "--load_run and --checkpoint are required"
    # ★seed guard: helpers.set_seed REPLACES seed==1 with a random seed (helpers.py:74-75).
    # A randomized seed silently gives B and No-DR different bucket/scenario/DR assignments,
    # voiding the identical-conditions protocol the comparison rests on. Force a fixed seed.
    if args.seed is None or args.seed == 1:
        print("[table2] WARNING: seed None/1 would be randomized by set_seed — forcing seed=12345")
        args.seed = 12345
    n_buckets = int(os.environ.get("EVAL_BUCKETS", "100"))
    difficulty = float(os.environ.get("EVAL_DIFFICULTY", "0.5"))
    steps_env = int(os.environ.get("EVAL_STEPS", "0"))
    nominal = os.environ.get("EVAL_NOMINAL", "0") == "1"

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.task, train_cfg.runner.experiment_name)
    _apply_saved_env_cfg(env_cfg, train_cfg, args, log_root)

    env_cfg.env.num_envs = args.num_envs if args.num_envs else 512
    # deterministic, perturbation-free eval conditions (identical for every policy)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.add_random_load = False
    # held-out morphology set (or single nominal robot for the sanity row)
    env_cfg.domain_rand.randomize_morphology = True
    env_cfg.domain_rand.morphology_regenerate = True
    if nominal:
        env_cfg.domain_rand.morphology_num_buckets = 1
        env_cfg.domain_rand.morphology_thigh_scale_range = [1.0, 1.0]
        env_cfg.domain_rand.morphology_shank_scale_range = [1.0, 1.0]
        env_cfg.domain_rand.morphology_prefix = "morphnom"
        n_buckets = 1
    else:
        env_cfg.domain_rand.morphology_num_buckets = n_buckets
        # EVAL_PREFIX lets concurrent evals (e.g. two difficulty levels on two GPUs) use distinct
        # morph*.urdf files so regeneration doesn't race. seed=777 → same scales regardless of prefix.
        env_cfg.domain_rand.morphology_prefix = os.environ.get("EVAL_PREFIX", "morphev")
        # EVAL_THIGH/EVAL_SHANK: single FIXED test morphology (fine-tune efficiency validation on one xi)
        _th, _sh = os.environ.get("EVAL_THIGH", ""), os.environ.get("EVAL_SHANK", "")
        if _th and _sh:
            env_cfg.domain_rand.morphology_num_buckets = 1
            env_cfg.domain_rand.morphology_thigh_scale_range = [float(_th), float(_th)]
            env_cfg.domain_rand.morphology_shank_scale_range = [float(_sh), float(_sh)]
            n_buckets = 1
            print(f"[table2] single test morphology xi=({_th},{_sh})")
    env_cfg.domain_rand.morphology_seed = EVAL_MORPH_SEED
    # EVAL_MOTOR: fixed per-env motor torque scale k (3D co-design candidate; motor must match the fine-tune)
    _mo = os.environ.get("EVAL_MOTOR", "")
    if _mo:
        env_cfg.domain_rand.randomize_motor_design = True
        env_cfg.domain_rand.motor_torque_scale_range = [float(_mo), float(_mo)]
        print(f"[table2] fixed motor torque scale k={_mo}")
    # scenarios at FIXED difficulty (no curriculum at eval — §12 trap)
    env_cfg.scenario.partition = True
    env_cfg.scenario.curriculum = False
    env_cfg.scenario.curriculum_start_frac = difficulty

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history, commands, critic_obs = env.get_observations()
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, train_cfg=train_cfg, args=args)
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    assert getattr(env, "scenario_id", None) is not None, "scenario partition failed to initialize"
    assert getattr(env, "morph_bucket_ids", None) is not None, "morphology bucketing failed to initialize"

    max_ep = int(env.max_episode_length)
    T = steps_env if steps_env > 0 else max_ep + 250
    N = env.num_envs
    ep_rew = torch.zeros(N, device=env.device)
    ep_len = torch.ones(N, device=env.device)   # runner-init env.reset() already took 1 zero-action step
    recorded = torch.zeros(N, dtype=torch.bool, device=env.device)
    rec_bucket, rec_scen, rec_rew, rec_len, rec_fell = [], [], [], [], []
    # refresh obs post runner-init reset (the tuple fetched before make_alg_runner is stale)
    obs, obs_history, commands, critic_obs = env.get_observations()

    print(f"[table2] run={args.load_run} ckpt={args.checkpoint} envs={N} buckets={n_buckets} "
          f"difficulty={difficulty} T={T} (max_ep={max_ep}) nominal={nominal} seed={args.seed}")
    for i in range(T):
        with torch.no_grad():
            est = encoder(obs_history, obs)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(actions.detach())
        ep_rew += rews
        ep_len += 1
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            timeout = env.time_out_buf[done_ids].bool()
            for j, e in enumerate(done_ids.tolist()):
                # FIRST completed episode per env only — every env/morphology gets equal weight.
                # (Recording all episodes weights metrics by fall frequency + censors tail to fast falls.)
                if bool(recorded[e]):
                    continue
                recorded[e] = True
                rec_bucket.append(int(env.morph_bucket_ids[e]))
                rec_scen.append(int(env.scenario_id[e]))
                rec_rew.append(float(ep_rew[e]))
                rec_len.append(float(ep_len[e]))
                rec_fell.append(0 if bool(timeout[j]) else 1)
            ep_rew[done_ids] = 0.0
            ep_len[done_ids] = 0.0
        if bool(recorded.all()):
            print(f"[table2]  all {N} envs recorded at step {i} — done early")
            break
        if i % 500 == 0:
            print(f"[table2]  step {i}/{T}  envs recorded: {int(recorded.sum())}/{N}")
    n_miss = int((~recorded).sum())
    if n_miss > 0:
        print(f"[table2] WARNING: {n_miss} envs never completed an episode in T={T} steps (excluded)")

    bucket = np.array(rec_bucket); scen = np.array(rec_scen)
    rew = np.array(rec_rew); length = np.array(rec_len); fell = np.array(rec_fell)
    n_eps = len(rew)
    print(f"[table2] ================== TABLE-II EVAL ==================")
    print(f"[table2] run={args.load_run} ckpt={args.checkpoint}  episodes={n_eps}")
    print(f"[table2] {'scenario':<10} {'eps':>5} {'mean_rew':>9} {'fall%':>6} {'mean_len':>9}")
    for k, name in enumerate(SCEN_NAMES):
        m = scen == k
        if m.sum() == 0:
            print(f"[table2] {name:<10} {0:>5}      -      -        -"); continue
        print(f"[table2] {name:<10} {int(m.sum()):>5} {rew[m].mean():>9.2f} {100*fell[m].mean():>5.1f}% {length[m].mean():>9.1f}")
    print(f"[table2] {'ALL':<10} {n_eps:>5} {rew.mean():>9.2f} {100*fell.mean():>5.1f}% {length.mean():>9.1f}")
    # ★FITNESS f(xi) = equal-weight (25% each) mean of the 4 per-scenario mean-rewards (user-set weights)
    scen_means = [rew[scen == k].mean() for k in range(len(SCEN_NAMES)) if (scen == k).sum() > 0]
    fitness = float(np.mean(scen_means)) if len(scen_means) == len(SCEN_NAMES) else float("nan")
    print(f"[table2] ★ FITNESS (4-scenario equal-weight mean_rew) = {fitness:.2f}"
          + ("" if len(scen_means) == len(SCEN_NAMES) else f"  [WARN: only {len(scen_means)}/4 scenarios had episodes]"))
    # Table-II headline: mean of per-bucket means (each held-out morphology weighted equally)
    bucket_means = [rew[bucket == b].mean() for b in np.unique(bucket)]
    if len(bucket_means) < n_buckets:
        print(f"[table2] note: {n_buckets - len(bucket_means)} bucket(s) drew no envs (randint assignment) — "
              f"same subset for every policy under the same seed")
    print(f"[table2] mean-of-bucket-means = {np.mean(bucket_means):.2f}  over {len(bucket_means)} held-out morphologies")

    out = os.environ.get("EVAL_OUT", "")
    if not out:
        os.makedirs(os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "table2"), exist_ok=True)
        tag = f"{args.load_run}_ckpt{args.checkpoint}_d{difficulty:g}" + ("_nom" if nominal else "")
        out = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "table2", f"table2_{tag}.npz")
    meta = dict(load_run=str(args.load_run), checkpoint=int(args.checkpoint), buckets=n_buckets,
                difficulty=difficulty, seed=int(args.seed) if args.seed is not None else -1,
                num_envs=N, T=T, nominal=nominal, morph_seed=EVAL_MORPH_SEED)
    np.savez(out, bucket=bucket, scen=scen, rew=rew, length=length, fell=fell,
             morph_scales=env.morphology_scales.cpu().numpy(), meta=json.dumps(meta))
    print(f"[table2] saved -> {out}")


if __name__ == "__main__":
    main()
