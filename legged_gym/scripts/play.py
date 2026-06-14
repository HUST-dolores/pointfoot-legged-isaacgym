# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import json
import subprocess

import isaacgym
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    export_mlp_as_onnx,
    task_registry,
    Logger,
)

import numpy as np
import torch
import matplotlib.pyplot as plt


def _git_state():
    """Return (short_hash, is_dirty) for the current repo. Best-effort, never raises."""
    try:
        sh = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=LEGGED_GYM_ROOT_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        sh = "?"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=LEGGED_GYM_ROOT_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip())
    except Exception:
        dirty = False
    return sh, dirty


def _load_train_run_cfgs(run_dir):
    """Read env_cfg.json / train_cfg.json saved at train time. Returns (env_cfg_d, train_cfg_d) or (None, None)."""
    env_p = os.path.join(run_dir, "env_cfg.json")
    tr_p = os.path.join(run_dir, "train_cfg.json")
    env_d, tr_d = None, None
    if os.path.isfile(env_p):
        try:
            with open(env_p) as f:
                env_d = json.load(f)
        except Exception:
            pass
    if os.path.isfile(tr_p):
        try:
            with open(tr_p) as f:
                tr_d = json.load(f)
        except Exception:
            pass
    return env_d, tr_d


def _dig(d, *keys, default=None):
    """Safe nested-dict get. _dig(d, 'env', 'use_qs_in_obs')."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _apply_saved_env_cfg(env_cfg, train_cfg, args, log_root):
    """Override env_cfg/train_cfg with saved-at-train-time values for the loaded run.

    The checkpoint's actor/critic/encoder layers were sized for the env layout used
    during training. If the in-tree config differs (e.g., flipped use_qs_in_obs,
    changed encoder dim), `runner.load(resume_path)` fails with shape mismatch.
    This auto-aligns the critical fields before make_env/make_alg_runner.
    """
    load_run = getattr(args, "load_run", None) or ""
    if not load_run:
        return
    run_dir = os.path.join(log_root, load_run)
    env_p = os.path.join(run_dir, "env_cfg.json")
    tr_p = os.path.join(run_dir, "train_cfg.json")
    if not os.path.isfile(env_p):
        print(f"[PLAY] saved env_cfg.json not found in {run_dir}; skipping auto-override")
        return
    try:
        saved_env = json.load(open(env_p))
    except Exception as e:
        print(f"[PLAY] failed to read saved env_cfg.json: {e}")
        return
    saved_tr = None
    if os.path.isfile(tr_p):
        try:
            saved_tr = json.load(open(tr_p))
        except Exception:
            pass

    applied = []

    def _ovr(obj, attr, new):
        old = getattr(obj, attr, None)
        if old != new:
            applied.append(f"{obj.__class__.__name__}.{attr}: {old} -> {new}")
            setattr(obj, attr, new)

    # --- env dimensions (drives actor/critic input width) ---
    se = saved_env.get("env", {}) if isinstance(saved_env.get("env"), dict) else {}
    for k in ("num_observations", "num_critic_observations"):
        if se.get(k) is not None:
            _ovr(env_cfg.env, k, int(se[k]))
    # --- env flags affecting obs structure ---
    # 字段存在：使用 saved 值。
    # 字段缺失（older runs）：use_torques_in_obs 的默认是 True（该 flag 2026-05-24 才加，
    # 之前所有 ckpt 都有 raw torques）；其他 flag 若缺失则不动 in-tree 默认。
    _flag_defaults_when_missing = {"use_torques_in_obs": True}
    for k in ("use_qs_in_obs", "use_residual_learning", "use_torques_in_obs"):
        sv = se.get(k)
        if sv is not None:
            _ovr(env_cfg.env, k, bool(sv))
        elif k in _flag_defaults_when_missing:
            _ovr(env_cfg.env, k, _flag_defaults_when_missing[k])

    # --- algorithm flag affecting encoder interpretation ---
    if saved_tr is not None:
        sa = saved_tr.get("algorithm", {}) if isinstance(saved_tr.get("algorithm"), dict) else {}
        if sa.get("use_load_residual_estimation") is not None:
            _ovr(train_cfg.algorithm, "use_load_residual_estimation",
                 bool(sa["use_load_residual_estimation"]))
        # --- encoder input dim (drives encoder's first layer) ---
        sm = saved_tr.get("MLP_Encoder", {}) if isinstance(saved_tr.get("MLP_Encoder"), dict) else {}
        if sm.get("num_input_dim") is not None:
            _ovr(train_cfg.MLP_Encoder, "num_input_dim", int(sm["num_input_dim"]))
        if sm.get("obs_history_length") is not None:
            _ovr(train_cfg.MLP_Encoder, "obs_history_length", int(sm["obs_history_length"]))
        if sa.get("extra_loss_load_boost") is not None:
            _ovr(
                train_cfg.algorithm,
                "extra_loss_load_boost",
                float(sa["extra_loss_load_boost"]),
            )

    if applied:
        print(f"[PLAY] applied {len(applied)} cfg override(s) from saved cfg "
              f"(to match ckpt arch):")
        for s in applied:
            print(f"       - {s}")
    else:
        print(f"[PLAY] in-tree cfg already matches saved cfg; no overrides needed")


def _cmd_tag(vx, vy, yaw, eps=0.05):
    """Build a short command descriptor: 'static' if all zeros, else 'walk_vx0.5' / 'yaw0.3' / combined."""
    parts = []
    if abs(vx) > eps:
        parts.append(f"vx{vx:g}")
    if abs(vy) > eps:
        parts.append(f"vy{vy:g}")
    if abs(yaw) > eps:
        parts.append(f"yaw{yaw:g}")
    if not parts:
        return "static"
    return "walk_" + "_".join(parts)


def _build_auto_exp_tag(env, train_cfg, args, cmd_vx, cmd_vy, cmd_yaw):
    """Build exp_tag of form lb{N}_qs{0|1}_resid{0|1}_torq{0|1}_{cmd}_seed{S}_ckpt{C}_[load{lo}-{hi}].

    `torq` 后缀仅在 use_torques_in_obs=False 时插入（避免污染历史 tag 命名）。
    """
    lb_val = getattr(train_cfg.algorithm, "extra_loss_load_boost", None)
    lb_s = f"{lb_val:g}" if lb_val is not None else "x"
    qs = int(bool(getattr(env.cfg.env, "use_qs_in_obs", True)))
    resid = int(bool(getattr(train_cfg.algorithm, "use_load_residual_estimation", False)))
    torq = bool(getattr(env.cfg.env, "use_torques_in_obs", True))
    seed = getattr(env.cfg, "seed", "x")
    ckpt = getattr(args, "checkpoint", "x")
    torq_seg = "" if torq else "_torq0"
    tag = f"lb{lb_s}_qs{qs}_resid{resid}{torq_seg}_{_cmd_tag(cmd_vx, cmd_vy, cmd_yaw)}_seed{seed}_ckpt{ckpt}"
    # Append load range suffix only when user overrode it (OOD case).
    lm_min = float(getattr(args, "load_mass_min", -1.0) or -1.0)
    lm_max = float(getattr(args, "load_mass_max", -1.0) or -1.0)
    if lm_min >= 0.0 and lm_max >= 0.0:
        tag += f"_load{lm_min:g}-{lm_max:g}"
    if bool(getattr(args, "flat_terrain", False)):
        tag += "_flat"
    if bool(getattr(args, "no_load", False)):
        tag += "_noload"
    _sl = float(getattr(args, "slope_deg", 0.0) or 0.0)
    if abs(_sl) > 1e-6:
        tag += f"_slope{_sl:g}"
    _ev = float(getattr(args, "estop_vx", 0.0) or 0.0)
    if _ev > 1e-6:
        tag += f"_estop{_ev:g}"
    # Eccentric-load suffix (0.0 is a valid centered value, so do NOT use the `or` idiom).
    _lox = float(getattr(args, "load_offset_x", -99.0))
    _loy = float(getattr(args, "load_offset_y", -99.0))
    if _lox > -90 or _loy > -90:
        tag += f"_ecc{(_lox if _lox > -90 else 0.0):g}x{(_loy if _loy > -90 else 0.0):g}y"
    _cm = str(getattr(args, "corrupt_estimate", "none") or "none")
    if _cm != "none":
        _cs = str(getattr(args, "corrupt_estimate_scope", "obs") or "obs")
        scope_seg = "" if _cs == "obs" else f"{_cs}"
        tag += f"_corr{scope_seg}{_cm}{getattr(args, 'corrupt_val', 0.0):g}"
    _ldur = float(getattr(args, "load_dur", -1.0)); _lint = float(getattr(args, "load_int", -1.0))
    if _ldur > 0 and _lint > 0:
        tag += f"_cycle{_ldur:g}-{_lint:g}"
    # Append ext-force suffix (Exp A) so force runs don't collide with mass runs.
    eff_kg = float(getattr(args, "ext_force_down_kg", 0.0) or 0.0)
    eff_kg_min = float(getattr(args, "ext_force_down_kg_min", 0.0) or 0.0)
    eff_kg_max = float(getattr(args, "ext_force_down_kg_max", 0.0) or 0.0)
    efx = float(getattr(args, "ext_force_x", 0.0) or 0.0)
    efy = float(getattr(args, "ext_force_y", 0.0) or 0.0)
    efz = float(getattr(args, "ext_force_z", 0.0) or 0.0)
    if eff_kg_min > 0.0 and eff_kg_max >= eff_kg_min:
        fdir = str(getattr(args, "ext_force_dir", "down") or "down").lower()
        tag += f"_f{fdir}{eff_kg_min:g}-{eff_kg_max:g}kg"
    elif eff_kg > 0.0:
        tag += f"_fdown{eff_kg:g}kg"
    elif abs(efx) > 1e-9 or abs(efy) > 1e-9 or abs(efz) > 1e-9:
        tag += f"_fxyz{efx:g}-{efy:g}-{efz:g}N"
    return tag


def _corrupt_policy_estimate_latent(est, mode, val, env):
    """Corrupt mass/CoM entries in the policy-facing encoder latent.

    The latent layout follows the play logging convention: indices 3:7 are
    mass and CoM delta in normalized units. This is a test-time ablation only.
    """
    if mode == "none" or est is None or est.shape[-1] < 7:
        return est
    out = est.clone()
    mass_scale = float(env.cfg.normalization.obs_scales.mass_scale)
    if mode in ("zero", "zero_all"):
        out[:, 3:7] = 0.0
    elif mode == "scale":
        out[:, 3:7] = out[:, 3:7] * float(val)
    elif mode == "fixed":
        out[:, 3] = float(val) * mass_scale
        out[:, 4:7] = 0.0
    elif mode == "noise":
        out[:, 3] = out[:, 3] + float(val) * mass_scale * torch.randn_like(out[:, 3])
    elif mode == "shuffle":
        # 跨 env 互换 mass/CoM latent(固定排列,保持分布内但与本 env 真实负载错配)。
        if not hasattr(env, "_shuffle_perm") or env._shuffle_perm.shape[0] != out.shape[0]:
            env._shuffle_perm = torch.randperm(out.shape[0], device=out.device)
        out[:, 3:7] = est[env._shuffle_perm, 3:7]
    return out


def _sanitize_for_mat(obj):
    """Recursively replace None with '' so scipy.io.savemat can serialize the dict."""
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return {k: _sanitize_for_mat(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_mat(v) for v in obj]
    return obj


def _collect_run_verify(env, train_cfg, args, log_root):
    """Collect run-config verification info — saved-at-train-time vs current-play-time.

    Returns a dict suitable for both embedding into the .mat file's `meta` and
    feeding into `_print_run_verify_block` for human-readable output.
    """
    # --- locate the loaded run directory ---
    load_run = getattr(args, "load_run", None) or ""
    ckpt = getattr(args, "checkpoint", None)
    ckpt_str = str(ckpt) if ckpt not in (None, "") else "?"
    run_dir = os.path.join(log_root, load_run) if load_run else ""
    ckpt_path = os.path.join(run_dir, f"model_{ckpt_str}.pt") if run_dir else ""
    ckpt_exists = os.path.isfile(ckpt_path) if ckpt_path else False

    # --- read saved train-time configs ---
    saved_env, saved_tr = _load_train_run_cfgs(run_dir) if run_dir else (None, None)
    saved_found = (saved_env is not None) or (saved_tr is not None)

    # --- current (play-time) values ---
    cur_env_cfg = env.cfg
    cur = {
        "use_qs_in_obs":              bool(getattr(cur_env_cfg.env, "use_qs_in_obs", True)),
        "use_residual_learning":      bool(getattr(cur_env_cfg.env, "use_residual_learning", False)),
        "use_torques_in_obs":         bool(getattr(cur_env_cfg.env, "use_torques_in_obs", True)),
        "use_load_residual_estimation": bool(getattr(train_cfg.algorithm, "use_load_residual_estimation", False)),
        "num_observations":           int(getattr(cur_env_cfg.env, "num_observations", -1)),
        "extra_loss_load_boost":      float(getattr(train_cfg.algorithm, "extra_loss_load_boost", float("nan"))),
        "seed":                       int(getattr(cur_env_cfg, "seed", -1)),
        "num_envs":                   int(getattr(cur_env_cfg.env, "num_envs", -1)),
    }
    # --- saved (train-time) values ---
    if saved_found:
        saved = {
            "use_qs_in_obs":              _dig(saved_env, "env", "use_qs_in_obs"),
            "use_residual_learning":      _dig(saved_env, "env", "use_residual_learning"),
            "use_torques_in_obs":         _dig(saved_env, "env", "use_torques_in_obs"),
            "use_load_residual_estimation": _dig(saved_tr,  "algorithm", "use_load_residual_estimation"),
            "num_observations":           _dig(saved_env, "env", "num_observations"),
            "extra_loss_load_boost":      _dig(saved_tr,  "algorithm", "extra_loss_load_boost"),
            "seed":                       _dig(saved_env, "seed"),
            "num_envs":                   _dig(saved_env, "env", "num_envs"),
        }
    else:
        saved = {k: None for k in cur}

    # --- git state ---
    commit, dirty = _git_state()

    # --- detect critical mismatches ---
    # These flags MUST match between train and play, otherwise the encoder
    # output is interpreted differently and results are silently wrong.
    critical = ["use_qs_in_obs", "use_residual_learning", "use_torques_in_obs",
                "use_load_residual_estimation", "num_observations"]
    mismatches = []
    for k in critical:
        sv, cv = saved.get(k), cur.get(k)
        if sv is not None and sv != cv:
            mismatches.append({"key": k, "train": sv, "play": cv})

    # --- single-line paste version ---
    paste = (
        f"run={load_run} ckpt={ckpt_str} "
        f"qs={cur['use_qs_in_obs']} resid_learn={cur['use_residual_learning']} "
        f"torq={cur['use_torques_in_obs']} "
        f"resid_est={cur['use_load_residual_estimation']} "
        f"n_obs={cur['num_observations']} lb={cur['extra_loss_load_boost']:g} "
        f"seed={cur['seed']} n_envs={cur['num_envs']} "
        f"commit={commit}{'-dirty' if dirty else ''} "
        f"tag={getattr(args, 'exp_tag', '') or '-'}"
    )

    return {
        "exp_tag":      getattr(args, "exp_tag", "") or "",
        "load_run":     load_run,
        "checkpoint":   ckpt_str,
        "ckpt_path":    ckpt_path,
        "ckpt_exists":  bool(ckpt_exists),
        "run_dir":      run_dir,
        "saved_found":  bool(saved_found),
        "commit":       commit,
        "dirty":        bool(dirty),
        "saved_cfg":    saved,
        "play_cfg":     cur,
        "mismatches":   mismatches,
        "paste_line":   paste,
    }


def _print_run_verify_block(v):
    """Print the verification block from a verify dict produced by _collect_run_verify."""
    saved = v["saved_cfg"]
    cur = v["play_cfg"]
    print("[play] ============== run verification =============")
    print(f"  exp_tag        : {v['exp_tag'] or '(none)'}")
    print(f"  load_run       : {v['load_run'] or '(none)'}")
    print(f"  checkpoint     : model_{v['checkpoint']}.pt  {'OK' if v['ckpt_exists'] else 'NOT FOUND'}")
    print(f"  ckpt_path      : {v['ckpt_path']}")
    print(f"  code commit    : {v['commit']}{'  [DIRTY working tree]' if v['dirty'] else ''}")
    if not v["saved_found"]:
        print(f"  saved cfg      : NOT FOUND in {v['run_dir']}  -> can't verify train/play match")
    print(f"")
    print(f"  {'key':<32} {'train(saved)':<16} {'play(current)':<16} {'match':<6}")
    print(f"  {'-'*32} {'-'*16} {'-'*16} {'-'*6}")
    for k in ["use_qs_in_obs", "use_residual_learning", "use_torques_in_obs",
              "use_load_residual_estimation", "num_observations",
              "extra_loss_load_boost", "seed", "num_envs"]:
        sv, cv = saved.get(k), cur.get(k)
        sv_s = "N/A" if sv is None else str(sv)
        cv_s = str(cv)
        if sv is None:
            mark = "  ? "
        elif sv == cv:
            mark = "  ok"
        else:
            mark = " ** "  # critical mismatch
        # num_envs / seed mismatches are informational, not critical
        if k in ("num_envs", "seed") and sv is not None and sv != cv:
            mark = " (i)"
        print(f"  {k:<32} {sv_s:<16} {cv_s:<16} {mark}")
    print(f"")
    if v["mismatches"]:
        print(f"  ** WARNING: {len(v['mismatches'])} CRITICAL flag(s) differ from train-time config:")
        for m in v["mismatches"]:
            print(f"     - {m['key']}: train={m['train']}  play={m['play']}")
        print(f"     -> encoder outputs may be misinterpreted; results NOT trustworthy")
    elif v["saved_found"]:
        print(f"  all critical flags match train-time config")
    print(f"")
    print(f"  paste:  {v['paste_line']}")
    print("[play] =============================================")


def _print_metrics_pretty(m):
    """Pretty-print metrics returned by _compute_experiment_metrics."""
    def g(k1, k2):
        try:
            v = m[k1][k2]
            return f"{v:+.4f}" if v >= 0 else f"{v:+.4f}"
        except Exception:
            return "  N/A "

    def gpos(k1, k2):
        try:
            v = m[k1][k2]
            return f"{v:.4f}" if v >= 0 else "  N/A "
        except Exception:
            return "  N/A "

    print("[play] ============= experiment metrics =============")
    print(f"  num_envs = {m.get('num_envs', '?')}")
    print(f"  [QS] = Model C 解析公式  |  [RL] = Encoder 最终输出")
    print(f"")
    print(f"  ====== LOAD MASS (kg) ======")
    print(f"    [QS]  RMSE={gpos('rmse','mass')}  bias={g('bias','mass')}  "
          f"per_env={gpos('rmse_per_env_mean','mass')}±{gpos('rmse_per_env_std','mass')}")
    print(f"    [RL]  RMSE={gpos('rmse','enc_mass')}  bias={g('bias','enc_mass')}  "
          f"per_env={gpos('rmse_per_env_mean','enc_mass')}±{gpos('rmse_per_env_std','enc_mass')}")
    print(f"")
    print(f"  ====== LOAD POSITION in body frame (m) — [QS] only task ======")
    print(f"    [QS]  com_x RMSE={gpos('rmse','com_x')}  bias={g('bias','com_x')}")
    print(f"    [QS]  com_y RMSE={gpos('rmse','com_y')}  bias={g('bias','com_y')}")
    print(f"")
    print(f"  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======")
    print(f"    [QS]  x: RMSE={gpos('rmse','dcom_x_mc')}  bias={g('bias','dcom_x_mc')}  |  "
          f"y: RMSE={gpos('rmse','dcom_y_mc')}  bias={g('bias','dcom_y_mc')}")
    print(f"    [RL]  x: RMSE={gpos('rmse','dcom_x_e')}   bias={g('bias','dcom_x_e')}   |  "
          f"y: RMSE={gpos('rmse','dcom_y_e')}   bias={g('bias','dcom_y_e')}")
    print(f"")
    print(f"  ====== convergence (mass < {m.get('thresholds',{}).get('mass_kg','?')} kg, "
          f"hold ≥ {m.get('thresholds',{}).get('hold_s','?')} s) ======")
    print(f"    [QS]  reached={gpos('conv_time_reached_frac','mass')}  @ avg "
          f"{gpos('conv_time_per_env_mean','mass')} s")
    print(f"    [RL]  reached={gpos('conv_time_reached_frac','enc_mass')}  @ avg "
          f"{gpos('conv_time_per_env_mean','enc_mass')} s")
    print(f"")
    print(f"  ====== dynamic-phase RMSE (|vx| > "
          f"{m.get('thresholds',{}).get('motion_v_mps','?')} m/s) ======")
    print(f"    [QS]  mass={gpos('dyn_phase_per_env_mean','mass')} kg  "
          f"dcom_x={gpos('dyn_phase_per_env_mean','dcom_x_mc')} m  "
          f"dcom_y={gpos('dyn_phase_per_env_mean','dcom_y_mc')} m")
    print(f"    [RL]  mass={gpos('dyn_phase_per_env_mean','enc_mass')} kg  "
          f"dcom_x={gpos('dyn_phase_per_env_mean','dcom_x_e')} m  "
          f"dcom_y={gpos('dyn_phase_per_env_mean','dcom_y_e')} m")
    print("[play] =============================================")


def _compute_experiment_metrics(logger, dt):
    """Compute headline metrics from Logger.

    Uses logger.state_log_full when available (per-env arrays, shape (T, N)) to
    compute statistics over ALL envs at once. Falls back to state_log (robot 0
    only, shape (T,)) if full data is missing.

    Returns nested dict → MATLAB struct with:
      rmse:      {mass, com_x, com_y} overall RMSE across (T, N) samples (kg / m).
      rmse_per_env_mean/std: per-env RMSE distribution stats (N envs).
      bias:      signed mean error over (T, N).
      conv_time_per_env_mean/std: per-env convergence time stats (s); env=-1 if never.
      dyn_phase: per-env RMS error restricted to time steps with |base_lin_vel_x| > thr.
      num_envs:  effective env count.
    """
    import numpy as np

    metrics = {}

    # Prefer per-env data when available; else fall back to 1D series.
    have_full = (
        "payload_mass" in logger.state_log_full
        and "payload_mass_ref" in logger.state_log_full
    )

    def _arr_full(key):
        lst = logger.state_log_full.get(key)
        return np.stack(lst, axis=0) if lst else None  # (T, N)

    def _arr_1d(key):
        v = logger.state_log.get(key)
        return np.asarray(v) if v else None

    # (label, est_key, ref_key, threshold).
    # 三组对比：
    #   mass / enc_mass: 都是 load_mass (kg)，可直接比；ref 是真实 load mass。
    #   com_x / com_y:   Model C 预测的"载荷在机体系下的 x/y 位置"，ref 是真实载荷位置。
    #   dcom_*_mc / dcom_*_e: Model C 和 encoder 各自预测的 com_delta，ref 是真值 com_delta。
    # 注意 com_x / com_y 跟 dcom_x/y 不是同一物理量（前者是载荷位置，后者是总 CoM 偏移），
    # 但 dcom_*_mc vs dcom_*_e 是同一量，可以直接比较"残差网络好坏"。
    pairs = [
        ("mass",       "payload_mass",        "payload_mass_ref",   0.5),    # Model C load mass
        ("enc_mass",   "encoder_mass",        "payload_mass_ref",   0.5),    # Encoder load mass
        ("com_x",      "load_x",              "load_x_ref",         0.05),   # Model C load 位置 x
        ("com_y",      "load_y",              "load_y_ref",         0.05),   # Model C load 位置 y
        ("dcom_x_mc",  "modelC_com_delta_x",  "true_com_delta_x",   0.02),   # Model C com_delta x
        ("dcom_x_e",   "encoder_com_delta_x", "true_com_delta_x",   0.02),   # Encoder com_delta x
        ("dcom_y_mc",  "modelC_com_delta_y",  "true_com_delta_y",   0.02),   # Model C com_delta y
        ("dcom_y_e",   "encoder_com_delta_y", "true_com_delta_y",   0.02),   # Encoder com_delta y
    ]
    hold_s = 0.5
    hold_steps = max(1, int(round(hold_s / dt)))
    motion_thresh = 0.15

    rmse, rmse_per_env_mean, rmse_per_env_std = {}, {}, {}
    bias = {}
    conv_per_env_mean, conv_per_env_std, conv_frac_reached = {}, {}, {}
    dyn_per_env_mean = {}
    n_envs = -1

    if have_full:
        bv_all = _arr_full("base_lin_vel_x")        # (T, N) or None
        motion_mask_all = np.abs(bv_all) > motion_thresh if bv_all is not None else None

        for name, est_key, ref_key, thr in pairs:
            est = _arr_full(est_key)               # (T, N)
            ref = _arr_full(ref_key)
            if est is None or ref is None or est.shape != ref.shape:
                rmse[name] = -1.0; bias[name] = 0.0
                rmse_per_env_mean[name] = -1.0; rmse_per_env_std[name] = -1.0
                conv_per_env_mean[name] = -1.0; conv_per_env_std[name] = -1.0
                conv_frac_reached[name] = 0.0; dyn_per_env_mean[name] = -1.0
                continue
            err = est - ref                        # (T, N)
            T, N = err.shape
            n_envs = N
            rmse[name] = float(np.sqrt(np.mean(err ** 2)))
            bias[name] = float(np.mean(err))
            per_env = np.sqrt(np.mean(err ** 2, axis=0))  # (N,)
            rmse_per_env_mean[name] = float(per_env.mean())
            rmse_per_env_std[name] = float(per_env.std())

            # Per-env convergence time: first t with hold_steps consecutive |err|<thr.
            within = np.abs(err) < thr             # (T, N)
            ct = np.full(N, -1.0)
            for j in range(N):
                run = 0
                for i in range(T):
                    run = run + 1 if within[i, j] else 0
                    if run >= hold_steps:
                        ct[j] = (i - hold_steps + 1) * dt
                        break
            reached = ct >= 0
            conv_frac_reached[name] = float(reached.mean())
            conv_per_env_mean[name] = float(ct[reached].mean()) if reached.any() else -1.0
            conv_per_env_std[name] = float(ct[reached].std()) if reached.any() else -1.0

            # Dynamic-phase per-env: keep only timesteps where env is moving
            if motion_mask_all is not None and motion_mask_all.shape == err.shape:
                masked_sq = np.where(motion_mask_all, err ** 2, 0.0)
                cnt = motion_mask_all.sum(axis=0).astype(float)
                cnt_safe = np.where(cnt > 0, cnt, 1.0)
                per_env_dyn = np.sqrt(masked_sq.sum(axis=0) / cnt_safe)
                per_env_dyn = np.where(cnt > 0, per_env_dyn, -1.0)
                valid = per_env_dyn >= 0
                dyn_per_env_mean[name] = float(per_env_dyn[valid].mean()) if valid.any() else -1.0
            else:
                dyn_per_env_mean[name] = -1.0
    else:
        # Fallback to 1D (robot 0 only). Same structure but stats are degenerate.
        n_envs = 1
        bv = _arr_1d("base_lin_vel_x")
        if bv is None:
            bv = _arr_1d("base_vel_x")
        motion_mask = (np.abs(bv) > motion_thresh) if bv is not None else None
        for name, est_key, ref_key, thr in pairs:
            est, ref = _arr_1d(est_key), _arr_1d(ref_key)
            if est is None or ref is None or est.shape != ref.shape:
                rmse[name] = -1.0; bias[name] = 0.0
                rmse_per_env_mean[name] = -1.0; rmse_per_env_std[name] = 0.0
                conv_per_env_mean[name] = -1.0; conv_per_env_std[name] = 0.0
                conv_frac_reached[name] = 0.0; dyn_per_env_mean[name] = -1.0
                continue
            err = est - ref
            rmse[name] = float(np.sqrt(np.mean(err ** 2)))
            bias[name] = float(np.mean(err))
            rmse_per_env_mean[name] = rmse[name]; rmse_per_env_std[name] = 0.0
            within = np.abs(err) < thr
            ct = -1.0
            if within.any():
                run = 0
                for i, ok in enumerate(within):
                    run = run + 1 if ok else 0
                    if run >= hold_steps:
                        ct = (i - hold_steps + 1) * dt; break
            conv_per_env_mean[name] = ct
            conv_per_env_std[name] = 0.0
            conv_frac_reached[name] = 1.0 if ct >= 0 else 0.0
            if motion_mask is not None and motion_mask.shape == err.shape and motion_mask.any():
                dyn_per_env_mean[name] = float(np.sqrt(np.mean(err[motion_mask] ** 2)))
            else:
                dyn_per_env_mean[name] = -1.0

    metrics["rmse"] = rmse
    metrics["bias"] = bias
    metrics["rmse_per_env_mean"] = rmse_per_env_mean
    metrics["rmse_per_env_std"] = rmse_per_env_std
    metrics["conv_time_per_env_mean"] = conv_per_env_mean
    metrics["conv_time_per_env_std"] = conv_per_env_std
    metrics["conv_time_reached_frac"] = conv_frac_reached
    metrics["dyn_phase_per_env_mean"] = dyn_per_env_mean
    metrics["num_envs"] = n_envs
    metrics["thresholds"] = {
        "mass_kg": 0.5, "com_m": 0.05, "hold_s": hold_s, "motion_v_mps": motion_thresh,
    }
    return metrics


def play(args):
    # 单关节测试模式：通过位置控制让指定关节缓慢旋转，观察动力学
    test_joint_mode = getattr(args, 'test_joint_mode', False)
    test_joint_name = getattr(args, 'test_joint_name', None)
    test_joint_amplitude = getattr(args, 'test_joint_amplitude', 0.3)  # rad
    test_joint_period = getattr(args, 'test_joint_period', 4.0)  # seconds
    test_joint_offset = getattr(args, 'test_joint_offset', 0.0)  # rad
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Auto-align config with the loaded checkpoint's training-time cfg.
    # Without this, e.g. playing a history_only ckpt while in-tree cfg has
    # use_qs_in_obs=True will fail at runner.load() with shape mismatch.
    _log_root_for_align = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", args.task,
        train_cfg.runner.experiment_name,
    )
    _apply_saved_env_cfg(env_cfg, train_cfg, args, _log_root_for_align)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 60
    # 显式 --num_envs 则尊重其值(可>30,用于统计更平滑);未显式则默认上限 30(交互式)
    if getattr(args, "num_envs", None) is None:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 30)

    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    # 跟训练 config 对齐：使用相同的地形 patch 类型分布
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.terrain.max_init_terrain_level = 0  # 全部最低难度
    env_cfg.terrain.curriculum = True  # 保持 True 走 curiculum() → difficulty=i/num_rows=0

    # Trimesh can crash silently when the mesh is too large for GPU PhysX.
    # 加大 patch 和 border，让机器人走 40s × 0.5 m/s = 20m 都不到边界，避免 reset 打断。
    if env_cfg.terrain.mesh_type == "trimesh":
        env_cfg.terrain.num_rows = 1           # 只 1 级难度，无升级空间
        env_cfg.terrain.num_cols = 10
        env_cfg.terrain.terrain_length = 20.0  # X 方向每个 patch 20m (走得开)
        env_cfg.terrain.terrain_width = 10.0   # Y 方向每个 patch 10m
        env_cfg.terrain.border_size = 20       # 周围 20m 平坦 border 作余量
        env_cfg.terrain.max_init_terrain_level = 0  # 强制 level 0
        print(
            "[PLAY] trimesh layout (all level-0, non-flat, spacious for sustained walking): "
            f"rows={env_cfg.terrain.num_rows}, cols={env_cfg.terrain.num_cols}, "
            f"length={env_cfg.terrain.terrain_length}m, width={env_cfg.terrain.terrain_width}m, "
            f"border={env_cfg.terrain.border_size}m, max_init_level={env_cfg.terrain.max_init_terrain_level}"
        )

    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5
    env_cfg.domain_rand.push_curriculum = False
    env_cfg.domain_rand.max_push_vel_xy = 2.0
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.domain_rand.load_debug = True

    # CLI override: load mass range (for OOD experiments). Negative means no override.
    _lm_min = float(getattr(args, "load_mass_min", -1.0) or -1.0)
    _lm_max = float(getattr(args, "load_mass_max", -1.0) or -1.0)
    if _lm_min >= 0.0 and _lm_max >= 0.0:
        env_cfg.domain_rand.add_load_range = [_lm_min, _lm_max]
        print(f"[PLAY] override add_load_range = {env_cfg.domain_rand.add_load_range}")

    # CLI override: sustained external force (Exp A: 外力 vs 重物辨识). World frame, -Z = downward.
    _eff_kg = float(getattr(args, "ext_force_down_kg", 0.0) or 0.0)
    _eff_kg_min = float(getattr(args, "ext_force_down_kg_min", 0.0) or 0.0)
    _eff_kg_max = float(getattr(args, "ext_force_down_kg_max", 0.0) or 0.0)
    _ext_force_vec = [
        float(getattr(args, "ext_force_x", 0.0) or 0.0),
        float(getattr(args, "ext_force_y", 0.0) or 0.0),
        float(getattr(args, "ext_force_z", 0.0) or 0.0),
    ]
    # per-env sweep: each env a distinct downward force via linspace(min,max,num_envs)
    _ext_force_sweep = (_eff_kg_min > 0.0 and _eff_kg_max >= _eff_kg_min)
    if _eff_kg > 0.0:  # convenience: single downward force = _eff_kg of payload weight (all envs)
        _ext_force_vec[2] = -_eff_kg * 9.81
    _ext_force_on = _ext_force_sweep or any(abs(v) > 1e-9 for v in _ext_force_vec)
    _keep_load = bool(getattr(args, "keep_load", False))
    if _ext_force_on and not _keep_load:
        # 加力试验不带真实负载：关闭负载生成，使唯一的附加效应就是这个力（Exp A 默认）
        env_cfg.domain_rand.add_random_load = False
    elif _ext_force_on and _keep_load:
        print("[PLAY] --keep_load: ext_force 与真实负载同时保留(抗倾覆推-恢复实验)")
        if _ext_force_sweep:
            print(f"[PLAY] EXT-FORCE SWEEP ON: dir={str(getattr(args,'ext_force_dir','down'))} "
                  f"per-env force = linspace({_eff_kg_min},{_eff_kg_max}) kg-equiv; add_random_load -> False")
        else:
            print(f"[PLAY] EXT-FORCE mode ON: world force [N] = {_ext_force_vec} "
                  f"(down_kg={_eff_kg}); add_random_load -> False")

    # Exp B: 强制平地，使姿态指标只反映控制、不含地形跟随
    if bool(getattr(args, "flat_terrain", False)):
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.measure_heights = False
        print("[PLAY] FLAT terrain mode: terrain.mesh_type='plane'")

    # Exp B baseline: 无负载（隔离负载引起的控制退化）
    if bool(getattr(args, "no_load", False)):
        env_cfg.domain_rand.add_random_load = False
        print("[PLAY] NO-LOAD mode: domain_rand.add_random_load=False")

    # Ch5 斜坡标定：负载全程恒定（不 on/off 循环），使翻倒由负载决定而非瞬态
    if bool(getattr(args, "load_hold", False)):
        env_cfg.domain_rand.load_start_time_s = 0.5
        env_cfg.domain_rand.load_duration_range_s = [1.0e6, 1.0e6]
        env_cfg.domain_rand.load_interval_range_s = [1.0e6, 1.0e6]
        print("[PLAY] LOAD-HOLD: 负载 0.5s 起全程恒定挂着")

    # 时序图专用:固定 ON 时长 + spawn 间隔 → 所有 env 同步加/卸(看估计瞬态响应)
    _ldur = float(getattr(args, "load_dur", -1.0)); _lint = float(getattr(args, "load_int", -1.0))
    if _ldur > 0 and _lint > 0:
        _lstart = float(getattr(args, "load_start", -1.0))
        if _lstart >= 0: env_cfg.domain_rand.load_start_time_s = _lstart
        env_cfg.domain_rand.add_random_load = True
        env_cfg.domain_rand.load_duration_range_s = [_ldur, _ldur]
        env_cfg.domain_rand.load_interval_range_s = [_lint, _lint]
        print(f"[PLAY] LOAD-CYCLE(sync): on={_ldur}s every {_lint}s, start={env_cfg.domain_rand.load_start_time_s}s")

    # Ch5 斜坡标定：向前倾斜重力 slope_deg 度模拟坡度（平地+倾斜重力 ≡ 坡上）
    _slope_deg = float(getattr(args, "slope_deg", 0.0) or 0.0)
    if abs(_slope_deg) > 1e-6:
        _b = np.radians(_slope_deg); _g = 9.81
        env_cfg.sim.gravity = [float(_g * np.sin(_b)), 0.0, float(-_g * np.cos(_b))]
        print(f"[PLAY] SLOPE calib: {_slope_deg}° -> sim.gravity={env_cfg.sim.gravity}")

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Ch5 斜坡标定：obs 的"向下"方向也倾斜，与物理重力一致（否则 projected_gravity 与物理不符）
    if abs(_slope_deg) > 1e-6:
        _b = np.radians(_slope_deg)
        env.gravity_vec = to_torch([float(np.sin(_b)), 0.0, float(-np.cos(_b))],
                                   device=env.device).repeat(env.num_envs, 1)
        print(f"[PLAY] SLOPE calib: env.gravity_vec tilted {_slope_deg}°")

    # Ch5 偏心负载:固定负载质心偏置(平地制造 ∝ m·offset 的倾覆力矩,检验 CoM 估计头)
    _lox = float(getattr(args, "load_offset_x", -99.0))
    _loy = float(getattr(args, "load_offset_y", -99.0))
    if _lox > -90 or _loy > -90:
        ox = _lox if _lox > -90 else 0.0
        oy = _loy if _loy > -90 else 0.0
        env.load_offset_range["x"] = (ox, ox)
        env.load_offset_range["y"] = (oy, oy)
        env.load_offset_range["z"] = (0.10, 0.10)
        print(f"[PLAY] ECCENTRIC load offset fixed: x={ox} y={oy} (z=0.10)")

    # Ch5 测试时估计消融:篡改喂进 obs 的负载估计(权重不变),验证"估计→控制"的因果
    _cmode = str(getattr(args, "corrupt_estimate", "none") or "none")
    _cscope = str(getattr(args, "corrupt_estimate_scope", "obs") or "obs")
    if _cmode != "none":
        if _cscope in ("obs", "both"):
            env.qs_corrupt_mode = _cmode
            env.qs_corrupt_val = float(getattr(args, "corrupt_val", 0.0))
        print(f"[PLAY] ESTIMATE-CORRUPT: mode={_cmode} "
              f"scope={_cscope} val={float(getattr(args, 'corrupt_val', 0.0))}")

    # 应用持续外力（在建环境后写入 env 属性；默认关闭则无影响）
    # ext_force_vec 支持 [3]（全环境同）或 [N,3]（per-env sweep）；_apply_sustained_ext_force
    # 里 ext_force_vec * mask([N,1]) 两种 shape 都能广播成 [N,3]。
    # sweep 方向：磁量沿哪个轴施加（默认 down=-Z；水平向为实验2）
    _ext_force_dir = str(getattr(args, "ext_force_dir", "down") or "down").lower()
    _DIR2AXIS = {"down": (2, -1.0), "up": (2, 1.0), "fwd": (0, 1.0), "forward": (0, 1.0),
                 "back": (0, -1.0), "left": (1, 1.0), "right": (1, -1.0),
                 "x": (0, 1.0), "y": (1, 1.0), "z": (2, -1.0)}
    _ext_force_equiv_kg_per_env = None  # 每个环境的力当量 (kg)，供 per-env 分析做参考真值
    if _ext_force_on:
        env.ext_force_enable = True
        env.ext_force_body_idx = 0  # base
        _N = env.num_envs
        if _ext_force_sweep:
            _ax, _sg = _DIR2AXIS.get(_ext_force_dir, (2, -1.0))
            _kg = torch.linspace(_eff_kg_min, _eff_kg_max, _N, device=env.device)
            _vec = torch.zeros(_N, 3, device=env.device)
            _vec[:, _ax] = _sg * _kg * 9.81
            env.ext_force_vec = _vec
            _ext_force_equiv_kg_per_env = _kg.detach().cpu().numpy().tolist()
        else:
            env.ext_force_vec = to_torch(_ext_force_vec, device=env.device)
            _equiv = (-_ext_force_vec[2] / 9.81) if _ext_force_vec[2] < 0 else 0.0
            _ext_force_equiv_kg_per_env = [float(_equiv)] * _N

    if hasattr(env, "terrain_types"):
        terrain_types_cpu = env.terrain_types.detach().cpu()
        unique_types, counts = torch.unique(terrain_types_cpu, return_counts=True)
        per_type = {int(t.item()): int(c.item()) for t, c in zip(unique_types, counts)}
        print(f"[PLAY] terrain_types(count by type id): {per_type}")

        smooth_cnt = int(((terrain_types_cpu >= 0) & (terrain_types_cpu < 2)).sum().item())
        rough_cnt = int(((terrain_types_cpu >= 2) & (terrain_types_cpu < 4)).sum().item())
        stairs_up_cnt = int(((terrain_types_cpu >= 4) & (terrain_types_cpu < 11)).sum().item())
        stairs_down_cnt = int(((terrain_types_cpu >= 11) & (terrain_types_cpu < 16)).sum().item())
        discrete_cnt = int(((terrain_types_cpu >= 16) & (terrain_types_cpu < 20)).sum().item())
        print(
            "[PLAY] terrain group counts: "
            f"smooth={smooth_cnt}, rough={rough_cnt}, stairs_up={stairs_up_cnt}, "
            f"stairs_down={stairs_down_cnt}, discrete={discrete_cnt}"
        )

    # get robot_type
    robot_type = os.getenv("ROBOT_TYPE")
    # CLI-driven command: --cmd_vx / --cmd_vy / --cmd_yaw (all default 0.0 -> static)
    _cmd_vx = float(getattr(args, "cmd_vx", 0.0) or 0.0)
    _cmd_vy = float(getattr(args, "cmd_vy", 0.0) or 0.0)
    _cmd_yaw = float(getattr(args, "cmd_yaw", 0.0) or 0.0)
    if robot_type.startswith("PF"):
        commands_val = to_torch([_cmd_vx, _cmd_vy, _cmd_yaw, 0], device=env.device)
    elif robot_type == "WF_TRON1A":
        commands_val = to_torch([_cmd_vx, _cmd_vy, _cmd_yaw], device=env.device)
    else:
        commands_val = to_torch([_cmd_vx, _cmd_vy, _cmd_yaw, 0.0, 0.0], device=env.device)
    print(f"[PLAY] command = vx={_cmd_vx} vy={_cmd_vy} yaw={_cmd_yaw}")
    action_scale = env.cfg.control.action_scale_pos if robot_type == "WF_TRON1A"\
        else env.cfg.control.action_scale
    obs, obs_history, commands, critic_obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    # train_cfg.runner.checkpoint = -1

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)
    env.commands[:, :] = commands_val
    obs, obs_history, commands, critic_obs = env.get_observations()

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            path,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            path,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    
    # 单关节测试模式初始化
    test_joint_idx = None
    if test_joint_mode and test_joint_name:
        if test_joint_name in env.dof_names:
            test_joint_idx = env.dof_names.index(test_joint_name)
            print(f"[TEST-MODE] Enabled single-joint test mode: {test_joint_name} (DOF #{test_joint_idx})")
            print(
                f"[TEST-MODE] Amplitude={test_joint_amplitude:.3f} rad, "
                f"Offset={test_joint_offset:.3f} rad, Period={test_joint_period:.3f} s"
            )
            print(f"[TEST-MODE] All joints at neutral (0.0), {test_joint_name} will sweep.")
        else:
            print(f"[TEST-MODE] ERROR: Joint '{test_joint_name}' not found in dof_names: {env.dof_names}")
            test_joint_mode = False
    
    # 关节方向调试：用于确认 dof_pos 的零位与正方向
    debug_hip_sign = True
    debug_hip_every = 20
    hip_l_idx = env.dof_names.index("hip_L_Joint") if "hip_L_Joint" in env.dof_names else None
    hip_r_idx = env.dof_names.index("hip_R_Joint") if "hip_R_Joint" in env.dof_names else None
    prev_theta_l = None
    prev_theta_r = None
    if debug_hip_sign and (hip_l_idx is not None) and (hip_r_idx is not None):
        print(
            "[HIP-DEBUG] dof_pos is raw joint position from simulator. "
            "URDF axis: hip_L=[0,1,0], hip_R=[0,-1,0]."
        )
    stop_state_log = 2000  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1.0, 1.0, 0.0])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    est = None
    # Ch5 急停：速度指令方波（巡航 estop_vx 的 go 段 → 归零的 stop 段，反复）
    _estop_vx = float(getattr(args, "estop_vx", 0.0) or 0.0)
    _estop_go = float(getattr(args, "estop_go_s", 4.0) or 4.0)
    _estop_stop = float(getattr(args, "estop_stop_s", 4.0) or 4.0)
    _estop_on = _estop_vx > 1e-6
    if _estop_on:
        print(f"[PLAY] E-STOP mode: cruise vx={_estop_vx} go={_estop_go}s / stop={_estop_stop}s")
    for i in range(10 * int(env.max_episode_length)):
        if test_joint_mode and test_joint_idx is not None:
            # 单关节测试模式：使用位置控制，只让被测试的关节旋转
            # 计算时间和目标角度（正弦波）
            time_s = i * env.dt
            phase = 2.0 * np.pi * time_s / test_joint_period
            target_angle = test_joint_offset + test_joint_amplitude * np.sin(phase)
            
            # 创建动作张量：所有关节目标位置都是 0.0（中间位置），除了被测试的关节
            actions = torch.zeros((env.num_envs, env.num_dofs), device=env.device, dtype=torch.float32)
            actions[:, test_joint_idx] = target_angle / action_scale  # 需要除以 action_scale
            
            if (i % 100) == 0:
                print(f"[TEST-MODE] step={i} time={time_s:.2f}s target_angle={target_angle:.4f} rad phase={phase:.2f}")
        else:
            # 正常策略推理模式
            with torch.no_grad():
                est_for_action = encoder(obs_history, obs)
                if _cmode != "none" and _cscope in ("latent", "both"):
                    est_for_action = _corrupt_policy_estimate_latent(
                        est_for_action, _cmode, float(getattr(args, "corrupt_val", 0.0)), env
                    )
                actions = policy(torch.cat((est_for_action, obs, commands), dim=-1).detach())

        env.commands[:, :] = commands_val
        if _estop_on:
            _t = i * env.dt
            _go = (_t % (_estop_go + _estop_stop)) < _estop_go
            env.commands[:, 0] = _estop_vx if _go else 0.0

        obs, rews, dones, infos, obs_history, commands, critic_obs = env.step(
            actions.detach()
        )
        if test_joint_mode and test_joint_idx is not None:
            est = None
        else:
            # Recompute for logging after env.step(), so est / critic_obs / load_est
            # all describe the same state. The policy action above still uses the
            # pre-step estimate, as in training rollout collection.
            with torch.no_grad():
                est = encoder(obs_history, obs)
                if _cmode != "none" and _cscope in ("latent", "both"):
                    est = _corrupt_policy_estimate_latent(
                        est, _cmode, float(getattr(args, "corrupt_val", 0.0)), env
                    )
        if debug_hip_sign and (hip_l_idx is not None) and (hip_r_idx is not None):
            theta_l = env.dof_pos[robot_index, hip_l_idx].item()
            theta_r = env.dof_pos[robot_index, hip_r_idx].item()
            if prev_theta_l is None:
                dtheta_l = 0.0
                dtheta_r = 0.0
            else:
                dtheta_l = theta_l - prev_theta_l
                dtheta_r = theta_r - prev_theta_r
            prev_theta_l = theta_l
            prev_theta_r = theta_r

            # if (i % debug_hip_every) == 0:
            #     print(
            #         f"[HIP-DEBUG] step={i:5d} "
            #         f"theta_l={theta_l:+.5f} rad dtheta_l={dtheta_l:+.5f} | "
            #         f"theta_r={theta_r:+.5f} rad dtheta_r={dtheta_r:+.5f}"
            #     )
        load_est = None
        if hasattr(env, "last_load_estimates"):
            load_est = env.last_load_estimates
        elif hasattr(env, "_compute_load_estimates"):
            load_est = env._compute_load_estimates()
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            target_position[2] = 0
            camera_position = target_position + camera_offset
            # env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            # 实时计算真实负载质量：负载在机体上时才计入
            load_on_body = False
            if hasattr(env, "is_load_on_body"):
                load_on_body_mask = env.is_load_on_body(torch.tensor([robot_index], device=env.device, dtype=torch.long))
                load_on_body = bool(load_on_body_mask[0].item())
                if hasattr(env, "has_load"):
                    load_on_body = load_on_body and bool(env.has_load[robot_index].item())

            if hasattr(env, "base_mass0") and hasattr(env, "load_mass"):
                if hasattr(env, "load_mass_per_env"):
                    _lm_robot = float(env.load_mass_per_env[robot_index].item())
                else:
                    _lm_robot = float(env.load_mass)
                current_true_load_mass = _lm_robot if load_on_body else 0.0
                current_true_mass = float(env.base_mass0[robot_index].item()) + current_true_load_mass
            elif hasattr(env, "base_mass") and hasattr(env, "base_mass0"):
                current_true_mass = float(env.base_mass[robot_index].item())
                current_true_load_mass = float(
                    current_true_mass - env.base_mass0[robot_index].item()
                )
            else:
                current_true_mass = 0.0
                current_true_load_mass = 0.0

            if load_est is not None:
                payload_mass_est = load_est["payload_mass"][robot_index].item()
                load_y_est = load_est["load_y"][robot_index].item()
                load_x_est = load_est["load_x"][robot_index].item()
                # raw (pre-filter) QS estimates for comparison
                payload_mass_raw = env.estimated_payload_mass_raw[robot_index].item() if hasattr(env, "estimated_payload_mass_raw") else payload_mass_est
                load_x_raw = env.estimated_load_x_raw[robot_index].item() if hasattr(env, "estimated_load_x_raw") else load_x_est
                load_y_raw = env.estimated_load_y_raw[robot_index].item() if hasattr(env, "estimated_load_y_raw") else load_y_est
                body_mass_for_load_estimation = float(
                    getattr(env.cfg.asset, "load_estimation_body_mass", 9.58)
                )
                body_com0 = env.base_com0[robot_index]
                estimated_total_mass = body_mass_for_load_estimation + payload_mass_est
                estimated_total_mass_safe = max(estimated_total_mass, 1e-6)
                estimated_com_x = (
                    body_mass_for_load_estimation * body_com0[0].item()
                    + payload_mass_est * load_x_est
                ) / estimated_total_mass_safe
                estimated_com_y = (
                    body_mass_for_load_estimation * body_com0[1].item()
                    + payload_mass_est * load_y_est
                ) / estimated_total_mass_safe

                actual_load_x = 0.0
                actual_load_y = 0.0
                if load_on_body and hasattr(env, "actor_indices") and hasattr(env, "load_indices"):
                    base_actor_idx = env.actor_indices[robot_index]
                    load_actor_idx = env.load_indices[robot_index]
                    base_pos = env.root_states[base_actor_idx, 0:3]
                    base_quat = env.root_states[base_actor_idx, 3:7]
                    load_pos = env.root_states[load_actor_idx, 0:3]
                    rel_body = quat_rotate_inverse(
                        base_quat.unsqueeze(0), (load_pos - base_pos).unsqueeze(0)
                    )[0]
                    actual_load_x = rel_body[0].item()
                    actual_load_y = rel_body[1].item()
                actual_total_mass = body_mass_for_load_estimation + current_true_load_mass
                actual_total_mass_safe = max(actual_total_mass, 1e-6)
                actual_com_x = (
                    body_mass_for_load_estimation * body_com0[0].item()
                    + current_true_load_mass * actual_load_x
                ) / actual_total_mass_safe
                actual_com_y = (
                    body_mass_for_load_estimation * body_com0[1].item()
                    + current_true_load_mass * actual_load_y
                ) / actual_total_mass_safe

                est_payload_mass = min(payload_mass_est, 20.0)
                ref_payload_mass = min(current_true_load_mass, 10.0)

                est_load_y = min(load_y_est, 0.5)
                ref_load_y = min(actual_load_y, 0.5)

                est_load_x = min(load_x_est, 0.5)
                ref_load_x = min(actual_load_x, 0.5)

                payload_mass_left = load_est["payload_mass_left"][robot_index].item() if "payload_mass_left" in load_est else 0.0
                payload_mass_right = load_est["payload_mass_right"][robot_index].item() if "payload_mass_right" in load_est else 0.0

                logger.log_states(
                    {
                        "payload_mass": est_payload_mass,
                        "payload_mass_ref": ref_payload_mass,
                        "payload_mass_left": payload_mass_left,
                        "payload_mass_right": payload_mass_right,
                        "load_y": est_load_y,
                        "load_y_ref": ref_load_y,
                        "load_x": est_load_x,
                        "load_x_ref": ref_load_x,
                        "payload_mass_est": payload_mass_est,
                        "payload_mass_raw": payload_mass_raw,
                        "payload_mass_actual": current_true_load_mass,
                        "load_x_est": load_x_est,
                        "load_x_raw": load_x_raw,
                        "load_x_actual": actual_load_x,
                        "load_y_est": load_y_est,
                        "load_y_raw": load_y_raw,
                        "load_y_actual": actual_load_y,
                        "robot_mass_est": estimated_total_mass,
                        "robot_mass_actual": actual_total_mass,
                        "robot_com_x_est": estimated_com_x,
                        "robot_com_x_actual": actual_com_x,
                        "robot_com_y_est": estimated_com_y,
                        "robot_com_y_actual": actual_com_y,
                    }
                )
                if (i % 50) == 0:
                    print(
                        f"[PLAY] t={i * env.dt:.2f}s "
                        f"payload_mass={est_payload_mass:.4f}/{ref_payload_mass:.4f} "
                        f"load_x={est_load_x:.4f}/{ref_load_x:.4f} "
                        f"load_y={est_load_y:.4f}/{ref_load_y:.4f}"
                    )

                # --- per-env logging: write all-env tensors to logger.state_log_full ---
                # Saved as <key>_all of shape (T, num_envs) in .mat. Plots still use robot 0.
                _N = env.num_envs
                _dev = env.device

                # All-env load-on-body mask
                if hasattr(env, "is_load_on_body"):
                    _all_ids = torch.arange(_N, device=_dev, dtype=torch.long)
                    _on_body = env.is_load_on_body(_all_ids)
                    if hasattr(env, "has_load"):
                        _on_body = _on_body & env.has_load
                else:
                    _on_body = torch.zeros(_N, dtype=torch.bool, device=_dev)

                # True load mass per env.  With per_env_load_mass=True each
                # env's load actor is created from its own density/mass asset.
                if hasattr(env, "load_mass_per_env"):
                    _true_mass = env.load_mass_per_env.to(device=_dev).float()
                else:
                    _true_mass = torch.full((_N,), float(env.load_mass), device=_dev)
                _true_mass = torch.where(_on_body, _true_mass, torch.zeros_like(_true_mass))

                # Actual (x, y) of load in each env's body frame; 0 if not on body
                if hasattr(env, "actor_indices") and hasattr(env, "load_indices"):
                    _base_pos = env.root_states[env.actor_indices, 0:3]
                    _base_quat = env.root_states[env.actor_indices, 3:7]
                    _load_pos = env.root_states[env.load_indices, 0:3]
                    _rel_body = quat_rotate_inverse(_base_quat, _load_pos - _base_pos)
                    _actual_lx = torch.where(_on_body, _rel_body[:, 0], torch.zeros(_N, device=_dev))
                    _actual_ly = torch.where(_on_body, _rel_body[:, 1], torch.zeros(_N, device=_dev))
                else:
                    _actual_lx = torch.zeros(_N, device=_dev)
                    _actual_ly = torch.zeros(_N, device=_dev)

                full_dict = {
                    "payload_mass": load_est["payload_mass"],     # Model C load mass (filtered, kg)
                    "payload_mass_ref": _true_mass,                # truth load mass (kg)
                    "load_x": load_est["load_x"],                  # Model C load 位置 x (m)
                    "load_x_ref": _actual_lx,
                    "load_y": load_est["load_y"],
                    "load_y_ref": _actual_ly,
                    "load_on_body": _on_body.float(),
                    "base_lin_vel_x": env.base_lin_vel[:, 0],
                    "base_lin_vel_y": env.base_lin_vel[:, 1],
                    "base_ang_vel_z": env.base_ang_vel[:, 2],
                    "command_x": env.commands[:, 0],
                    "command_yaw": env.commands[:, 2],
                }

                # 姿态/高度（实验B：控制稳定性指标，per-env）
                _bq = env.base_quat  # [N,4] xyzw
                _qx, _qy, _qz, _qw = _bq[:, 0], _bq[:, 1], _bq[:, 2], _bq[:, 3]
                full_dict["base_roll"] = torch.atan2(2.0 * (_qw * _qx + _qy * _qz),
                                                     1.0 - 2.0 * (_qx * _qx + _qy * _qy))
                full_dict["base_pitch"] = torch.asin(torch.clamp(2.0 * (_qw * _qy - _qz * _qx), -1.0, 1.0))
                full_dict["base_lin_vel_z"] = env.base_lin_vel[:, 2]
                full_dict["base_height"] = env.root_states[env.actor_indices, 2]
                # 急停滑移距离用：轮子累计转角(rad, ×Rw=位移) + 真实机体 x 位置(per-env)
                try:
                    _wl = env.dof_names.index("wheel_L_Joint"); _wr = env.dof_names.index("wheel_R_Joint")
                    full_dict["wheel_angle"] = 0.5 * (env.dof_pos[:, _wl] + env.dof_pos[:, _wr])
                except Exception:
                    pass
                full_dict["base_pos_x"] = env.root_states[env.actor_indices, 0]
                full_dict["base_pos_y"] = env.root_states[env.actor_indices, 1]

                # Encoder 输出反归一化到物理单位。
                # get_inference_encoder(obs_history, obs) already returns the
                # policy-facing latent. In residual mode this means QS baseline
                # has already been added back by PPO.encode_for_policy().
                if est is not None and est.shape[-1] >= 4:
                    _mass_scale = float(env.cfg.normalization.obs_scales.mass_scale)
                    _com_scale = float(env.cfg.normalization.obs_scales.com_scale)
                    full_dict["encoder_mass"] = est[:, 3] / _mass_scale  # 最终质量估计 (kg)
                    if est.shape[-1] >= 7:
                        _enc_com_scaled = est[:, 4:7]
                        full_dict["encoder_com_delta_x"] = _enc_com_scaled[:, 0] / _com_scale
                        full_dict["encoder_com_delta_y"] = _enc_com_scaled[:, 1] / _com_scale
                        full_dict["encoder_com_delta_z"] = _enc_com_scaled[:, 2] / _com_scale

                # Model C 的 com_delta 预测（已存在 env.last_load_estimates["qs_com_delta"] 里）
                if "qs_com_delta" in load_est:
                    _qs_cd = load_est["qs_com_delta"]
                    full_dict["modelC_com_delta_x"] = _qs_cd[:, 0]
                    full_dict["modelC_com_delta_y"] = _qs_cd[:, 1]
                    full_dict["modelC_com_delta_z"] = _qs_cd[:, 2]

                # 真值 com_delta = (base_com - base_com0)，跟 encoder 预测同单位
                if hasattr(env, "base_com") and hasattr(env, "base_com0"):
                    _cd_true = env.base_com - env.base_com0
                    full_dict["true_com_delta_x"] = _cd_true[:, 0]
                    full_dict["true_com_delta_y"] = _cd_true[:, 1]
                    full_dict["true_com_delta_z"] = _cd_true[:, 2]

                logger.log_states_full(full_dict)

            logger.log_states(
                    {
                        "dof_pos_target": actions[robot_index, joint_index].item() * action_scale,
                        "dof_pos": (
                            env.dof_pos[robot_index, joint_index]
                            - env.raw_default_dof_pos[joint_index]
                        ).item(),
                        "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                        "filtered_dof_vel": env.filtered_obs_buf[robot_index, 12 + joint_index].item() / env.cfg.normalization.obs_scales.dof_vel if hasattr(env, "filtered_obs_buf") else 0.0,
                        "dof_torque": env.torques[robot_index, joint_index].item(),
                        # filtered_dof_torque 仅在 use_torques_in_obs=True 时存在于 filtered_obs_buf 中
                        "filtered_dof_torque": (
                            env.filtered_obs_buf[robot_index, 20 + joint_index].item() / env.cfg.normalization.obs_scales.torque
                            if hasattr(env, "filtered_obs_buf") and bool(getattr(env.cfg.env, "use_torques_in_obs", True))
                            else 0.0
                        ),
                        "command_x": env.commands[robot_index, 0].item(),
                        "command_y": env.commands[robot_index, 1].item(),
                        "command_yaw": env.commands[robot_index, 2].item(),
                        "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                        "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                        "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                        "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                        "filtered_base_vel_yaw": env.filtered_obs_buf[robot_index, 2].item() / env.cfg.normalization.obs_scales.ang_vel if hasattr(env, "filtered_obs_buf") else 0.0,
                        "power": torch.sum(env.power[robot_index, :]).item(),
                        "joint_pos_abad_L": (env.dof_pos[robot_index, 0] - env.raw_default_dof_pos[0]).item(),
                        "joint_pos_hip_L": (env.dof_pos[robot_index, 1] - env.raw_default_dof_pos[1]).item(),
                        "joint_pos_knee_L": (env.dof_pos[robot_index, 2] - env.raw_default_dof_pos[2]).item(),
                        "joint_pos_wheel_L": (env.dof_pos[robot_index, 3] - env.raw_default_dof_pos[3]).item(),
                        "joint_pos_abad_R": (env.dof_pos[robot_index, 4] - env.raw_default_dof_pos[4]).item(),
                        "joint_pos_hip_R": (env.dof_pos[robot_index, 5] - env.raw_default_dof_pos[5]).item(),
                        "joint_pos_knee_R": (env.dof_pos[robot_index, 6] - env.raw_default_dof_pos[6]).item(),
                        "joint_pos_wheel_R": (env.dof_pos[robot_index, 7] - env.raw_default_dof_pos[7]).item(),
                        "torque_abad_L": env.torques[robot_index, 0].item(),
                        "torque_hip_L": env.torques[robot_index, 1].item(),
                        "torque_knee_L": env.torques[robot_index, 2].item(),
                        "torque_wheel_L": env.torques[robot_index, 3].item(),
                        "torque_abad_R": env.torques[robot_index, 4].item(),
                        "torque_hip_R": env.torques[robot_index, 5].item(),
                        "torque_knee_R": env.torques[robot_index, 6].item(),
                        "torque_wheel_R": env.torques[robot_index, 7].item(),
                        "mass": current_true_mass,
                        "CoM_x": env.base_com[robot_index, 0].item(),
                        "CoM_y": env.base_com[robot_index, 1].item(),
                        "CoM_z": env.base_com[robot_index, 2].item(),
                        "load_on_body": float(load_on_body),
                        # "inertia_xx": env.base_inertia[robot_index, 0].item(),
                        # "inertia_yy": env.base_inertia[robot_index, 1].item(),
                        # "inertia_zz": env.base_inertia[robot_index, 2].item(),   
                        
                        "contact_forces_z": env.contact_forces[
                            robot_index, env.feet_indices, 2
                        ]
                        .cpu()
                        .numpy(),
                    }
                )
            # print(torch.sum(env.power[robot_index, :]).item())
            if est is not None:                # 计算并记录 extra_loss 分量（速度 / 质量 / 质心）
                extra_loss_vel = (
                    (est[:, 0:3] - critic_obs[:, 0:3]).pow(2).mean().item()
                )
                extra_loss_mass = (
                    (est[:, 3] - critic_obs[:, 3]).pow(2).mean().item()
                )
                extra_loss_com = (
                    (est[:, 4:7] - critic_obs[:, 4:7]).pow(2).mean().item()
                )
                logger.log_states(
                    {
                        "extra_loss_vel": extra_loss_vel,
                        "extra_loss_mass": extra_loss_mass,
                        "extra_loss_com": extra_loss_com,
                    }
                )
                if (i % 50) == 0:
                    # obs_vec = obs[robot_index].detach().cpu().tolist()
                    # cmd_vec = env.commands[robot_index].detach().cpu().tolist()
                    print(
                        f"[PLAY] t={i * env.dt:.2f}s "
                        f"extra_loss_vel={extra_loss_vel:.6f} "
                        f"extra_loss_mass={extra_loss_mass:.6f} "
                        f"extra_loss_com={extra_loss_com:.6f}"
                    )
                    # print(f"[PLAY] obs[{robot_index}]={obs_vec}")
                    # print(f"[PLAY] commands[{robot_index}]={cmd_vec}")
                logger.log_states(
                    {
                        "est_lin_vel_x": est[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": est[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": est[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "mass_est": (
                            est[robot_index, 3].item() / env.cfg.normalization.obs_scales.mass_scale
                            + float(env.base_mass0[robot_index].item())
                        ),
                        "CoM_est_x": (
                            est[robot_index, 4].item() / env.cfg.normalization.obs_scales.com_scale
                            + float(env.base_com0[robot_index, 0].item())
                        ),
                        "CoM_est_y": (
                            est[robot_index, 5].item() / env.cfg.normalization.obs_scales.com_scale
                            + float(env.base_com0[robot_index, 1].item())
                        ),
                        "CoM_est_z": (
                            est[robot_index, 6].item() / env.cfg.normalization.obs_scales.com_scale
                            + float(env.base_com0[robot_index, 2].item())
                        ),
                    }
                )
            elif test_joint_mode and test_joint_idx is not None:
                # 单关节测试模式：记录被测试关节的详细数据
                logger.log_states(
                    {
                        "dof_pos_target": actions[robot_index, test_joint_idx].item() * action_scale,
                        "dof_pos": (
                            env.dof_pos[robot_index, test_joint_idx]
                            - env.raw_default_dof_pos[test_joint_idx]
                        ).item(),
                        "dof_vel": env.dof_vel[robot_index, test_joint_idx].item(),
                        "dof_torque": env.torques[robot_index, test_joint_idx].item(),
                    }
                )
        elif i == stop_state_log:
            # Build a unique filename so successive runs never overwrite each other.
            # Format: play_data_<YYYYMMDD-HHMMSS>[_<exp_tag>].mat
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d-%H%M%S")
            tag = getattr(args, "exp_tag", "") or ""
            if not tag:
                tag = _build_auto_exp_tag(env, train_cfg, args, _cmd_vx, _cmd_vy, _cmd_yaw)
                print(f"[play] auto exp_tag = {tag}")
            tag_suffix = f"_{tag}" if tag else ""
            mat_name = f"play_data_{ts}{tag_suffix}.mat"
            mat_dir = os.path.join(
                LEGGED_GYM_ROOT_DIR, "logs", args.task,
                train_cfg.runner.experiment_name, "exported",
            )
            os.makedirs(mat_dir, exist_ok=True)
            mat_path = os.path.join(mat_dir, mat_name)
            # Plot output dir mirrors the .mat name: same tag, same timestamp.
            plot_dir = os.path.join(mat_dir, f"plots_{ts}_{tag}")
            # Inject experiment-level metadata + computed metrics before save.
            metrics = _compute_experiment_metrics(logger, env.dt)
            _log_root_for_verify = os.path.join(
                LEGGED_GYM_ROOT_DIR, "logs", args.task,
                train_cfg.runner.experiment_name,
            )
            verify = _collect_run_verify(env, train_cfg, args, _log_root_for_verify)
            run_meta = {
                "exp_tag": tag,
                "timestamp": ts,
                "load_run": str(getattr(args, "load_run", "") or ""),
                "checkpoint": str(getattr(args, "checkpoint", "") or ""),
                "dt": float(env.dt),
                "cmd_vx": _cmd_vx, "cmd_vy": _cmd_vy, "cmd_yaw": _cmd_yaw,
                "ext_force_on": bool(_ext_force_on),
                "keep_load": bool(_keep_load),
                "ext_force_vec_N": list(_ext_force_vec),     # [Fx,Fy,Fz] world frame, N (单值模式)
                "ext_force_down_kg": float(_eff_kg),          # 力当量质量 (kg), 0 = 非单值竖直当量模式
                "ext_force_sweep": bool(_ext_force_sweep),    # True = per-env 力当量扫描
                "ext_force_dir": str(_ext_force_dir),         # 力方向 down/up/fwd/back/left/right
                "ext_force_down_kg_min": float(_eff_kg_min),
                "ext_force_down_kg_max": float(_eff_kg_max),
                # 每个环境的力当量 (kg)，per-env 分析的参考真值（不要跨环境平均后再算 RMSE）
                "ext_force_equiv_kg_per_env": (_ext_force_equiv_kg_per_env
                                               if _ext_force_equiv_kg_per_env is not None else []),
                # 固定的阶跃时序（力/负载共用），供 MATLAB 重建阶跃参考线
                "load_start_time_s": float(getattr(env.cfg.domain_rand, "load_start_time_s", 0.0)),
                "load_duration_s_used": float(getattr(env.cfg.domain_rand, "load_duration_range_s", [0.0, 0.0])[0]),
                "load_interval_s_used": float(getattr(env.cfg.domain_rand, "load_interval_range_s", [0.0, 0.0])[0]),
                "load_mass_range_used": list(getattr(env.cfg.domain_rand, "add_load_range", [-1, -1])),
                "load_offset_x": float(_lox if _lox > -90 else np.nan),
                "load_offset_y": float(_loy if _loy > -90 else np.nan),
                "slope_deg": float(_slope_deg),
                "estop_vx": float(_estop_vx),
                "corrupt_estimate": str(_cmode),
                "corrupt_estimate_scope": str(_cscope),
                "corrupt_val": float(getattr(args, "corrupt_val", 0.0)),
                "use_qs_in_obs": bool(getattr(env.cfg.env, "use_qs_in_obs", True)),
                "use_residual_learning": bool(getattr(env.cfg.env, "use_residual_learning", False)),
                "use_torques_in_obs": bool(getattr(env.cfg.env, "use_torques_in_obs", True)),
                "use_load_residual_estimation": bool(
                    getattr(train_cfg.algorithm, "use_load_residual_estimation", False)
                ),
                "plot_dir": plot_dir,
                "verify": _sanitize_for_mat(verify),
            }
            logger.save_to_mat(mat_path, extra={"metrics": metrics, "meta": run_meta})
            print(f"[play] saved: {mat_path}")
            _print_metrics_pretty(metrics)
            _print_run_verify_block(verify)
            logger.plot_states(save_dir=plot_dir, show=not bool(getattr(args, "headless", False)))

            # Exit immediately under headless / --exit_after_save so batch runs
            # don't spin on the post-save loop until stop_rew_log.
            if bool(getattr(args, "exit_after_save", False)) or bool(getattr(args, "headless", False)):
                print(f"[play] exit_after_save=True (headless={bool(getattr(args,'headless',False))}); returning.")
                return

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
