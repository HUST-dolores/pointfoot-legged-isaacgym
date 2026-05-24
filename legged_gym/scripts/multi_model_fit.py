"""Run multiple QS ansatz on ALL calibration .npz files, compare mass RMSE.

Variants tried (5+2 = 7):
  A. basic                       (4 params: aL, aR, gL, gR)
  B. + beta_pitch * sin(pitch)   (+1)
  C. + beta_hip * hip_diff       (+1)  ← deployed in wheelfoot_flat.py
  E. + beta_abad * abad_diff     (+1)
  G. C + B + E                   (7 params)
  C_sym. C with aL==aR, gL==gR enforced (3 params: a, g_sum, beta_hip)
  G_sym. G with same symmetry (5 params)

Plus the knee-weight variant on the BEST per-policy ansatz:
  K. load_torque = -alpha_k·power_knee - alpha_h·power_hip
     (lets lstsq learn the knee/hip relative weight)

Reports:
  - per-cal-dir × per-model RMSE table
  - average RMSE across all policies (for picking the universal best)
"""
import os
import sys
import numpy as np
import glob

ROOT = "/home/xu/limx_rl/pointfoot-legged-gym/logs/calibration"


def load_concat(cal_dir):
    """Load all calib_m*.npz in cal_dir and concatenate."""
    files = sorted(glob.glob(os.path.join(cal_dir, "calib_m*.npz")))
    if not files:
        return None
    parts = [np.load(f, allow_pickle=True) for f in files]
    keys = parts[0].keys()
    out = {}
    for k in keys:
        try:
            out[k] = np.concatenate([np.asarray(p[k]) for p in parts])
        except Exception:
            out[k] = np.asarray(parts[0][k])  # scalar fields
    return out


def _fit_mass_y(data, y_weight=1.0, with_pitch=False, with_hip=False, with_abad=False,
                symmetric=False):
    """Mass + y joint lstsq. If symmetric, enforce aL=aR=a, gL=gR=g."""
    v = data["valid"].astype(bool)
    R_L = data["R_L"][v]
    R_R = data["R_R"][v]
    yfL = data["y_foot_L"][v]
    yfR = data["y_foot_R"][v]
    m_true = data["mass_true"][v]
    y_true = data["y_rel_true"][v]
    n = len(R_L)

    if symmetric:
        # mass row: a*(R_L+R_R) + 2*g [+ extras] = m_true
        cols_mass = [R_L + R_R, 2.0 * np.ones(n)]
        dyL = yfL - y_true
        dyR = yfR - y_true
        # y row symmetric: a*(R_L*dyL+R_R*dyR) + g*(dyL+dyR) = 0
        cols_y = [R_L * dyL + R_R * dyR, dyL + dyR]
    else:
        cols_mass = [R_L, R_R, np.ones(n), np.ones(n)]
        dyL = yfL - y_true
        dyR = yfR - y_true
        cols_y = [R_L * dyL, R_R * dyR, dyL, dyR]

    extras = []
    if with_pitch:
        sin_pitch = np.sin(data["pitch"][v])
        cols_mass.append(2.0 * sin_pitch)
        cols_y.append((yfL + yfR - 2.0 * y_true) * sin_pitch)
        extras.append("beta_pitch")
    if with_hip:
        hip_diff = data["theta_lhip"][v] - data["theta_rhip"][v]
        cols_mass.append(2.0 * hip_diff)
        cols_y.append((yfL + yfR - 2.0 * y_true) * hip_diff)
        extras.append("beta_hip")
    if with_abad:
        abad_diff = data["theta_labad"][v] - data["theta_rabad"][v]
        cols_mass.append(2.0 * abad_diff)
        cols_y.append((yfL + yfR - 2.0 * y_true) * abad_diff)
        extras.append("beta_abad")

    A_mass = np.stack(cols_mass, axis=1)
    A_y = np.stack(cols_y, axis=1)
    b_mass = m_true
    b_y = np.zeros(n)
    W = float(np.std(np.concatenate([dyL, dyR]))) or 0.1
    y_scale = y_weight / W
    A = np.concatenate([A_mass, A_y * y_scale], axis=0)
    b = np.concatenate([b_mass, b_y * y_scale], axis=0)
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)

    if symmetric:
        out = dict(alpha=float(theta[0]), gamma_sum=float(2.0 * theta[1]))
        for i, name in enumerate(extras):
            out[name] = float(theta[2 + i])
    else:
        out = dict(
            alpha_L=float(theta[0]), alpha_R=float(theta[1]),
            gamma_L=float(theta[2]), gamma_R=float(theta[3]),
        )
        for i, name in enumerate(extras):
            out[name] = float(theta[4 + i])
    return out


def _predict_mass(data, p, symmetric=False):
    """Compute m_est for every (valid AND invalid) sample, return array."""
    R_L = data["R_L"]; R_R = data["R_R"]
    if symmetric:
        m = p["alpha"] * (R_L + R_R) + p["gamma_sum"]
    else:
        m = p["alpha_L"] * R_L + p["alpha_R"] * R_R + p["gamma_L"] + p["gamma_R"]
    if "beta_pitch" in p:
        m = m + 2.0 * p["beta_pitch"] * np.sin(data["pitch"])
    if "beta_hip" in p:
        m = m + 2.0 * p["beta_hip"] * (data["theta_lhip"] - data["theta_rhip"])
    if "beta_abad" in p:
        m = m + 2.0 * p["beta_abad"] * (data["theta_labad"] - data["theta_rabad"])
    return m


def fit_and_eval(data, **kw):
    p = _fit_mass_y(data, **kw)
    v = data["valid"].astype(bool)
    m_pred = _predict_mass(data, p, symmetric=kw.get("symmetric", False))
    err = m_pred[v] - data["mass_true"][v]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    return p, rmse, bias


def fit_knee_weight(data, with_hip=True):
    """Variant K: let lstsq learn knee/hip torque weight rather than fixed -1.
       load_torque_left = -aK*power_lknee - aH*power_lhip
       load_torque_right = +aK*power_rknee + aH*power_rhip
       R_L = load_torque_left / denom_L,  similarly R_R.

    Implementation: instead of using pre-computed R_L = T_L/denom_L,
    we compute T_L_decomposed = (knee_L_signed, hip_L_signed) / denom_L.
    Then mass row: aK*(-knee_L/dL + knee_R/dR)·alpha_L_effective + ...

    Simpler: solve for (alpha_knee, alpha_hip) jointly with mass eq.
    Need raw T_L/T_R AND raw knee/hip components. Our npz has T_L, T_R but
    not separately stored knee/hip torques (collect script combined them).
    → cannot do this without modifying collect script. Skip for now.
    """
    return None


def run_all_models(data):
    """Run all variants on this cal dir's data. Return dict of (name -> (params, rmse, bias))."""
    results = {}
    variants = [
        ("A_basic",       dict(with_pitch=False, with_hip=False, with_abad=False, symmetric=False)),
        ("B_pitch",       dict(with_pitch=True,  with_hip=False, with_abad=False, symmetric=False)),
        ("C_hip",         dict(with_pitch=False, with_hip=True,  with_abad=False, symmetric=False)),
        ("E_abad",        dict(with_pitch=False, with_hip=False, with_abad=True,  symmetric=False)),
        ("F_hip_abad",    dict(with_pitch=False, with_hip=True,  with_abad=True,  symmetric=False)),
        ("G_all",         dict(with_pitch=True,  with_hip=True,  with_abad=True,  symmetric=False)),
        ("C_sym",         dict(with_pitch=False, with_hip=True,  with_abad=False, symmetric=True)),
        ("G_sym",         dict(with_pitch=True,  with_hip=True,  with_abad=True,  symmetric=True)),
    ]
    for name, kw in variants:
        try:
            p, rmse, bias = fit_and_eval(data, **kw)
            results[name] = dict(params=p, rmse=rmse, bias=bias, n_params=len(p))
        except Exception as e:
            results[name] = dict(error=str(e))
    return results


SHORT = {
    "exper_qs_resi_load_boost_3_seed_42": "main_lb3_s42",
    "exper_qs_resi_load_boost_3_seed_43": "main_lb3_s43",
    "exper_qs_resi_load_boost_6":         "main_lb6",
    "exper_qs_noresi_load_boost_3":       "direct",
    "exper_history_only_load_boost_3":    "histonly",
}


def _label(d, data):
    """Prefer npz `load_run`; fall back to dir name."""
    if "load_run" in data:
        v = str(data["load_run"])
        if v in SHORT:
            return SHORT[v]
        if v:
            return v[:20]
    return d


def main():
    # By default: only include the 5 fresh canonical cals (5月23 17:53-17:58).
    # Set CAL_FILTER to override.
    cal_filter = os.environ.get("CAL_FILTER", "5月23_17-5")
    cal_dirs = [d for d in sorted(os.listdir(ROOT))
                if os.path.isdir(os.path.join(ROOT, d)) and cal_filter in d]
    print(f"Found {len(cal_dirs)} cal dirs matching filter '{cal_filter}'")

    all_results = {}
    labels = {}
    for d in cal_dirs:
        data = load_concat(os.path.join(ROOT, d))
        if data is None:
            continue
        if "valid" not in data:
            print(f"  skip {d}: no 'valid' field")
            continue
        all_results[d] = run_all_models(data)
        labels[d] = _label(d, data)
        print(f"  {d} -> {labels[d]}")

    # Print RMSE table: rows = cal dirs, cols = models
    model_names = ["A_basic","B_pitch","C_hip","E_abad","F_hip_abad","G_all","C_sym","G_sym"]
    print(f"\n{'policy':<14}", *[f"{m:>10}" for m in model_names])
    print("-" * (14 + 11 * len(model_names)))
    for d in sorted(all_results.keys()):
        row = []
        for m in model_names:
            r = all_results[d].get(m, {})
            if "rmse" in r:
                row.append(f"{r['rmse']:>10.4f}")
            else:
                row.append(f"{'err':>10}")
        print(f"{labels[d]:<14}", *row)

    # Average over QS-USING policies (exclude histonly since QS not in obs there).
    qs_using = [d for d in all_results if "histonly" not in labels[d]]
    print(f"\n{'AVG (qs-using)':<14}", end="")
    for m in model_names:
        vals = [all_results[d][m]["rmse"] for d in qs_using if "rmse" in all_results[d].get(m, {})]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"{avg:>10.4f}", end="")
        else:
            print(f"{'-':>10}", end="")
    print()
    print(f"{'AVG (all)':<14}", end="")
    for m in model_names:
        vals = [all_results[d][m]["rmse"] for d in all_results if "rmse" in all_results[d].get(m, {})]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"{avg:>10.4f}", end="")
        else:
            print(f"{'-':>10}", end="")
    print()

    # Print one detailed example: G_all params for each policy
    print("\n\n=== G_all params per cal_dir ===")
    print(f"{'cal_dir':<22} {'aL':>7} {'aR':>7} {'gL':>7} {'gR':>7} {'beta_p':>8} {'beta_h':>8} {'beta_a':>8} {'rmse':>7}")
    for d in sorted(all_results.keys()):
        r = all_results[d].get("G_all", {})
        if "params" in r:
            p = r["params"]
            print(f"{d:<22} {p['alpha_L']:>+7.3f} {p['alpha_R']:>+7.3f} {p['gamma_L']:>+7.3f} {p['gamma_R']:>+7.3f}"
                  f" {p['beta_pitch']:>+8.3f} {p['beta_hip']:>+8.3f} {p['beta_abad']:>+8.3f} {r['rmse']:>7.3f}")

    # G_sym params (per policy + variation across policies) — only QS-using policies for variation stats
    print("\n\n=== G_sym (recommended) params per policy ===")
    print(f"{'policy':<14} {'alpha':>7} {'g_sum':>7} {'beta_p':>8} {'beta_h':>8} {'beta_a':>8} {'rmse':>7}")
    gsym_params = {k: [] for k in ("alpha","gamma_sum","beta_pitch","beta_hip","beta_abad")}
    qs_using = [d for d in all_results if "histonly" not in labels[d]]
    for d in sorted(all_results.keys()):
        r = all_results[d].get("G_sym", {})
        if "params" in r:
            p = r["params"]
            print(f"{labels[d]:<14} {p['alpha']:>+7.3f} {p['gamma_sum']:>+7.3f}"
                  f" {p['beta_pitch']:>+8.3f} {p['beta_hip']:>+8.3f} {p['beta_abad']:>+8.3f} {r['rmse']:>7.3f}")
            if d in qs_using:
                for k in gsym_params:
                    gsym_params[k].append(p[k])

    print("\n--- G_sym cross-policy variation ---")
    print(f"{'param':<14} {'min':>9} {'max':>9} {'mean':>9} {'std':>9} {'rel_std%':>10} {'range/|mean|':>12}")
    for k, vals in gsym_params.items():
        arr = np.array(vals)
        mn, mx = arr.min(), arr.max()
        mu, sd = arr.mean(), arr.std()
        rel = (sd/abs(mu)*100) if abs(mu) > 1e-9 else float('nan')
        rng = (mx - mn) / max(abs(mu), 1e-9)
        print(f"{k:<14} {mn:>+9.3f} {mx:>+9.3f} {mu:>+9.3f} {sd:>9.3f} {rel:>9.1f}% {rng:>12.2f}")

    # Compare: if we DEPLOY a SINGLE "average G_sym" across all policies,
    # how much RMSE penalty does each policy pay?
    print("\n--- Universal G_sym = avg(per-policy fits) — penalty per policy ---")
    universal = {k: float(np.mean(vals)) for k, vals in gsym_params.items()}
    print(f"  Universal coefs: alpha={universal['alpha']:+.3f}  g_sum={universal['gamma_sum']:+.3f}"
          f"  beta_p={universal['beta_pitch']:+.3f}  beta_h={universal['beta_hip']:+.3f}  beta_a={universal['beta_abad']:+.3f}")
    print(f"\n{'policy':<14} {'own_rmse':>10} {'univ_rmse':>10} {'penalty':>10}")
    for d in sorted(all_results.keys()):
        r = all_results[d].get("G_sym", {})
        if "rmse" not in r: continue
        data = load_concat(os.path.join(ROOT, d))
        if data is None: continue
        m_pred = _predict_mass(data, universal, symmetric=True)
        v = data["valid"].astype(bool)
        err = m_pred[v] - data["mass_true"][v]
        univ_rmse = float(np.sqrt(np.mean(err ** 2)))
        own = r["rmse"]
        pen = univ_rmse - own
        note = ""
        if "histonly" in labels[d]:
            note = "  ← excluded from universal fit (QS not in obs)"
        print(f"{labels[d]:<14} {own:>10.4f} {univ_rmse:>10.4f} {pen:>+10.4f}{note}")

    # Universal G_all comparison (7 params, lowest avg RMSE) — same penalty test
    gall_params = {k: [] for k in ("alpha_L","alpha_R","gamma_L","gamma_R","beta_pitch","beta_hip","beta_abad")}
    for d in qs_using:
        r = all_results[d].get("G_all", {})
        if "params" in r:
            for k in gall_params:
                gall_params[k].append(r["params"][k])
    univ_gall = {k: float(np.mean(vals)) for k, vals in gall_params.items()}
    print(f"\n--- Universal G_all = avg(per-policy fits, QS-using only) ---")
    print(f"  aL={univ_gall['alpha_L']:+.3f} aR={univ_gall['alpha_R']:+.3f} gL={univ_gall['gamma_L']:+.3f} gR={univ_gall['gamma_R']:+.3f}"
          f" b_p={univ_gall['beta_pitch']:+.3f} b_h={univ_gall['beta_hip']:+.3f} b_a={univ_gall['beta_abad']:+.3f}")
    print(f"\n{'policy':<14} {'own_gall':>10} {'univ_gall':>10} {'penalty':>10}")
    for d in sorted(all_results.keys()):
        r = all_results[d].get("G_all", {})
        if "rmse" not in r: continue
        data = load_concat(os.path.join(ROOT, d))
        m_pred = _predict_mass(data, univ_gall, symmetric=False)
        v = data["valid"].astype(bool)
        err = m_pred[v] - data["mass_true"][v]
        univ_rmse = float(np.sqrt(np.mean(err ** 2)))
        pen = univ_rmse - r["rmse"]
        note = "  ← excluded" if "histonly" in labels[d] else ""
        print(f"{labels[d]:<14} {r['rmse']:>10.4f} {univ_rmse:>10.4f} {pen:>+10.4f}{note}")

    # Also try universal C_hip (5-params, deployed ansatz) for comparison
    chip_params = {k: [] for k in ("alpha_L","alpha_R","gamma_L","gamma_R","beta_hip")}
    for d in qs_using:
        r = all_results[d].get("C_hip", {})
        if "params" in r:
            for k in chip_params:
                chip_params[k].append(r["params"][k])
    univ_chip = {k: float(np.mean(vals)) for k, vals in chip_params.items()}
    print(f"\n--- Universal C_hip = avg(per-policy fits, QS-using only) ---")
    print(f"  alpha_L={univ_chip['alpha_L']:+.3f}  alpha_R={univ_chip['alpha_R']:+.3f}"
          f"  gamma_L={univ_chip['gamma_L']:+.3f}  gamma_R={univ_chip['gamma_R']:+.3f}"
          f"  beta_hip={univ_chip['beta_hip']:+.3f}")
    print(f"\n{'policy':<14} {'own_chip':>10} {'univ_chip':>10} {'penalty':>10}")
    for d in sorted(all_results.keys()):
        r = all_results[d].get("C_hip", {})
        if "rmse" not in r: continue
        data = load_concat(os.path.join(ROOT, d))
        if data is None: continue
        m_pred = _predict_mass(data, univ_chip, symmetric=False)
        v = data["valid"].astype(bool)
        err = m_pred[v] - data["mass_true"][v]
        univ_rmse = float(np.sqrt(np.mean(err ** 2)))
        own = r["rmse"]
        pen = univ_rmse - own
        note = "  ← excluded" if "histonly" in labels[d] else ""
        print(f"{labels[d]:<14} {own:>10.4f} {univ_rmse:>10.4f} {pen:>+10.4f}{note}")


if __name__ == "__main__":
    main()
