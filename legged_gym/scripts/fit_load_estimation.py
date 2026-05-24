# Fit the linear coefficients of the load-estimation formula via least squares.
#
# Usage:
#   python legged_gym/scripts/fit_load_estimation.py \
#       --inputs logs/calibration/<dt>/calib_m2.000.npz \
#                logs/calibration/<dt>/calib_m3.333.npz \
#                logs/calibration/<dt>/calib_m4.667.npz \
#                logs/calibration/<dt>/calib_m6.000.npz
#
# Fits, jointly across all input NPZs:
#   m_L = alpha_L * R_L + gamma_L
#   m_R = alpha_R * R_R + gamma_R
#   load_y = (y_foot_L*m_L + y_foot_R*m_R) / (m_L + m_R)            [moment-balance form]
#   load_x = (tau_diff - T_body) / (m_total * g * cos_pitch)
#            - k_pitch * tan_pitch + x_offset                       [x form]
#
# Reports per-condition residuals so structural errors are visible.

import argparse
import glob
import os
from collections import defaultdict

import numpy as np


def load_all(paths):
    arrays = []
    for p in paths:
        for fp in sorted(glob.glob(p)):
            d = dict(np.load(fp))
            # Tighten validity: load must not have escaped in x/y either.
            # Some PhysX runs leave a few outliers with z OK but x/y huge.
            v = d["valid"].copy()
            outlier = (np.abs(d["x_rel_true"]) > 0.5) | (np.abs(d["y_rel_true"]) > 0.5)
            n_outlier = int((v & outlier).sum())
            v[outlier] = False
            d["valid"] = v
            extra_msg = f"  (rejected {n_outlier} xy-outlier)" if n_outlier else ""
            print(f"[fit] loaded {fp}: N={len(d['mass_true'])} valid={int(v.sum())}{extra_msg}")
            arrays.append(d)
    if not arrays:
        raise ValueError("no input NPZs matched")
    # Concatenate. Assume scalar fields (delta etc.) are identical across files.
    keys_scalar = {"delta", "thigh_len", "gravity", "robot_width",
                   "leg_eff_length", "abad_R_sign", "zero_thigh_angle"}
    # String provenance fields added by recent collect_calibration_data.py.
    keys_string = {"load_run", "checkpoint", "task"}
    out = {}
    for k in arrays[0].keys():
        if k in keys_scalar:
            out[k] = float(arrays[0][k])
        elif k in keys_string:
            out[k] = str(arrays[0][k])
        else:
            try:
                out[k] = np.concatenate([a[k] for a in arrays], axis=0)
            except Exception:
                # Skip any other 0-dim / non-concatable field
                out[k] = arrays[0][k]
    return out


def fit_mass_and_y(data, y_weight=1.0, with_pitch=False, with_hip=False, with_abad=False):
    """Joint least squares for mass coefficients + optional symmetric extras.

    Basic model:
        m_L = alpha_L * R_L + gamma_L
        m_R = alpha_R * R_R + gamma_R

    Optional symmetric extras (each adds 1 parameter, applied to BOTH legs equally):
        with_pitch: + beta_pitch * sin(pitch)
        with_hip:   + beta_hip   * (theta_lhip - theta_rhip)
        with_abad:  + beta_abad  * (theta_labad - theta_rabad)
    """
    v = data["valid"]
    R_L = data["R_L"][v]
    R_R = data["R_R"][v]
    yfL = data["y_foot_L"][v]
    yfR = data["y_foot_R"][v]
    m_true = data["mass_true"][v]
    y_true = data["y_rel_true"][v]  # actual settled load position in body frame

    n = len(R_L)

    # Mass rows: alpha_L*R_L + alpha_R*R_R + gamma_L + gamma_R [+ 2*beta_pitch*sin_pitch] = m_true
    cols_mass = [R_L, R_R, np.ones(n), np.ones(n)]
    # Y rows (moment balance, kg*m units): see derivation in code comments.
    dyL = yfL - y_true
    dyR = yfR - y_true
    cols_y = [R_L * dyL, R_R * dyR, dyL, dyR]

    extra_names = []
    if with_pitch:
        sin_pitch = np.sin(data["pitch"][v])
        cols_mass.append(2.0 * sin_pitch)
        cols_y.append((yfL + yfR - 2.0 * y_true) * sin_pitch)
        extra_names.append("beta_pitch")
    if with_hip:
        hip_diff = data["theta_lhip"][v] - data["theta_rhip"][v]
        cols_mass.append(2.0 * hip_diff)
        cols_y.append((yfL + yfR - 2.0 * y_true) * hip_diff)
        extra_names.append("beta_hip")
    if with_abad:
        abad_diff = data["theta_labad"][v] - data["theta_rabad"][v]
        cols_mass.append(2.0 * abad_diff)
        cols_y.append((yfL + yfR - 2.0 * y_true) * abad_diff)
        extra_names.append("beta_abad")

    A_mass = np.stack(cols_mass, axis=1)
    b_mass = m_true
    A_y = np.stack(cols_y, axis=1)
    b_y = np.zeros(n)

    # Weight: mass eq residual is in kg; y eq residual is in kg*m. Scale y eq to comparable.
    W = float(np.std(np.concatenate([dyL, dyR]))) or 0.1
    y_scale = y_weight / W
    A = np.concatenate([A_mass, A_y * y_scale], axis=0)
    b = np.concatenate([b_mass, b_y * y_scale], axis=0)

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    out = dict(
        alpha_L=float(theta[0]), alpha_R=float(theta[1]),
        gamma_L=float(theta[2]), gamma_R=float(theta[3]),
    )
    for i, name in enumerate(extra_names):
        out[name] = float(theta[4 + i])
    return out


def fit_x(data, mass_params):
    """Fit (T_body, k_pitch, x_offset) using estimated m_total from mass_params."""
    v = data["valid"]
    R_L = data["R_L"][v]
    R_R = data["R_R"][v]
    tau_lhip = data["tau_lhip"][v]
    tau_rhip = data["tau_rhip"][v]
    cos_pitch = data["cos_pitch"][v]
    tan_pitch = data["tan_pitch"][v]
    x_true = data["x_rel_true"][v]  # actual settled load x in body frame
    g = data["gravity"]

    aL = mass_params["alpha_L"]; aR = mass_params["alpha_R"]
    gL = mass_params["gamma_L"]; gR = mass_params["gamma_R"]
    m_est = aL * R_L + gL + aR * R_R + gR
    if "beta_pitch" in mass_params:
        sin_pitch = np.sin(data["pitch"][v])
        m_est = m_est + 2.0 * mass_params["beta_pitch"] * sin_pitch
    if "beta_hip" in mass_params:
        hip_diff = data["theta_lhip"][v] - data["theta_rhip"][v]
        m_est = m_est + 2.0 * mass_params["beta_hip"] * hip_diff
    if "beta_abad" in mass_params:
        abad_diff = data["theta_labad"][v] - data["theta_rabad"][v]
        m_est = m_est + 2.0 * mass_params["beta_abad"] * abad_diff

    # Discard samples where m_est is too close to zero (formula blows up there).
    safe = np.abs(m_est) > 0.5
    R_L, R_R = R_L[safe], R_R[safe]
    tau_lhip, tau_rhip = tau_lhip[safe], tau_rhip[safe]
    cos_pitch, tan_pitch = cos_pitch[safe], tan_pitch[safe]
    x_true = x_true[safe]
    m_est = m_est[safe]

    tau_diff = tau_lhip - tau_rhip
    a_x = tau_diff / (m_est * g * cos_pitch)
    # (a_x) - T_body/(m_est*g*cos_pitch) - k_pitch*tan_pitch + offset = x_true
    # Unknowns: T_body, k_pitch, offset
    col_T = -1.0 / (m_est * g * cos_pitch)
    col_k = -tan_pitch
    col_o = np.ones_like(a_x)
    A = np.stack([col_T, col_k, col_o], axis=1)
    b = x_true - a_x

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    T_body, k_pitch, x_offset = theta
    return dict(T_body=float(T_body), k_pitch=float(k_pitch), x_offset=float(x_offset))


def predict(data, mass_params, x_params):
    """Compute m_est, y_est, x_est on every sample (valid or not) using fitted params."""
    R_L = data["R_L"]; R_R = data["R_R"]
    yfL = data["y_foot_L"]; yfR = data["y_foot_R"]
    tau_lhip = data["tau_lhip"]; tau_rhip = data["tau_rhip"]
    cos_pitch = data["cos_pitch"]; tan_pitch = data["tan_pitch"]
    g = data["gravity"]

    m_L = mass_params["alpha_L"] * R_L + mass_params["gamma_L"]
    m_R = mass_params["alpha_R"] * R_R + mass_params["gamma_R"]
    if "beta_pitch" in mass_params:
        sin_pitch_all = np.sin(data["pitch"])
        m_L = m_L + mass_params["beta_pitch"] * sin_pitch_all
        m_R = m_R + mass_params["beta_pitch"] * sin_pitch_all
    if "beta_hip" in mass_params:
        hip_diff_all = data["theta_lhip"] - data["theta_rhip"]
        m_L = m_L + mass_params["beta_hip"] * hip_diff_all
        m_R = m_R + mass_params["beta_hip"] * hip_diff_all
    if "beta_abad" in mass_params:
        abad_diff_all = data["theta_labad"] - data["theta_rabad"]
        m_L = m_L + mass_params["beta_abad"] * abad_diff_all
        m_R = m_R + mass_params["beta_abad"] * abad_diff_all
    m_total = m_L + m_R

    m_safe = np.where(np.abs(m_total) < 1e-3, np.sign(m_total + 1e-9) * 1e-3, m_total)
    y_est = (yfL * m_L + yfR * m_R) / m_safe
    x_est = (
        (tau_lhip - tau_rhip - x_params["T_body"]) / (m_safe * g * cos_pitch)
        - x_params["k_pitch"] * tan_pitch
        + x_params["x_offset"]
    )
    return m_total, y_est, x_est


def residual_report(data, m_est, y_est, x_est):
    """Print per-condition residuals (mean+/-std) so structural error is visible."""
    v = data["valid"]
    cond_id = data.get("cond_id")  # only present in per-mass files; concatenated keeps it
    m_true = data["mass_true"]; x_true = data["x_rel_true"]; y_true = data["y_rel_true"]

    em = m_est - m_true
    ex = x_est - x_true
    ey = y_est - y_true

    print("\n=== overall (valid only) ===")
    print(f"  mass:  bias={em[v].mean():+.3f}kg  rmse={np.sqrt((em[v]**2).mean()):.3f}kg")
    print(f"  x   :  bias={ex[v].mean():+.3f}m   rmse={np.sqrt((ex[v]**2).mean()):.3f}m")
    print(f"  y   :  bias={ey[v].mean():+.3f}m   rmse={np.sqrt((ey[v]**2).mean()):.3f}m")

    # Per-(m,x,y) condition breakdown — bucket by SETPOINT (x_set/y_set),
    # not actual rel position (which jitters env-to-env).
    keys = list(zip(
        data["mass_true"].round(2),
        data["x_set"].round(3),
        data["y_set"].round(3),
    ))
    by_cond = defaultdict(list)
    for i, k in enumerate(keys):
        if v[i]:
            by_cond[k].append(i)

    print("\n=== per-condition (mass, x_set, y_set) ===")
    print(f"  {'m':>5}  {'x':>6}  {'y':>6}  | {'m_err':>10} {'x_err':>10} {'y_err':>10}  n")
    for k in sorted(by_cond.keys()):
        idxs = by_cond[k]
        if not idxs:
            continue
        em_k = em[idxs]; ex_k = ex[idxs]; ey_k = ey[idxs]
        print(
            f"  {k[0]:>5.2f}  {k[1]:>+6.3f}  {k[2]:>+6.3f}  | "
            f"{em_k.mean():>+5.2f}±{em_k.std():<4.2f} "
            f"{ex_k.mean():>+5.3f}±{ex_k.std():<5.3f} "
            f"{ey_k.mean():>+5.3f}±{ey_k.std():<5.3f} "
            f"{len(idxs)}"
        )


def print_config_snippet(mass_params, x_params, data):
    """Print the fitted values in a copy-pasteable form."""
    aL = mass_params["alpha_L"]; aR = mass_params["alpha_R"]
    gL = mass_params["gamma_L"]; gR = mass_params["gamma_R"]
    delta = data["delta"]; thigh_len = data["thigh_len"]

    print("\n=== fitted parameters ===")
    print(f"alpha_L  = {aL:.6f}")
    print(f"alpha_R  = {aR:.6f}")
    print(f"gamma_L  = {gL:.6f}   (= -alpha_L*mass_offset_L + beta_L)")
    print(f"gamma_R  = {gR:.6f}   (= -alpha_R*mass_offset_R + beta_R)")
    if "beta_pitch" in mass_params:
        print(f"beta_pitch = {mass_params['beta_pitch']:.6f}   (per-leg symmetric pitch correction)")
    if "beta_hip" in mass_params:
        print(f"beta_hip   = {mass_params['beta_hip']:.6f}   (per-leg symmetric hip-diff correction)")
    if "beta_abad" in mass_params:
        print(f"beta_abad  = {mass_params['beta_abad']:.6f}   (per-leg symmetric abad-diff correction)")
    print(f"T_body_x = {x_params['T_body']:.6f}")
    print(f"k_pitch  = {x_params['k_pitch']:.6f}   (com_x_bias in current code)")
    print(f"x_offset = {x_params['x_offset']:.6f}")

    # Translate to the per-leg form used in wheelfoot_flat.py if you want to swap in:
    print("\n=== drop-in formula form (matches wheelfoot_flat.py shape) ===")
    if "beta_pitch" in mass_params:
        bp = mass_params["beta_pitch"]
        print("payload_mass_left  = alpha_L * (T_L / ((thigh_len * cos_l + delta) * g * cos_abad_L)) + gamma_L + beta_pitch * sin(pitch)")
        print("payload_mass_right = alpha_R * (T_R / ((thigh_len * cos_r + delta) * g * cos_abad_R)) + gamma_R + beta_pitch * sin(pitch)")
        print(f"# with: alpha_L={aL:.4f}, alpha_R={aR:.4f}, gamma_L={gL:.4f}, gamma_R={gR:.4f}, beta_pitch={bp:.4f}, delta={delta:.5f}")
    else:
        print("payload_mass_left  = alpha_L * (load_torque_left  / ((thigh_len * cos_l + delta) * g * cos_abad_L)) + gamma_L")
        print("payload_mass_right = alpha_R * (load_torque_right / ((thigh_len * cos_r + delta) * g * cos_abad_R)) + gamma_R")
        print(f"# with: alpha_L={aL:.4f}, alpha_R={aR:.4f}, gamma_L={gL:.4f}, gamma_R={gR:.4f}, delta={delta:.5f}")
    print(
        f"load_x = (-power_rhip + power_lhip - {x_params['T_body']:.4f}) "
        f"/ payload_mass_safe / gravity / cos_pitch "
        f"- {x_params['k_pitch']:.4f} * tan_pitch + {x_params['x_offset']:.4f}"
    )


def run_one_model(data, label, y_weight, with_pitch=False, with_hip=False, with_abad=False):
    print("\n" + "#" * 70)
    print(f"# {label}")
    print("#" * 70)
    mass_params = fit_mass_and_y(
        data, y_weight=y_weight, with_pitch=with_pitch,
        with_hip=with_hip, with_abad=with_abad,
    )
    x_params = fit_x(data, mass_params)
    m_est, y_est, x_est = predict(data, mass_params, x_params)
    residual_report(data, m_est, y_est, x_est)
    print_config_snippet(mass_params, x_params, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths or glob patterns for calib_m*.npz files")
    parser.add_argument("--y_weight", type=float, default=1.0,
                        help="Relative weight of y-balance equations (default 1.0)")
    parser.add_argument("--model",
                        choices=["basic", "pitch", "hip", "abad", "hip_abad", "all_extras", "all"],
                        default="all",
                        help="Which model(s) to fit. 'all' runs basic + pitch + hip + abad + hip+abad + all_extras.")
    args = parser.parse_args()

    data = load_all(args.inputs)
    if data["valid"].sum() == 0:
        raise RuntimeError("No valid samples after loading. Check collection.")

    if args.model in ("basic", "all"):
        run_one_model(data, "MODEL A: basic (4 mass params)",
                      y_weight=args.y_weight)
    if args.model in ("pitch", "all"):
        run_one_model(data, "MODEL B: + beta_pitch * sin(pitch) (5 mass params)",
                      y_weight=args.y_weight, with_pitch=True)
    if args.model in ("hip", "all"):
        run_one_model(data, "MODEL C: + beta_hip * (theta_lhip - theta_rhip) (5 mass params)",
                      y_weight=args.y_weight, with_hip=True)
    if args.model in ("abad", "all"):
        run_one_model(data, "MODEL E: + beta_abad * (theta_labad - theta_rabad) (5 mass params)",
                      y_weight=args.y_weight, with_abad=True)
    if args.model in ("hip_abad", "all"):
        run_one_model(data, "MODEL F: hip_diff + abad_diff (6 mass params)",
                      y_weight=args.y_weight, with_hip=True, with_abad=True)
    if args.model in ("all_extras", "all"):
        run_one_model(data, "MODEL G: pitch + hip_diff + abad_diff (7 mass params)",
                      y_weight=args.y_weight, with_pitch=True, with_hip=True, with_abad=True)


if __name__ == "__main__":
    main()
