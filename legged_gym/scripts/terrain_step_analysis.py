#!/usr/bin/env python3
"""Audit physical terrain rollouts and plot their actual paths and camera frames."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from legged_gym.scripts.payload_step_analysis import write_csv, mean, rms
from legged_gym.scripts.terrain_step_protocol import (
    DT, POLICIES, TERRAINS, SEEDS, X0, X1, Y0, Y1, FRAME_TIMES,
    sample_height_numpy, sha256_file, array_digest, write_json,
)


def load_validate(root, preflight=False):
    root = Path(root)
    records = {}
    expected = {(p, t, s) for p in POLICIES for t in TERRAINS for s in ((12345,) if preflight else SEEDS)}
    folder = root / ("preflight" if preflight else "batches")
    audits = []
    scenes = {s: dict(np.load(root / "scenarios" / f"seed_{s}.npz", allow_pickle=False))
              for s in ((12345,) if preflight else SEEDS)}
    for path in sorted(folder.glob("*/result.json")):
        meta = json.loads(path.read_text())
        if path.parent.name != meta["batch_id"]:
            continue
        if meta["exit_status"] != "completed" or not meta.get("cleanup_completed", False):
            raise ValueError(f"incomplete process: {path}")
        key = (meta["policy_id"], meta["terrain"], meta["evaluation_seed"])
        with np.load(path.parent / "trace.npz", allow_pickle=False) as saved:
            z = {k: v.copy() for k, v in saved.items()}
        if sha256_file(path.parent / "trace.npz") != meta["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        if array_digest(scenes[key[2]]) != meta["scene_array_sha256"]:
            raise ValueError("scene differs from the frozen tables")
        steps, count = z["root_state"].shape[:2]
        np.testing.assert_allclose(z["time_s"], np.arange(1, steps+1)*DT, atol=1e-6)
        np.testing.assert_allclose(z["task_commands"], np.broadcast_to([0, 0, 0, .4], (steps, count, 4)), atol=1e-7)
        np.testing.assert_array_equal(z["actor_commands"], 0)
        np.testing.assert_allclose(z["observations"][..., -3], .4, atol=1e-7)
        if z["has_load"].any() or z["patch_exited"].any():
            raise ValueError("load contamination or trajectory left its assigned terrain patch")
        if (np.diff(z["first_failure"].astype(int), axis=0) < 0).any():
            raise ValueError("a failed trajectory was reset")
        if not np.isfinite(z["actions"]).all():
            raise ValueError("nonfinite policy output")
        scene = scenes[key[2]]
        ids = np.broadcast_to(np.arange(count)[None], (steps, count))
        local = z["root_state"][..., :2] - scene["mesh_anchors_xy"][None]
        expected_height = sample_height_numpy(scene["height_"+key[1]], ids, local)
        np.testing.assert_allclose(z["ground_at_base"], expected_height, atol=5e-6)
        feet = z["rigid_body_state"][:, :, meta["feet_indices"], :]
        matrix = Rotation.from_quat(feet[..., 3:7].reshape(-1, 4)).as_matrix().reshape(steps, count, 2, 3, 3)
        center = feet[..., :3] + np.einsum("tnwij,wj->tnwi", matrix, np.asarray(meta["wheel_collision_offset_m"]))
        axis = matrix[..., :, 1]
        low = center.copy()
        radius = np.asarray(meta["wheel_radius_m"])
        width = np.asarray(meta["wheel_width_m"])
        if meta["wheel_collision_geometry"] != "capsule":
            raise ValueError("this frozen protocol expects the original capsule conversion")
        low[..., 2] -= radius[None, None]
        low -= .5 * width[None, None, :, None] * np.sign(axis[..., 2])[..., None] * axis
        np.testing.assert_allclose(z["wheel_lowest_world"], low, atol=3e-5)
        foot_ids = np.broadcast_to(np.arange(count)[None, :, None], (steps, count, 2))
        local_low = z["wheel_lowest_world"][..., :2] - scene["mesh_anchors_xy"][None, :, None]
        ground_low = sample_height_numpy(scene["height_"+key[1]], foot_ids, local_low)
        np.testing.assert_allclose(z["ground_at_lowest"], ground_low, atol=5e-6)
        np.testing.assert_allclose(z["wheel_clearance_signed"], z["wheel_lowest_world"][..., 2] - ground_low, atol=5e-6)
        valid = ~z["first_failure"]
        inside = ((local[..., 0] >= X0+.5) & (local[..., 0] <= X1-.5)
                  & (local[..., 1] >= Y0+.5) & (local[..., 1] <= Y1-.5))
        if not inside[valid].all():
            raise ValueError("valid sample outside assigned patch")
        records[key] = (meta, z, path.parent)
        audits.append({"batch_id": meta["batch_id"], "samples": steps*count,
            "constant_commands": True, "exact_triangle_height": True, "capsule_clearance": True,
            "inside_own_patch": True, "no_load_no_reset": True, "failed": int(z["first_failure"][-1].sum())})
    if set(records) != expected:
        raise ValueError(f"incomplete matrix: missing={expected-set(records)}")
    for terrain in TERRAINS:
        for seed in ((12345,) if preflight else SEEDS):
            a, az, adir = records[("base", terrain, seed)]
            b, bz, bdir = records[("adapted", terrain, seed)]
            if a["mesh_sha256"] != b["mesh_sha256"]:
                raise ValueError("policies saw different meshes")
            for key in ("base_mass0", "base_com0", "imu_offset_quaternion", "noise_injected", "noise_effective"):
                np.testing.assert_array_equal(az[key], bz[key], err_msg=f"unpaired {key}")
            np.testing.assert_allclose(az["root_state"][0], bz["root_state"][0], atol=1e-6)
            for name in ("actual_body_properties.json", "actual_shape_properties.json"):
                if json.loads((adir/name).read_text()) != json.loads((bdir/name).read_text()):
                    raise ValueError("unpaired physical properties")
    return records, scenes, {"status": "passed", "batch_count": len(records), "audits": audits,
                             "scope": "preflight" if preflight else "formal"}


def angles(z):
    q = z["root_state"][..., 3:7].astype(np.float64)
    x, y, zz, w = [q[..., i] for i in range(4)]
    roll = np.rad2deg(np.arctan2(2*(w*x+y*zz), 1-2*(x*x+y*y)))
    pitch = np.rad2deg(np.arcsin(np.clip(2*(w*y-zz*x), -1, 1)))
    yaw = np.rad2deg(np.arctan2(2*(w*zz+x*y), 1-2*(y*y+zz*zz)))
    return roll, pitch, yaw


def metrics(z, env, mask, terrain):
    keys = ("speed_rmse_mps", "pitch_rms_deg", "roll_rms_deg", "pitch_sd_deg", "roll_sd_deg",
            "left_contact_fraction", "right_contact_fraction", "double_support_fraction",
            "left_only_fraction", "right_only_fraction", "no_contact_fraction", "speed_mean_mps",
            "world_forward_mean_mps", "yaw_rms_deg", "pitch_mean_deg", "roll_mean_deg")
    if not mask.any():
        return dict.fromkeys(keys)
    roll, pitch, yaw = angles(z)
    vx = z["body_velocity"][mask, env, 0].astype(float)
    force = z["wheel_forces"][mask, env, :, 2]
    left, right = force[:, 0] > 1., force[:, 1] > 1.
    slope = np.deg2rad(5 if terrain == "slope5" else 0)
    forward = z["root_state"][mask, env, 7] * np.cos(slope) + z["root_state"][mask, env, 9] * np.sin(slope)
    values = [rms(vx-.4), rms(pitch[mask, env]), rms(roll[mask, env]),
              float(np.std(pitch[mask, env])), float(np.std(roll[mask, env])),
              float(left.mean()), float(right.mean()), float((left & right).mean()),
              float((left & ~right).mean()), float((~left & right).mean()), float((~left & ~right).mean()),
              float(vx.mean()), float(forward.mean()), rms(yaw[mask, env]),
              float(np.mean(pitch[mask, env])), float(np.mean(roll[mask, env]))]
    return dict(zip(keys, values))


def analyze(records, output):
    rows, path_rows, pair_rows, direction_rows = [], [], [], []
    for (policy, terrain, seed), (meta, z, directory) in sorted(records.items()):
        initial_x = json.loads((directory / "effective_env_cfg.json").read_text())["init_state"]["pos"][0]
        for env in range(30):
            valid = ~z["first_failure"][:, env]
            early_fail = np.flatnonzero(~valid)
            identity = {"policy_id": policy, "terrain": terrain, "evaluation_seed": seed, "env_id": env}
            row = {**identity, "completed_6s": int(not early_fail.size),
                "first_failure_s": None if not early_fail.size else float(z["time_s"][early_fail[0]]),
                "valid_samples": int(valid.sum()), "sample_equivalent_s": float(valid.sum()*DT),
                **metrics(z, env, valid, terrain)}
            post = valid & (z["time_s"] >= 1.-1e-6)
            row.update({"post1s_"+k: v for k, v in metrics(z, env, post, terrain).items()})
            rows.append(row)
            last_valid = np.flatnonzero(valid)
            displacement = (float(z["root_state"][last_valid[-1], env, 0]
                                   - z["mesh_anchors_xy"][env, 0] - initial_x)
                            if last_valid.size else None)
            direction_rows.append({**identity, "completed_6s": bool(not early_fail.size),
                "last_valid_time_s": float(z["time_s"][last_valid[-1]]) if last_valid.size else None,
                "delta_world_x_m": displacement,
                "positive_world_x": displacement > 0 if displacement is not None else None})
        if seed == 12345:
            local = z["root_state"][:, 0, :2] - z["mesh_anchors_xy"][0]
            arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=-1))))
            roll, pitch, yaw = angles(z)
            for i, t in enumerate(z["time_s"]):
                path_rows.append({"policy_id": policy, "terrain": terrain, "evaluation_seed": seed,
                    "env_id": 0, "time_s": float(t), "valid": int(not z["first_failure"][i, 0]),
                    "path_horizontal_arc_m": float(arc[i]), "path_local_x_m": float(local[i, 0]),
                    "path_local_y_m": float(local[i, 1]), "actual_ground_z_m": float(z["ground_at_base"][i, 0]),
                    "pitch_deg": float(pitch[i, 0]), "roll_deg": float(roll[i, 0]),
                    "left_Fz_N": float(z["wheel_forces"][i, 0, 0, 2]),
                    "right_Fz_N": float(z["wheel_forces"][i, 0, 1, 2]),
                    "left_clearance_m": float(z["wheel_clearance_signed"][i, 0, 0]),
                    "right_clearance_m": float(z["wheel_clearance_signed"][i, 0, 1])})
    for terrain in TERRAINS:
        for seed in SEEDS:
            a, b = records[("base", terrain, seed)][1], records[("adapted", terrain, seed)][1]
            for env in range(30):
                mask = ~(a["first_failure"][:, env] | b["first_failure"][:, env])
                am, bm = metrics(a, env, mask, terrain), metrics(b, env, mask, terrain)
                pair = {"terrain": terrain, "evaluation_seed": seed, "env_id": env,
                        "common_samples": int(mask.sum())}
                pair.update({"adapted_minus_base_"+k: None if am[k] is None else bm[k]-am[k] for k in am})
                pair_rows.append(pair)
    numerical = [k for k in rows[0] if k not in ("policy_id", "terrain", "evaluation_seed", "env_id", "first_failure_s")]
    seed_groups, groups = defaultdict(list), defaultdict(list)
    for row in rows:
        seed_groups[(row["policy_id"], row["terrain"], row["evaluation_seed"])].append(row)
        groups[(row["policy_id"], row["terrain"])].append(row)
    seed_rows = []
    for (policy, terrain, seed), members in sorted(seed_groups.items()):
        seed_rows.append({"policy_id": policy, "terrain": terrain, "evaluation_seed": seed,
                          **{k: mean(r[k] for r in members) for k in numerical}})
    summary = []
    for (policy, terrain), members in sorted(groups.items()):
        seed_units = [r for r in seed_rows if r["policy_id"] == policy and r["terrain"] == terrain]
        summary.append({"policy_id": policy, "terrain": terrain, "planned_trajectories": len(members),
            "completed_count": sum(r["completed_6s"] for r in members),
            **{k: mean(r[k] for r in seed_units) for k in numerical},
            **{k+"_sd_across_env": float(np.std([r[k] for r in members if r[k] is not None])) for k in numerical}})
    for name, data in (("trajectory_metrics", rows), ("evaluation_seed_summary", seed_rows),
                       ("summary", summary), ("paired_common_window_metrics", pair_rows),
                       ("preregistered_path_and_timeseries", path_rows),
                       ("forward_progress_diagnostics", direction_rows)):
        write_csv(output / f"{name}.csv", data)
    write_json(output / "summary.json", {"rows": summary, "trajectories": len(rows),
        "pairs": len(pair_rows), "independent_training_replicates_per_policy": 1,
        "training_relation": "adapted policy resumes the base policy; not a single-factor causal ablation"})
    table = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Terrain & Policy & No termination & Speed RMSE & Pitch RMS & Roll RMS & L contact & R contact & Both \\",
             r" & & & (m/s) & (deg) & (deg) & (\%) & (\%) & (\%) \\", r"\midrule"]
    terrain_names = {"flat": "Flat", "slope5": r"$5^\circ$ slope", "rough5cm": r"Rough $\pm5$ cm"}
    for terrain in TERRAINS:
        for policy in POLICIES:
            row = next(r for r in summary if r["policy_id"] == policy and r["terrain"] == terrain)
            table.append(f"{terrain_names[terrain]} & {POLICIES[policy][0]} & {row['completed_count']}/90 & "
                         f"{row['speed_rmse_mps']:.4f} & {row['pitch_rms_deg']:.4f} & {row['roll_rms_deg']:.4f} & "
                         f"{100*row['left_contact_fraction']:.2f} & {100*row['right_contact_fraction']:.2f} & "
                         f"{100*row['double_support_fraction']:.2f} " + r"\\")
    table.extend((r"\bottomrule", r"\end{tabular}",
                  "% No termination is not successful uphill progress. One training lineage per policy."))
    (output / "summary_table.tex").write_text("\n".join(table)+"\n", encoding="utf-8")
    from scipy.io import savemat
    matlab = {"geometry": "original capsule-converted wheel collider; local vertical clearance in metres"}
    for terrain in TERRAINS:
        for policy in POLICIES:
            z = records[(policy, terrain, 12345)][1]
            roll, pitch, yaw = angles(z)
            matlab["time_s"] = z["time_s"]
            matlab[f"{terrain}_{policy}"] = {
                "root_state": z["root_state"][:, 0], "pitch_deg": pitch[:, 0], "roll_deg": roll[:, 0],
                "clearance_m": z["wheel_clearance_signed"][:, 0], "wheel_forces_N": z["wheel_forces"][:, 0],
                "wheel_lowest_world_m": z["wheel_lowest_world"][:, 0], "ground_z_m": z["ground_at_base"][:, 0],
                "first_failure": z["first_failure"][:, 0], "task_commands": z["task_commands"][:, 0]}
    savemat(output / "preregistered_process_examples.mat", matlab, do_compression=True, long_field_names=True)
    return summary


def render_figures(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.text import Text
    from PIL import Image
    import shutil
    from legged_gym.scripts.figs_chx.fig0_sixd_design_variables import configure_fonts

    original = next(iter(records.values()))[2].parents[1] / "analysis/fonts"
    if (original / "CodesignNotoSansSC-Regular.ttf").exists():
        (output / "fonts").mkdir(exist_ok=True)
        for source in original.glob("*.ttf"):
            target = output / "fonts" / source.name
            if target.resolve() != source.resolve():
                shutil.copy2(source, target)
            font_manager.fontManager.addfont(str(target))
        plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Codesign Noto Sans SC", "DejaVu Sans"],
                             "pdf.fonttype": 42, "svg.fonttype": "path", "axes.unicode_minus": False})
    else:
        configure_fonts(output, extra_text=Path(__file__).read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 14, "axes.titleweight": "normal", "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = {"base": "#347BB4", "adapted": "#1C927C"}
    for language in ("zh", "en"):
        zh = language == "zh"
        names = {"base": "基础策略" if zh else "Base", "adapted": "地形适配策略" if zh else "Terrain-adapted"}
        for terrain in TERRAINS:
            terrain_label = ({"flat": "平地", "slope5": "5°连续上坡", "rough5cm": "±5 cm粗糙地面"} if zh else
                             {"flat": "Flat", "slope5": "Continuous 5° uphill slope", "rough5cm": "Rough ground (±5 cm)"})[terrain]
            fig = plt.figure(figsize=(14.4, 9.1))
            profile = fig.add_axes([.09, .765, .88, .16])
            for row, policy in enumerate(POLICIES):
                meta, z, directory = records[(policy, terrain, 12345)]
                valid = ~z["first_failure"][:, 0]
                xy = z["root_state"][:, 0, :2]
                arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))))
                profile.plot(arc[valid], z["ground_at_base"][valid, 0]*100,
                             color=colors[policy], lw=1.4, label=names[policy])
                for col, time_s in enumerate(FRAME_TIMES):
                    ax = fig.add_axes([.075+col*.225, .405-row*.32, .21, .245])
                    path = directory / "frames" / f"t_{time_s:.2f}.png"
                    if not path.exists():
                        raise ValueError(f"missing real camera frame: {path}")
                    with Image.open(path) as im:
                        rgb = np.asarray(im)
                        saturated = rgb.max(-1).astype(int) - rgb.min(-1).astype(int) > 40
                        yy, xx = np.where(saturated)
                        if not len(xx) or xx.min() < 200 or xx.max() >= 760 or yy.min() < 140 or yy.max() >= 560:
                            raise ValueError("fixed screenshot crop would cut robot geometry")
                        ax.imshow(rgb)
                        ax.set_xlim(200, 760)
                        ax.set_ylim(560, 140)
                    ax.set_axis_off()
                    sample = int(round(time_s/DT))-1
                    failed = bool(z["first_failure"][sample, 0])
                    ax.text(.5, -.055, f"{time_s:.2f} s" + (" *" if failed else ""),
                            transform=ax.transAxes, ha="center", va="top", fontsize=14,
                            color="#B55348" if failed else "#29343B")
                fig.text(.024, .527-row*.32, names[policy], rotation=90, ha="center", va="center", fontsize=14)
            profile.set_ylabel("地面高程 (cm)" if zh else "Ground elevation (cm)")
            profile.set_xlabel("沿实际路径的水平弧长 (m)" if zh else "Horizontal arc length along the actual path (m)")
            profile.grid(axis="y", color="#DDE3E7", lw=.6)
            fig.legend(*profile.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
                       bbox_to_anchor=(.54, .995))
            fig.text(.09, .985, terrain_label, ha="left", va="top", fontsize=14)
            save(fig, output / f"terrain_profile_frames_{terrain}_{language}")
            plt.close(fig)

            fig, axes = plt.subplots(4, 2, figsize=(13.8, 9.6), sharex=True,
                                     gridspec_kw={"height_ratios": [.55, 1, 1, 1]})
            for col, policy in enumerate(POLICIES):
                _, z, _ = records[(policy, terrain, 12345)]
                t = z["time_s"]
                valid = ~z["first_failure"][:, 0]
                contact = z["wheel_forces"][:, 0, :, 2] > 1
                roll, pitch, _ = angles(z)
                for wheel, color in enumerate(("#7386B3", "#BA9B52")):
                    y0 = 1-wheel
                    axes[0, col].fill_between(t, y0, y0+.65, where=contact[:, wheel] & valid,
                                              step="mid", color=color)
                    clearance = z["wheel_clearance_signed"][:, 0, wheel]*100
                    axes[1, col].plot(t, np.where(valid, clearance, np.nan), color=color, lw=.6, alpha=.25)
                    axes[1, col].plot(t, np.where(valid & ~contact[:, wheel], clearance, np.nan),
                                      color=color, lw=1.4, label=("左轮" if wheel == 0 else "右轮") if zh else ("L" if wheel == 0 else "R"))
                axes[0, col].set_yticks([.325, 1.325])
                axes[0, col].set_yticklabels(["右轮", "左轮"] if zh else ["R", "L"])
                axes[0, col].set_ylim(-.1, 1.85)
                axes[2, col].plot(t, np.where(valid, pitch[:, 0], np.nan), color="#B45C52", lw=1.3)
                axes[3, col].plot(t, np.where(valid, roll[:, 0], np.nan), color="#427E9E", lw=1.3)
                failure = np.flatnonzero(~valid)
                for row in range(4):
                    ax = axes[row, col]
                    ax.axvspan(FRAME_TIMES[0], FRAME_TIMES[-1], color="#E8ECEF", alpha=.5, zorder=0)
                    if failure.size:
                        ax.axvline(t[failure[0]], color="#B55348", lw=1)
                    ax.grid(axis="y", color="#DDE3E7", lw=.6)
                axes[-1, col].set_xlim(0, 6)
                axes[-1, col].set_xlabel("时间 (s)" if zh else "Time (s)")
            for row in (1, 2, 3):
                bounds = [ax.get_ylim() for ax in axes[row]]
                for ax in axes[row]:
                    ax.set_ylim(min(v[0] for v in bounds), max(v[1] for v in bounds))
            labels = (["实际轮接触", "轮缘竖向净空 (cm)", "机体俯仰 (deg)", "机体横滚 (deg)"] if zh else
                      ["Measured contact", "Vertical rim clearance (cm)", "Body pitch (deg)", "Body roll (deg)"])
            for row, text in enumerate(labels):
                axes[row, 0].set_ylabel(text)
            fig.subplots_adjust(left=.105, right=.98, bottom=.08, top=.895, hspace=.22, wspace=.17)
            for col, policy in enumerate(POLICIES):
                box = axes[0, col].get_position()
                fig.text((box.x0+box.x1)/2, .925, names[policy], ha="center", va="bottom", fontsize=14)
            fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
                       bbox_to_anchor=(.54, .995))
            fig.text(.105, .985, terrain_label, ha="left", va="top", fontsize=14)
            save(fig, output / f"terrain_contact_clearance_attitude_{terrain}_{language}")
            plt.close(fig)


def save(fig, stem):
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ignored = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not min(limits)-1e-8 <= tick.get_loc() <= max(limits)+1e-8:
                    ignored.update((tick.label1, tick.label2))
    for item in fig.findobj(Text):
        if item not in ignored and item.get_visible() and item.get_text().strip():
            bbox = item.get_window_extent(renderer)
            if min(bbox.x0, bbox.y0) < -1 or bbox.x1 > fig.bbox.width+1 or bbox.y1 > fig.bbox.height+1:
                raise ValueError(f"text outside figure: {stem}: {item.get_text()}")
    for legend in fig.legends:
        box = legend.get_window_extent(renderer)
        for text in fig.texts:
            if box.overlaps(text.get_window_extent(renderer)):
                raise ValueError(f"legend overlaps figure label: {stem}")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(str(stem)+"."+extension, dpi=300, facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, scenes, validation = load_validate(args.input, args.preflight)
    write_json(args.output / ("preflight_validation.json" if args.preflight else "validation.json"), validation)
    if not args.preflight:
        analyze(records, args.output)
        if not args.no_figures:
            render_figures(records, args.output)
    print(json.dumps({"validation": "passed", "batches": len(records), "preflight": args.preflight,
                      "output": str(args.output)}, indent=2))
