#!/usr/bin/env python3
"""Validate paired physical trials and plot all-assigned posture, events and drift."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.io import savemat

from legged_gym.scripts.eccentric_payload_protocol import (
    DT, POLICIES, MASSES, PLACEMENTS, SEEDS, MAIN_MASS, MAIN_SEED, MAIN_ENV,
    angles, events, array_digest, sha256_file, write_json,
)
from legged_gym.scripts.payload_step_analysis import write_csv, mean, rms


def metrics(root, roll, pitch, env, mask, reference):
    names = ("pitch_rms_deg", "roll_rms_deg", "pitch_sd_deg", "roll_sd_deg",
             "pitch_mean_deg", "roll_mean_deg", "max_planar_drift_m", "final_planar_drift_m")
    if not mask.any():
        return dict.fromkeys(names)
    p, r = pitch[mask,env], roll[mask,env]
    drift = np.linalg.norm(root[mask,env,:2]-reference, axis=-1)
    return dict(zip(names, [rms(p), rms(r), float(p.std()), float(r.std()), float(p.mean()),
                           float(r.mean()), float(drift.max()), float(drift[-1])]))


def analyze(root, output, preflight=False):
    seeds = (12345,) if preflight else SEEDS
    masses = (8,26) if preflight else MASSES
    expected = {(p,m,c,s) for p in POLICIES for m in masses for c in PLACEMENTS for s in seeds}
    seen, audits, rows, selected, signatures = set(), [], [], {}, {}
    source_audit = {}
    for path in sorted((root / ("preflight" if preflight else "batches")).glob("*/result.json")):
        meta = json.loads(path.read_text())
        key = (meta["policy_id"],meta["mass_kg"],meta["placement"],meta["evaluation_seed"])
        if key in seen:
            raise ValueError("duplicate batch")
        seen.add(key)
        if not meta["cleanup_completed"] or meta["exit_status"] != "completed":
            raise ValueError("incomplete process")
        if sha256_file(path.parent / "trace.npz") != meta["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        with np.load(path.parent / "trace.npz", allow_pickle=False) as saved:
            z = {k: v.copy() for k,v in saved.items()}
        t = z["time_s"]
        steps, count = z["root_state"].shape[:2]
        np.testing.assert_allclose(t, np.arange(1,steps+1)*DT, atol=3e-6)
        np.testing.assert_array_equal(z["task_commands"], 0)
        np.testing.assert_array_equal(z["actor_commands"], 0)
        np.testing.assert_allclose(meta["measured_payload_mass_kg"], key[1], atol=1e-5)
        if np.any(np.diff(z["first_failure"].astype(int), axis=0)<0):
            raise ValueError("a failed trial was reset")
        if not np.isfinite(z["root_state"]).all() or not np.isfinite(z["box_state"]).all():
            raise ValueError("nonfinite physical trajectory")
        if not np.isfinite(z["actions"]).all():
            raise ValueError("nonfinite action")
        # The tensor snapshot is taken inside the actual spawning callback, before integration.
        spawn_delta = z["spawn_box_state"][:,:3]-z["spawn_root_state"][:,:3]
        spawn_local = Rotation.from_quat(z["spawn_root_state"][:,3:7]).inv().apply(spawn_delta)
        np.testing.assert_allclose(spawn_local, np.tile([PLACEMENTS[key[2]],0,.1], (count,1)), atol=3e-6)
        np.testing.assert_allclose(z["spawn_box_state"][:,7:13], 0, atol=1e-7)
        np.testing.assert_allclose(z["mass_label"], z["on_body"]*key[1], atol=1e-5)
        factor = z["mass_label"]/(z["base_mass0"][None]+z["mass_label"])
        np.testing.assert_allclose(z["com_label"], factor[...,None]*(z["box_relative_position"]-z["base_com0"][None]), atol=2e-5)
        for i in range(count):
            spawn = np.flatnonzero(z["has_load"][:,i])
            if not spawn.size or abs(float(t[spawn[0]])-.5)>1e-5 or not z["has_load"][spawn[0]:,i].all():
                raise ValueError("spawn schedule or active unload mismatch")
        scene = dict(np.load(root / "scenarios" / f"seed_{key[3]}.npz", allow_pickle=False))
        scene["box_offset_body"][:,0] = PLACEMENTS[key[2]]
        if array_digest(scene) != meta["used_scene_array_sha256"]:
            raise ValueError("unfrozen scene")
        fingerprint = {
            "noise": array_digest({k:z[k] for k in ("noise_injected", "noise_effective", "imu_offset_quaternion")}),
            "initial": array_digest({"root":z["root_state"][0]}),
            "bodies": sha256_file(path.parent / "actual_body_properties.json"),
            "shapes": sha256_file(path.parent / "actual_shape_properties.json"),
        }
        prior = signatures.setdefault(key[3], fingerprint)
        if fingerprint != prior:
            raise ValueError(f"unpaired scene/physics/noise: {key}")
        source_audit[key[0]] = {"run_name":meta["run_name"], "checkpoint":meta["checkpoint"],
            "sha256":meta["checkpoint_sha256"], "obs_dim":meta["obs_dim"],
            "encoder_input_dim":meta["encoder_input_dim"], "training_seed":meta["training_seed"],
            "archived_source_sha256":meta["archived_source_sha256"]}
        audits.append({"batch_id":meta["batch_id"], "steps":steps, "envs":count,
            "zero_commands":True, "physical_mass_verified":True, "spawn_pose_verified":True,
            "paired_physics_noise":True, "no_reset_or_unload":True,
            "cleanup_completed":True})
        if not preflight:
            roll,pitch,up = angles(z["root_state"])
            ref_i = int(np.argmin(np.abs(t-.48)))
            for env in range(count):
                event = events(z,env)
                ref = z["root_state"][ref_i,env,:2]
                row = {"policy_id":key[0], "method":meta["method"], "mass_kg":key[1],
                    "placement":key[2], "evaluation_seed":key[3], "env_id":env, **event}
                masks = {"all":t>=.5-1e-6, "early":(t>=.5-1e-6)&(t<=2+1e-6),
                    "late":t>=35-1e-6, "at_risk":(t>=.5-1e-6)&(t<event["at_risk_end_exclusive_s"]-1e-6)}
                for name,mask in masks.items():
                    row.update({name+"_"+k:v for k,v in metrics(z["root_state"],roll,pitch,env,mask,ref).items()})
                    row[name+"_samples"] = int(mask.sum())
                row["on_body_fraction_postload"] = float(z["on_body"][masks["all"],env].mean())
                row["upright_fraction_postload"] = float((up[masks["all"],env]>=.5).mean())
                rows.append(row)
            if key[1] == MAIN_MASS and key[3] == MAIN_SEED:
                # Store only the preselected environment; do not retain all batches in RAM.
                selected[(key[0],key[2])] = {k:v[:,MAIN_ENV].copy() if v.ndim>=2 and v.shape[0]==steps
                    else v.copy() for k,v in z.items() if k in
                    ("time_s","root_state","box_state","box_relative_position","on_body","has_load",
                     "first_failure","wheel_forces","estimate","mass_label","com_label")}
    if seen != expected:
        raise ValueError(f"incomplete matrix: missing={expected-seen}; extra={seen-expected}")
    audit = {"status":"passed", "batches":len(seen), "audits":audits,
             "source_policies":source_audit, "all_noise_physics_pairs_verified":True}
    write_json(output / ("preflight_validation.json" if preflight else "validation.json"), audit)
    if preflight:
        return [], {}, []
    write_csv(output / "per_environment_metrics.csv", rows)
    excluded = {"policy_id","method","mass_kg","placement","evaluation_seed","env_id"}
    metric_keys = [k for k in rows[0] if k not in excluded and not k.endswith("_s")]
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in ("policy_id","mass_kg","placement","evaluation_seed"))].append(r)
    seed_rows=[]
    for key,members in sorted(groups.items()):
        r=dict(zip(("policy_id","mass_kg","placement","evaluation_seed"),key))
        r["n_assigned"]=len(members)
        for k in metric_keys:
            r[k]=mean(m.get(k) for m in members)
            r["n_"+k]=sum(m.get(k) is not None for m in members)
        seed_rows.append(r)
    write_csv(output / "per_evaluation_seed_metrics.csv", seed_rows)
    summary=[]
    for p in POLICIES:
        for mass in MASSES:
            for c in PLACEMENTS:
                members=[r for r in seed_rows if (r["policy_id"],r["mass_kg"],r["placement"])==(p,mass,c)]
                r={"policy_id":p,"method":POLICIES[p][0],"mass_kg":mass,"placement":c,
                   "n_assigned":90,"n_evaluation_seeds":3,"n_training_seeds":1}
                for k in metric_keys:
                    values=[m[k] for m in members if m[k] is not None]
                    r[k]=mean(values)
                    r[k+"_eval_seed_sd"]=float(np.std(values,ddof=1)) if len(values)>1 else None
                    r["n_"+k]=sum(m["n_"+k] for m in members)
                summary.append(r)
    write_csv(output / "summary.csv", summary)
    pair_rows=[]
    lookup={(r["policy_id"],r["mass_kg"],r["placement"],r["evaluation_seed"],r["env_id"]):r for r in rows}
    for p in POLICIES:
        for m in MASSES:
            for s in SEEDS:
                for e in range(30):
                    a,b=[lookup[(p,m,c,s,e)] for c in PLACEMENTS]
                    r={"policy_id":p,"mass_kg":m,"evaluation_seed":s,"env_id":e}
                    for k in metric_keys:
                        if a.get(k) is not None and b.get(k) is not None:
                            r["eccentric_minus_centered_"+k]=b[k]-a[k]
                    pair_rows.append(r)
    write_csv(output / "placement_paired_increments.csv", pair_rows)
    for (p,c),z in selected.items():
        savemat(output / f"main_m13_{p}_{c}.mat", z, do_compression=True)
    write_json(output / "summary.json", {"rows":summary,"figure_case":{
        "mass_kg":MAIN_MASS,"evaluation_seed":MAIN_SEED,"env_id":MAIN_ENV},
        "all_trajectories_retained":len(rows),"posture_window":"all assigned physical samples, 0.5--40 s",
        "uncertainty":"sample SD of 3 evaluation-seed means; one training seed per method"})
    return summary,selected,seed_rows


def render(summary,selected,output,font_source=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from legged_gym.scripts.figs_chx.fig0_sixd_design_variables import configure_fonts
    from legged_gym.scripts.terrain_step_analysis import save
    from matplotlib import font_manager
    import shutil
    if font_source is not None and (font_source / "CodesignNotoSansSC-Regular.ttf").exists():
        dest=output / "fonts"; dest.mkdir(exist_ok=True)
        for source in font_source.iterdir():
            target=dest / source.name
            if source.resolve()!=target.resolve():
                shutil.copy2(source,target)
        for font in dest.glob("*.ttf"):
            font_manager.fontManager.addfont(str(font))
    else:
        configure_fonts(output, extra_text=Path(__file__).read_text())
    plt.rcParams.update({"font.family":"Codesign Noto Sans SC","font.size":14,"axes.labelsize":14,
        "xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,"axes.titlesize":14,
        "pdf.fonttype":42,"svg.fonttype":"path","axes.unicode_minus":False,
        "axes.spines.top":False,"axes.spines.right":False})
    colors={"rl":"#AC5549","mg":"#267C9B"}
    for language in ("zh","en"):
        zh=language=="zh"
        c_names={"centered":"居中放置" if zh else "Initially centered",
                 "eccentric":"初始纵向偏心 +0.15 m" if zh else "Initial longitudinal offset +0.15 m"}
        for c in (PLACEMENTS if summary else ()):
            fig,axs=plt.subplots(2,2,figsize=(11.5,8.5))
            fields=("all_pitch_rms_deg","all_roll_rms_deg","fall","drop")
            labels=(["俯仰 RMS (deg)","横滚 RMS (deg)","跌倒率 (%)","掉载率 (%)"] if zh else
                    ["Pitch RMS (deg)","Roll RMS (deg)","Fall rate (%)","Payload-drop rate (%)"])
            for ax,k,label in zip(axs.flat,fields,labels):
                for p in POLICIES:
                    data=sorted([r for r in summary if r["policy_id"]==p and r["placement"]==c],key=lambda r:r["mass_kg"])
                    factor=100 if k in ("fall","drop") else 1
                    y=np.array([r[k] for r in data])*factor
                    sd=np.array([r[k+"_eval_seed_sd"] for r in data])*factor
                    ax.plot(MASSES,y,color=colors[p],marker="o",lw=1.5,ms=5,label=POLICIES[p][0])
                    ax.fill_between(MASSES,y-sd,y+sd,color=colors[p],alpha=.13,lw=0)
                ax.set_ylabel(label)
                ax.set_xlabel("负载质量 (kg)" if zh else "Payload mass (kg)")
                ax.set_xticks(MASSES)
                ax.grid(axis="y",color="#DFE5E8",lw=.6)
                if k in ("fall","drop"):
                    ax.set_ylim(-.3,10 if k=="fall" else 25)
                else:
                    ax.set_ylim(bottom=0)
            fig.subplots_adjust(left=.10,right=.97,bottom=.09,top=.88,hspace=.42,wspace=.29)
            fig.text(.10,.965,c_names[c],va="top")
            fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper right",bbox_to_anchor=(.985,.985),
                       frameon=False,ncol=2)
            save(fig,output / f"center_eccentric_posture_{c}_{language}")
            plt.close(fig)

        fig,axs=plt.subplots(4,2,figsize=(13.5,10.2),sharex=True)
        trace_rows=[]
        early_series=[[] for _ in range(4)]
        placement_colors={"centered":"#347F9D","eccentric":"#B76B41"}
        for col,p in enumerate(POLICIES):
            for c in PLACEMENTS:
                z=selected[(p,c)]; t=z["time_s"]
                valid=t>=.5-1e-6
                roll,pitch,_=angles(z["root_state"])
                reference=z["root_state"][np.argmin(np.abs(t-.48)),:2]
                delta=z["root_state"][:,:2]-reference
                drift=np.linalg.norm(delta,axis=-1)
                series=(z["box_relative_position"][:,0]*100,z["box_relative_position"][:,1]*100,pitch,drift)
                for row,y in enumerate(series):
                    early_series[row].append(y[valid & (t<=8.+1e-6)])
                # No censoring or survivor substitution: events and later response remain visible.
                expanded={k:v[:,None] if k!="time_s" else v for k,v in z.items()}
                event=events(expanded,0)
                color=placement_colors[c]
                for ax,y in zip(axs[:,col],series):
                    ax.plot(t,np.where(valid,y,np.nan),color=color,lw=1.1,label=c_names[c])
                    for field,style in (("drop_start_s",":"),("fall_start_s","--")):
                        if event[field] is not None:
                            ax.axvline(event[field],color=color,ls=style,lw=.9)
                    ax.axvline(.5,color="#7D878D",lw=.7)
                    ax.grid(axis="y",color="#DFE5E8",lw=.6)
                for i,time_s in enumerate(t):
                    trace_rows.append({"policy_id":p,"placement":c,"time_s":float(time_s),
                        "box_relative_x_m":float(z["box_relative_position"][i,0]),
                        "box_relative_y_m":float(z["box_relative_position"][i,1]),
                        "pitch_deg":float(pitch[i]),"roll_deg":float(roll[i]),
                        "robot_delta_x_m":float(delta[i,0]),"robot_delta_y_m":float(delta[i,1]),
                        "robot_planar_displacement_m":float(drift[i]),"on_body":int(z["on_body"][i]),
                        "mass_estimate_kg":float(z["estimate"][i,3])})
            axs[0,col].set_title(POLICIES[p][0],pad=9)
            axs[-1,col].set_xlim(0,40)
            axs[-1,col].set_xlabel("时间 (s)" if zh else "Time (s)")
        labels=(["箱体相对 x (cm)","箱体相对 y (cm)","机体俯仰 (deg)","平面位移 (m)"] if zh else
                ["Box relative x (cm)","Box relative y (cm)","Body pitch (deg)","Planar displacement (m)"])
        for i,label in enumerate(labels):
            axs[i,0].set_ylabel(label)
            lo=min(ax.get_ylim()[0] for ax in axs[i]); hi=max(ax.get_ylim()[1] for ax in axs[i])
            for ax in axs[i]:
                ax.set_ylim(lo,hi)
        handles=[Line2D([0],[0],color=placement_colors[c],label=c_names[c]) for c in PLACEMENTS]
        handles += [Line2D([0],[0],color="#58626A",ls=":",label="掉载" if zh else "Drop"),
                    Line2D([0],[0],color="#58626A",ls="--",label="跌倒" if zh else "Fall")]
        fig.subplots_adjust(left=.105,right=.98,bottom=.075,top=.88,hspace=.30,wspace=.16)
        fig.legend(handles=handles,loc="upper center",ncol=4,frameon=False,bbox_to_anchor=(.53,.995))
        save(fig,output / f"center_eccentric_dynamics_m13_{language}")
        for row in range(4):
            values=np.concatenate(early_series[row])
            low,high=float(values.min()),float(values.max())
            margin=max((high-low)*.07, .02 if row==3 else .1)
            for ax in axs[row]:
                ax.set_xlim(0,8)
                ax.set_xticks([0,2,4,6,8])
                ax.set_ylim(low-margin,high+margin)
        save(fig,output / f"center_eccentric_dynamics_m13_loading_{language}")
        plt.close(fig)
        if zh:
            write_csv(output / "main_m13_timeseries.csv",trace_rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--no-figures",action="store_true")
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    summary,selected,_=analyze(a.input,a.output,a.preflight)
    if not a.preflight and not a.no_figures:
        render(summary,selected,a.output,a.input / "analysis/fonts")
    print(f"Validation passed; output={a.output}")
