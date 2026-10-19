#!/usr/bin/env python3
"""Validate actually executed commands and summarize one policy's staged response."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.spatial.transform import Rotation

from legged_gym.scripts.roll_step_response_protocol import (
    DT,RUN,CHECKPOINT,SEEDS,COMMANDS,PHASE_NAMES,MAIN_SEED,MAIN_ENV,
    phase_at_step,first_failure,angles,array_digest,sha256_file,write_json,
)
from legged_gym.scripts.payload_step_analysis import write_csv,mean,rms


def motion_metrics(z,env,mask):
    names=("speed_rmse_mps","speed_mean_mps","world_x_speed_mean_mps","pitch_rms_deg","roll_rms_deg",
           "pitch_sd_deg","roll_sd_deg","pitch_mean_deg","roll_mean_deg","left_swing_fraction",
           "right_swing_fraction","double_support_fraction","left_only_fraction","right_only_fraction",
           "neither_contact_fraction","left_air_clearance_fraction","right_air_clearance_fraction",
           "roll_proxy_mean_mps","step_proxy_mean_mps")
    if not mask.any(): return dict.fromkeys(names)
    roll,pitch,_=angles(z["root_state"][mask,env])
    vx=z["body_velocity"][mask,env,0]
    contact=z["wheel_forces"][mask,env,:,2]>1.
    left,right=contact[:,0],contact[:,1]
    gap=z["wheel_clearance_signed"][mask,env]
    vals=[rms(vx-.5),float(vx.mean()),float(z["root_state"][mask,env,7].mean()),rms(pitch),rms(roll),
        float(pitch.std()),float(roll.std()),float(pitch.mean()),float(roll.mean()),
        float((~left).mean()),float((~right).mean()),float((left&right).mean()),
        float((left&~right).mean()),float((~left&right).mean()),float((~left&~right).mean()),
        float((~left&(gap[:,0]>.01)).mean()),float((~right&(gap[:,1]>.01)).mean()),
        float(z["v_roll_proxy"][mask,env].mean()),float(z["v_step_proxy"][mask,env].mean())]
    return dict(zip(names,vals))


def validate(root,output,preflight=False):
    records={}; audits=[]
    seeds=(12345,) if preflight else SEEDS
    folder=root/("preflight" if preflight else "batches")
    for path in sorted(folder.glob("*/result.json")):
        meta=json.loads(path.read_text()); seed=meta["evaluation_seed"]
        if seed in records: raise ValueError("duplicate seed")
        if not meta["cleanup_completed"] or meta["exit_status"]!="completed": raise ValueError("unfinished sim")
        assert meta["run_name"]==RUN and meta["checkpoint"]==CHECKPOINT
        if sha256_file(path.parent/"trace.npz")!=meta["trace_sha256"]: raise ValueError("trace hash mismatch")
        with np.load(path.parent/"trace.npz",allow_pickle=False) as saved: z=dict(saved)
        n=len(z["time_s"]); steps=np.arange(1,n+1)
        np.testing.assert_allclose(z["time_s"],steps*DT,atol=3e-6)
        np.testing.assert_array_equal(z["phase_for_next_action"],np.broadcast_to(phase_at_step(steps)[:,None],(n,30)))
        np.testing.assert_array_equal(z["phase_of_preceding_interval"],np.broadcast_to(phase_at_step(steps,True)[:,None],(n,30)))
        now=np.broadcast_to(COMMANDS[phase_at_step(steps)][:,None],(n,30,2))
        old=np.broadcast_to(COMMANDS[phase_at_step(steps,True)][:,None],(n,30,2))
        np.testing.assert_array_equal(z["task_roll_step"],now)
        np.testing.assert_array_equal(z["applied_previous_interval_roll_step"],old)
        np.testing.assert_array_equal(z["observations"][...,28],now[...,1])
        np.testing.assert_allclose(z["actor_commands"][...,0],now[...,0]*2,atol=1e-7)
        np.testing.assert_array_equal(z["actor_commands"][...,1:],0)
        np.testing.assert_array_equal(z["policy_input"][...,7:38],z["observations"])
        np.testing.assert_array_equal(z["policy_input"][...,38:41],z["actor_commands"])
        np.testing.assert_array_equal(z["history_step_commands"][...,-1],now[...,1])
        np.testing.assert_array_equal(z["history_step_commands"][1:,:,:-1],z["history_step_commands"][:-1,:,1:])
        if z["has_load"].any() or np.any(np.diff(z["first_failure"].astype(int),axis=0)<0):
            raise ValueError("load contamination or reset")
        if not np.isfinite(z["actions"]).all() or not np.isfinite(z["root_state"]).all(): raise ValueError("nonfinite rollout")
        scene=dict(np.load(root/"scenarios"/f"seed_{seed}.npz",allow_pickle=False))
        if array_digest(scene)!=meta["scene_array_sha256"]: raise ValueError("unfrozen scene")
        expected_delta=np.broadcast_to(scene["gait"][:,0]*DT,(n-1,30))
        actual_delta=np.remainder(np.diff(z["gait_phase"],axis=0),1.)
        np.testing.assert_allclose(actual_delta,expected_delta,atol=1e-6)
        feet=z["foot_state"]
        matrix=Rotation.from_quat(feet[...,3:7].reshape(-1,4)).as_matrix().reshape(n,30,2,3,3)
        center=feet[...,:3]+np.einsum("tnwij,wj->tnwi",matrix,np.asarray(meta["wheel_collision_offset_m"]))
        clearance=center[...,2]-np.asarray(meta["wheel_radius_m"])[None,None]
        clearance-=.5*np.asarray(meta["wheel_width_m"])[None,None]*np.abs(matrix[...,2,1])
        np.testing.assert_allclose(z["wheel_clearance_signed"],clearance,atol=3e-5)
        contact=(z["wheel_forces"][...,2]>1.).astype(float)
        wheel_speed=z["dof_velocity"][...,[3,7]]*.127
        count=contact.sum(-1)
        roll=np.where(count>0,(wheel_speed*contact).sum(-1)/np.maximum(count,1),wheel_speed.mean(-1))
        np.testing.assert_allclose(z["v_roll_proxy"],roll,atol=2e-6)
        np.testing.assert_allclose(z["v_step_proxy"],z["body_velocity"][...,0]-roll,atol=2e-6)
        records[seed]=(meta,z)
        audits.append({"evaluation_seed":seed,"samples":n*30,"actual_policy_commands_verified":True,
            "history_continuous":True,"gait_phase_continuous":True,"capsule_clearance_verified":True,
            "roll_step_proxy_verified":True,"no_load_no_reset":True,"all_finite":True})
    if set(records)!=set(seeds): raise ValueError(f"missing/extra seeds {set(seeds)^set(records)}")
    result={"status":"passed","batches":len(records),"audits":audits}
    write_json(output/("preflight_validation.json" if preflight else "validation.json"),result)
    return records


def aggregate(rows,groups):
    identities={"evaluation_seed","env_id","phase","phase_name","condition","roll_command_mps","step_command_mps"}
    metrics=[k for k in rows[0] if k not in identities]
    buckets=defaultdict(list)
    for row in rows: buckets[tuple(row[g] for g in groups)].append(row)
    result=[]
    for key,items in sorted(buckets.items()):
        row=dict(zip(groups,key)); row["n_assigned"]=len(items)
        for k in metrics:
            values=[i[k] for i in items if i[k] is not None]
            row[k]=mean(values); row["n_"+k]=len(values)
        result.append(row)
    return result,metrics


def summarize(rows,groups,output,prefix):
    seed_rows,keys=aggregate(rows,[*groups,"evaluation_seed"])
    write_csv(output/f"{prefix}_evaluation_seeds.csv",seed_rows)
    result=[]
    for group in sorted(set(tuple(r[g] for g in groups) for r in seed_rows)):
        items=[r for r in seed_rows if tuple(r[g] for g in groups)==group]
        row=dict(zip(groups,group)); row.update(n_assigned=90,n_training_seeds=1,n_evaluation_seeds=3)
        for k in keys:
            values=[r[k] for r in items if r[k] is not None]
            row[k]=mean(values)
            row[k+"_eval_seed_sd"]=float(np.std(values,ddof=1)) if len(values)>1 else None
            row["n_"+k]=sum(r["n_"+k] for r in items)
        result.append(row)
    write_csv(output/f"{prefix}_summary.csv",result)
    return result


def analyze(records,output):
    phases=[]; conditions=[]; events=[]; boundary=[]
    for seed,(meta,z) in sorted(records.items()):
        phase=z["phase_of_preceding_interval"][:,0]; t=z["time_s"]
        for env in range(30):
            event=first_failure(z,env)
            events.append({"evaluation_seed":seed,"env_id":env,**event})
            alive=np.ones(len(t),bool) if event["first_invalid_s"] is None else t<event["first_invalid_s"]-1e-6
            def row_for(mask,identity):
                valid=mask&alive
                row={"evaluation_seed":seed,"env_id":env,**identity,"valid_samples":int(valid.sum()),
                    "valid_sample_equiv_s":float(valid.sum()*DT),"all_samples":int(mask.sum()),
                    "valid_trial":int(valid.any()),"window_completed":int(alive[mask].all())}
                row.update({"valid_"+k:v for k,v in motion_metrics(z,env,valid).items()})
                row.update({"all_"+k:v for k,v in motion_metrics(z,env,mask).items()})
                return row
            for p,name in enumerate(PHASE_NAMES):
                phases.append(row_for(phase==p,{"phase":p,"phase_name":name,
                    "roll_command_mps":float(COMMANDS[p,0]),"step_command_mps":float(COMMANDS[p,1])}))
            for name,mask in (("roll_pooled",(phase==0)|(phase==3)),("mixed",phase==1),("step_dominant",phase==2)):
                conditions.append(row_for(mask,{"condition":name}))
            for time_s in (10,20,30):
                for label,mask in (("pre_2s",(t>time_s-2)&(t<=time_s)),
                                   ("post_2s",(t>time_s)&(t<=time_s+2))):
                    boundary.append(row_for(mask,{"phase":time_s,"phase_name":label}))
    write_csv(output/"per_environment_phases.csv",phases)
    write_csv(output/"per_environment_conditions.csv",conditions)
    write_csv(output/"first_events.csv",events)
    write_csv(output/"command_boundary_windows.csv",boundary)
    summary=summarize(phases,["phase","phase_name"],output,"phase")
    pooled=summarize(conditions,["condition"],output,"condition")
    meta,z=records[MAIN_SEED]
    main={k:(v[:,MAIN_ENV] if v.ndim>=2 and v.shape[0]==len(z["time_s"]) else v) for k,v in z.items()
          if k in ("time_s","root_state","body_velocity","task_roll_step","applied_previous_interval_roll_step",
                   "phase_of_preceding_interval","wheel_forces","wheel_clearance_signed","first_failure",
                   "gait_phase","desired_contact","v_roll_proxy","v_step_proxy","actor_commands","observations")}
    savemat(output/"preregistered_command_response.mat",main,do_compression=True)
    roll,pitch,_=angles(main["root_state"])
    trace_rows=[]
    for i,t in enumerate(main["time_s"]):
        trace_rows.append({"time_s":float(t),"roll_command_mps":float(main["task_roll_step"][i,0]),
            "step_command_mps":float(main["task_roll_step"][i,1]),"actual_body_vx_mps":float(main["body_velocity"][i,0]),
            "left_Fz_N":float(main["wheel_forces"][i,0,2]),"right_Fz_N":float(main["wheel_forces"][i,1,2]),
            "left_clearance_m":float(main["wheel_clearance_signed"][i,0]),
            "right_clearance_m":float(main["wheel_clearance_signed"][i,1]),
            "pitch_deg":float(pitch[i]),"roll_deg":float(roll[i]),
            "original_termination_latched":int(main["first_failure"][i])})
    write_csv(output/"preregistered_timeseries.csv",trace_rows)
    write_json(output/"summary.json",{"phases":summary,"conditions":pooled,"trajectories":90,
        "phase_rows":len(phases),"total_original_terminations":sum(r["original_termination_s"] is not None for r in events),
        "total_geometric_falls":sum(r["fall_start_s"] is not None for r in events),
        "main_case":{"evaluation_seed":MAIN_SEED,"env_id":MAIN_ENV},
        "primary_mask":"before first original termination or sustained geometric-fall onset; all assigned rows retained",
        "speed_definition":"body-frame finite-difference vx; total target 0.5 m/s",
        "swing_definition":"Fz<=1 N contact-defined swing proxy, not guaranteed positive clearance"})
    table(summary,pooled,output)
    return main


def velocity_audit(records,output):
    rows=[]
    for seed,(_,z) in sorted(records.items()):
        for phase in range(4):
            physical=z["phase_of_preceding_interval"][:,0]==phase
            actor=z["phase_for_next_action"][:,0]==phase
            rows.append({"evaluation_seed":seed,"phase":phase,
                "fd_body_vx_mean_mps":float(z["body_velocity"][physical,:,0].mean()),
                "root_body_vx_mean_mps":float(z["body_velocity_from_root"][physical,:,0].mean()),
                "actor_next_phase_roll_min_mps":float(z["actor_commands"][actor,:,0].min()/2),
                "actor_next_phase_roll_max_mps":float(z["actor_commands"][actor,:,0].max()/2),
                "actor_next_phase_step_min_mps":float(z["observations"][actor,:,28].min()),
                "actor_next_phase_step_max_mps":float(z["observations"][actor,:,28].max())})
    write_csv(output/"velocity_definition_check.csv",rows)


def table(summary,pooled,output):
    def formatted(value,scale=1.):
        return "缺失" if value is None else f"{value*scale:.4f}"
    lines=["# 命令响应接触与运动统计","", "单策略、单训练seed；3评测seed各30环境。条件统计保留全部分配行。",
        "主数值使用首次终止/几何跌倒前样本，缺失、有效暴露与失败另列；完整无删样本统计在CSV。","",
        "| 时段s | roll+step m/s | 有效环境/90 | 完整时段/90 | 速度RMSE m/s | 左失接触% | 右失接触% | 双支撑% | pitch SD deg | roll SD deg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        p=r["phase"]
        values=[formatted(r["valid_speed_rmse_mps"]),formatted(r["valid_left_swing_fraction"],100),
                formatted(r["valid_right_swing_fraction"],100),formatted(r["valid_double_support_fraction"],100),
                formatted(r["valid_pitch_sd_deg"]),formatted(r["valid_roll_sd_deg"])]
        lines.append(f"| {p*10}–{(p+1)*10} | {COMMANDS[p,0]:.1f}+{COMMANDS[p,1]:.1f} | "
            f"{round(r['valid_trial']*90)} | {round(r['window_completed']*90)} | "+" | ".join(values)+" |")
    lines += ["", "## 三命令条件合并表", "", "纯滚动将首段/末段在同一环境内合并后重算，不平均两个RMSE，也不视为独立复验。",
        "其总暴露20 s，另外两条件各10 s；前后纯滚动仍以上表分列为准。", "",
        "| 条件 | 左失接触% | 右失接触% | 双支撑% | 左悬空且净空>1cm% | 右悬空且净空>1cm% |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in pooled:
        keys=("valid_left_swing_fraction","valid_right_swing_fraction","valid_double_support_fraction",
              "valid_left_air_clearance_fraction","valid_right_air_clearance_fraction")
        lines.append("| "+r["condition"]+" | "+" | ".join(formatted(r[k],100) for k in keys)+" |")
    (output/"CONTACT_AND_MOTION_TABLE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def render(main,output,font_source=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import shutil
    from legged_gym.scripts.figs_chx.fig0_sixd_design_variables import configure_fonts
    from legged_gym.scripts.terrain_step_analysis import save
    if font_source is not None and (font_source/"CodesignNotoSansSC-Regular.ttf").exists():
        dest=output/"fonts"; dest.mkdir(exist_ok=True)
        for source in font_source.iterdir():
            if source.resolve()!=(dest/source.name).resolve(): shutil.copy2(source,dest/source.name)
        for path in dest.glob("*.ttf"): font_manager.fontManager.addfont(str(path))
    else: configure_fonts(output,extra_text=Path(__file__).read_text())
    plt.rcParams.update({"font.family":"Codesign Noto Sans SC","font.size":14,"axes.labelsize":14,
        "axes.titlesize":14,"xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,
        "axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"path"})
    t=main["time_s"]; contact=main["wheel_forces"][...,2]>1
    roll,pitch,_=angles(main["root_state"])
    event=first_failure({k:v[:,None] if k!="time_s" else v for k,v in main.items()},0)
    valid=np.ones(len(t),bool) if event["first_invalid_s"] is None else t<event["first_invalid_s"]-1e-6
    for language in ("zh","en"):
        zh=language=="zh"
        fig,axs=plt.subplots(5,1,figsize=(13.4,10),sharex=True,
                             gridspec_kw={"height_ratios":[1,1.4,.65,1,1]})
        axs[0].step(t,main["task_roll_step"][:,0],where="post",color="#327C9C",lw=1.4,label="滚动命令" if zh else "Rolling command")
        axs[0].step(t,main["task_roll_step"][:,1],where="post",color="#B18A39",lw=1.4,label="迈步命令" if zh else "Stepping command")
        axs[0].plot(t,main["task_roll_step"].sum(-1),color="#515B63",ls="--",lw=1.,label="总速度目标" if zh else "Total target")
        axs[1].plot(t,np.where(valid,main["body_velocity"][:,0],np.nan),color="#2C8873",lw=.9)
        axs[1].plot(t,np.where(~valid,main["body_velocity"][:,0],np.nan),color="#9AA2A8",lw=.7)
        axs[1].axhline(.5,color="#515B63",ls="--",lw=.9)
        for side,color in ((0,"#477F9D"),(1,"#AE7067")):
            y=1-side
            axs[2].fill_between(t,y,y+.65,where=contact[:,side]&valid,color=color,step="mid")
        axs[2].set_yticks([.325,1.325]); axs[2].set_yticklabels(["右轮","左轮"] if zh else ["R","L"])
        axs[2].set_ylim(-.1,1.8)
        for ax,y,color in ((axs[3],pitch,"#B46B48"),(axs[4],roll,"#786C9D")):
            ax.plot(t,np.where(valid,y,np.nan),color=color,lw=.9)
            ax.plot(t,np.where(~valid,y,np.nan),color="#9AA2A8",lw=.7)
        labels=(["命令 (m/s)","实际前向速度 (m/s)","实际接触","俯仰 (deg)","横滚 (deg)"] if zh else
                ["Command (m/s)","Actual forward speed (m/s)","Measured contact","Pitch (deg)","Roll (deg)"])
        for ax,label in zip(axs,labels):
            ax.set_ylabel(label); ax.grid(axis="y",color="#DFE5E8",lw=.6)
            for change in (10,20,30): ax.axvline(change,color="#747E86",ls=":",lw=1)
            if event["first_invalid_s"] is not None: ax.axvspan(event["first_invalid_s"],40,color="#EBEDEF",alpha=.55,zorder=0)
        axs[-1].set_xlabel("时间 (s)" if zh else "Time (s)")
        axs[-1].set_xlim(0,40); axs[-1].set_xticks(np.arange(0,41,5))
        fig.subplots_adjust(left=.12,right=.98,top=.93,bottom=.07,hspace=.30)
        fig.legend(*axs[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.54,.995))
        save(fig,output/f"unified_command_response_full_{language}")
        for change in (10,20,30):
            for ax in axs:
                ax.set_xlim(change-2,change+2)
                ax.set_xticks(np.arange(change-2,change+2.01,1))
            save(fig,output/f"unified_command_response_change{change}_{language}")
        plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--preflight",action="store_true"); p.add_argument("--no-figures",action="store_true")
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    records=validate(a.input,a.output,a.preflight)
    if not a.preflight:
        main=analyze(records,a.output)
        velocity_audit(records,a.output)
        if not a.no_figures: render(main,a.output,a.input/"analysis/fonts")
    print(f"Validation passed: {len(records)} batch(es); output={a.output}")
