#!/usr/bin/env python3
"""Coulomb-contact probes: one finite force pulse or a static inclined plate."""

import argparse
import json
from pathlib import Path
import time

from .geometry import write_json


PAIRS=((.5,.5),(.1,.1),(.05,.05),(.02,.02),(.005,.005),(0.,0.),(1.,0.),(0.,1.))


def run(args):
    import isaacgym  # noqa: F401
    from isaacgym import gymapi,gymtorch
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.cuda.set_device(0)
    args.output.mkdir(parents=True,exist_ok=True)
    gym=gymapi.acquire_gym(); p=gymapi.SimParams()
    p.dt=.005; p.substeps=1; p.up_axis=gymapi.UP_AXIS_Z; p.gravity=gymapi.Vec3(0,0,-9.81)
    p.use_gpu_pipeline=p.physx.use_gpu=True; p.physx.num_threads=2
    p.physx.num_position_iterations=8; p.physx.num_velocity_iterations=1
    p.physx.contact_collection=gymapi.CC_ALL_SUBSTEPS
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,p)
    if sim is None: raise RuntimeError("GPU simulation creation failed")
    rows=[]; handles=[]; start=time.monotonic()
    try:
        ground=gymapi.PlaneParams(); ground.normal=gymapi.Vec3(0,0,1)
        ground.static_friction=ground.dynamic_friction=.8; ground.restitution=0
        gym.add_ground(sim,ground)
        for mechanism in ("pulse","incline","no_drive"):
            pairs=PAIRS if mechanism!="no_drive" else ((0.,0.),)
            for mp,mb in pairs:
                for repeat in range(3):
                    env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,2),8)
                    origin=gym.get_env_origin(env)
                    plate_opt=gymapi.AssetOptions(); plate_opt.fix_base_link=True
                    plate=gym.create_box(sim,.5,.5,.005,plate_opt)
                    cube_opt=gymapi.AssetOptions(); cube_opt.density=args.mass/.001
                    cube_opt.linear_damping=cube_opt.angular_damping=0.
                    cube=gym.create_box(sim,.1,.1,.1,cube_opt)
                    for asset,friction in ((plate,mp),(cube,mb)):
                        props=gym.get_asset_rigid_shape_properties(asset)
                        for v in props:
                            v.friction=friction; v.restitution=v.rolling_friction=v.torsion_friction=0.
                            v.contact_offset=.002; v.rest_offset=0.
                        gym.set_asset_rigid_shape_properties(asset,props)
                    theta=np.deg2rad(10. if mechanism=="incline" else 0.)
                    pose=gymapi.Transform(); pose.p=gymapi.Vec3(0,0,.8)
                    pose.r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0),theta)
                    plate_h=gym.create_actor(env,plate,pose,"plate",len(rows),0)
                    local=np.array([-.12 if mechanism=="incline" else 0.,(repeat-1)*.04,.0535])
                    rotation=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
                    cpose=gymapi.Transform(); offset=rotation@local
                    cpose.p=gymapi.Vec3(offset[0],offset[1],.8+offset[2]); cpose.r=pose.r
                    box_h=gym.create_actor(env,cube,cpose,"free_box",len(rows),0)
                    box_id=gym.get_actor_index(env,box_h,gymapi.DOMAIN_SIM)
                    body_id=gym.get_actor_rigid_body_index(env,box_h,0,gymapi.DOMAIN_SIM)
                    actual=gym.get_actor_rigid_body_properties(env,box_h)[0].mass
                    props=[gym.get_actor_rigid_shape_properties(env,h)[0] for h in (plate_h,box_h)]
                    rows.append({"case_id":len(rows),"mechanism":mechanism,"plate_friction":mp,"box_friction":mb,
                        "repeat":repeat,"incline_deg":float(np.rad2deg(theta)),"mass_kg":actual,
                        "actual_frictions":[v.friction for v in props],"env_layout_origin_m":[origin.x,origin.y,origin.z],
                        "plate_quaternion_xyzw":[pose.r.x,pose.r.y,pose.r.z,pose.r.w],
                        "pulse_start_s":1.,"pulse_duration_s":.04,"pulse_force_N":args.mass*.25/.04 if mechanism=="pulse" else 0.,
                        "box_actor_index":box_id,"box_body_index":body_id})
                    handles.append((env,plate_h,box_h))
        gym.prepare_sim(sim)
        roots=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        contact=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
        count=gym.get_sim_rigid_body_count(sim)
        forces=torch.zeros((count,3),device="cuda:0")
        ids=torch.tensor([r["box_actor_index"] for r in rows],device="cuda:0",dtype=torch.long)
        plate_ids=torch.tensor([gym.get_actor_index(e,h,gymapi.DOMAIN_SIM) for e,h,_ in handles],device="cuda:0")
        body=torch.tensor([r["box_body_index"] for r in rows],device="cuda:0",dtype=torch.long)
        pulse=torch.tensor([r["pulse_force_N"] for r in rows],device="cuda:0")
        records={"root":[],"plate_root":[],"contact_force":[],"time_s":[],"applied_force":[]}
        for step in range(int(round(args.duration/p.dt))):
            t=step*p.dt
            forces.zero_()
            if 200<=step<208: forces[body,0]=pulse
            gym.apply_rigid_body_force_tensors(sim,gymtorch.unwrap_tensor(forces),None,gymapi.ENV_SPACE)
            gym.simulate(sim); gym.fetch_results(sim,True)
            gym.refresh_actor_root_state_tensor(sim); gym.refresh_net_contact_force_tensor(sim)
            records["root"].append(roots[ids].cpu().numpy().copy())
            records["plate_root"].append(roots[plate_ids].cpu().numpy().copy())
            records["contact_force"].append(contact[body].cpu().numpy().copy())
            records["time_s"].append((step+1)*p.dt)
            records["applied_force"].append(forces[body].cpu().numpy().copy())
        np.savez_compressed(args.output/"trace.npz",**{k:np.asarray(v) for k,v in records.items()})
        write_json(args.output/"cases.json",rows)
        metadata={"cases":len(rows),"dt_s":p.dt,"duration_s":args.duration,"elapsed_s":time.monotonic()-start,
            "gpu":torch.cuda.get_device_name(0),"solver_position_iterations":8,"solver_velocity_iterations":1,
            "force_is_zero_after_pulse":True,"no_post_creation_pose_writes":True,"box_free_not_fixed":True}
    finally: gym.destroy_sim(sim)
    metadata["cleanup_completed"]=True
    write_json(args.output/"result.json",metadata)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--mass",type=float,default=5.93)
    p.add_argument("--duration",type=float,default=4.)
    run(p.parse_args())
