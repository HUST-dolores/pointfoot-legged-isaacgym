#!/usr/bin/env python3
"""Record a frozen unified actor under a four-part deterministic command schedule."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from legged_gym.scripts.roll_step_response_protocol import (
    DT, RUN, CHECKPOINT, SEEDS, make_scene, array_digest, sha256_file, write_json,
)
from legged_gym.scripts.payload_step_experiment import configure, environment_type, timestamp


ROOT=Path(__file__).resolve().parents[2]


def prepare(output):
    import numpy as np
    folder=output/"scenarios"; folder.mkdir(parents=True,exist_ok=True)
    result={}
    for seed in SEEDS:
        scene=make_scene(seed)
        path=folder/f"seed_{seed}.npz"
        if path.exists():
            assert array_digest(dict(np.load(path,allow_pickle=False)))==array_digest(scene)
        else:
            np.savez_compressed(path,**scene)
        result[str(seed)]={"sha256":sha256_file(path),"array_sha256":array_digest(scene)}
    write_json(folder/"manifest.json",{"seeds":result,"num_envs":30,"steps":2000,"dt_s":DT})


def run(a):
    import isaacgym  # noqa: F401
    from isaacgym import gymapi,gymutil
    from isaacgym.torch_utils import quat_rotate,quat_rotate_inverse
    import numpy as np
    import torch
    from legged_gym.envs.wheelfoot_flat.wheelfoot_flat_config import BipedCfgWF
    from legged_gym.algorithm.actor_critic import ActorCritic
    from legged_gym.algorithm.mlp_encoder import MLP_Encoder
    from legged_gym.utils.helpers import class_to_dict,set_seed
    from legged_gym.scripts.terrain_step_protocol import wheel_geometry

    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
    set_seed(a.seed)
    source=ROOT/"logs/wheelfoot_flat/WF_TRON1A"/RUN
    saved=json.loads((source/"env_cfg.json").read_text())
    train=json.loads((source/"train_cfg.json").read_text())
    assert saved["env"]["num_observations"]==31 and train["MLP_Encoder"]["num_input_dim"]==540
    assert not train["algorithm"]["use_load_residual_estimation"]
    cfg=configure(BipedCfgWF(),saved,SimpleNamespace(num_envs=30,condition="empty"))
    cfg.rewards.scales=SimpleNamespace(**saved["rewards"]["scales"])
    batch=f"unified_eval{a.seed}"
    folder=a.output/a.phase/batch
    if (folder/"result.json").exists(): raise FileExistsError(folder)
    folder.mkdir(parents=True,exist_ok=True)
    scene_path=a.output/"scenarios"/f"seed_{a.seed}.npz"
    scene=dict(np.load(scene_path,allow_pickle=False))
    spec=importlib.util.spec_from_file_location("legged_gym.envs.wheelfoot_flat._response_archived",source/"wheelfoot_flat.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    Matched=environment_type(module.BipedWF)

    class Scheduled(Matched):
        def _resample_commands(self,env_ids):
            phases=(self.episode_length_buf[env_ids]//500).clamp(0,3)
            table=torch.tensor([[.5,0.],[.3,.2],[.1,.4],[.5,0.]],device=self.device)
            command=table[phases]
            self.commands[env_ids]=0.
            self.commands[env_ids,0]=command[:,0]
            self.cmd_step[env_ids,0]=command[:,1]

        def _post_physics_step_callback(self):
            self._resample_commands(torch.arange(self.num_envs,device=self.device))
            super()._post_physics_step_callback()

    sim=gymapi.SimParams(); gymutil.parse_sim_config(class_to_dict(cfg.sim),sim)
    sim.use_gpu_pipeline=sim.physx.use_gpu=True; sim.physx.num_threads=4
    started=timestamp(); before=time.monotonic()
    env=Scheduled(scene,cfg=cfg,sim_params=sim,physics_engine=gymapi.SIM_PHYSX,sim_device="cuda:0",headless=True)
    try:
        encoder=MLP_Encoder(**train["MLP_Encoder"]).to(env.device).eval()
        actor=ActorCritic(41,48,8,**train["policy"]).to(env.device).eval()
        model=source/f"model_{CHECKPOINT}.pt"
        state=torch.load(model,map_location=env.device,weights_only=False)
        encoder.load_state_dict(state["encoder_state_dict"],strict=True)
        actor.load_state_dict(state["model_state_dict"],strict=True)
        encoder.requires_grad_(False); actor.requires_grad_(False)
        geom=wheel_geometry(ROOT/"resources/robots/WF_TRON1A/urdf/robot.urdf")
        offsets,radii,widths=[torch.as_tensor(v,device=env.device) for v in geom]
        records={}
        def add(k,v): records.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
        with torch.inference_mode():
            env.reset(); env.recording=True; env.first_failure[:]=False
            initial_history=env.encoder_obs_history.cpu().numpy().copy()
            previous=torch.tensor([.5,0.],device=env.device).expand(30,-1).clone()
            for index in range(a.steps):
                obs,history,command,critic=env.get_observations()
                estimate=encoder.encode(history)
                policy_input=torch.cat((estimate,obs,command),-1)
                action=actor.act_inference(policy_input)
                step=env.episode_length_buf
                phase=(step//500).clamp(0,3)
                expected=torch.tensor([[.5,0.],[.3,.2],[.1,.4],[.5,0.]],device=env.device)[phase]
                if not (torch.allclose(command[:,0]/env.commands_scale[0],expected[:,0])
                        and torch.allclose(obs[:,28],expected[:,1])
                        and torch.allclose(history.view(30,10,54)[:,-1,28],expected[:,1])):
                    raise ValueError("actual actor or recurrent command does not match schedule")
                if not bool(torch.isfinite(action).all()): raise ValueError("nonfinite policy output")
                feet=env.rigid_body_state[:,env.feet_indices]
                quat=feet[...,3:7].reshape(-1,4)
                center=feet[...,:3]+quat_rotate(quat,offsets[None].expand(30,-1,-1).reshape(-1,3)).reshape(30,2,3)
                axis=torch.zeros_like(center); axis[...,1]=1.
                axis=quat_rotate(quat,axis.reshape(-1,3)).reshape_as(center)
                assert cfg.asset.replace_cylinder_with_capsule
                clearance=center[...,2]-radii[None]-.5*widths[None]*axis[...,2].abs()
                root=env.root_states[env.actor_indices]
                current=torch.stack((env.commands[:,0],env.cmd_step[:,0]),-1)
                for k,v in {"time_s":step[0].float()*env.dt,"root_state":root,
                    "body_velocity":env.base_lin_vel,
                    "body_velocity_from_root":quat_rotate_inverse(root[:,3:7],root[:,7:10]),
                    "wheel_forces":env.contact_forces[:,env.feet_indices],"wheel_clearance_signed":clearance,
                    "foot_state":feet,"actions":action,"policy_input":policy_input,"observations":obs,
                    "actor_commands":command,"task_roll_step":current,
                    "applied_previous_interval_roll_step":previous,
                    "phase_for_next_action":phase,"phase_of_preceding_interval":((step-1)//500).clamp(0,3),
                    "history_step_commands":history.view(30,10,54)[:,:,28],
                    "v_roll_proxy":env.v_roll,"v_step_proxy":env.v_step,
                    "desired_contact":env.desired_contact_states,"gait_phase":env.gait_indices,
                    "first_failure":env.first_failure,"failure_reason":env.failure_reason,
                    "fail_counter":env.fail_buf,"has_load":env.has_load,
                    "dof_position":env.dof_pos,"dof_velocity":env.dof_vel,"torque":env.torques,
                    "noise_injected":env.last_noise_injected[:,:20],
                    "noise_effective":env.last_noise_effective[:,:20]}.items(): add(k,v)
                if index+1<a.steps:
                    previous=current.clone()
                    env.step(action)
                if (index+1)%500==0:
                    print(f"[EVAL] {batch} {index+1}/{a.steps} wall={time.monotonic()-before:.1f}s",flush=True)
        arrays={k:np.stack(v) for k,v in records.items()}
        arrays["initial_encoder_history"]=initial_history
        for k,v in (("base_mass0",env.base_mass0),("base_com0",env.base_com0),
            ("imu_offset_quaternion",env.random_imu_offset),("first_failure_time_s",env.failure_time)):
            arrays[k]=v.cpu().numpy().copy()
        np.savez_compressed(folder/"trace.npz",**arrays)
        props=[]
        for i,(handle,robot) in enumerate(zip(env.envs,env.actor_handles)):
            bodies=env.gym.get_actor_rigid_body_properties(handle,robot)
            shapes=env.gym.get_actor_rigid_shape_properties(handle,robot)
            props.append({"env_id":i,"mass_kg":[v.mass for v in bodies],
                "inertia_diagonal":[[v.inertia.x.x,v.inertia.y.y,v.inertia.z.z] for v in bodies],
                "shape_friction":[v.friction for v in shapes],"shape_restitution":[v.restitution for v in shapes]})
        write_json(folder/"actual_physics.json",props)
        write_json(folder/"effective_env_cfg.json",class_to_dict(cfg))
        result={"batch_id":batch,"evaluation_seed":a.seed,"phase":a.phase,"num_envs":30,
            "steps":a.steps,"dt_s":env.dt,"started":started,"finished":timestamp(),
            "elapsed_s":time.monotonic()-before,"run_name":RUN,"checkpoint":CHECKPOINT,"training_seed":train["seed"],
            "checkpoint_sha256":sha256_file(model),"archived_task_sha256":sha256_file(source/"wheelfoot_flat.py"),
            "scene_array_sha256":array_digest(scene),"trace_sha256":sha256_file(folder/"trace.npz"),
            "visible_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),"gpu_name":torch.cuda.get_device_name(0),
            "obs_dim":31,"history_dim":540,"policy_input_dim":41,"no_reset_after_failure":True,
            "wheel_radius_m":geom[1].tolist(),"wheel_width_m":geom[2].tolist(),
            "wheel_collision_offset_m":geom[0].tolist(),"foot_names":[env.body_names[i] for i in env.feet_indices.cpu().tolist()],
            "gait_not_reset_at_command_changes":True,"first_sample_after_zero_action_s":DT}
    finally:
        env.gym.destroy_sim(env.sim)
    result.update(cleanup_completed=True,exit_status="completed")
    write_json(folder/"result.json",result)
    print(f"[DONE] {batch} wall={result['elapsed_s']:.2f}s",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prepare",action="store_true")
    p.add_argument("--phase",choices=("preflight","batches"),default="batches")
    p.add_argument("--seed",type=int,choices=SEEDS,default=12345)
    p.add_argument("--steps",type=int,default=2000)
    a=p.parse_args(); a.output=a.output.resolve()
    if a.prepare: prepare(a.output)
    else:
        if a.phase=="batches" and a.steps!=2000: p.error("formal trials must have 2000 samples")
        run(a)
