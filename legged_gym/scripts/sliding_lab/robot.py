#!/usr/bin/env python3
"""Free-box sliding on a robot under a frozen standing policy, not retention training."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET
from datetime import datetime

from .geometry import make_variant,urdf_audit,write_json,sha256


ROOT=Path(__file__).resolve().parents[3]
RUN="Jun04_23-18-34_wide2-30_model_guided_seed1"
PRESETS={"original":(None,.5,True),"box_only_low":(None,.02,True),
    "mu_0p1":(.1,.1,True),"mu_0p05":(.05,.05,True),"mu_0p02":(.02,.02,True),
    "mu_0p005":(.005,.005,True),"mu_zero":(0.,0.,True),"no_pulse":(.02,.02,False)}
DIRECTIONS={"x":(1.,0.,0.),"minus_x":(-1.,0.,0.),"y":(0.,1.,0.)}


def variant(a):
    nominal=ROOT/"resources/robots/WF_TRON1A/urdf/robot.urdf"
    dest=nominal.with_name(f"sliding_lab_{a.geometry}_collision.urdf")
    make_variant(nominal,dest,.06 if a.geometry=="raised" else .02)
    tree=ET.parse(dest)
    palette={"base":(.35,.58,.78),"plate":(.22,.57,.74),"abad":(.78,.48,.56),
             "hip":(.88,.69,.30),"knee":(.29,.68,.56),"wheel":(.36,.35,.42)}
    for link in tree.getroot().findall("link"):
        collisions=link.findall("collision")
        if not collisions: continue
        for old in link.findall("visual"): link.remove(old)
        for c in collisions:
            v=ET.SubElement(link,"visual")
            for part in c: v.append(copy.deepcopy(part))
            color=next((color for name,color in palette.items() if name in link.get("name")),(.5,.6,.7))
            material=ET.SubElement(v,"material",name="slide_"+link.get("name"))
            ET.SubElement(material,"color",rgba=" ".join(map(str,(*color,1.))))
    tree.write(dest,encoding="utf-8",xml_declaration=True)
    return dest


def run(a):
    import isaacgym  # noqa: F401
    from isaacgym import gymapi,gymtorch,gymutil
    from isaacgym.torch_utils import quat_rotate
    import numpy as np
    import torch
    from legged_gym.algorithm.actor_critic import ActorCritic
    from legged_gym.algorithm.mlp_encoder import MLP_Encoder
    from legged_gym.envs.wheelfoot_flat.wheelfoot_flat_config import BipedCfgWF
    from legged_gym.utils.helpers import class_to_dict,set_seed
    from legged_gym.scripts.payload_step_experiment import configure,environment_type
    from legged_gym.scripts.payload_step_protocol import make_scene

    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.cuda.set_device(0); set_seed(12345)
    if (a.output/"result.json").exists(): raise FileExistsError(f"preserve completed experiment: {a.output}")
    a.output.mkdir(parents=True,exist_ok=True)
    urdf=variant(a)
    definitions=[]
    for name in (PRESETS if a.sweep else (a.preset,)):
        for direction in (DIRECTIONS if a.sweep else (a.direction,)):
            mp,mb,pulse=PRESETS[name]
            definitions.append({"case_id":len(definitions),"preset":name,"direction":direction,
                "plate_friction":mp,"box_friction":mb,"pulse_enabled":pulse,
                "excitation":a.excitation,"geometry":a.geometry})
    n=len(definitions)
    source=ROOT/"logs/wheelfoot_flat/WF_TRON1A"/RUN
    saved=json.loads((source/"env_cfg.json").read_text()); train=json.loads((source/"train_cfg.json").read_text())
    cfg=configure(BipedCfgWF(),saved,SimpleNamespace(num_envs=n,condition="free_box"))
    cfg.env.use_gait_stepping=cfg.env.use_gait_phase=False
    cfg.rewards.scales=SimpleNamespace(**saved["rewards"]["scales"])
    cfg.asset.file=str(urdf)
    cfg.domain_rand.add_load_range=[a.mass,a.mass]; cfg.domain_rand.randomize_inertia=False
    cfg.noise.add_noise=False
    cfg.sim.physx.num_position_iterations=8; cfg.sim.physx.num_velocity_iterations=1
    scene=make_scene(12345,num_envs=n,steps=int(a.duration/.02)+10)
    scene["body_mass_inertia_scale"][:]=1.; scene["initial_joint_offset"][:]=0.
    scene["initial_root_velocity"][:]=0.; scene["imu_pitch_roll_rad"][:]=0.; scene["noise_uniform"][:]=0.
    plate_z=.06 if a.geometry=="raised" else .02
    scene["box_offset_body"][:]=[0.,0.,plate_z+.0025+.05+.003]
    spec=importlib.util.spec_from_file_location("legged_gym.envs.wheelfoot_flat._slide_archived",source/"wheelfoot_flat.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    Matched=environment_type(module.BipedWF)

    class SlidingRobot(Matched):
        def create_sim(self):
            if a.capture or a.viewer: self.graphics_device_id=0
            return super().create_sim()

        def _create_envs(self):
            super()._create_envs()
            self.material_audit=[]
            for i,(handle,actor,box) in enumerate(zip(self.envs,self.actor_handles,self.load_handles)):
                ranges=self.gym.get_actor_rigid_body_shape_indices(handle,actor)
                base_idx=self.body_names.index("base_Link")
                base_shapes=list(range(ranges[base_idx].start,ranges[base_idx].start+ranges[base_idx].count))
                before=self.gym.get_actor_rigid_shape_properties(handle,actor)
                old=[float(v.friction) for v in before]
                mu=definitions[i]["plate_friction"]
                for s in base_shapes:
                    if mu is not None: before[s].friction=mu
                    before[s].restitution=0.
                    before[s].rolling_friction=before[s].torsion_friction=0.
                    before[s].contact_offset=.002; before[s].rest_offset=0.
                self.gym.set_actor_rigid_shape_properties(handle,actor,before)
                props=self.gym.get_actor_rigid_shape_properties(handle,box)
                for v in props:
                    v.friction=definitions[i]["box_friction"]; v.restitution=v.rolling_friction=v.torsion_friction=0.
                    v.contact_offset=.002; v.rest_offset=0.
                self.gym.set_actor_rigid_shape_properties(handle,box,props)
                after=self.gym.get_actor_rigid_shape_properties(handle,actor)
                for s in range(len(after)):
                    if s not in base_shapes: assert after[s].friction==old[s]
                box_props=self.gym.get_actor_rigid_shape_properties(handle,box)
                self.material_audit.append({**definitions[i],"base_shape_indices":base_shapes,
                    "robot_friction_before":old,"robot_friction_after":[float(v.friction) for v in after],
                    "box_friction_actual":[float(v.friction) for v in box_props],
                    "support_contact_offset_m":[float(after[s].contact_offset) for s in base_shapes],
                    "box_contact_offset_m":[float(v.contact_offset) for v in box_props],
                    "wheel_and_leg_shapes_unchanged":True,
                    "box_mass_actual_kg":self.gym.get_actor_rigid_body_properties(handle,box)[0].mass})

        def _resample_commands(self,ids):
            self.commands[ids]=0.
            if a.excitation=="drive":
                times=self.episode_length_buf[ids]*self.dt
                self.commands[ids,0]=((times>=2.)&(times<3.)).float()*a.drive_speed

        def _post_physics_step_callback(self):
            self._resample_commands(torch.arange(self.num_envs,device=self.device))
            super()._post_physics_step_callback()

        def _maybe_spawn_loads(self):
            times=self.episode_length_buf*self.dt
            ids=((~self.has_load)&(times>=.5)).nonzero(as_tuple=False).flatten()
            if not ids.numel(): return
            root=self.root_states[self.actor_indices[ids]]
            offset=quat_rotate(root[:,3:7],self.scene["box_offset_body"][ids])
            state=self.root_states[self.load_indices[ids]]
            state[:,:3]=root[:,:3]+offset; state[:,3:7]=root[:,3:7]
            state[:,7:10]=root[:,7:10]+torch.cross(root[:,10:13],offset,dim=-1)
            state[:,10:13]=root[:,10:13]
            self.root_states[self.load_indices[ids]]=state
            actor_ids=self.load_indices[ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(actor_ids),len(ids))
            self.has_load[ids]=True; self.loads_spawned_count[ids]+=1
            self.load_remove_time[ids]=self.next_load_spawn_time[ids]=1e6
            self.load_contact_grace_counter[ids]=self.load_contact_grace_steps

        def step(self,actions):
            forces=torch.zeros_like(self.contact_forces)
            if a.excitation=="pulse":
                t=float(self.episode_length_buf[0])*self.dt
                if 2.-1e-6<=t<2.04-1e-6:
                    local=torch.tensor([DIRECTIONS[r["direction"]] for r in definitions],device=self.device)
                    pulse=torch.tensor([r["pulse_enabled"] for r in definitions],device=self.device)
                    force=quat_rotate(self.base_quat,local)*(a.mass*a.impulse_dv/.04)*pulse[:,None]
                    forces[:,self.num_bodies]=force
            self.force_request=forces[:,self.num_bodies].clone()
            self.active_external_forces=forces
            return super().step(actions)

        def _compute_torques(self,actions):
            # Called once per physics substep: the finite force pulse is not a kinematic trajectory.
            if hasattr(self,"active_external_forces"):
                self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(self.active_external_forces.view(-1,3)),
                                                       None,gymapi.ENV_SPACE)
            return super()._compute_torques(actions)

    sim=gymapi.SimParams(); gymutil.parse_sim_config(class_to_dict(cfg.sim),sim)
    sim.use_gpu_pipeline=sim.physx.use_gpu=True; sim.physx.num_threads=2
    start=time.monotonic()
    env=SlidingRobot(scene,cfg=cfg,sim_params=sim,physics_engine=gymapi.SIM_PHYSX,
                     sim_device="cuda:0",headless=not a.viewer)
    camera=None; camera_frames=[]; records={}
    def add(k,v): records.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
    try:
        encoder=MLP_Encoder(**train["MLP_Encoder"]).to(env.device).eval()
        actor=ActorCritic(env.num_obs+10,env.num_critic_obs+10,8,**train["policy"]).to(env.device).eval()
        model=source/"model_11000.pt"
        state=torch.load(model,map_location=env.device,weights_only=False)
        actor.load_state_dict(state["model_state_dict"],strict=True)
        encoder.load_state_dict(state["encoder_state_dict"],strict=True)
        if a.capture:
            props=gymapi.CameraProperties(); props.width=960; props.height=720
            props.horizontal_fov=50; props.use_collision_geometry=True
            camera=env.gym.create_camera_sensor(env.envs[0],props)
            if camera<0: raise RuntimeError("camera creation failed")
            (a.output/"frames").mkdir(exist_ok=True)
        for handle,box in zip(env.envs,env.load_handles):
            env.gym.set_rigid_body_color(handle,box,0,gymapi.MESH_VISUAL_AND_COLLISION,gymapi.Vec3(.95,.43,.12))
        palette={"base":(.35,.58,.78),"abad":(.78,.48,.56),"hip":(.88,.69,.30),
                 "knee":(.29,.68,.56),"wheel":(.35,.34,.41)}
        for handle,robot in zip(env.envs,env.actor_handles):
            for body,name in enumerate(env.body_names):
                color=next((color for prefix,color in palette.items() if prefix in name),(.5,.6,.7))
                env.gym.set_rigid_body_color(handle,robot,body,gymapi.MESH_VISUAL_AND_COLLISION,gymapi.Vec3(*color))
        with torch.inference_mode():
            env.reset(); env.recording=True; env.first_failure[:]=False
            playback_start=time.monotonic()
            for index in range(int(round(a.duration/.02))):
                obs,history,commands,critic=env.get_observations()
                latent=encoder.encode(history); latent[:,3:7]+=obs[:,36:40]
                action=actor.act_inference(torch.cat((latent,obs,commands),-1))
                if not bool(torch.isfinite(action).all()): raise ValueError("nonfinite actor output")
                for k,v in {"time_s":env.episode_length_buf[0].float()*env.dt,
                    "root":env.root_states[env.actor_indices],"box":env.root_states[env.load_indices],
                    "box_contact_force":env.contact_forces[:,env.num_bodies],
                    "force_previous_interval":env.force_request,"first_failure":env.first_failure,
                    "has_load":env.has_load,"wheel_forces":env.contact_forces[:,env.feet_indices],
                    "commands":commands,"actions":action,"observations":obs,
                    "gait_phase":env.gait_indices}.items(): add(k,v)
                now=float(env.episode_length_buf[0])*env.dt
                if camera is not None and index%5==0:
                    from PIL import Image
                    root=env.root_states[env.actor_indices[0],:3].cpu().numpy()
                    target=root+np.array([0.,0.,-.25])
                    position=target+np.array([1.1,-1.5,1.05])
                    env.gym.set_camera_location(camera,env.envs[0],gymapi.Vec3(*position),gymapi.Vec3(*target))
                    env.gym.fetch_results(env.sim,True); env.gym.step_graphics(env.sim); env.gym.render_all_camera_sensors(env.sim)
                    rgb=np.asarray(env.gym.get_camera_image(env.sim,env.envs[0],camera,gymapi.IMAGE_COLOR)).reshape(720,960,4)[...,:3]
                    if rgb.std()<3: raise RuntimeError("blank camera frame")
                    name=f"t_{now:.2f}.png"; Image.fromarray(rgb).save(a.output/"frames"/name)
                    camera_frames.append({"time_s":now,"file":name})
                if a.viewer:
                    plate=env.root_states[env.actor_indices[0],:3].cpu().numpy()
                    target=plate+np.array([0.,0.,-.25])
                    env.gym.viewer_camera_look_at(env.viewer,env.envs[0],gymapi.Vec3(*(target+np.array([1.1,-1.5,1.05]))),gymapi.Vec3(*target))
                    if env.gym.query_viewer_has_closed(env.viewer): break
                if index+1<int(round(a.duration/.02)): env.step(action)
                if a.viewer and env.enable_viewer_sync:
                    delay=(index+1)*env.dt-(time.monotonic()-playback_start)
                    if delay>0: time.sleep(delay)
        np.savez_compressed(a.output/"trace.npz",**{k:np.stack(v) for k,v in records.items()})
        write_json(a.output/"materials.json",env.material_audit)
        write_json(a.output/"effective_cfg.json",class_to_dict(cfg))
        metadata={"cases":definitions,"mass_kg":a.mass,"geometry":urdf_audit(urdf),"urdf":urdf.name,
            "urdf_sha256":sha256(urdf),"nominal_sha256":sha256(ROOT/"resources/robots/WF_TRON1A/urdf/robot.urdf"),
            "policy_run":RUN,"checkpoint":11000,"checkpoint_sha256":sha256(model),
            "dt_s":env.dt,"duration_s":a.duration,"elapsed_s":time.monotonic()-start,
            "excitation":a.excitation,"impulse_dv_nominal_mps":a.impulse_dv,
            "pulse_interval_s":[2.,2.04],"drive_command_mps":a.drive_speed,
            "camera_frames":camera_frames,"no_box_state_overwrites_after_spawn":True,
            "not_a_retention_controller":True,"visible_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "runner_source_sha256":sha256(Path(__file__)),"robot_base_fixed":cfg.asset.fix_base_link,
            "materials_updated_before_prepare_sim":True,"common_support_and_box_contact_offset_m":.002}
    finally:
        if camera is not None: env.gym.destroy_camera_sensor(env.sim,env.envs[0],camera)
        if env.viewer is not None: env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
    metadata["cleanup_completed"]=True; write_json(a.output/"result.json",metadata)
    print(json.dumps({k:metadata[k] for k in ("elapsed_s","excitation","cleanup_completed")}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path); p.add_argument("--geometry",choices=("original","raised"),default="raised")
    p.add_argument("--sweep",action="store_true"); p.add_argument("--preset",choices=PRESETS,default="mu_0p05")
    p.add_argument("--direction",choices=DIRECTIONS,default="y")
    p.add_argument("--excitation",choices=("pulse","drive","none"),default="pulse")
    p.add_argument("--mass",type=float,default=5.93); p.add_argument("--impulse-dv",type=float,default=.25)
    p.add_argument("--drive-speed",type=float,default=.6); p.add_argument("--duration",type=float,default=6.)
    p.add_argument("--capture",action="store_true"); p.add_argument("--viewer",action="store_true")
    p.add_argument("--clean-exit",action="store_true")
    a=p.parse_args()
    if a.mass<=0 or a.duration<=.5: p.error("positive mass and duration>.5 required")
    if a.output is None: a.output=ROOT/"logs/sliding_lab_play"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run(a)
    if a.clean_exit and (a.capture or a.viewer):
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
