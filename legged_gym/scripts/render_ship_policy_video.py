"""Native Isaac Gym robot/deck checkpoint video, with unedited failure counts.

A single environment is a visual demonstration, not statistical success evidence.
Use ship_motion_probe for the matched multi-environment survival evaluation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('ROBOT_TYPE','WF_TRON1A')
from isaacgym import gymapi,gymutil
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship import BipedWFShip
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_config import BipedCfgWFShip,BipedCfgPPOWFShip
from legged_gym.algorithm.on_policy_runner import OnPolicyRunner
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.saved_config import apply_config_dict
from legged_gym.utils.ship_motion import surface_pose
from legged_gym.scripts.render_ship_deck_video import euler,FONT,ffmpeg_environment


class NativeCameraCapture:
    def create_sim(self):
        # Keep viewer headless, but enable real native offscreen cameras.
        self.graphics_device_id=self.sim_device_id
        super().create_sim()

    def reset_idx(self,env_ids):
        # Capture physical post-step states BEFORE reset overwrites the shared
        # root tensor with next-tick commands. These match the native camera.
        if hasattr(self,'deck_indices'):
            self.camera_robot_root=self.root_states[self.actor_indices].clone()
            self.camera_deck_root=self.root_states[self.deck_indices].clone()
            self.camera_deck_imu=self.deck_imu.clone()
        super().reset_idx(env_ids)


class CameraShip(NativeCameraCapture, BipedWFShip):
    pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--duration',type=float,default=20.)
    p.add_argument('--seed',type=int,default=2026101501)
    p.add_argument('--motion-scale',type=float,default=1.)
    p.add_argument('--roll-deg',type=float,default=20.)
    p.add_argument('--pitch-deg',type=float,default=12.)
    p.add_argument('--heave-m',type=float,default=.15)
    p.add_argument('--fixed-frequency-hz',type=float,nargs=3,metavar=('ROLL','PITCH','HEAVE'))
    p.add_argument('--imu-off',action='store_true')
    a=p.parse_args()
    if a.duration<=0 or a.motion_scale<0 or not a.device.startswith('cuda'):
        p.error('positive duration, nonnegative scale and a CUDA device are required')
    if a.fixed_frequency_hz is not None and any(v<=0 for v in a.fixed_frequency_hz):
        p.error('fixed frequencies must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed);torch.set_num_threads(2)
    cfg,train=BipedCfgWFShip(),BipedCfgPPOWFShip()
    saved_env=json.loads((a.checkpoint.parent/'env_cfg.json').read_text())
    locomotion_mode=saved_env.get('ship_mobility',{}).get('mode','balance')
    env_type=CameraShip
    if saved_env.get('ship_model_prior',{}).get('enabled',False):
        if locomotion_mode!='balance':
            raise ValueError('model prior video is standing balance only')
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_prior import BipedWFShipPrior
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_prior_config import BipedCfgWFShipPrior,BipedCfgPPOWFShipPrior
        class PriorCameraShip(NativeCameraCapture,BipedWFShipPrior):
            pass
        env_type=PriorCameraShip
        cfg,train=BipedCfgWFShipPrior(),BipedCfgPPOWFShipPrior()
    if locomotion_mode!='balance':
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_mobility import BipedWFShipMobility
        from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_mobility_config import (
            BipedCfgWFShipRoll,BipedCfgWFShipStep,BipedCfgPPOWFShipMobility,
        )
        class MobilityCameraShip(NativeCameraCapture,BipedWFShipMobility):
            pass
        env_type=MobilityCameraShip
        cfg=(BipedCfgWFShipRoll if locomotion_mode=='roll' else BipedCfgWFShipStep)()
        train=BipedCfgPPOWFShipMobility()
    apply_config_dict(cfg,saved_env)
    apply_config_dict(train,json.loads((a.checkpoint.parent/'train_cfg.json').read_text()))
    if locomotion_mode!='balance':
        cfg.ship_mobility.speed_curriculum=False
    cfg.seed=a.seed;cfg.env.num_envs=1
    cfg.env.episode_length_s=a.duration+1.
    cfg.ship.curriculum=False;cfg.ship.training_schedule=False
    cfg.ship.corner_fraction=0.  # demonstration is not a training anchor assignment
    cfg.ship.initial_difficulty=1.;cfg.ship.imu_enabled=not a.imu_off
    cfg.ship.amplitude_max=[math.radians(a.roll_deg)*a.motion_scale,
                            math.radians(a.pitch_deg)*a.motion_scale,a.heave_m*a.motion_scale]
    cfg.ship.amplitude_min=list(cfg.ship.amplitude_max)
    if a.fixed_frequency_hz is not None:
        cfg.ship.frequency_min=list(a.fixed_frequency_hz)
        cfg.ship.frequency_max=list(a.fixed_frequency_hz)
    params=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    params.use_gpu_pipeline=True;params.physx.use_gpu=True
    params.physx.num_position_iterations=8;params.physx.num_velocity_iterations=2
    env=None;encoder=None
    result={'checkpoint':str(a.checkpoint),'seed':a.seed,'num_envs':1,
            'native_camera':True,'physics_gpu':True,'fixed_amplitude':True,
            'imu_enabled':cfg.ship.imu_enabled,'balance_success_claim':False,
            'amplitude_rad_m':list(cfg.ship.amplitude_max),
            'frequency_min_hz':list(cfg.ship.frequency_min),
            'frequency_max_hz':list(cfg.ship.frequency_max),
            'reference_mode':getattr(cfg.ship,'reference_mode','body'),
            'locomotion_mode':locomotion_mode,
            'scope':'single environment visual demonstration; not statistical survival evidence'}
    try:
        env=env_type(cfg,params,gymapi.SIM_PHYSX,a.device,True)
        runner=OnPolicyRunner(env,class_to_dict(train),str(a.output/'policy_log'),a.device)
        runner.load(str(a.checkpoint),load_optimizer=False)
        runner.alg.actor_critic.eval();runner.alg.encoder.eval()
        cam_cfg=gymapi.CameraProperties();cam_cfg.width=960;cam_cfg.height=540
        cam_cfg.horizontal_fov=55
        camera=env.gym.create_camera_sensor(env.envs[0],cam_cfg)
        if camera<0:raise RuntimeError('native offscreen camera unavailable')
        origin=env.ship_pivot[0].detach().cpu().tolist()
        env.gym.set_camera_location(camera,env.envs[0],
            gymapi.Vec3(origin[0]+6.3,origin[1]-7.5,origin[2]+4.0),
            gymapi.Vec3(origin[0],origin[1],origin[2]+.5))
        env.gym.set_rigid_body_color(env.envs[0],env.load_handles[0],0,
            gymapi.MESH_VISUAL_AND_COLLISION,gymapi.Vec3(.15,.58,.64))
        fps=25;frame_stride=int(round(1/(fps*env.dt)))
        if abs(frame_stride*env.dt-1/fps)>1e-7:raise RuntimeError('policy timestep does not divide video frame interval')
        video=a.output/'robot_ship_motion.mp4'
        encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','960x640','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2',
            '-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
            stdin=subprocess.PIPE,stderr=subprocess.PIPE,env=ffmpeg_environment())
        font=ImageFont.truetype(FONT,20)
        roots=[];decks=[];sensors=[];observations=[];actions_log=[];dones_log=[];commands_log=[]
        frames=failures=0;max_roll=max_pitch=max_heave=0.
        with torch.inference_mode():
            env.reset()
            for step in range(int(round(a.duration/env.dt))):
                obs,history,commands,_=env.get_observations()
                latent=runner.alg.encode_for_policy(history,obs)
                actions=runner.alg.actor_critic.act_inference(torch.cat((latent,obs,commands),dim=-1))
                observations.append(obs[0].cpu().numpy().copy())
                commands_log.append(env.commands[0,:3].cpu().numpy().copy())
                data=env.step(actions)
                done=data[2].bool();timeout=data[3]['time_outs'].bool()
                failures+=int((done & ~timeout).sum())
                roots.append(env.camera_robot_root[0].cpu().numpy().copy())
                deck=env.camera_deck_root[0]
                decks.append(deck.cpu().numpy());sensors.append(env.camera_deck_imu[0].cpu().numpy().copy())
                actions_log.append(actions[0].cpu().numpy().copy());dones_log.append(bool(done[0]))
                angles=euler(deck[None])[0]*180/math.pi
                top,_=surface_pose(deck[None],cfg.ship.thickness)
                heave=float(top[0,2]-env.ship_pivot[0,2])
                max_roll=max(max_roll,float(angles[0].abs()));max_pitch=max(max_pitch,float(angles[1].abs()))
                max_heave=max(max_heave,abs(heave))
                if step%frame_stride:continue
                env.gym.fetch_results(env.sim,True)
                env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
                rgb=env.gym.get_camera_image(env.sim,env.envs[0],camera,gymapi.IMAGE_COLOR).reshape(540,960,4)[:,:,:3]
                frame=Image.new('RGB',(960,640),(20,25,34));frame.paste(Image.fromarray(rgb.copy()),(0,0))
                draw=ImageDraw.Draw(frame)
                draw.text((18,548),f'真实 GPU 仿真 | {a.checkpoint.name} | {locomotion_mode} | 甲板 IMU {"开" if cfg.ship.imu_enabled else "关"}',font=font,fill='white')
                draw.text((18,578),f't={(step+1)*env.dt:.2f}s  横摇={float(angles[0]):+.1f}°  纵摇={float(angles[1]):+.1f}°  升沉={heave:+.3f}m',font=font,fill='white')
                draw.text((18,608),f'失败 reset：{failures}  | 单环境展示，不代表整体成功率',font=font,fill=(255,205,115))
                encoder.stdin.write(frame.tobytes());frames+=1
                if frames==1 or step>=int(a.duration/env.dt)//2 and not (a.output/'preview.png').exists():
                    frame.save(a.output/('preview_start.png' if frames==1 else 'preview.png'))
        encoder.stdin.close();error=encoder.stderr.read().decode(errors='replace');code=encoder.wait(timeout=30)
        if code:raise RuntimeError('ffmpeg failed: '+error[-1200:])
        np.savez_compressed(a.output/'actual_states.npz',robot=np.asarray(roots),deck=np.asarray(decks),
                            deck_imu=np.asarray(sensors),obs_before_action=np.asarray(observations),
                            actions=np.asarray(actions_log),dones=np.asarray(dones_log),
                            commands_before_action=np.asarray(commands_log),dt=env.dt)
        result.update(passed=True,video=str(video),frames=frames,fps=fps,failure_resets=failures,
            max_actual_roll_deg=max_roll,max_actual_pitch_deg=max_pitch,max_actual_heave_m=max_heave,
            max_deck_position_error_m=float(env.ship_pose_error_max.max()))
        result['state_sampling']='physical post-step root/IMU before reset; ordinary observations before action'
        (a.output/'env_cfg.json').write_text(json.dumps(class_to_dict(cfg),indent=2))
    except Exception as error:
        result.update(passed=False,error=repr(error));raise
    finally:
        if encoder is not None and encoder.poll() is None:
            if not encoder.stdin.closed:encoder.stdin.close()
            try:encoder.wait(timeout=30)
            except subprocess.TimeoutExpired:encoder.terminate();encoder.wait(timeout=10)
        (a.output/'metadata.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
