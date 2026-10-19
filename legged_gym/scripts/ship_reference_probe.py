"""Check that a reward-reference choice does not alter frozen-policy physics."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('ROBOT_TYPE','WF_TRON1A')
from isaacgym import gymapi,gymutil
import torch
import numpy as np
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship import BipedWFShip
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_config import BipedCfgWFShip,BipedCfgPPOWFShip
from legged_gym.algorithm.on_policy_runner import OnPolicyRunner
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.saved_config import apply_config_dict


class PhysicalSnapshotShip(BipedWFShip):
    def reset_idx(self,ids):
        if hasattr(self,'deck_indices'):
            self.physical_roots=self.root_states.clone()
        super().reset_idx(ids)


def run(checkpoint,mode,seed,steps,num_envs):
    torch.manual_seed(seed);torch.set_num_threads(1)
    cfg,train=BipedCfgWFShip(),BipedCfgPPOWFShip()
    apply_config_dict(cfg,json.loads((checkpoint.parent/'env_cfg.json').read_text()))
    apply_config_dict(train,json.loads((checkpoint.parent/'train_cfg.json').read_text()))
    cfg.seed=seed;cfg.env.num_envs=num_envs;cfg.ship.reference_mode=mode
    cfg.ship.curriculum=False;cfg.ship.training_schedule=False;cfg.ship.initial_difficulty=1.
    cfg.ship.amplitude_min=cfg.ship.amplitude_max=[4*np.pi/180,0.,0.]
    cfg.env.episode_length_s=max(20.,steps*.02+1)
    params=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    params.use_gpu_pipeline=False;params.physx.use_gpu=False;params.physx.num_threads=1
    params.physx.num_position_iterations=8;params.physx.num_velocity_iterations=2
    env=PhysicalSnapshotShip(cfg,params,gymapi.SIM_PHYSX,'cpu',True)
    try:
        runner=OnPolicyRunner(env,class_to_dict(train),None,'cpu')
        saved=torch.load(checkpoint,map_location='cpu')
        runner.alg.actor_critic.load_state_dict(saved['model_state_dict'])
        runner.alg.encoder.load_state_dict(saved['encoder_state_dict'])
        runner.alg.actor_critic.eval();runner.alg.encoder.eval()
        values={k:[] for k in ['root','obs','actions','dones','reward']}
        with torch.inference_mode():
            env.reset()
            for _ in range(steps):
                obs,history,commands,_=env.get_observations()
                values['obs'].append(obs.clone())
                latent=runner.alg.encode_for_policy(history,obs)
                actions=runner.alg.actor_critic.act_inference(torch.cat((latent,obs,commands),dim=-1))
                values['actions'].append(actions.clone())
                data=env.step(actions)
                values['root'].append(env.physical_roots.clone())
                values['dones'].append(data[2].clone());values['reward'].append(data[1].clone())
        return {k:torch.stack(v) for k,v in values.items()}
    finally:env.gym.destroy_sim(env.sim)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2026101503);p.add_argument('--steps',type=int,default=200)
    p.add_argument('--num-envs',type=int,default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    legacy=run(a.checkpoint,'body',a.seed,a.steps,a.num_envs)
    projected=run(a.checkpoint,'gravity_projection',a.seed,a.steps,a.num_envs)
    differences={k:float((legacy[k]-projected[k]).abs().max()) for k in ['root','obs','actions','reward']}
    same_done=torch.equal(legacy['dones'],projected['dones'])
    passed=(differences['root']<1e-5 and differences['obs']<1e-5 and differences['actions']<1e-5 and same_done)
    result={'passed':passed,'seed':a.seed,'steps':a.steps,'num_envs':a.num_envs,'device':'cpu',
            'fixed_roll_deg':4,'pitch_deg':0,'heave_m':0,'differences_max_abs':differences,
            'identical_dones':same_done,'checkpoint':str(a.checkpoint),
            'scope':'reference changes reward/privileged estimation target, not frozen-policy physics',
            'balance_success_claim':False}
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(a.output/'physical_reference_comparison.npz',
                        **{f'{mode}_{key}':tensor.numpy() for mode,data in [('body',legacy),('projection',projected)] for key,tensor in data.items()})
    print(json.dumps(result,indent=2),flush=True)
    if not passed:raise RuntimeError('reference choice unexpectedly altered physical policy trajectory')


if __name__=='__main__':main()
