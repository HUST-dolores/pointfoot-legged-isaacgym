"""Isolate post-PPO supervised encoder drift on a real simulated rollout.

Uses a loaded COPY, never modifies checkpoints or running training jobs.
Encoder Adam state is fresh: old checkpoints do not store that optimizer.
This is a diagnostic, not reproduction of an actual late-training update.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('ROBOT_TYPE','WF_TRON1A')
from isaacgym import gymapi,gymutil
import torch
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship import BipedWFShip
from legged_gym.envs.wheelfoot_ship.wheelfoot_ship_config import BipedCfgWFShip,BipedCfgPPOWFShip
from legged_gym.algorithm.on_policy_runner import OnPolicyRunner
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.saved_config import apply_config_dict


def gaussian_mean_shift_kl(before,after,sigma):
    """KL for unchanged diagonal variance, summed over actions."""
    return ((after-before).square()/(2*sigma.square())).sum(-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--settle-steps',type=int,default=200)
    p.add_argument('--seed',type=int,default=2026101506)
    p.add_argument('--encoder-kl-limit',type=float)
    p.add_argument('--encoder-update-blocks',type=int,default=1)
    a=p.parse_args()
    if a.num_envs<4 or a.settle_steps<0:p.error('need at least four environments and nonnegative settling')
    if a.encoder_update_blocks<1:p.error('encoder-update-blocks must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed);torch.set_num_threads(2)
    cfg,train=BipedCfgWFShip(),BipedCfgPPOWFShip()
    apply_config_dict(cfg,json.loads((a.checkpoint.parent/'env_cfg.json').read_text()))
    apply_config_dict(train,json.loads((a.checkpoint.parent/'train_cfg.json').read_text()))
    cfg.seed=a.seed;cfg.env.num_envs=a.num_envs
    if a.encoder_kl_limit is not None:train.algorithm.encoder_kl_limit=a.encoder_kl_limit
    cfg.ship.training_schedule=False;cfg.ship.curriculum=False;cfg.ship.initial_difficulty=1.
    cfg.ship.corner_fraction=0.
    cfg.ship.amplitude_min=list(cfg.ship.amplitude_max)
    cfg.ship.frequency_min=list(cfg.ship.frequency_max)
    params=gymapi.SimParams();gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
    params.use_gpu_pipeline=a.device.startswith('cuda');params.physx.use_gpu=params.use_gpu_pipeline
    params.physx.num_threads=2;params.physx.num_position_iterations=8;params.physx.num_velocity_iterations=2
    env=BipedWFShip(cfg,params,gymapi.SIM_PHYSX,a.device,True)
    try:
        runner=OnPolicyRunner(env,class_to_dict(train),None,a.device)
        runner.load(str(a.checkpoint));alg=runner.alg
        if alg.extra_optimizer is None:raise RuntimeError('checkpoint configuration has no trainable encoder')
        for group in alg.optimizer.param_groups:group['lr']=0.
        alg.schedule='fixed';alg.anneal_lr=False
        original_actor={k:v.detach().clone() for k,v in alg.actor_critic.state_dict().items()}
        original_encoder={k:v.detach().clone() for k,v in alg.encoder.state_dict().items()}
        with torch.inference_mode():
            env.reset()
            for _ in range(a.settle_steps):
                obs,hist,cmd,_=env.get_observations()
                mu=alg.actor_critic.act_inference(torch.cat((alg.encode_for_policy(hist,obs),obs,cmd),-1))
                env.step(mu)
            for _ in range(runner.num_steps_per_env):
                obs,hist,cmd,critic=env.get_observations()
                action=alg.act(obs,hist,cmd,critic)
                data=env.step(action)
                alg.process_env_step(data[1],data[2],data[3],data[0])
            obs,hist,cmd,critic=env.get_observations()
            critic_input=torch.cat((critic,cmd),-1)
            if alg.critic_take_latent:critic_input=torch.cat((critic_input,alg.encode_for_policy(hist,obs)),-1)
            alg.compute_returns(critic_input)
        storage=alg.storage
        # Storage clear() only resets its step cursor; both updates reuse this rollout.
        observed=storage.observations.flatten(0,1)
        history=storage.observation_history.flatten(0,1)
        commands=storage.commands.flatten(0,1)
        def means():
            with torch.inference_mode():
                chunks=[]
                # Match rollout batching: different CUDA GEMM/GRU batch shapes
                # can otherwise introduce numerical differences before ANY update.
                for start in range(0,len(observed),a.num_envs):
                    o=observed[start:start+a.num_envs];h=history[start:start+a.num_envs];c=commands[start:start+a.num_envs]
                    chunks.append(alg.actor_critic.act_inference(torch.cat((alg.encode_for_policy(h,o),o,c),-1)).clone())
                return torch.cat(chunks)
        before=means()
        old_mu=storage.mu.flatten(0,1).clone();sigma=storage.sigma.flatten(0,1).clone()
        snapshot_error=float((before-old_mu).abs().max())
        with torch.inference_mode():
            batched=alg.actor_critic.act_inference(torch.cat((alg.encode_for_policy(history,observed),observed,commands),-1))
            batched_snapshot_error=float((batched-old_mu).abs().max())
        (a.output/'pre_update_snapshot.json').write_text(json.dumps({'same_batch_max_error':snapshot_error,'flattened_batch_max_error':batched_snapshot_error},indent=2))
        assert snapshot_error<1e-5,'unchanged rollout likelihood mismatch'
        extra_optimizer=alg.extra_optimizer
        alg.extra_optimizer=None
        control=alg.update()
        control_error=float((means()-before).abs().max())
        assert control_error==0.,'actor-zero-LR/frozen-encoder control changed actions'
        alg.extra_optimizer=extra_optimizer
        update_blocks=[]
        for block in range(a.encoder_update_blocks):
            block_before=means()
            storage.step=runner.num_steps_per_env
            updated=alg.update()
            block_after=means()
            update_blocks.append({'block':block+1,**updated[4],
                                  'full_rollout_encoder_kl':float(gaussian_mean_shift_kl(block_before,block_after,sigma).mean())})
        after=means()
        actor_change=max(float((v-original_actor[k]).abs().max()) for k,v in alg.actor_critic.state_dict().items())
        encoder_change=max(float((v-original_encoder[k]).abs().max()) for k,v in alg.encoder.state_dict().items())
        assert actor_change==0.
        if a.encoder_kl_limit is None:assert encoder_change>0.
        elif all(b.get('encoder_update_rejected') for b in update_blocks):assert encoder_change==0.
        kl=gaussian_mean_shift_kl(before,after,sigma)
        result={'checkpoint':str(a.checkpoint),'seed':a.seed,'device':a.device,
                'num_envs':a.num_envs,'rollout_steps':runner.num_steps_per_env,
                'imu_enabled':cfg.ship.imu_enabled,'amplitude':cfg.ship.amplitude_max,
                'frequency_hz':cfg.ship.frequency_max,'actor_lr':0.,
                'encoder_optimizer_state':'fresh Adam; checkpoint does not store encoder optimizer',
                'encoder_lr':extra_optimizer.param_groups[0]['lr'],
                'rollout_snapshot_mean_error':snapshot_error,'frozen_encoder_control_mean_error':control_error,
                'flattened_batch_snapshot_error':batched_snapshot_error,
                'actor_max_parameter_change':actor_change,'encoder_max_parameter_change':encoder_change,
                'post_encoder_mean_kl':float(kl.mean()),'post_encoder_max_kl':float(kl.max()),
                'post_encoder_action_max_change':float((after-before).abs().max()),
                'desired_kl':alg.desired_kl,'post_encoder_fraction_over_kl_stop':float((kl>1.5*alg.desired_kl).float().mean()),
                'control_ppo_logged_kl':control[3],'encoder_update_ppo_logged_kl':updated[3],
                'encoder_update_loss':updated[1],
                'encoder_guard_metrics':updated[4],
                'encoder_update_blocks':update_blocks,
                'scope':'isolated diagnostic; not proof of historical training failure cause'}
        (a.output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
    finally:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
