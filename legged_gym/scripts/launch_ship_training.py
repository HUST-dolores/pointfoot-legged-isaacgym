"""Start matched ship runs on the two verified RTX 6000 Ada GPUs.

Source must already be staged in an isolated remote directory. This launcher
does not synchronize or overwrite source and does not stop any existing job.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-directory',type=Path,required=True)
    p.add_argument('--remote-root',default='/home/ps/mydrive/xu/ship_motion_20261014')
    p.add_argument('--host',default='codesign-ws')
    p.add_argument('--session-prefix',default='ship')
    p.add_argument('--timer-unit',default='ship-motion-hourly-20261014-v3.timer')
    p.add_argument('--num-envs',type=int,default=2048)
    p.add_argument('--iterations',type=int,default=12000)
    p.add_argument('--episode-length-s',type=float)
    p.add_argument('--yaw-tracking-scale',type=float)
    p.add_argument('--heading-yaw-tracking-scale',type=float)
    p.add_argument('--finetune-ship-policy',help='Absolute remote standing ship checkpoint, same source for both runs')
    p.add_argument('--encoder-kl-limit',type=float)
    p.add_argument('--locomotion-mode',choices=['balance','roll','step'],default='balance')
    p.add_argument('--initial-ship-policy',help='Absolute remote 37-observation ship checkpoint shared by both mobility runs')
    p.add_argument('--single-pair',action='store_true')
    p.add_argument('--pair-label',help='Explicit cohort label, e.g. lr1e4_s02')
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=2026101402)
    p.add_argument('--imu-gpu',type=int,choices=[0,1],default=1)
    p.add_argument('--evaluation-source')
    p.add_argument('--fixed-eval-amplitude',action='store_true')
    p.add_argument('--secondary-learning-rate',type=float,default=1e-3)
    p.add_argument('--secondary-pair-label',default='lr1e3_s02')
    p.add_argument('--reference-mode',choices=['body','gravity_projection'],default=None)
    p.add_argument('--schedule',choices=['fixed','adaptive'],default='fixed')
    p.add_argument('--kl-early-stop',action='store_true')
    p.add_argument('--freeze-encoder',action='store_true')
    p.add_argument('--entropy-coef',type=float,default=.01)
    p.add_argument('--max-noise-std',type=float)
    p.add_argument('--corner-fraction',type=float,default=0.)
    p.add_argument('--warmup-iterations',type=int,default=1000)
    p.add_argument('--motion-ramp-iterations',type=int,default=4000)
    a=p.parse_args()
    if a.heading_yaw_tracking_scale is not None and (not math.isfinite(a.heading_yaw_tracking_scale) or a.heading_yaw_tracking_scale<0):
        p.error('heading yaw tracking scale must be finite and nonnegative')
    if a.heading_yaw_tracking_scale is not None and a.yaw_tracking_scale is not None:
        p.error('choose heading-rate or body-yaw reward, not both')
    if a.finetune_ship_policy is not None and (a.locomotion_mode!='balance' or a.initial_ship_policy is not None or not Path(a.finetune_ship_policy).is_absolute()):
        p.error('finetune-ship-policy must be absolute, balance-only, and exclusive with initial-ship-policy')
    if a.locomotion_mode!='balance' and (not a.initial_ship_policy or not Path(a.initial_ship_policy).is_absolute()):
        p.error('mobility requires an absolute remote --initial-ship-policy shared by the pair')
    if a.locomotion_mode=='balance' and a.initial_ship_policy is not None:
        p.error('initial-ship-policy is for mobility, not the legacy flat-to-balance migration')
    if a.reference_mode is None:
        a.reference_mode='body' if a.locomotion_mode=='balance' else 'gravity_projection'
    if a.encoder_kl_limit is not None and (not math.isfinite(a.encoder_kl_limit) or a.encoder_kl_limit<=0):
        p.error('encoder KL limit must be finite and positive')
    if a.yaw_tracking_scale is not None and (not math.isfinite(a.yaw_tracking_scale) or a.yaw_tracking_scale<0):
        p.error('yaw tracking scale must be finite and nonnegative')
    if a.episode_length_s is not None and (not math.isfinite(a.episode_length_s) or a.episode_length_s<=0):
        p.error('episode length must be finite and positive')
    if a.single_pair and (not a.pair_label or a.learning_rate<=0):
        p.error('single-pair requires a pair-label and positive learning rate')
    a.manifest_directory.mkdir(parents=True,exist_ok=False)
    root=a.remote_root
    python='/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python'
    source=root+'/source'
    path='/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:/usr/local/cuda/bin:/usr/bin:/bin'
    plan=[('imu_lr3e4_s01',0,True,3e-4,2026101401),
          ('noimu_lr3e4_s01',1,False,3e-4,2026101401),
          ('imu_lr1e3_s02',1,True,1e-3,2026101402),
          ('noimu_lr1e3_s02',0,False,1e-3,2026101402)]
    plan[2:]=[('imu_'+a.secondary_pair_label,1,True,a.secondary_learning_rate,2026101402),
              ('noimu_'+a.secondary_pair_label,0,False,a.secondary_learning_rate,2026101402)]
    if a.single_pair:
        plan=[('imu_'+a.pair_label,a.imu_gpu,True,a.learning_rate,a.seed),
              ('noimu_'+a.pair_label,1-a.imu_gpu,False,a.learning_rate,a.seed)]
    jobs=[]
    for label,gpu,imu,lr,seed in plan:
        output=root+'/jobs/'+label
        session=a.session_prefix+'_'+label
        job={'label':label,'host':a.host,'gpu':gpu,'imu_enabled':imu,'learning_rate':lr,
             'seed':seed,'output':output,'python':python,'source':source,'session':session,
             'path':path,'pythonpath':source,'torch_extensions_dir':root+'/torch_extensions',
             'roll_deg':20,'pitch_deg':12,'heave_m':.15}
        job['locomotion_mode']=a.locomotion_mode
        job['initial_ship_policy']=a.initial_ship_policy
        job['finetune_ship_policy']=a.finetune_ship_policy
        job['heading_yaw_tracking_scale']=a.heading_yaw_tracking_scale
        if a.evaluation_source:job['evaluation_source']=a.evaluation_source
        job['fixed_eval_amplitude']=a.fixed_eval_amplitude
        job['reference_mode']=a.reference_mode
        job['episode_length_s']=a.episode_length_s
        job['yaw_tracking_scale']=a.yaw_tracking_scale
        job['encoder_kl_limit']=a.encoder_kl_limit
        job['stabilization']={'schedule':a.schedule,'kl_early_stop':a.kl_early_stop,
                              'freeze_encoder':a.freeze_encoder,'entropy_coef':a.entropy_coef,
                              'max_noise_std':a.max_noise_std,'corner_fraction':a.corner_fraction}
        command=['env',f'PATH={path}','ROBOT_TYPE=WF_TRON1A',f'CUDA_VISIBLE_DEVICES={gpu}',
                 f'PYTHONPATH={source}',f'TORCH_EXTENSIONS_DIR={root}/torch_extensions',
                 'PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4',python,
                 source+'/legged_gym/scripts/train_ship_motion.py','--output',output,'--label',label,
                 '--device','cuda:0','--num-envs',str(a.num_envs),'--iterations',str(a.iterations),
                 '--learning-rate',str(lr),'--seed',str(seed)]
        if a.locomotion_mode=='balance':
            if a.finetune_ship_policy is not None:
                command.extend(['--ship-finetune',a.finetune_ship_policy])
            else:
                command.extend(['--warm-start',root+'/initial_policy/model_11000.pt'])
        else:
            command.extend(['--locomotion-mode',a.locomotion_mode,'--ship-warm-start',a.initial_ship_policy])
        if not imu:command.append('--imu-off')
        if a.reference_mode != 'body':command.extend(['--reference-mode',a.reference_mode])
        if a.schedule!='fixed':command.extend(['--schedule',a.schedule])
        if a.kl_early_stop:command.append('--kl-early-stop')
        if a.freeze_encoder:command.append('--freeze-encoder')
        if a.entropy_coef!=.01:command.extend(['--entropy-coef',str(a.entropy_coef)])
        if a.max_noise_std is not None:command.extend(['--max-noise-std',str(a.max_noise_std)])
        if a.corner_fraction:command.extend(['--corner-fraction',str(a.corner_fraction)])
        if a.warmup_iterations!=1000:command.extend(['--warmup-iterations',str(a.warmup_iterations)])
        if a.motion_ramp_iterations!=4000:command.extend(['--motion-ramp-iterations',str(a.motion_ramp_iterations)])
        if a.episode_length_s is not None:command.extend(['--episode-length-s',str(a.episode_length_s)])
        if a.yaw_tracking_scale is not None:command.extend(['--yaw-tracking-scale',str(a.yaw_tracking_scale)])
        if a.heading_yaw_tracking_scale is not None:command.extend(['--heading-yaw-tracking-scale',str(a.heading_yaw_tracking_scale)])
        if a.encoder_kl_limit is not None:command.extend(['--encoder-kl-limit',str(a.encoder_kl_limit)])
        shell=' '.join(shlex.quote(x) for x in command)
        shell+=' >'+shlex.quote(output+'/console.log')+' 2>&1; code=$?; printf "%s\\n" "$code" >'+shlex.quote(output+'/EXIT_CODE')+'; exit "$code"'
        remote=('test ! -e '+shlex.quote(output)+' && mkdir -p '+shlex.quote(output)+' && '+
                ' '.join(shlex.quote(x) for x in ['tmux','new-session','-d','-s',session,shell]))
        launched=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',a.host,remote],
                                text=True,capture_output=True,timeout=40)
        job['launch_returncode']=launched.returncode
        if launched.returncode:job['launch_error']=launched.stderr[-500:]
        jobs.append(job)
        print(label,'launched' if launched.returncode==0 else 'FAILED',flush=True)
        manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,
                  'num_envs_per_run':a.num_envs,'iterations_per_run':a.iterations,
                  'episode_length_s':a.episode_length_s,
                  'locomotion_mode':a.locomotion_mode,'initial_ship_policy':a.initial_ship_policy,
                  'finetune_ship_policy':a.finetune_ship_policy,'heading_yaw_tracking_scale':a.heading_yaw_tracking_scale,
                  'warmup_iterations':a.warmup_iterations,
                  'motion_ramp_iterations':a.motion_ramp_iterations,
                  'timer_unit':a.timer_unit,
                  'pro6000_preflight':'Torch 2.4.1 sm_120 CUDA kernel test failed; existing environment unchanged',
                  'protocol':'paired IMU on/off; static adaptation gate then roll/pitch/heave ramp'}
        (a.manifest_directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if launched.returncode:raise RuntimeError('stopped launch after failure; existing jobs preserved')
    files=['legged_gym/envs/base/base_task.py',
           'legged_gym/algorithm/ppo.py',
           'legged_gym/envs/wheelfoot_ship/wheelfoot_ship.py',
           'legged_gym/envs/wheelfoot_ship/wheelfoot_ship_config.py',
           'legged_gym/utils/ship_motion.py','legged_gym/utils/ship_warm_start.py',
           'legged_gym/scripts/train_ship_motion.py',
           'legged_gym/scripts/ship_motion_probe.py']
    if a.locomotion_mode!='balance':
        files.extend(['legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py',
                      'legged_gym/envs/wheelfoot_ship/wheelfoot_ship_mobility.py',
                      'legged_gym/envs/wheelfoot_ship/wheelfoot_ship_mobility_config.py'])
    # Record what ACTUALLY runs remotely, not the evolving local evaluation code.
    script=('import hashlib,json\nfrom pathlib import Path\n'
            +f'root=Path({source!r})\nfiles={files!r}\n'
            +'print(json.dumps({f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}))\n')
    read=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',a.host,'python3 -'],
                        input=script,text=True,capture_output=True,timeout=40)
    if read.returncode:raise RuntimeError('jobs launched, but remote hash collection failed; preserve and inspect jobs')
    hashes=json.loads(read.stdout)
    (a.manifest_directory/'source_hashes.json').write_text(json.dumps(hashes,indent=2))


if __name__=='__main__':main()
