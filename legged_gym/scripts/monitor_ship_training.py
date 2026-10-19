"""Hourly read-only fleet collection, with optional isolated checkpoint evaluation.

Uses SSH aliases and existing keys; no credentials are stored in the manifest.
Training jobs are never killed or restarted. Only this monitor's timer may
stop itself after all jobs have reached a terminal state.
"""
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temporary.replace(path)


def collect(host,jobs):
    script='''import json,os,subprocess,time
from pathlib import Path
jobs=json.loads(JOBS)
out=[]
for j in jobs:
 p=Path(j['output'])
 try:
  progress=json.loads((p/'progress.json').read_text())
 except (OSError,ValueError):
  progress={'state':'starting'}
 pid=progress.get('pid')
 alive=False
 if pid:
  try:
   command=Path('/proc')/str(pid)/'cmdline'
   alive=b'train_ship_motion.py' in command.read_bytes()
  except OSError:pass
 if progress.get('state')=='running' and not alive:
  progress['state']='exited_without_completion'
 exit_code=None
 try:exit_code=int((p/'EXIT_CODE').read_text().strip())
 except (OSError,ValueError):pass
 buffer_errors=0
 try:
  with (p/'console.log').open(errors='replace') as log:
   buffer_errors=sum('PxgAABBManager.cpp' in line and 'invalid parameter' in line for line in log)
 except OSError:pass
 checkpoints=[x for x in p.glob('model_*.pt') if time.time()-x.stat().st_mtime>3]
 latest=max(checkpoints,key=lambda x:int(x.stem.split('_')[1])) if checkpoints else None
 out.append({**j,'progress':progress,'process_alive':alive,'exit_code':exit_code,'physx_buffer_errors':buffer_errors,'latest_checkpoint':str(latest) if latest else None})
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.free','--format=csv,noheader'],text=True,capture_output=True)
print(json.dumps({'jobs':out,'gpu':gpu.stdout.strip()}))
'''.replace('JOBS',repr(json.dumps(jobs)))
    last_error=''
    for attempt in range(3):
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',host,'python3 -'],
                                  input=script,text=True,capture_output=True,timeout=75)
            if result.returncode==0:return json.loads(result.stdout)
            last_error=result.stderr.strip()[-500:]
        except (subprocess.TimeoutExpired,ValueError) as error:
            last_error=str(error)
        if attempt<2:time.sleep(2)
    raise RuntimeError(last_error)


def evaluate(job,stamp,motion_scale):
    steps=job.get('evaluation_steps',1000)
    if type(steps) is not int or steps<1:
        raise ValueError('evaluation_steps must be a positive integer')
    seed=job.get('evaluation_seed',2026101499)
    if type(seed) is not int:
        raise ValueError('evaluation_seed must be an integer')
    output=f"{job['output']}/hourly_eval/{stamp}_scale{motion_scale:g}"
    source=job.get('evaluation_source',job['source'])
    args=[job['python'],f"{source}/legged_gym/scripts/ship_motion_probe.py",
          '--mode','eval','--checkpoint',job['latest_checkpoint'],'--device','cuda:0',
          '--num-envs','64','--steps',str(steps),'--seed',str(seed),'--motion-scale',str(motion_scale),
          '--roll-deg',str(job['roll_deg']),'--pitch-deg',str(job['pitch_deg']),
          '--heave-m',str(job['heave_m']),
          '--output',output]
    if job.get('fixed_eval_amplitude',False):args.append('--fixed-amplitude')
    if job.get('reference_mode','body')!='body':args.extend(['--reference-mode',job['reference_mode']])
    if job.get('fixed_eval_frequency_hz'):
        args.extend(['--fixed-frequency-hz',*[str(v) for v in job['fixed_eval_frequency_hz']]])
    if not job['imu_enabled']:args.append('--imu-off')
    env=['ROBOT_TYPE=WF_TRON1A',f"CUDA_VISIBLE_DEVICES={job['gpu']}",
         f"PYTHONPATH={source}",'PYTHONDONTWRITEBYTECODE=1']
    if job.get('path'):env.append(f"PATH={job['path']}")
    if job.get('torch_extensions_dir'):env.append(f"TORCH_EXTENSIONS_DIR={job['torch_extensions_dir']}")
    if job.get('ld_library_path'):env.append(f"LD_LIBRARY_PATH={job['ld_library_path']}")
    command=' '.join(shlex.quote(x) for x in ['env',*env,*args])
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',job['host'],command],
                          text=True,capture_output=True,timeout=600)
    command='cat '+shlex.quote(output+'/result.json')
    read=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',job['host'],command],
                        text=True,capture_output=True,timeout=30)
    if read.returncode==0:
        value=json.loads(read.stdout)
        value['execution_returncode']=result.returncode
        return value
    return {'state':'eval_failed','error':result.stderr[-1200:],'execution_returncode':result.returncode}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--eval-scales',type=float,nargs='+',default=[0.,.25,1.],
                        help='Static, mild and full motion by default')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    folder=args.manifest.parent
    (folder/'hourly').mkdir(exist_ok=True)
    now=datetime.now(timezone.utc)
    stamp=now.strftime('%Y%m%dT%H%M%SZ')
    report={'updated_utc':now.isoformat(),'updated_shanghai':now.astimezone(timezone(timedelta(hours=8))).isoformat(),
            'jobs':[],'hosts':{}}
    for host in sorted({j['host'] for j in manifest['jobs']}):
        jobs=[j for j in manifest['jobs'] if j['host']==host]
        try:
            result=collect(host,jobs)
            report['hosts'][host]={'gpu':result['gpu'],'state':'reachable'}
            report['jobs'].extend(result['jobs'])
        except Exception as error:
            report['hosts'][host]={'state':'unreachable','error':str(error)}
            report['jobs'].extend({**j,'progress':{'state':'unreachable'}} for j in jobs)
    # A quick liveness poll must not erase the last expensive policy test.
    try:
        previous=json.loads((folder/'status.json').read_text())
        previous_jobs={j['output']:j for j in previous.get('jobs',[])}
    except (OSError,ValueError,KeyError):previous_jobs={}
    for job in report['jobs']:
        old=previous_jobs.get(job['output'],{})
        if old.get('evaluation'):
            job['evaluation']=old['evaluation']
            job['evaluation_checkpoint']=old.get('evaluation_checkpoint',old.get('latest_checkpoint'))
            job['evaluation_updated_utc']=old.get('evaluation_updated_utc',previous.get('updated_utc'))
    if args.evaluate:
        for job in report['jobs']:
            if job.get('latest_checkpoint') and job['progress'].get('state') in ('running','complete','failed','exited_without_completion'):
                job['evaluation']={}
                job['evaluation_checkpoint']=job['latest_checkpoint']
                job['evaluation_updated_utc']=now.isoformat()
                for scale in args.eval_scales:
                    try:job['evaluation'][f'{scale:g}']=evaluate(job,stamp,scale)
                    except Exception as error:job['evaluation'][f'{scale:g}']={'state':'eval_failed','error':str(error)}
    write_json(folder/'status.json',report)
    write_json(folder/'hourly'/f'{stamp}.json',report)
    lines=['# 船舶训练小时跟踪','',f"更新时间（上海）：{report['updated_shanghai']}",'',
           '| 实验 | 参考 | 工作站/GPU | 状态 | 更新次数 | 平均 episode 秒 | Reward | 课程横/纵/升比例 | 最新模型 |',
           '| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |']
    diagnostic_lines=[]
    for j in report['jobs']:
        p=j['progress']
        length=p.get('mean_episode_length_s')
        reward=p.get('mean_reward')
        checkpoint=Path(j['latest_checkpoint']).name if j.get('latest_checkpoint') else '-'
        reference={'body':'机身','gravity_projection':'重力投影'}.get(j.get('reference_mode','body'),j.get('reference_mode'))
        phase=' / '.join(f'{100*x:.0f}%' for x in p['motion_scale']) if p.get('motion_scale') else '-'
        lines.append(f"| {j['label']} | {reference} | {j['host']}/{j['gpu']} | {p.get('state')} | {p.get('iteration','-')} | {('-' if length is None else f'{length:.2f}')} | {('-' if reward is None else f'{reward:.3f}')} | {phase} | {checkpoint} |")
        if j.get('exit_code') is not None:diagnostic_lines.extend(['',f"{j['label']} 退出码：{j['exit_code']}"])
        if j.get('physx_buffer_errors'):diagnostic_lines.extend(['',f"{j['label']} 仿真无效警告：{j['physx_buffer_errors']} 次 PhysX 缓冲溢出。"])
        if p.get('stop_reason'):diagnostic_lines.extend(['',f"{j['label']} 停止原因：{p['stop_reason']}"])
    lines.extend(diagnostic_lines)
    lines.extend(['','课程比例是计划幅值上限比例；每个episode的实际幅值采样并冻结，实测幅值见独立评估。'])
    evaluated=[j for j in report['jobs'] if j.get('evaluation')]
    if evaluated:
        lines.extend(['','## 独立策略评估','',
                      '相同评估 seed，控制周期 20 ms；实际测试时长见表，完整原始指标保存在 status.json。', '',
                      '| 实验 | 条件 | 时长 s | 完整幅值 | 无失败比例 | 失败次数 | 实测横摇/纵摇 ° | 实测升沉 m | 平均漂移 m | 执行 |',
                      '| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |'])
        for j in evaluated:
            for scale,e in j['evaluation'].items():
                label={'0':'静止','0.25':'轻摇','1':'剧烈'}.get(scale,scale)
                survival=e.get('survival_fraction')
                angles='/'.join('-' if e.get(key) is None else f'{e[key]:.1f}'
                                for key in ['max_actual_roll_deg','max_actual_pitch_deg'])
                heave=e.get('max_actual_heave_m')
                drift=e.get('mean_deck_relative_position_error_m')
                duration=None if e.get('steps') is None else e['steps']*.02
                execution=('通过' if e.get('execution_returncode')==0 else e.get('state','失败'))
                lines.append(f"| {j['label']} | {label} | {'-' if duration is None else f'{duration:.1f}'} | {'是' if e.get('fixed_amplitude') else '否/未记录'} | "
                             f"{'-' if survival is None else f'{100*survival:.1f}%'} | {e.get('failure_resets','-')} | "
                             f"{angles} | {'-' if heave is None else f'{heave:.3f}'} | "
                             f"{'-' if drift is None else f'{drift:.3f}'} | {execution} |")
        for j in evaluated:
            lines.extend(['',f"{j['label']} 评估模型：{Path(j.get('evaluation_checkpoint') or j.get('latest_checkpoint') or '-').name}；"
                          f"评估采集时间：{j.get('evaluation_updated_utc','未记录')}。"])
        yaw_rows=[(j,scale,e['yaw_metrics']) for j in evaluated for scale,e in j['evaluation'].items() if e.get('yaw_metrics')]
        if yaw_rows:
            lines.extend(['','## 航向诊断','',
                '只统计2秒后、每环境首次失败之前；累计yaw正确展开，不计reset跳变。较小转角不能代替无失败率。','',
                '| 实验 | 条件 | 航向速率 RMS rad/s | 最大累计yaw中位数 ° | 最大累计yaw P95 ° |',
                '| --- | --- | ---: | ---: | ---: |'])
            for j,scale,y in yaw_rows:
                values=[y.get(k) for k in ['heading_rate_rms_rad_s','peak_heading_change_median_deg','peak_heading_change_p95_deg']]
                cells=['-' if v is None else f'{v:.3f}' for v in values]
                lines.append('| '+j['label']+' | '+scale+' | '+' | '.join(cells)+' |')
    lines.extend(['','`passed` 只表示执行/观测检查通过；平衡收益以存活、跌倒和漂移指标判断。',
                  '有/无 IMU 同 seed、同学习率、同预算配对；不同参数组属于探索，不直接作统计结论。'])
    (folder/'status.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    states={j['progress'].get('state') for j in report['jobs']}
    if states and states.issubset({'complete','failed','exited_without_completion','early_stopped'}):
        timer=manifest.get('timer_unit')
        if timer:subprocess.run(['systemctl','--user','stop',timer],check=False)
    if states=={'unreachable'}:
        raise SystemExit(1)


if __name__=='__main__':
    main()
