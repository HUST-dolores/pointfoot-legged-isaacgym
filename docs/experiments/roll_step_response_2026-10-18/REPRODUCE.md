# 复现与字段

从完整交付包根目录运行，分析不需要Isaac Gym或GPU，不会启动训练：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/code/repo" \
 MPLCONFIGDIR=/tmp/codesign_figures_mpl \
 /home/xu/anaconda3/envs/pointfoot_legged_gym/bin/python \
 -m legged_gym.scripts.roll_step_response_analysis \
 --input "$PWD/experiment" --output /tmp/roll_step_response_reproduction

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/code/repo" \
 /home/xu/anaconda3/envs/pointfoot_legged_gym/bin/python \
 -m unittest legged_gym.scripts.test_roll_step_response_protocol -v

sha256sum -c --quiet SHA256SUMS
```

输入使用NPZ且不允许pickle。PNG/CSV/JSON可逐字节核验；MAT比较数值数组，PDF忽略生成日期。
归档内字体可直接读取，无需联网。代码/检查点/机器人资产完整保存，
任务层来自run快照，共享框架是本次兼容版本，不能声称恢复原训练时全部依赖版本。

- `task_roll_step`：当前策略接收的物理roll/step命令，m/s。
- `actor_commands`：实际缩放roll/y/yaw命令；obs[28]才是step，不能只记录commands[:,0]。
- `policy_input`：编码7＋观测31＋命令3，共41维，保存实际actor输入以核验。
- `history_step_commands`：GRU10帧中step命令，跨切点严格移位，不清空历史。
- `applied_previous_interval_roll_step`：刚结束的物理区间所用命令，不冒充当前新命令。
- `phase_of_preceding_interval`：状态对应的统计阶段；`phase_for_next_action`：当前动作对应阶段。
- `body_velocity`：框架20 ms位移差分、旋转到当前机体系的速度。
- `body_velocity_from_root`：仿真根世界线速度旋转至机体，另存诊断，不与主口径混算。
- `v_roll_proxy`、`v_step_proxy`：保存原实现的接触加权轮速与剩余量，仅为任务代理。
- `wheel_forces`、`wheel_clearance_signed`：实际接触力与胶囊最低点对平地的有符号净空。
- `first_failure`：原终止首次触发后锁存；失败后不复位，保持同一条物理轨迹，几何跌倒另计算。

图内轨迹只有预登记主例，统计使用全部90条；`per_environment_phases.csv`有360行，
`per_environment_conditions.csv`有270行，两表不是360/270条独立试验。
