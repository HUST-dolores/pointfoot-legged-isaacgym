# 复现与文件说明

完整交付包根目录含`experiment/`原始轨迹/相机帧、`code/repo/`独立源码与两个
冻结检查点、`protocol/`所有说明、`figures/`中英文图20/21的三种地形版本。

## 离线重算

在交付包根目录执行，不需要GPU或Isaac Gym，不连接工作站，不训练。

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codesign_figures_mpl \
 PYTHONPATH="$PWD/code/repo" \
 /home/xu/anaconda3/envs/pointfoot_legged_gym/bin/python \
 -m legged_gym.scripts.terrain_step_analysis \
 --input "$PWD/experiment" --output /tmp/terrain_step_reproduced
```

使用归档字体，输出完整540条/270组统计、6个过程主例MAT与12个中英文PNG及PDF/SVG。
MAT头与PDF元数据可能包含生成时刻；数值数组和PNG应能独立一致复现。

```bash
sha256sum -c --quiet SHA256SUMS
```

## GPU重跑

需要既有Isaac Gym/PyTorch环境。切换到`code/repo`，写新结果目录，不覆盖归档：

```bash
export PYTHONPATH="$PWD"
export ROBOT_TYPE=WF_TRON1A
python -m legged_gym.scripts.terrain_step_protocol --output /tmp/terrain_step_new
python -m legged_gym.scripts.terrain_step_launch --output /tmp/terrain_step_new --phase preflight --capture-main
python -m legged_gym.scripts.terrain_step_launch --output /tmp/terrain_step_new --phase formal --capture-main
```

正式比较前必须核验预检。GPU0采固定主例相机，其他批次在双GPU队列并行。
相机退出兼容处理只在显式清理成功后生效，详情见`PROBLEMS.md`，不可用来掩盖异常。

## 数值与图的对应

- `trace.npz`按(T,N,...)保存，`allow_pickle=False`读取；全程300时刻、30环境。
- `root_state`为世界位置/xyzw/世界线速度/角速度；`rigid_body_state`为真实逐刚体状态。
- `body_velocity`为原任务使用的机体坐标速度；坡向速度用世界速度投影到+X上坡切向。
- `task_commands=[roll, lateral, yaw_rate, step]`恒为[0,0,0,0.4]；actor缩放命令3维均0。
- `wheel_forces`为实际三维力；`wheel_lowest_world`、`ground_at_lowest`与
  `wheel_clearance_signed`构成有符号竖向净空，米，不裁掉负值。
- `ground_at_base`与实际XY轨迹形成图20剖面；不能替换成地图任意截线。
- `first_failure`首次原终止后保持真；失败后不重置。原始后续状态保留但不作为有效运动。
- `frames/metadata.json`保存真实采图时刻、相机位置/目标、失效状态及RGB离散性；
  原始PNG不经美化，拼图仅统一视窗裁剪。
- `preregistered_process_examples.mat`只导出两个策略x三地形的env0主例；完整环境在NPZ。
- `forward_progress_diagnostics.csv`为额外符号/净位移核验，不是事后新增的成功筛选门槛。
- 配对表、逐seed表、全体表均保留；环境SD仅为描述性离散性。

`source_audit.json`给出两run配置差异及输入哈希；`delivery_manifest.json`给出最终源码哈希。
冻结后的分析变化仅涉及额外导出、方向诊断、字体/排版/统一裁剪和核验，主指标/失败规则未改。
