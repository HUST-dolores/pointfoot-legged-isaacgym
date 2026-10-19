# 独立复现与数据读取

完整交付包结构：`experiment/`为原始数据和结果，`code/repo/`为隔离源码/资源/
冻结检查点，`protocol/`为计划、审计、执行口径、问题记录和图注，`figures/`为图片入口。
根目录`audit_manifest.json`保留最初静态审计，不能将其中`evaluations_started=false`
误认为实验尚未完成；实际状态看`experiment/formal_status.json`与`README.md`。

## 离线重算与制图

在完整交付包根目录运行；不需要GPU、Isaac Gym或联网，不会启动训练。
保存的字体已可直接复用。输入为NPZ，使用`allow_pickle=False`。

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/codesign_figures_mpl \
 PYTHONPATH="$PWD/code/repo" \
 /home/xu/anaconda3/envs/pointfoot_legged_gym/bin/python \
 -m legged_gym.scripts.payload_step_analysis \
 --input "$PWD/experiment" --output /tmp/payload_step_reproduced
```

输出包含720条逐轨迹、360组配对、各级汇总、预登记主例CSV/MAT及两类中英文图。
MAT仅导出两条主例，完整30环境数据仍在各批`trace.npz`；不要将MAT主例当作全部样本。
MAT头含生成时刻，复现时比较其数组而不是要求文件字节完全相同。

```bash
sha256sum -c --quiet SHA256SUMS
```

## 重跑GPU仿真

需要原有Isaac Gym环境及两张GPU。先切换到`code/repo/`并将其加入PYTHONPATH，
不要覆盖原始`experiment/`，而使用新目录。设置ROBOT_TYPE=WF_TRON1A。

```bash
export PYTHONPATH="$PWD"
export ROBOT_TYPE=WF_TRON1A
python -m legged_gym.scripts.payload_step_experiment --prepare --output /tmp/payload_step_new
python -m legged_gym.scripts.payload_step_launch --output /tmp/payload_step_new --phase preflight
python -m legged_gym.scripts.payload_step_launch --output /tmp/payload_step_new --phase formal
```

调度器将每个子进程限制为一张可见物理GPU，进程内cuda:0，逐批保存stdout与退出码。
当前脚本遇到已完成同名批次不会覆盖；调度器可复用完整结果。正式比较前应复查
所有预检断言，而不是直接把上例当作无条件可以跳过审核的流水线。

## 记录字段

- `time_s`：0.02--40.00 s；所有状态、估计和接触共用当前时刻。
- `root_state`/`box_state`：xyz、xyzw、世界线速度、世界角速度；箱体离体后不归零。
- `mass_label`/`com_label`：原监督质量与等效基座质心增量，kg/m，不是箱体坐标。
- `estimate`：物理单位速度3、质量1、质心增量3；MG为已经融合的策略输入。
- `network_raw_mass_com`：RL直接预测或MG残差，语义由result.json的encoder_output区分。
- `analytic_prior`：当前原始解析基线；MG实际策略基线应读`observations[:,:,36:40]`
  并反归一化，以包括观测裁剪。图18使用后者。
- `noise_injected`/`noise_effective`：原实现IMU覆盖前后实际噪声，配对公共通道完全一致。
- `first_failure`：首次机器人终止后持续为真，不重新置零；`terminal_root`保存首次终端状态。
- `actions`：当前时刻推理候选；40 s最后一帧不再向前执行。它不是电机实际力矩。
- `torque`：上一个物理推进所用的关节力矩采样；不代表控制周期内的冲量积分。

接触、时间窗和失败判据以`EXECUTION_PROTOCOL.md`与冻结源码为准。
本文档不宣称策略在所有命令、负载或硬件下有效，也不提供15 kg结果的替代证据。
