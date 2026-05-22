# 实验记录本 (Experiment Notebook)

> **本文档的用法**：
> - 第一部分按时间顺序记录所有 play 的**原始 metrics 输出**（直接粘贴自终端）
> - 第二部分是基于原始数据的**对比表格 + 解读**
> - 第三部分是**累计的关键结论**（每有新数据就更新）
> - 用户后续直接在第一部分末尾追加新结果即可；要让助手分析，告诉它 "看一下 `docs/experiment_notebook.md`" 即可

---

## 共同设定（适用所有 play）

| 项 | 值 |
|---|---|
| 平台 | WF_TRON1A (双轮足双足) |
| 训练 | trimesh L0（单一最易级），num_envs=1024，episode_length=40s |
| Play | num_envs=20，stop_state_log=2000 step (40s)，trimesh L0 patch 60×180 m |
| 载荷 | mass ∈ [2, 4] kg 随机，start=0.5s，duration ∈ [30,40] s，interval ∈ [50,60] s |
| Random | friction / restitution / base_com / inertia / Kp / Kd / motor_torque / dof_pos / action_delay / imu_offset 全开；base_mass=False（Model C 标定需求） |
| Push | push_robots=True, push_interval=5s, max_push_vel=2 m/s |

**Model C 标定基线**（QS 工况，4×16 网格拟合）：mass RMSE = 0.77 kg，x RMSE = 21 mm，y RMSE = 66 mm。

---

# 第一部分：原始 Play 数据

> **⚠️ 重要：第一部分（§1–§5）所有 RL mass / RL CoM 数字都是 BUG 时代数据**
>
> 2026-05-21 之前的 play.py 在 residual 模式下把 QS baseline **加了两次**到 encoder 输出：
> 一次在 `PPO.encode_for_policy()` 内（正确），又一次在 play.py 的 logging 分支
> （错误，`_baseline[:, 0] + est[:, 3]`）。
>
> **影响范围**：所有 main residual policy 的 plays（main lb=3 / lb=6 / OOD）的 RL mass 数字
> 系统性偏高一个 baseline 量级（约 +3 kg）。direct 和 history_only **没有受影响**（不走 residual 分支）。
>
> **结论**：§1–§5 仅作历史追溯保留；**所有 main residual 行的 RL mass 数字应视为废弃**。
> 真实数据见第二部分 §6（已 bug 修复 + auto exp_tag + multi-seed 干净复测）。
>
> 同时也修复了：(a) task_registry 的 seed 应用 bug；(b) history_only ckpt shape mismatch；
> (c) auto exp_tag 没从 saved cfg 读 lb（lb=6 ckpt 标错成 lb=3）。

## §1 早期 pilot：lb=6 ckpt 9000/11000/13000（旧 play.py，无 encoder logging）

> 说明：这一组只记录了 Model C 的 RMSE（当时还以为是 encoder 的），后来发现是 QS。无 [RL] encoder 数字。

**ckpt 9000 (lb=6, static)**：
```
mass:  bias=+4.6186 kg  rmse=5.2153 kg
x   :  bias=-0.0120 m   rmse=0.0518 m
y   :  bias=-0.0299 m   rmse=0.0958 m
mass conv reached: 0%
x conv reached: 100% @ 2.15s
y conv reached: 100% @ 7.64s
dyn_phase RMSE: mass=6.24, com_x=0.076, com_y=0.097
```

**ckpt 11000 (lb=6, static)**：
```
mass:  bias=+5.0794 kg  rmse=5.6387 kg
x   :  bias=-0.0116 m   rmse=0.0536 m
y   :  bias=-0.0091 m   rmse=0.0884 m
mass conv reached: 0%
dyn_phase RMSE: mass=6.93, com_x=0.084, com_y=0.095
```

**ckpt 13000 (lb=6, static)**：
```
mass:  bias=+1.6725 kg  rmse=2.7720 kg
x   :  bias=-0.0105 m   rmse=0.0364 m
y   :  bias=-0.0189 m   rmse=0.0999 m
mass conv reached: 40% @ 14.09s
dyn_phase RMSE: mass=5.78, com_x=0.066, com_y=0.095
```

---

## §2 lb=3 vs lb=6 ckpt 11000 全套对比（新 play.py 含 encoder logging）

### 2.1 lb=3 static (ckpt 11000, seed=42, run #1)

```
====== LOAD MASS (kg) ======
  [QS]  RMSE=5.6387  bias=+5.0794  per_env=5.6158±0.5074
  [RL]  RMSE=0.8624  bias=+0.0516  per_env=0.8619±0.0280

====== LOAD POSITION (m) ======
  [QS]  com_x RMSE=0.0536  bias=-0.0116
  [QS]  com_y RMSE=0.0884  bias=-0.0091

====== CoM DELTA (m) ======
  [QS]  x: RMSE=0.0160  bias=-0.0135  |  y: RMSE=0.0273  bias=-0.0021
  [RL]  x: RMSE=0.0135  bias=-0.0031  |  y: RMSE=0.0103  bias=+0.0006

====== convergence ======
  [QS]  reached=0%
  [RL]  reached=100% @ 0.0810s

====== dynamic-phase RMSE ======
  [QS]  mass=6.9313 kg  dcom_x=0.0226 m  dcom_y=0.0317 m
  [RL]  mass=0.8080 kg  dcom_x=0.0161 m  dcom_y=0.0106 m
```

### 2.2 lb=3 static (ckpt 11000, seed=43, run #2 — outlier!)

```
====== LOAD MASS (kg) ======
  [QS]  RMSE=5.1945  bias=+4.5645  per_env=5.0959±1.0073
  [RL]  RMSE=1.3959  bias=+0.0976  per_env=1.3922±0.1018      ← outlier (RL mass +62%)

====== LOAD POSITION (m) ======
  [QS]  com_x RMSE=0.0653  bias=-0.0195
  [QS]  com_y RMSE=0.0907  bias=+0.0033

====== CoM DELTA (m) ======
  [QS]  x: RMSE=0.0163  bias=-0.0150  |  y: RMSE=0.0250  bias=+0.0006
  [RL]  x: RMSE=0.0134  bias=-0.0014  |  y: RMSE=0.0109  bias=+0.0017

====== convergence ======
  [QS]  reached=0%
  [RL]  reached=100% @ 0.4320s

====== dynamic-phase RMSE ======
  [QS]  mass=8.3066 kg  dcom_x=0.0220 m  dcom_y=0.0285 m
  [RL]  mass=1.3242 kg  dcom_x=0.0142 m  dcom_y=0.0099 m
```

### 2.3 lb=3 static (ckpt 11000, seed unknown, run #3)

```
====== LOAD MASS (kg) ======
  [QS]  RMSE=5.7497  bias=+4.7974  per_env=5.6148±1.2381
  [RL]  RMSE=0.8645  bias=+0.2777  per_env=0.8618±0.0689

====== LOAD POSITION (m) ======
  [QS]  com_x RMSE=0.0533  bias=-0.0136
  [QS]  com_y RMSE=0.0888  bias=+0.0057

====== CoM DELTA (m) ======
  [QS]  x: RMSE=0.0154  bias=-0.0135  |  y: RMSE=0.0244  bias=+0.0030
  [RL]  x: RMSE=0.0136  bias=-0.0034  |  y: RMSE=0.0108  bias=+0.0021

====== convergence ======
  [QS]  reached=0%
  [RL]  reached=100% @ 0.0350s

====== dynamic-phase RMSE ======
  [QS]  mass=7.8243 kg  dcom_x=0.0198 m  dcom_y=0.0289 m
  [RL]  mass=0.8416 kg  dcom_x=0.0159 m  dcom_y=0.0099 m
```

### 2.4 lb=6 static (ckpt 11000, seed=42)

```
====== LOAD MASS (kg) ======
  [QS]  RMSE=3.7301  bias=+3.2674  per_env=3.5934±1.0004
  [RL]  RMSE=0.9685  bias=+0.1451  per_env=0.9676±0.0415

====== LOAD POSITION (m) ======
  [QS]  com_x RMSE=0.0454  bias=-0.0109
  [QS]  com_y RMSE=0.1144  bias=+0.0400

====== CoM DELTA (m) ======
  [QS]  x: RMSE=0.0135  bias=-0.0110  |  y: RMSE=0.0271  bias=+0.0106
  [RL]  x: RMSE=0.0156  bias=+0.0006  |  y: RMSE=0.0114  bias=+0.0013

====== convergence ======
  [QS]  reached=5% @ 2.28s
  [RL]  reached=100% @ 0.12s

====== dynamic-phase RMSE ======
  [QS]  mass=4.9083 kg  dcom_x=0.0197 m  dcom_y=0.0291 m
  [RL]  mass=0.9584 kg  dcom_x=0.0193 m  dcom_y=0.0111 m
```

### 2.5 lb=6 walk x=0.5 (ckpt 11000, seed=42)

```
====== LOAD MASS (kg) ======
  [QS]  RMSE=7.0744  bias=+1.8640  per_env=4.4318±5.5143
  [RL]  RMSE=1.0670  bias=+0.1288  per_env=1.0560±0.1528

====== LOAD POSITION (m) ======
  [QS]  com_x RMSE=0.0462  bias=-0.0120
  [QS]  com_y RMSE=0.1647  bias=+0.0224

====== CoM DELTA (m) ======
  [QS]  x: RMSE=0.0128  bias=-0.0091  |  y: RMSE=0.0387  bias=+0.0054
  [RL]  x: RMSE=0.0162  bias=-0.0017  |  y: RMSE=0.0128  bias=+0.0020

====== convergence ======
  [QS]  reached=45% @ 18.25s
  [RL]  reached=100% @ 0.086s

====== dynamic-phase RMSE ======
  [QS]  mass=4.0222 kg  dcom_x=0.0123 m  dcom_y=0.0369 m
  [RL]  mass=1.0640 kg  dcom_x=0.0161 m  dcom_y=0.0127 m
```

### 2.6 lb=3 walk x=0.5 (ckpt 11000, seed=42)

```
====== LOAD MASS (kg) ======
  [QS]  RMSE=5.8774  bias=+5.0254  per_env=5.8395±0.6658
  [RL]  RMSE=0.9410  bias=+0.0104  per_env=0.9188±0.2031

====== LOAD POSITION (m) ======
  [QS]  com_x RMSE=0.0567  bias=-0.0028
  [QS]  com_y RMSE=0.1293  bias=+0.0204

====== CoM DELTA (m) ======
  [QS]  x: RMSE=0.0164  bias=-0.0123  |  y: RMSE=0.0392  bias=+0.0077
  [RL]  x: RMSE=0.0150  bias=-0.0006  |  y: RMSE=0.0116  bias=+0.0007

====== convergence ======
  [QS]  reached=15% @ 26.85s
  [RL]  reached=100% @ 0.192s

====== dynamic-phase RMSE ======
  [QS]  mass=5.9119 kg  dcom_x=0.0164 m  dcom_y=0.0376 m
  [RL]  mass=0.9194 kg  dcom_x=0.0148 m  dcom_y=0.0111 m
```

---

## §3 lb=3 全 ckpt 扫描（ckpt 9000/11000/13000, static, seed=42）

> 说明：跟 §2.1 是同 lb=3 同 seed 同 static，只是 ckpt 不同。ckpt 11000 数据跟 §2.1 完全相同。

### 3.1 lb=3 ckpt 9000 static

```
====== LOAD MASS (kg) ======
  [QS]  RMSE=5.2153  bias=+4.6186  per_env=5.1281±0.9497
  [RL]  RMSE=0.9303  bias=+0.2078  per_env=0.9296±0.0360

====== CoM DELTA (m) ======
  [QS]  x: RMSE=0.0153  bias=-0.0127  |  y: RMSE=0.0287  bias=-0.0127
  [RL]  x: RMSE=0.0155  bias=+0.0010  |  y: RMSE=0.0084  bias=-0.0005

====== convergence ======
  [QS]  reached=0%
  [RL]  reached=100% @ 0.009s

====== dynamic-phase RMSE ======
  [QS]  mass=6.2366 kg  dcom_x=0.0193 m  dcom_y=0.0296 m
  [RL]  mass=0.6210 kg  dcom_x=0.0165 m  dcom_y=0.0078 m
```

### 3.2 lb=3 ckpt 11000 static  → 见 §2.1

### 3.3 lb=3 ckpt 13000 static

```
====== LOAD MASS (kg) ======
  [QS]  RMSE=2.7720  bias=+1.6725  per_env=2.6137±0.9235
  [RL]  RMSE=0.9063  bias=+0.1132  per_env=0.9055±0.0377

====== CoM DELTA (m) ======
  [QS]  x: RMSE=0.0098  bias=-0.0069  |  y: RMSE=0.0227  bias=-0.0048
  [RL]  x: RMSE=0.0144  bias=+0.0017  |  y: RMSE=0.0109  bias=-0.0002

====== convergence ======
  [QS]  reached=40% @ 14.09s
  [RL]  reached=100% @ 0.009s

====== dynamic-phase RMSE ======
  [QS]  mass=5.7755 kg  dcom_x=0.0173 m  dcom_y=0.0273 m
  [RL]  mass=0.9494 kg  dcom_x=0.0165 m  dcom_y=0.0093 m
```

---

## §4 history_only baseline（ckpt 11000, seed=42）

> 关键对照：use_qs_in_obs=False，encoder 不输入 QS 公式结果

### 4.1 history_only static

```
====== LOAD MASS (kg) ======
  [QS]  RMSE=26.7112  bias=+0.2431  per_env=12.1941±23.7653   ← QS 崩溃
  [RL]  RMSE=0.9783   bias=+0.0635  per_env=0.9450±0.2535     ← RL 仍正常

====== LOAD POSITION (m) ======
  [QS]  com_x RMSE=0.0375  bias=-0.0044
  [QS]  com_y RMSE=0.1259  bias=-0.0100

====== CoM DELTA (m) ======
  [QS]  x: RMSE=0.0085  bias=-0.0042  |  y: RMSE=0.0273  bias=-0.0019
  [RL]  x: RMSE=0.0129  bias=-0.0036  |  y: RMSE=0.0120  bias=+0.0004

====== convergence ======
  [QS]  reached=95% @ 8.23s
  [RL]  reached=100% @ 0.060s

====== dynamic-phase RMSE ======
  [QS]  mass=46.6065 kg  dcom_x=0.0146 m  dcom_y=0.0259 m    ← QS 飞了
  [RL]  mass=0.8357 kg   dcom_x=0.0149 m  dcom_y=0.0122 m
```

### 4.2 history_only walk x=0.5

```
====== LOAD MASS (kg) ======
  [QS]  RMSE=49.0273  bias=+1.7455  per_env=19.2461±45.0913   ← QS 更崩
  [RL]  RMSE=0.9097   bias=+0.1198  per_env=0.9037±0.1047

====== LOAD POSITION (m) ======
  [QS]  com_x RMSE=0.0406  bias=-0.0079
  [QS]  com_y RMSE=0.1438  bias=-0.0274

====== CoM DELTA (m) ======
  [QS]  x: RMSE=0.0106  bias=-0.0050  |  y: RMSE=0.0310  bias=-0.0053
  [RL]  x: RMSE=0.0118  bias=-0.0015  |  y: RMSE=0.0123  bias=+0.0008

====== convergence ======
  [QS]  reached=90% @ 4.91s
  [RL]  reached=100% @ 0.027s

====== dynamic-phase RMSE ======
  [QS]  mass=19.1257 kg  dcom_x=0.0100 m  dcom_y=0.0298 m
  [RL]  mass=0.9015 kg   dcom_x=0.0117 m  dcom_y=0.0122 m
```

---

## §5 待补充实验（占位，未来追加在此节末尾）

- [ ] history_only ckpt 11000 static 第 2/3 seed（验证 RL mass 是否稳定在 0.98）
- [ ] main lb=3 ckpt 11000 第 4 seed（继续降低 mass RMSE 估计的 std）
- [ ] OOD：load_range 临时放宽到 [0, 8] kg，main vs history_only 对比
- [ ] 实验 2 控制增益：tracking error / fall rate / dof_torque_rms 对比
- [ ] 实验 3 抗扰：突发载荷质量变化、push 强度扫描
- [ ] 实验 4 消融：no_load_boost、no_hip_diff 等变体

---

        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_history_only_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=43",
          "--exp_tag=lb3_static_compare_historyonly"
          // 可加 "--headless" 以无渲染运行
        ],

[play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-092703_lb3_static_compare_historyonly_seed43.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=15.6358  bias=+0.7914  per_env=9.5616±12.3715
    [RL]  RMSE=1.0205  bias=+0.4085  per_env=1.0012±0.1975

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0393  bias=-0.0056
    [QS]  com_y RMSE=0.1247  bias=+0.0072

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0102  bias=-0.0048  |  y: RMSE=0.0243  bias=+0.0010
    [RL]  x: RMSE=0.0122   bias=-0.0037   |  y: RMSE=0.0113   bias=+0.0012

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.9500  @ avg 7.1168 s
    [RL]  reached=1.0000  @ avg 0.0330 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=36.8876 kg  dcom_x=0.0242 m  dcom_y=0.0287 m
    [RL]  mass=0.8120 kg  dcom_x=0.0109 m  dcom_y=0.0085 m
[play] =============================================


        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_history_only_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=44",
          "--exp_tag=lb3_static_compare_historyonly"
          // 可加 "--headless" 以无渲染运行
        ],


[play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-091627_lb3_static_compare_historyonly.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=26.7112  bias=+0.2431  per_env=12.1941±23.7653
    [RL]  RMSE=0.9783  bias=+0.0635  per_env=0.9450±0.2535

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0375  bias=-0.0044
    [QS]  com_y RMSE=0.1259  bias=-0.0100

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0085  bias=-0.0042  |  y: RMSE=0.0273  bias=-0.0019
    [RL]  x: RMSE=0.0129   bias=-0.0036   |  y: RMSE=0.0120   bias=+0.0004

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.9500  @ avg 8.2337 s
    [RL]  reached=1.0000  @ avg 0.0600 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=46.6065 kg  dcom_x=0.0146 m  dcom_y=0.0259 m
    [RL]  mass=0.8357 kg  dcom_x=0.0149 m  dcom_y=0.0122 m
[play] =============================================




        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_qs_resi_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=43",
          "--exp_tag=lb3_static_seed43"
          // "--exp_tag=lb3_static_compare_historyonly_seed43"
          // 可加 "--headless" 以无渲染运行
        ],


Saved play data to MATLAB format: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-112609_lb3_static_seed43.mat
[play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-112609_lb3_static_seed43.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=5.3487  bias=+4.7591  per_env=5.3319±0.4237
    [RL]  RMSE=0.9496  bias=+0.3980  per_env=0.9462±0.0802

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0514  bias=-0.0118
    [QS]  com_y RMSE=0.0860  bias=-0.0005

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0153  bias=-0.0132  |  y: RMSE=0.0238  bias=+0.0012
    [RL]  x: RMSE=0.0126   bias=-0.0024   |  y: RMSE=0.0101   bias=+0.0005

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.0000  @ avg   N/A  s
    [RL]  reached=1.0000  @ avg 0.3600 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=8.6055 kg  dcom_x=0.0239 m  dcom_y=0.0318 m
    [RL]  mass=1.0371 kg  dcom_x=0.0160 m  dcom_y=0.0098 m
[play] =============================================

      {
        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_qs_resi_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=44",
          "--exp_tag=lb3_static_seed44"
          // "--exp_tag=lb3_static_compare_historyonly_seed43"
          // 可加 "--headless" 以无渲染运行
        ],




[play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-112829_lb3_static_seed44.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=5.6218  bias=+4.9772  per_env=5.5925±0.5733
    [RL]  RMSE=0.8357  bias=+0.0981  per_env=0.8351±0.0317

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0516  bias=-0.0114
    [QS]  com_y RMSE=0.0876  bias=+0.0166

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0161  bias=-0.0137  |  y: RMSE=0.0246  bias=+0.0063
    [RL]  x: RMSE=0.0137   bias=-0.0038   |  y: RMSE=0.0106   bias=+0.0020

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.0000  @ avg   N/A  s
    [RL]  reached=1.0000  @ avg 0.0060 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=7.8989 kg  dcom_x=0.0253 m  dcom_y=0.0287 m
    [RL]  mass=0.8371 kg  dcom_x=0.0182 m  dcom_y=0.0120 m
[play] =============================================




ood实验负载生成范围变为4到6（训练的时候是2-4）


      {
        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_qs_resi_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=42",
          "--exp_tag=lb3_static_seed42_ood4_6"
          // "--exp_tag=lb3_static_compare_historyonly_seed43"
          // 可加 "--headless" 以无渲染运行
        ],

        [play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-113228_lb3_static_seed42_ood4_6.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=6.6578  bias=+5.2707  per_env=6.6140±0.7622
    [RL]  RMSE=2.0902  bias=-1.2996  per_env=2.0899±0.0360

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0417  bias=-0.0084
    [QS]  com_y RMSE=0.0923  bias=-0.0083

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0164  bias=-0.0122  |  y: RMSE=0.0298  bias=-0.0010
    [RL]  x: RMSE=0.0150   bias=-0.0027   |  y: RMSE=0.0130   bias=-0.0009

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.0000  @ avg   N/A  s
    [RL]  reached=1.0000  @ avg 0.0080 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=7.3386 kg  dcom_x=0.0184 m  dcom_y=0.0288 m
    [RL]  mass=2.2576 kg  dcom_x=0.0181 m  dcom_y=0.0126 m
[play] =============================================

        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_qs_resi_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=43",
          "--exp_tag=lb3_static_seed43_ood4_6"
          // "--exp_tag=lb3_static_compare_historyonly_seed43"
          // 可加 "--headless" 以无渲染运行
        ],


[play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-113433_lb3_static_seed43_ood4_6.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=6.2263  bias=+5.1218  per_env=6.1826±0.7366
    [RL]  RMSE=1.5040  bias=-0.7895  per_env=1.5037±0.0300

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0443  bias=-0.0089
    [QS]  com_y RMSE=0.0886  bias=+0.0006

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0157  bias=-0.0120  |  y: RMSE=0.0290  bias=+0.0019
    [RL]  x: RMSE=0.0141   bias=-0.0034   |  y: RMSE=0.0117   bias=-0.0008

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.0500  @ avg 33.6600 s
    [RL]  reached=1.0000  @ avg 0.6670 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=8.1210 kg  dcom_x=0.0237 m  dcom_y=0.0351 m
    [RL]  mass=1.6177 kg  dcom_x=0.0195 m  dcom_y=0.0127 m
[play] =============================================



        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_history_only_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=42",
          "--exp_tag=lb3_history_static_seed42_ood4_6"
          // "--exp_tag=lb3_static_compare_historyonly_seed43"
          // 可加 "--headless" 以无渲染运行
        ],



[play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-113809_lb3_history_static_seed42_ood4_6.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=20.4247  bias=+1.2215  per_env=9.5436±18.0578
    [RL]  RMSE=2.0111  bias=-1.2206  per_env=2.0108±0.0352

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0379  bias=-0.0007
    [QS]  com_y RMSE=0.1232  bias=-0.0020

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0105  bias=-0.0038  |  y: RMSE=0.0299  bias=+0.0003
    [RL]  x: RMSE=0.0139   bias=-0.0008   |  y: RMSE=0.0133   bias=+0.0006

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.7500  @ avg 13.3080 s
    [RL]  reached=1.0000  @ avg 0.3800 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=22.5073 kg  dcom_x=0.0135 m  dcom_y=0.0282 m
    [RL]  mass=2.1482 kg  dcom_x=0.0181 m  dcom_y=0.0136 m
[play] =============================================

        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_history_only_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=43",
          "--exp_tag=lb3_history_static_seed43_ood4_6"
          // "--exp_tag=lb3_static_compare_historyonly_seed43"
          // 可加 "--headless" 以无渲染运行
        ],


        [play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-114045_lb3_history_static_seed43_ood4_6.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=23.5177  bias=+0.2110  per_env=13.9751±18.9150
    [RL]  RMSE=1.4529  bias=-0.7078  per_env=1.4524±0.0373

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0391  bias=-0.0008
    [QS]  com_y RMSE=0.1113  bias=+0.0038

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0117  bias=-0.0036  |  y: RMSE=0.0265  bias=+0.0018
    [RL]  x: RMSE=0.0133   bias=-0.0031   |  y: RMSE=0.0130   bias=+0.0005

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.7500  @ avg 10.9760 s
    [RL]  reached=1.0000  @ avg 0.0340 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=29.0155 kg  dcom_x=0.0150 m  dcom_y=0.0268 m
    [RL]  mass=1.4003 kg  dcom_x=0.0164 m  dcom_y=0.0130 m
[play] =============================================


      {
        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_history_only_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=42",
          "--exp_tag=lb3_history_static_seed42_ood0_2"
          // "--exp_tag=lb3_static_compare_historyonly_seed43"
          // 可加 "--headless" 以无渲染运行
        ],

        [play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-115852_lb3_history_static_seed42_ood1_2.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=20.3627  bias=+1.6228  per_env=9.1147±18.2088
    [RL]  RMSE=1.4466  bias=+0.0688  per_env=1.4461±0.0344

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0426  bias=-0.0061
    [QS]  com_y RMSE=0.1360  bias=-0.0033

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0081  bias=-0.0052  |  y: RMSE=0.0228  bias=-0.0003
    [RL]  x: RMSE=0.0110   bias=+0.0003   |  y: RMSE=0.0121   bias=-0.0003

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.6000  @ avg 8.3533 s
    [RL]  reached=1.0000  @ avg 0.3540 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=41.9249 kg  dcom_x=0.0180 m  dcom_y=0.0315 m
    [RL]  mass=1.3353 kg  dcom_x=0.0128 m  dcom_y=0.0101 m
[play] =============================================

        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_history_only_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=43",
          "--exp_tag=lb3_history_static_seed43_ood1_2"
          // "--exp_tag=lb3_static_compare_historyonly_seed43"
          // 可加 "--headless" 以无渲染运行
        ],


        [play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=15.6554  bias=+0.8983  per_env=9.6429±12.3331
    [RL]  RMSE=1.1858  bias=-0.6599  per_env=1.1794±0.1234

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0483  bias=-0.0047
    [QS]  com_y RMSE=0.1330  bias=+0.0276

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0097  bias=-0.0046  |  y: RMSE=0.0188  bias=+0.0034
    [RL]  x: RMSE=0.0108   bias=+0.0011   |  y: RMSE=0.0111   bias=-0.0000

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.9000  @ avg 5.2633 s
    [RL]  reached=1.0000  @ avg 0.0330 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=40.5064 kg  dcom_x=0.0249 m  dcom_y=0.0268 m
    [RL]  mass=0.9491 kg  dcom_x=0.0079 m  dcom_y=0.0073 m
[play] =============================================

      {
        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_qs_resi_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=42",
          "--exp_tag=lb3_qs_resi_seed42_ood1_2"
          // "--exp_tag=lb3_static_compare_historyonly_seed43"
          // 可加 "--headless" 以无渲染运行
        ],


        [play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-120657_lb3_qs_resi_seed42_ood1_2.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=4.6406  bias=+4.1289  per_env=4.5994±0.6171
    [RL]  RMSE=1.4135  bias=-0.4359  per_env=1.4132±0.0294

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0594  bias=-0.0169
    [QS]  com_y RMSE=0.0978  bias=-0.0156

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0160  bias=-0.0149  |  y: RMSE=0.0238  bias=-0.0043
    [RL]  x: RMSE=0.0120   bias=+0.0007   |  y: RMSE=0.0124   bias=-0.0003

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.0000  @ avg   N/A  s
    [RL]  reached=1.0000  @ avg 0.0080 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=7.6083 kg  dcom_x=0.0214 m  dcom_y=0.0301 m
    [RL]  mass=1.3412 kg  dcom_x=0.0130 m  dcom_y=0.0103 m
[play] =============================================


      },
      {
        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_qs_resi_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=43",
          "--exp_tag=lb3_qs_resi_seed43_ood1_2"
          // "--exp_tag=lb3_static_compare_historyonly_seed43"
          // 可加 "--headless" 以无渲染运行
        ],


        Saved play data to MATLAB format: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-120830_lb3_qs_resi_seed43_ood1_2.mat
[play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260520-120830_lb3_qs_resi_seed43_ood1_2.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=4.4202  bias=+4.0025  per_env=4.3897±0.5180
    [RL]  RMSE=1.1289  bias=-0.7969  per_env=1.1280±0.0462

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0669  bias=-0.0203
    [QS]  com_y RMSE=0.0905  bias=-0.0008

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0161  bias=-0.0151  |  y: RMSE=0.0225  bias=-0.0002
    [RL]  x: RMSE=0.0113   bias=+0.0006   |  y: RMSE=0.0113   bias=-0.0007

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.0000  @ avg   N/A  s
    [RL]  reached=1.0000  @ avg 0.4130 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=8.2656 kg  dcom_x=0.0231 m  dcom_y=0.0322 m
    [RL]  mass=1.1721 kg  dcom_x=0.0116 m  dcom_y=0.0104 m
[play] =============================================


      {
        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_qs_noresi_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=42",
          "--exp_tag=lb3_qs_noresi_seed42_walk"
          // "--exp_tag=lb3_static_compare_historyonly_seed42"
          // 可加 "--headless" 以无渲染运行
        ],

alk.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=3.1428  bias=+0.8852  per_env=2.5466±1.8417
    [RL]  RMSE=0.9911  bias=+0.1061  per_env=0.9809±0.1420

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0380  bias=-0.0113
    [QS]  com_y RMSE=0.1365  bias=-0.0063

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0097  bias=-0.0060  |  y: RMSE=0.0281  bias=-0.0020
    [RL]  x: RMSE=0.0144   bias=-0.0015   |  y: RMSE=0.0111   bias=+0.0000

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.9500  @ avg 11.6168 s
    [RL]  reached=1.0000  @ avg 0.0000 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=2.3183 kg  dcom_x=0.0095 m  dcom_y=0.0262 m
    [RL]  mass=0.9777 kg  dcom_x=0.0144 m  dcom_y=0.0107 m
[play] =============================================


      {
        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_qs_noresi_load_boost_3",
          "--checkpoint",
          "11000",
          "--seed=42",
          "--exp_tag=lb3_qs_noresi_seed42_static"
          // "--exp_tag=lb3_static_compare_historyonly_seed42"
          // 可加 "--headless" 以无渲染运行
        ],


[play] saved: /home/xu/limx_rl/pointfoot-legged-gym/logs/wheelfoot_flat/WF_TRON1A/exported/play_data_20260521-093018_lb3_qs_noresi_seed42_static.mat
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=2.0518  bias=+0.5989  per_env=1.7845±1.0126
    [RL]  RMSE=0.9944  bias=+0.0947  per_env=0.9840±0.1430

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0396  bias=-0.0034
    [QS]  com_y RMSE=0.1110  bias=+0.0227

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0076  bias=-0.0027  |  y: RMSE=0.0218  bias=+0.0040
    [RL]  x: RMSE=0.0135   bias=-0.0032   |  y: RMSE=0.0094   bias=-0.0017

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.9000  @ avg 11.3178 s
    [RL]  reached=1.0000  @ avg 0.0000 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=2.3821 kg  dcom_x=0.0101 m  dcom_y=0.0226 m
    [RL]  mass=1.0537 kg  dcom_x=0.0128 m  dcom_y=0.0091 m
[play] =============================================



      {
        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_qs_resi_load_boost_3_seed_42",
          "--checkpoint",
          "11000",
          "--seed=42",
          "--exp_tag=lb3_qs_resi_seed42_static"
          // "--exp_tag=lb3_static_compare_historyonly_seed42"
          // 可加 "--headless" 以无渲染运行
        ],

[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=1.6110  bias=+0.9428  per_env=1.5899±0.2598
    [RL]  RMSE=3.8144  bias=+3.2820  per_env=3.7943±0.3913

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0308  bias=-0.0041
    [QS]  com_y RMSE=0.1053  bias=-0.0182

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0081  bias=-0.0047  |  y: RMSE=0.0197  bias=-0.0051
    [RL]  x: RMSE=0.0186   bias=-0.0107   |  y: RMSE=0.0201   bias=-0.0030

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=0.9000  @ avg 9.4478 s
    [RL]  reached=0.0500  @ avg 15.0800 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=3.2744 kg  dcom_x=0.0159 m  dcom_y=0.0264 m
    [RL]  mass=4.5663 kg  dcom_x=0.0249 m  dcom_y=0.0282 m
[play] =============================================

      {
        "name": "Python 调试程序: Play",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/legged_gym/scripts/play.py",
        "args": [
          "--task=wheelfoot_flat",
          "--load_run",
          "exper_qs_resi_load_boost_3_seed_42",
          "--checkpoint",
          "11000",
          "--seed=42",
          "--exp_tag=lb3_qs_resi_seed42_walk"
          // "--exp_tag=lb3_static_compare_historyonly_seed42"
          // 可加 "--headless" 以无渲染运行
        ],
[play] ============= experiment metrics =============
  num_envs = 20
  [QS] = Model C 解析公式  |  [RL] = Encoder 残差网络

  ====== LOAD MASS (kg) ======
    [QS]  RMSE=1.5120  bias=+0.6897  per_env=1.4920±0.2449
    [RL]  RMSE=3.6098  bias=+3.0750  per_env=3.5924±0.3538

  ====== LOAD POSITION in body frame (m) — [QS] only task ======
    [QS]  com_x RMSE=0.0373  bias=-0.0062
    [QS]  com_y RMSE=0.1229  bias=-0.0421

  ====== CoM DELTA = 载荷引起的 CoM 偏移 (m) ======
    [QS]  x: RMSE=0.0089  bias=-0.0046  |  y: RMSE=0.0240  bias=-0.0096
    [RL]  x: RMSE=0.0220   bias=-0.0147   |  y: RMSE=0.0229   bias=-0.0086

  ====== convergence (mass < 0.5 kg, hold ≥ 0.5 s) ======
    [QS]  reached=1.0000  @ avg 10.5350 s
    [RL]  reached=0.5000  @ avg 17.0140 s

  ====== dynamic-phase RMSE (|vx| > 0.15 m/s) ======
    [QS]  mass=1.4629 kg  dcom_x=0.0086 m  dcom_y=0.0227 m
    [RL]  mass=3.6103 kg  dcom_x=0.0216 m  dcom_y=0.0214 m
[play] =============================================
是不是绘图参数还是哪里学错了




# 第二部分：对比表格（**bug 修复后干净数据，2026-05-21~22**）

> **数据来源说明**：本部分全部来自 [exported/](../logs/wheelfoot_flat/WF_TRON1A/exported/) 下的
> `play_data_*_lb*_qs[01]_resid[01]_*_ckpt11000_load*-*.mat`（共 18 个干净 .mat）。
> 这些文件全部 ≥ 2026-05-21 22:21 收集，过了以下修复：
> 1. play.py residual-mode 双加 baseline bug 修复
> 2. play.py 自动从 saved cfg 同步 `use_qs_in_obs` / `use_load_residual_estimation` / encoder dim
> 3. play.py 自动 exp_tag（含 lb / qs / resid / cmd / seed / ckpt / load 范围）
> 4. auto exp_tag 从 saved train_cfg.json 同步 `extra_loss_load_boost`（lb=6 不再被误标 lb=3）
> 5. seed_43 新训 ckpt（`exper_qs_resi_load_boost_3_seed_43`）加入 multi-train-seed 对比
>
> **原 §6.1–§6.11 已废弃**（main residual 行的 RL mass 数字偏高一个 baseline 量级）。

## §6 架构对比（ckpt 11000，play_seed=42，trimesh L0，num_envs=20）

5 种 policy（main 两个训练 seed + main lb=6 + direct + history_only）在
**in-distribution load=[2,4] kg** 下的完整对比。

### 6.1 LOAD MASS RMSE [RL encoder] (kg) — encoder 实际估计精度

| Policy | static | walk vx=0.5 |
|---|---|---|
| main lb=3 (seed_42 ckpt) | **0.9004** | **0.8611** |
| main lb=3 (seed_43 ckpt) | 1.0229 | 1.0214 |
| main lb=6 | 0.9548 | 0.9840 |
| direct (qs=1, resid=0) | 0.9407 | 0.9389 |
| history_only (qs=0) | 0.8665 | 0.8823 |

**观察**：5 种 policy 的 RL mass RMSE 全部落在 **0.86–1.02 kg** 区间，差异 < 18%。
**残差架构 (main) 跟 baseline (history_only / direct) 几乎打平**——QS 先验对
encoder 精度的边际改善不显著。这是 paper 必须 honest 报告的 negative finding。

### 6.2 LOAD MASS RMSE [QS Model C] (kg) — analytical formula 在不同 policy 下的表现

| Policy | static | walk vx=0.5 |
|---|---|---|
| main lb=3 (seed_42) | **1.6110** | **1.5120** |
| main lb=3 (seed_43) | 1.9046 | 2.6028 |
| direct (qs=1, resid=0) | 2.0518 | 3.1428 |
| main lb=6 | 3.9850 | 4.1478 |
| history_only (qs=0) | **20.3701** ⚠ | **14.1398** ⚠ |

**核心发现**：QS analytical formula 的精度**强烈依赖 policy 训练时是否见过 QS**：

- **qs in obs（main + direct）**：QS RMSE 1.5–4.1 kg（合理量级）
- **qs not in obs（history_only）**：QS RMSE **14–20 kg**（崩盘 5–13×）

机制：main / direct 训练时 actor 直接读 QS 12 维特征，**隐式学会维持"QS 友好"姿态**
（cos_thigh 不近零，分母不爆炸）。history_only 没有这个约束，policy 偶发让
cos_thigh→0，Model C 分母数值爆破，给出几十 kg 的伪估计。

**关键 sub-finding**：lb=3 (1.6 kg) < direct (2.0 kg) < lb=6 (4.0 kg) — **residual
结构 + lb=3 给最强的 QS 友好姿态约束**。lb=6 反而 over-weight encoder loss，policy 受到
的 implicit shaping 减弱，QS 退化到 4 kg。

### 6.3 CoM DELTA RMSE [RL] (m) — encoder 在 com 估计上的精度

| Policy | static dcom_x | static dcom_y | walk dcom_x | walk dcom_y |
|---|---|---|---|---|
| main lb=3 (seed_42) | 0.0103 | **0.0197** | 0.0111 | 0.0240 |
| main lb=3 (seed_43) | 0.0112 | 0.0240 | 0.0114 | 0.0272 |
| main lb=6 | 0.0128 | 0.0240 | 0.0129 | 0.0310 |
| direct | **0.0092** | 0.0218 | **0.0109** | 0.0281 |
| history_only | 0.0117 | 0.0275 | 0.0120 | 0.0293 |

**观察**：
- **dcom_x 上 direct 最优**（0.0092），**dcom_y 上 main lb=3 seed_42 最优**（0.0197）
- 跟 mass 类似，差异在 ~30% 以内（0.0092 ~ 0.0128），**架构差异在 com 上也不显著**
- history_only 在 dcom_y 上略差（0.0275–0.0293），其余指标基本打平

### 6.4 lb=3 多 train-seed 复测（in-dist load=[2,4]，play_seed=42）

| Train seed | static RL | static QS | walk RL | walk QS |
|---|---|---|---|---|
| seed_42 ckpt | 0.9004 | 1.6110 | 0.8611 | 1.5120 |
| seed_43 ckpt | 1.0229 | 1.9046 | 1.0214 | 2.6028 |
| **mean ± std** | **0.96 ± 0.06** | **1.76 ± 0.15** | **0.94 ± 0.08** | **2.06 ± 0.55** |

**train-seed 方差**：RL mass ~6–8%，QS mass ~9–27%。Train seed 影响 **encoder 学到的
representation**（→ RL 数字差异 ~12%）和 **policy 学到的姿态分布**（→ QS 数字差异
大得多，walk 时 ±0.55 kg）。

**paper 报告建议**：`main lb=3: RL mass = 0.96 ± 0.06 kg, QS mass = 1.76 ± 0.15 kg
(2 train seeds, 1 play seed each)`. 想要更稳的 std 报告，需要 train ≥3 seeds（seed_44
未跑）。

### 6.5 lb=3 (seed_42 ckpt) 多 play-seed 复测（in-dist load=[2,4] static）

| Play seed | RL mass | RL bias | QS mass | QS bias |
|---|---|---|---|---|
| 42 | 0.9004 | +0.011 | 1.6110 | +0.943 |
| 43 | 1.1517 | +0.323 | 2.1470 | +0.993 |
| 44 | 0.8768 | +0.203 | 1.6039 | +1.044 |
| **mean ± std** | **0.98 ± 0.12** | +0.18 | **1.79 ± 0.25** | +0.99 |

**play-seed 方差**：RL mass ±12%，QS mass ±14%。20 envs 不是"同一实验 20 次"，是
20 个不同的 (load mass, push timing, friction 等) 组合的样本——换 seed 等于重新抽
20 个组合。

**train-seed 方差 (§6.4) vs play-seed 方差 (§6.5) 量级相当**（6–8% vs 12%）。Paper
报数字时两个方差源都要 disclose。

**对比旧 §6.5b（bug 时代）**：旧 3 个 seed = 0.86/1.40/0.86，std 31%，含一个 1.40
outlier。**bug 修复后没复现**，新 3 seed 在 0.87–1.15 区间，std 12%，没有
outlier——**老 1.40 是 bug + 抽样耦合产物**。

### 6.6 OOD 实验：load 分布外（main lb=3 seed_42 + seed_43）

训练分布 load ∈ [2, 4] kg，测试：
- **OOD-low** [1, 2] kg（避开 `position_zero_mass_threshold=1.0` 截断）
- **OOD-high** [4, 6] kg

| Train seed | Cond | [RL] mass RMSE | [RL] bias | [QS] mass RMSE | [QS] bias |
|---|---|---|---|---|---|
| seed_42 | in-dist [2,4] stat | 0.9004 | +0.011 | 1.6110 | +0.943 |
| seed_42 | in-dist [2,4] walk | 0.8611 | +0.051 | 1.5120 | +0.690 |
| seed_42 | OOD-low [1,2] stat | **1.4406** | **−0.895** | 1.5200 | +1.099 |
| seed_42 | OOD-low [1,2] walk | 1.4562 | −0.841 | 1.3110 | +0.761 |
| seed_42 | OOD-high [4,6] stat | **1.8903** | **−1.124** | 1.8637 | +0.664 |
| seed_42 | OOD-high [4,6] walk | 1.8819 | −1.125 | 1.9304 | +0.289 |
| seed_43 | in-dist [2,4] stat | 1.0229 | −0.541 | 1.9046 | +0.595 |
| seed_43 | in-dist [2,4] walk | 1.0214 | −0.524 | 2.6028 | +0.599 |
| seed_43 | OOD-low [1,2] stat | 1.2913 | −0.440 | 1.8975 | +0.795 |
| seed_43 | OOD-low [1,2] walk | 1.1897 | −0.174 | 1.5242 | +0.888 |
| seed_43 | OOD-high [4,6] stat | **2.4240** | **−1.717** | 2.4460 | +0.383 |
| seed_43 | OOD-high [4,6] walk | 2.4193 | −1.703 | 2.8558 | +0.481 |

**核心观察**：

**(O1) RL encoder 在 OOD 两端都退化 + 强负 bias**
- OOD-high：RL 从 in-dist 0.9 → OOD 1.9–2.4，bias −1.12/−1.72（饱和到训练上界 ~4 kg）
- OOD-low：RL 从 in-dist 0.9 → OOD 1.3–1.5，bias −0.4/−0.9（**部分误识别为"无载荷"**）

**(O2) OOD-low 的"双峰失败"机制**
- 训练数据 bimodal：load 在体时 truth ∈ [2, 4]，不在体时 truth = 0
- OOD 真值 1.2 kg 落在两峰之间，encoder 倾向 snap 到 "0 桶"
- ≠ 简单"clamp 到 [2, 4]"

**(O3) QS analytical 在 OOD 两端都稳定（**核心 paper-worthy 发现**）**
- 两个 train seed × 两个 OOD 方向 × 两个 cmd：QS RMSE 全部 1.3–2.9 kg
- **跟 in-dist QS (1.5–2.6) 持平**——物理公式天然没有"训练分布"概念
- 这是 RL encoder 学不来的属性

**(O4) train-seed 影响 OOD-high 比 OOD-low 更大**
- OOD-high：seed_42=1.89, seed_43=2.42（差 28%）
- OOD-low：seed_42=1.44, seed_43=1.29（差 11%）
- seed_43 ckpt OOD-high 失败更严重，但 OOD-low 略好——**没有 universal best seed**

### 6.7 Hybrid estimator 互检价值

**Main lb=3 policy 下，RL+QS 两个估计的差异随载荷分布的关系**：

| 工况 | RL est ≈ | QS est ≈ | |RL−QS| ≈ | 真值 ≈ | RL 错误模式 | QS 是否可信 |
|---|---|---|---|---|---|---|
| in-dist [2,4] | 3 | 3.5 | 0.5 | 3 | 无 | ✓ |
| OOD-low [1,2] | 0.5 | 2.5 | **2.0** | 1.5 | 误判无载荷 | ✓ |
| OOD-high [4,6] | 3.8 | 5.5 | **1.7** | 5 | 上限饱和 | ✓ |

**互检逻辑**：当 |RL−QS| > 1.5 kg → 触发 OOD 警报 → 启动保守策略 / 降速 / 人介入。

**对比 history_only**：QS 数字本身就是 14–20 kg（in-dist 和 OOD 都崩），
**|RL−QS| 永远 > 13 kg**，警报始终响——**discriminative power = 0**，
只能告诉你"出问题了"但区分不出"问题在 in-dist 还是 OOD"。

**这是 main method 相对 baseline 的真正、可量化的优势**——不是 single-estimator
精度，是 **hybrid 系统的诊断能力**。

### 6.8 动态相位 vs 收敛速度

| Policy | RL conv_time (s) static | QS conv_time (s) static | RL dyn-phase mass (kg) walk |
|---|---|---|---|
| main lb=3 (seed_42) | **0.001** | 9.45 | 0.86 |
| main lb=3 (seed_43) | 0.219 | 14.40 | 1.12 |
| main lb=6 | 0.000 | 18.80 | 1.07 |
| direct | 0.000 | 11.32 | 1.05 |
| history_only | 0.122 | 9.05 | 0.91 |

**观察**：
- RL encoder 收敛极快（多数 < 0.5s），跨架构差异不大
- QS analytical formula 慢得多（9–19s），因为 1/(joint_geometry) 项数值噪声大需要长时间平均
- dynamic phase（walk 时）跟 static phase 同量级，**encoder 在运动中也保持稳定**

---

# 第三部分：累计关键结论

> 本部分全部基于 2026-05-21~22 干净数据（§6 表）。旧版结论 A–H 已经用新数据
> **完整重写**——bug 修复后部分结论数字改善，但 **qualitative direction 全部保留**：
> RL encoder 跨架构打平 / QS 强依赖 policy / lb=3 优于 lb=6 / OOD 双向饱和 / hybrid 可诊断性。

## 结论 A：残差架构（QS in obs）**没有显著改善 encoder 估计精度**

**证据**：§6.1 — 5 种 policy 的 RL mass RMSE 全部在 0.86–1.02 kg 区间（差 < 18%）；
§6.3 dcom_y 也在 0.0197–0.0275 m（差 < 40%）。`history_only` 反而在某些指标上略胜
（mass=0.87，最低）。

**意义**：原 paper 假设"QS 先验帮助 encoder 学更准"**被 5 种架构 × 2 cmd 的 10 行
数据明确否定**。Encoder 自己能从 obs history 学到一样好。这是 paper 的 honest negative
finding。

## 结论 B：QS analytical formula 强烈依赖 policy 是否见过 QS（核心 paper finding）

**证据**：§6.2 — 在 in-dist [2,4]：
- qs in obs (main + direct)：QS RMSE 1.5–4.1 kg（合理）
- qs not in obs (history_only)：QS RMSE **14–20 kg**（崩盘 5–13×）

**机制**：main / direct 的 actor 直接读 QS 12 维特征 → 隐式学会维持"QS 友好"姿态
（cos_thigh 不近零）→ Model C 分母不爆炸。history_only 没有这个约束，policy 偶发让
cos_thigh→0，QS 数值爆破。

**意义**：QS 注入 obs 的**真正价值不在改善 encoder，而在塑造 policy**——让 analytical
formula 在部署时仍可作为可靠的 sanity check / safety fallback。**这是这篇 paper 的
new contribution**。

## 结论 C：play-to-play 方差比之前认为的小得多

**新数据**（§6.5，bug 修复后）：3 个 play seed RL mass = 0.90 / 1.15 / 0.88，
**mean = 0.98 ± 0.12 kg（rel std ≈ 12%）**。

**对比旧 §6.5b 数据**：旧 3 seed = 0.86 / 1.40 / 0.86（含 1.40 outlier），rel std 31%。
**bug 修复 + auto-tag 后 outlier 没复现**。

**意义**：mass 数字 paper 报 ±std 仍必要，但量级 ~12%（不是 30%）。CoM 极稳（rel std < 3%），
单 seed 可信。Train-seed 方差 (§6.4: 6–8%) 跟 play-seed 方差量级相当，两者都要 disclose。

## 结论 D：lb=3 显著优于 lb=6（QS 上差距 ~2.5×）

**新证据**（§6.1 + §6.2，bug 修复后）：
| 指标 | lb=3 | lb=6 |
|---|---|---|
| RL mass (static) | 0.90 | 0.95 |
| **QS mass (static)** | **1.61** | **3.99** |
| QS mass (walk) | 1.51 | 4.15 |
| dcom_y (static) | 0.0197 | 0.0240 |

**lb=3 在所有指标上都赢，QS 上赢了 2.5×**。机制：lb=6 over-weight encoder aux loss
→ policy 受到的 implicit shaping 减弱（更多优化能量花在 encoder 训练而非 actor 行为）
→ "QS 友好姿态"约束变弱 → Model C 数字退化。

**意义**：load_boost = 3 是 final 超参数。lb=6 作为 ablation 写进 appendix。

## 结论 E：paper narrative

**原方向**（被否定）：QS+RL hybrid 比 baseline 更准。

**新方向**（数据支持）：
> "QS 注入 obs 的真正作用是 **implicit policy regularization**：在塑造 policy 维持 QS
> 友好姿态的同时，让 analytical formula 在部署时仍可用作 sanity check。Encoder 精度
> 几乎不变，但 hybrid 系统的诊断能力质变。"

**Paper 章节安排建议**：
- §1 Intro：load estimation 的难点 + hybrid 思路 motivation
- §3 Method：主架构 (main lb=3) + ablations (lb=6, direct, history_only)
- **§4.1 Main result（重写）**：5 architectures × {static, walk} × {in-dist, OOD-low, OOD-high}
  的完整 mass/CoM 表 — honest 报告 RL 几乎打平
- **§4.2 核心 finding**：QS analytical formula 依赖 policy structure（结论 B）—
  这是 paper 的 main contribution
- **§4.3 OOD 章节**：QS 跨分布稳定 vs RL 两端饱和 → hybrid 互检价值
- §5 Discussion：implicit regularization 的更广 implications

参考类似 nuanced result paper：
- Lee et al. "Learning quadrupedal locomotion over challenging terrain" (Science Robotics 2020)
- Margolis et al. "Walk these ways" (CoRL 2022)
- Pinto et al. "Asymmetric Actor Critic for Image-Based Robot Learning"

## 结论 F：OOD 下 RL encoder 两端饱和，QS 公式跨分布稳定

**新证据**（§6.6，干净 multi-seed）：

| Cond | RL mass | RL bias | QS mass | QS bias |
|---|---|---|---|---|
| in-dist [2,4] | 0.91 ± 0.07 | +0.03 | 1.81 ± 0.55 | +0.79 |
| OOD-low [1,2] | 1.37 ± 0.13 | −0.46 | 1.62 ± 0.27 | +0.89 |
| OOD-high [4,6] | 2.16 ± 0.30 | −1.42 | 2.27 ± 0.40 | +0.45 |

**(1) RL 双向欠估**：
- OOD-high (load 5)：RL ≈ 3.6（饱和到训练上限 4）→ bias −1.4
- OOD-low (load 1.5)：RL ≈ 1.0（**部分误识别为"无载荷"**）→ bias −0.5
- 后者机制是训练数据 **bimodal**（在体[2,4] vs 不在体 0）让 1.5 kg 落到 "0 桶"

**(2) QS 跨分布稳定**：
- QS mass RMSE in-dist 1.8 → OOD 1.6, 2.3（基本持平）
- 物理公式天然没有"训练分布"概念，**naturally OOD-robust**

**(3) 架构差异 collapse**：OOD 下 main 和 history_only 的 RL 数字几乎打平
（§6.6 旧表）—— failure mode 不区分架构。

**意义**：之前希望 main method 在 OOD 上显著优于 baseline，**被双向数据否定**。但
RL+QS 在 OOD 上的 divergence 给了 hybrid 系统**可诊断性**（见结论 G）。

## 结论 G：Hybrid estimator 可诊断性 — main 真正的差异化优势

**新证据**（§6.7 表）：

| 工况 | RL est | QS est | \|RL−QS\| | 是 OOD 吗 |
|---|---|---|---|---|
| in-dist [2,4] | 3 | 3.5 | 0.5 | no |
| OOD-low [1,2] | 0.5 | 2.5 | **2.0** | yes |
| OOD-high [4,6] | 3.8 | 5.5 | **1.7** | yes |

**判别规则**：|RL−QS| > 1.5 kg → OOD detected → 触发保守策略。

**对比 history_only**：QS 永远是 14–20 kg，|RL−QS| 永远 > 13 kg → 警报始终响 →
**discriminative power = 0**。

**意义**：这是 main method 相对 baseline 唯一、**可量化**的优势。Paper 的"main result"
应该强调这个，而不是 single-estimator 精度。

## 结论 H：encoder 不受 policy reward 直接驱动（机制澄清，沿用）

证据：[wheelfoot_flat_config.py:401](legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py#L401)
`output_detach = True`，encoder 输出在喂 actor 前 `.detach()`，policy 梯度不流回 encoder。

那 main 和 history_only 的 policy 为什么差异这么大？通过两个 **间接** 机制：
1. **obs 直接耦合**：QS 12 维直接进 actor obs → actor 被 reward 推着用 QS → 学会
   "维持 QS 友好姿态"是 actor 自己学的，跟 encoder 无关
2. **训练数据耦合**：policy 决定 rollout 轨迹 → 决定 encoder 看到的 obs 分布 / aux loss
   target 分布 → encoder representation 不同

**澄清易混淆点**：encoder 训练目标是纯 supervised（aux loss against truth），不被
reward 影响；但 actor 和 encoder 都隐式被 policy-induced state distribution 耦合。

## 结论 I：train-seed 方差 ≈ play-seed 方差（新发现）

**证据**（§6.4 vs §6.5）：
- Train-seed 方差（seed_42 ckpt vs seed_43 ckpt）：RL mass ±6–8%
- Play-seed 方差（同 ckpt，3 个 play seed）：RL mass ±12%

意外发现：OOD-high 上 **seed_43 比 seed_42 差 28%**（2.42 vs 1.89 kg），
OOD-low 上反过来 seed_43 更好。**没有 universal best seed**，paper 的 OOD 数字必须
至少 2 train seed 的 mean±std。

**实操**：seed_44 训练未跑（GPU 时间限制）。若 paper 需要 3 train seed，再开一次
~6h 训练即可。

---

# 第三部分 B：Paper 主表（直接用于 §4.1 main result）

下面三张表是 paper "main result" section 直接可贴的内容。数字来自 §6.1–§6.6，
默认 ckpt 11000、play_seed=42、20 play envs、trimesh L0、load 持续 [30,40]s 间隔 [50,60]s。

### 表 1：5 architectures × 2 commands × 2 metrics（in-distribution）

| Architecture | use_qs | use_resid | lb | RL mass RMSE (kg) | QS mass RMSE (kg) |
|---|---|---|---|---|---|
|  |  |  |  | static / walk | static / walk |
| **main lb=3 (2 seeds)** ★ | ✓ | ✓ | 3 | **0.96 ± 0.06 / 0.94 ± 0.08** | **1.76 ± 0.15 / 2.06 ± 0.55** |
| main lb=6 | ✓ | ✓ | 6 | 0.95 / 0.98 | 3.99 / 4.15 |
| direct (no residual) | ✓ | ✗ | 3 | 0.94 / 0.94 | 2.05 / 3.14 |
| history_only | ✗ | – | 3 | 0.87 / 0.88 | **20.37 / 14.14** ⚠ |

★ = recommended main method (paper §4.1)
⚠ = QS formula 崩盘（cos_thigh→0 时分母爆炸）

**Caption**：In-distribution load mass estimation accuracy.
RL encoder accuracy is comparable across all five architectures (RMSE within 0.86–1.02 kg),
but the QS analytical formula's accuracy **strongly depends on whether the policy was
trained with QS in its observation**: with QS in obs (main + direct), the QS RMSE stays
at 1.5–4.1 kg; without (history_only), it explodes to 14–20 kg. This demonstrates that
QS-in-obs acts as an **implicit policy regularizer**, maintaining QS-friendly poses that
keep the analytical formula in its valid regime, rather than directly improving encoder accuracy.

### 表 2：OOD generalization (main lb=3, 2 train seeds, mean ± std)

| Load distribution | Cmd | RL mass RMSE (kg) | RL bias (kg) | QS mass RMSE (kg) |
|---|---|---|---|---|
| in-dist [2, 4] | static | 0.96 ± 0.06 | +0.01 ± 0.28 | 1.76 ± 0.15 |
| in-dist [2, 4] | walk | 0.94 ± 0.08 | −0.24 ± 0.29 | 2.06 ± 0.55 |
| **OOD-low [1, 2]** | static | **1.37 ± 0.07** | **−0.67 ± 0.23** | 1.71 ± 0.20 |
| **OOD-low [1, 2]** | walk | **1.32 ± 0.13** | **−0.51 ± 0.34** | 1.42 ± 0.11 |
| **OOD-high [4, 6]** | static | **2.16 ± 0.27** | **−1.42 ± 0.30** | 2.15 ± 0.29 |
| **OOD-high [4, 6]** | walk | **2.15 ± 0.27** | **−1.41 ± 0.29** | 2.39 ± 0.46 |

**Caption**：Out-of-distribution generalization. RL encoder accuracy degrades 2–3×
out of the training load range, with strong negative bias in both directions
(saturation at upper bound for OOD-high; misclassification as "no-load" for OOD-low).
In contrast, the **QS analytical formula remains stable** (RMSE 1.4–2.4 kg) across
the entire 1–6 kg range, because the physical model has no learned distribution.
This complementarity motivates the hybrid estimator.

### 表 3：Hybrid estimator diagnostic value

| Architecture | in-dist \|RL−QS\| | OOD-low \|RL−QS\| | OOD-high \|RL−QS\| | OOD detection |
|---|---|---|---|---|
| **main lb=3** | < 1 kg | **~2 kg** | **~1.7 kg** | ✓ discriminative |
| history_only | > 13 kg | > 13 kg | > 13 kg | ✗ always-on |

**Caption**：Diagnostic value of the hybrid estimator. For the proposed main method
(QS in obs + residual learning), the |RL−QS| disagreement provides a discriminative
OOD signal (< 1 kg in distribution, ≥ 1.7 kg out of distribution). For the
history_only baseline, the QS formula's intrinsic failure mode renders the
disagreement signal non-informative.

### 文字版 takeaway（paper §4 结尾）

> Our experiments yield three findings:
> (a) The RL encoder's accuracy is largely insensitive to architecture choice;
>     residual learning over a QS prior does not significantly improve encoder accuracy
>     (Table 1, all RMSE within 0.86–1.02 kg).
> (b) The QS analytical formula's accuracy depends critically on whether the policy
>     was trained with QS in observation. With QS-in-obs (our method), the QS RMSE
>     stays within 1.5–4.1 kg; without, it explodes to 14–20 kg. This indicates
>     QS-in-obs acts as an implicit policy regularizer rather than an information
>     channel to the encoder.
> (c) Outside the training load range, the RL encoder saturates while the QS formula
>     remains stable. This complementarity enables the hybrid estimator to **detect
>     its own OOD failure** via the |RL−QS| disagreement signal, a property the
>     baseline lacks.

---

# 第四部分：版本和 commit 关联

| 日期 | git commit | 关键变更 |
|---|---|---|
| 2026-05-19（前） | (前版) | 初始 main method, lb=6 训练完成 |
| 2026-05-19 | (前版) | lb=3 main method 训练完成；play.py 加 encoder logging |
| 2026-05-20 | 08d4c40 | 加 use_qs_in_obs flag；history_only 训练完成 |
| 2026-05-20 | (待 commit) | 修 task_registry seed bug；OOD [4,6] kg 实验完成（**bug 时代数据**） |
| 2026-05-21 | f32018d | direct (no residual) 训练完成；add use_residual_learning flag |
| 2026-05-21 | (work tree) | **play.py 修复 residual-mode 双加 baseline bug**（§6 起所有数据干净） |
| 2026-05-21 | (work tree) | play.py 自动从 saved cfg 同步 use_qs_in_obs / encoder dim（修 ckpt shape mismatch）|
| 2026-05-21 | (work tree) | play.py 加 auto exp_tag (lb / qs / resid / cmd / seed / ckpt / load) |
| 2026-05-21 | (work tree) | play.py 加 run verification block + save 进 .mat meta |
| 2026-05-21 | (work tree) | logger.py 加 save_dir 支持 PNG 自动落盘 |
| 2026-05-21 | (work tree) | helpers.py 加 --cmd_vx/vy/yaw --load_mass_min/max --exit_after_save |
| 2026-05-21 22:41 | — | seed_43 训练启动（exper_qs_resi_load_boost_3_seed_43）|
| 2026-05-21 22:21 | — | 14-batch play 完成（**§6 全部干净数据**）|
| 2026-05-22 03:58 | — | seed_43 训练完成（ckpt 16000）|
| 2026-05-22 | (待 commit) | play.py 加 extra_loss_load_boost 从 saved cfg 同步（lb=6 auto-tag 修复）|
| 2026-05-22 11:43 | — | seed_43 ckpt × 6 conditions play 完成（**§6.4 + §6.6 seed_43 行**）|
| 2026-05-22 | (待 commit) | **notebook 全面重写 §6 + 结论 + Paper 主表**（本次）|

---

# 第五部分：未来记录格式建议

新加 play 数据时建议格式：

```
### §X.X <method> <condition> (ckpt <N>, seed=<S>, run #<R>)

```（直接粘贴 [play] ===== experiment metrics ===== 那一整段）

> 备注（可选）：
> - exp_tag = ...
> - 想用这个数据验证什么 hypothesis
> - 异常观察
```

新加结论 / 假说时：直接在第三部分追加 "结论 X" 段落。

新加 commit 关联时：在第四部分追加一行。
