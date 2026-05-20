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


# 第二部分：对比表格

## §6 三种 policy 在 ckpt 11000 的完整对比

> 数据来源：§2.1（main lb=3 static），§2.4（main lb=6 static），§4.1（history_only static），§2.6（main lb=3 walk），§2.5（main lb=6 walk），§4.2（history_only walk）

### 6.1 LOAD MASS RMSE [RL encoder] (kg) — encoder 实际估计精度

| Policy | static | walk@0.5 |
|---|---|---|
| main lb=6 | 0.97 | 1.07 |
| **main lb=3** | **0.86 / 1.40 / 0.86** (3 seed) | **0.94** |
| **history_only** | **0.98** | **0.91** |

**观察**：三种 policy 在 RL mass 上**几乎打平**（0.86–1.07 范围），都跟 Model C 标定的天花板 0.77 kg 接近。残差架构（lb=3 main）**没有显著优于 baseline（history_only）**。

### 6.2 LOAD MASS RMSE [QS Model C] (kg) — Model C 在不同 policy 下的表现

| Policy | static | walk@0.5 |
|---|---|---|
| main lb=6 | 3.73 | 7.07 |
| main lb=3 | 5.64 / 5.19 / 5.75 | 5.88 |
| **history_only** | **26.71** | **49.03** ⚠️ |

**观察**：**核心发现** —— 当 policy 训练时 QS 不在 obs 里（history_only），Model C 公式在 play 时彻底崩坏（5-8× 恶化）。因为 history_only 的 policy 没有"维持 QS 友好姿态"的隐式动机，cos_thigh 等几何量趋零时 Model C 分母爆炸。

### 6.3 CoM DELTA RMSE [RL] (m)

| Policy | static dcom_x | static dcom_y | walk dcom_x | walk dcom_y |
|---|---|---|---|---|
| main lb=6 | 0.0156 | 0.0114 | 0.0162 | 0.0128 |
| **main lb=3** | 0.0135 | **0.0103** | 0.0150 | 0.0116 |
| history_only | 0.0129 | 0.0120 | 0.0118 | 0.0123 |

**观察**：dcom_y 上 main lb=3 略胜（0.0103 < 0.0120），但优势小。dcom_x 上 history_only 略胜。整体 **encoder 精度差异在 ~15% 内**。

### 6.4 收敛速度 conv_time [RL] (s)

| Policy | static | walk@0.5 |
|---|---|---|
| main lb=6 | 0.12 | 0.086 |
| main lb=3 | 0.08 / 0.43 / 0.035 | 0.19 |
| history_only | 0.060 | 0.027 |

**观察**：所有 policy 的 RL encoder 都在 < 0.5s 内收敛，**history_only 反而最快**（0.027s, 0.060s）。

### 6.5 同 policy play-to-play 方差（lb=3 ckpt 11000 static, 3 seed）

| 指标 | 3 次值 | mean | std | rel std |
|---|---|---|---|---|
| RL mass | 0.86, 1.40, 0.86 | 1.04 | 0.31 | **30%** ⚠️ |
| RL dcom_x | 0.0135, 0.0134, 0.0136 | 0.0135 | 0.0001 | 0.7% |
| RL dcom_y | 0.0103, 0.0109, 0.0108 | 0.0107 | 0.0003 | 3% |
| RL conv_time | 0.08, 0.43, 0.035 | 0.18 | 0.22 | **120%** ⚠️ |

**观察**：mass / conv_time 单次 play 方差极大；CoM 极稳。**paper mass 必须报 ≥3 seed 均值±std**。

> **注**：上面 3 次 play 中，"seed=43"那次实际上因 task_registry 的 seed bug 没生效，跟"seed unknown"那次实质是不同随机抽样。2026-05-20 修复了 seed bug 后，明确的 seed=42/43/44 才真正可控。

### 6.5b lb=3 static 修 seed bug 后的多 seed 复测（ckpt 11000）

| seed | RL mass | RL mass bias | RL per_env std | conv_time | dcom_y |
|---|---|---|---|---|---|
| 42（早期 §2.1） | 0.86 | +0.05 | 0.03 | 0.08s | 0.0103 |
| 43（今天，bug 修复后） | 0.95 | +0.40 | 0.08 | 0.36s | 0.0101 |
| 44（今天，bug 修复后） | 0.84 | +0.10 | 0.03 | 0.006s | 0.0106 |
| **mean** | **0.88** | +0.18 | — | — | **0.0103** |
| **std** | **0.06** | — | — | — | **0.0003** |

**观察**：bug 修复后，3 seed mass RMSE = 0.88 ± 0.06 (rel std ~7%)，**比之前 §6.5 的 30% std 稳定得多**。之前的 1.40 outlier 没再复现，可能是没控住 seed 时的偶发。CoM 仍极稳。**这是 paper 应该用的 lb=3 mass 数字**。

### 6.6 OOD 实验：训练分布外载荷质量（载荷范围 [4, 6] kg，训练时是 [2, 4]）

| Policy | Seed | [QS] mass RMSE | [RL] mass RMSE | [RL] mass bias | [RL] dcom_y |
|---|---|---|---|---|---|
| **main lb=3 (OOD 4-6)** | 42 | 6.66 | **2.09** | **−1.30** | 0.0130 |
| **main lb=3 (OOD 4-6)** | 43 | 6.23 | **1.50** | **−0.79** | 0.0117 |
| **history_only (OOD 4-6)** | 42 | 20.42 | **2.01** | **−1.22** | 0.0133 |
| **history_only (OOD 4-6)** | 43 | 23.52 | **1.45** | **−0.71** | 0.0130 |
| 对照 in-dist (lb=3, §6.5b 均值) | — | ~5.5 | ~0.88 | +0.18 | ~0.0103 |
| 对照 in-dist (history_only, §4.1) | 42 | 26.71 | 0.98 | +0.06 | 0.0120 |

**四个关键观察**：

**(O1) OOD 让 RL mass 退化 ~2 倍**：in-dist 0.85–1.0 → OOD 1.45–2.09 kg。两个方法都退化。

**(O2) RL mass bias 强烈负偏（−0.7 到 −1.3 kg）= 训练上限饱和现象**：
- 训练 load ∈ [2, 4] kg → encoder 输出范围近似 [0, 4]
- OOD 测试 load ∈ [4, 6] kg → encoder 仍输出 ~4 kg → 系统欠估 1–2 kg
- 典型"NN 不会外推"现象

**(O3) main 和 history_only 在 OOD 上 RL mass 几乎完全打平**：
- seed=42: main=2.09 vs history=2.01（差 4%）
- seed=43: main=1.50 vs history=1.45（差 3%）
- **否定了"main method OOD 更鲁棒"的假设**——两种 encoder 都被训练上限同样限制

**(O4) QS 在 OOD 仍延续 in-distribution 趋势**：
- main policy 下：QS 6 kg，跟真值（5 kg）差 1 kg
- history_only policy 下：QS 20–23 kg 仍崩坏
- **"QS 友好姿态"性质跟载荷分布无关，由 policy 训练时是否见过 QS 决定**——OOD 也确认

### 6.7 OOD 下"两个 estimator 互检"的潜在价值

OOD 工况下 main policy 给出两个估计：
- RL encoder：1.5–2.1 kg（饱和到训练上限）
- Model C QS：6 kg（数值上仍跟真值 5 kg 在同量级）

**互检逻辑**：当两个估计差异 > 阈值（比如 > 3 kg），系统应该识别"我在 OOD"，触发保守策略 / 降速 / 人工介入。

history_only policy 下 QS 是 20+ kg，跟 RL 差 18+ kg——也会触发"OOD"警报，但 QS 数值本身没参考价值，**只能告诉你"出问题了"，但不能告诉你"问题在哪个方向"**。

**这是 main method 相对 baseline 的真正、可量化的好处**——不是 estimation 精度，而是 **hybrid estimator 系统的可诊断性**。

### 6.8 OOD 低端：载荷质量 [1, 2] kg（训练下限以下）

> **关于范围选择**：用 [1, 2] 而非 [0, 2] 是为了避开 Model C 公式的截止机制——
> [wheelfoot_flat.py:451-453](../legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py#L451-L453) 当估计质量 < `position_zero_mass_threshold = 1.0` 时强制将 load_x/load_y 置零（防 1/小数 爆炸）。
> 设 true_mass ∈ [1, 2] 保证 QS 估计稳定在阈值之上，OOD 对比反映"分布偏移"，不混入"小质量被截断"的噪声。

| Policy | Seed | [QS] mass RMSE | [QS] bias | [RL] mass RMSE | [RL] bias | [RL] per_env std |
|---|---|---|---|---|---|---|
| **main lb=3 (OOD 1-2)** | 42 | 4.64 | **+4.13** | **1.41** | **−0.44** | 0.029 |
| **main lb=3 (OOD 1-2)** | 43 | 4.42 | **+4.00** | **1.13** | **−0.80** | 0.046 |
| **history_only (OOD 1-2)** | 42 | 20.36 | +1.62 | 1.45 | **+0.07** | 0.034 |
| **history_only (OOD 1-2)** | 43 | 15.66 | +0.90 | 1.19 | **−0.66** | 0.123 |
| 对照 in-dist (lb=3) | — | ~5.5 | +5.0 | 0.88 | +0.18 | — |
| 对照 OOD high [4,6] (lb=3) | 42 | 6.66 | +5.27 | 2.09 | −1.30 | — |
| 对照 OOD high [4,6] (lb=3) | 43 | 6.23 | +5.12 | 1.50 | −0.79 | — |

### 6.9 OOD-low 出乎预料的发现

> 真实负载范围 [1, 2] kg（避开 position_zero_mass_threshold=1.0 的截断）。真实平均（考虑 load_on_body 时长比例）约 1.2–1.3 kg。

**预测 vs 实际**：
- ✗ **预测**：encoder 应该饱和到训练下限 ~2 kg，bias 强正（encoder > true ≈ +0.7）
- ✓ **实际**：bias 多为负（−0.44 ~ −0.80），encoder **低于**真值，**没有简单"硬饱和"**

**从 bias 反推 encoder 输出**（per-env 真值平均 ≈ 1.2 kg）：
- main lb=3 seed=42: bias=−0.44 → encoder 输出 ≈ 0.8 kg
- main lb=3 seed=43: bias=−0.80 → encoder 输出 ≈ 0.4 kg
- history_only seed=42: bias=+0.07 → encoder 输出 ≈ 1.3 kg（几乎准确！但 seed=43 又掉到 0.5）

**这意味着什么**：
- encoder 不是简单"clamp 到 [2, 4]"，而是有更复杂的行为
- 推测：训练中真值有两种分布——**载荷在体时 truth ∈ [2, 4]，载荷不在体时 truth = 0**，**双峰分布**
- OOD 真值 1–2 kg 落在两个峰之间，encoder 倾向于**判到 "0" 桶**（小载荷被部分识别为"无载荷"）
- 跟 OOD-high 的饱和到 4 不同，OOD-low 是"部分误识别成 0"

**关键现象**：**OOD-low 和 OOD-high 都让 encoder 欠估**，但机制不同：
- OOD-high (load 5, encoder ≈ 3.5–4)：训练上限截断
- OOD-low (load 1.5, encoder ≈ 0.4–1.3)：部分被识别为"载荷不在体"

**main 和 history_only 在 OOD-low 也几乎打平**（main: 1.13/1.41，history: 1.19/1.45；差异 < 5%），跟 OOD-high 一致——再次证实 **encoder 失败模式与架构无关**。

### 6.10 OOD-low 下 hybrid estimator 互检的 sanity check

在 main policy 下 OOD-low 工况：
- RL: ~0.5 kg（误判为弱载荷）
- QS: ~4.5 kg（QS 公式正常工作，给出训练分布附近的估计）
- 两者差 ~4 kg → 跟 OOD-high 类似量级 → 触发互检警报 ✓

在 history_only policy 下：
- RL: ~0.5 kg
- QS: 15–20 kg（cos_thigh 趋零导致 QS 公式爆，跟 in-dist 一样）
- 差 ~15–20 kg → 警报触发但 QS 数字仍无参考价值

**与 OOD-high 对称**：**main method 的 hybrid estimator 在两个 OOD 方向（低/高）都能保持互检的诊断价值**。这是双向证据，强化了结论 G 的"diagnosable hybrid system"卖点。

### 6.11 一个意外但重要的细节

history_only seed=42 在 OOD-low 下 RL bias = **+0.07**（几乎零偏），但 seed=43 又是 **−0.66**。**单 seed 极不可信**。

这反过来印证了**结论 C 的"mass 必须多 seed"原则**——在 OOD 边缘情况下方差更大，更不能单 seed 下结论。Paper 写 OOD 实验时这点尤其要强调。

---

# 第三部分：累计关键结论

## 结论 A：残差架构（QS in obs）**没有显著改善 encoder 估计精度**

证据：§6.1 显示三种 policy 的 RL mass RMSE 都在 0.86–1.07 范围；§6.3 dcom 差异在 15% 内。

意义：原 paper 假设"QS 先验帮助 encoder 学得更准"**被数据否定**。

## 结论 B：QS 公式表现取决于 **policy 是否被 QS 影响**

证据：§6.2 — 当 policy 训练时 QS 在 obs（main method），QS RMSE = 5 kg；当 policy 没见过 QS（history_only），同样的 Model C 公式 RMSE 跳到 27–49 kg。

意义：**这是真正的新发现**。意味着把 QS 注入 obs 的**主要价值不在改善 encoder，而在塑造 policy**——让 policy 维持"QS 友好"的姿态分布，从而让 analytical formula 仍可作为可靠的 sanity check / safety fallback。

## 结论 C：play-to-play 方差对 mass 估计影响巨大（~30%）

证据：§6.5 — 同 policy 同 ckpt static 三次 play mass RMSE = 0.86 / 1.40 / 0.86。CoM 几乎不变。

原因：20 envs 不是"同一实验 20 次"，而是 20 个不同 (load mass / push timing / friction / 推力等) 组合的样本。换 seed 等于重新抽 20 个组合。

意义：**paper 报 mass 数字必须 ≥3 seed 均值±std**；CoM 单 seed 可信。

## 结论 D：lb=3 在 mass 和 com 上都不输 lb=6，且训练更稳

证据：
- mass RMSE：lb=3 (0.86–1.40) ≈ lb=6 (0.97)，方差内
- dcom_y：lb=3 (0.0103) < lb=6 (0.0114)
- mass bias：lb=3 (+0.05) < lb=6 (+0.15) 持续更接近 0
- 训练 loss 曲线：lb=3 比 lb=6 平滑（pilot 观察）

意义：**load_boost = 3 选为 final 超参数**。

## 结论 E：paper narrative 需要重构

原方向：**"QS+RL hybrid 比 nostalgia baseline 更准"** → 不成立。

新方向：**"QS 注入 obs 的真正作用是 implicit policy regularization：塑造 policy 维持 QS 友好姿态，让 analytical formula 在部署时仍可用作 sanity check"**

参考类似 nuanced result paper：
- Lee et al. "Learning quadrupedal locomotion over challenging terrain" (Science Robotics 2020)
- Margolis et al. "Walk these ways" (CoRL 2022)
- Pinto et al. "Asymmetric Actor Critic for Image-Based Robot Learning"

## 结论 F：OOD 下 RL encoder **双向失败**，两个架构同样受限

证据：§6.6 + §6.8 — 训练 load ∈ [2, 4]，测试两个 OOD 方向：

**OOD-high (load ∈ [4, 6])**：
- main lb=3: RL RMSE 0.88（in-dist）→ 1.50–2.09，bias 强负 (−0.7 ~ −1.3) → encoder 饱和到训练上限 ~4 kg
- history_only: 同方向、同量级（差 < 4%）

**OOD-low (load ∈ [0, 2])**：
- main lb=3: RL RMSE 0.88 → 1.13–1.41，bias 也是负 (−0.44 ~ −0.80) → encoder **误判为"无载荷"**（输出 ≈ 0）
- history_only: 同方向、同量级（差 < 5%）
- 跟预期"饱和到下限 2"完全不同，机制是 **bimodal 训练分布**（in-body=load 2-4 vs not-on-body=0）让 encoder 把小载荷归类到 "0" 桶

**统一现象**：
- 两个方向 OOD 都让 encoder 欠估（负 bias）——但机制不同：高端是"输出上限截断"，低端是"误识别为无载荷"
- **residual 架构在两个 OOD 方向都不比 baseline 好**——结论 A 被进一步加强
- "NN 不会外推"普遍成立

意义：
- 之前希望 main method 在 OOD 上显著优于 baseline，**被双向数据否定**
- 但两个 encoder **失败模式同样可预测** → 失败可识别
- paper 的 framing：**hybrid 系统的价值不是 encoder 更准，而是 OOD 工况下两个 estimator 互检能识别失败方向**

## 结论 G：Hybrid estimator **可诊断性**——main method 真正的差异化优势

证据：§6.6 + §6.7 — OOD 工况下：
- main: RL=2.0 kg, QS=6 kg，两个估计差 ~4 kg 但 QS 仍在合理量级
- history_only: RL=2.0 kg, QS=20 kg，两个差 ~18 kg 但 QS 数值已无参考价值

意义：
- main 的 hybrid 系统可以做"两个 estimator 差异 > 阈值 → 触发保守策略 / 降速 / 人工介入"
- history_only 也能 detect 异常（差 18+ kg 显然异常），但**只能告诉你"出问题了"，不能说"问题在哪个方向"**
- **这是 paper 应该重点 sell 的差异**——不是 single-estimator 精度，是 **diagnosable hybrid system**

## 结论 H：encoder 没有被 policy reward 直接驱动（机制澄清）

证据：[wheelfoot_flat_config.py:389](legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py#L389) `output_detach = True`，encoder 输出在喂 actor 前 `.detach()`，policy 梯度不流回 encoder。

那 main 和 history_only 的 policy 为什么差异这么大？通过两个**间接**机制：
1. **obs 直接耦合**：QS 12 维特征直接进 actor obs → actor 直接被 reward 推着用 QS → 学会"维持 QS 友好姿态"是 actor 自己学的，跟 encoder 无关
2. **训练数据耦合**：policy 决定 rollout 轨迹 → 决定 encoder 看到的 obs / aux loss target 分布 → encoder 训练数据不同 → encoder representation 不同

**这澄清了一个易混淆点**：encoder 训练目标是纯 supervised（aux loss against truth），不被 reward 影响；但 actor 和 encoder 都隐式被 policy-induced state distribution 耦合。

---

# 第四部分：版本和 commit 关联

| 日期 | git commit | 关键变更 |
|---|---|---|
| 2026-05-19（前） | (前版) | 初始 main method, lb=6 训练完成 |
| 2026-05-19 | (前版) | lb=3 main method 训练完成；play.py 加 encoder logging |
| 2026-05-20 | 08d4c40 | 加 use_qs_in_obs flag；history_only 训练完成 |
| 2026-05-20 | (待 commit) | 修 task_registry seed bug；OOD [4,6] kg 实验完成 |

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
