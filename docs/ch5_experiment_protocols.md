# 第5章实验流程（protocols）—— 抗倾覆能力系列

本文件给出每个实验的**工况 / 机器人数量 / 判定条件 / 数据处理 / 数据来源**,便于复现与写论文方法节。
所有实验均为 **play-only**(加载已训基底回放,不再训练)。结果与讨论见 `experiment_notebook2.md` §6–§7。

---

## 0. 通用设置(所有实验共享)

**四个宽负载基底(2–30kg 训练,单 seed)**
| 变体 | 标志 qs/resid/torq | run 目录 | ckpt | n_obs |
|---|---|---|---|---|
| Model-guided | T/T/T | `Jun04_23-18-34_wide2-30_model_guided_seed1` | 11000 | 48 |
| Estimate-guided | T/F/T | `Jun06_00-18-09_wide2-30_estimate_guided_seed1` | 16000 | 48 |
| Source-guided | F/F/T | `Jun05_17-50-01_wide2-30_source_guided_seed42` | 16000 | 36 |
| RL-only | F/F/F | `Jun05_03-01-38_wide2-30_rl_only_seed1` | 11000 | 28 |
（变体按 `use_load_residual_estimation`=qs∧resid 分类;Source seed1 曾坍缩,改 seed42 重训成功。）

**回放工具**:`legged_gym/scripts/play.py`
- **自动架构对齐**:从 run 目录的 `env_cfg.json/train_cfg.json` 读回 qs/resid/torq,自动设 n_obs/encoder 维度并加载 ckpt(无需手改 config)。
- **机器人数量**:`--num_envs 30`(play 上限 30)。每 env 用 `per_env_load_mass=True` 各自采样自身负载。
- **回合长度**:`episode_length_s=40` → 2000 步 @ `dt=0.02s`。
- **负载**:`--load_mass_min/max` 覆盖 `add_load_range`;`--load_hold` 把 load_duration/interval 设成 1e6 → **负载 0.5s 起全程恒定**(否则 on/off 循环会造成瞬态假倒)。
- **平地**:`--flat_terrain`(mesh=plane),使姿态反映控制而非地形跟随。

**终止/复位条件(关键)**:`check_termination`(base_task.py)= 躯干接触力>10N **或** `projected_gravity_z>-0.1`(倾倒>~84°)。
→ 25° "直立锥"远严于此:斜站 ~30° 既不触地也没到 84°,**不复位**;倒到底/触地则复位重生。24° 坡复位极频繁(26/30 env、~150–210 次/run)。

**每-env 日志(`full_dict[...]_all`,写入 .mat)**:`base_roll/base_pitch/base_lin_vel_z/base_height/wheel_angle/base_pos_x/base_pos_y/command_x/payload_mass_ref` 等,形状 [2000, 30]。
**数据落盘**:`logs/wheelfoot_flat/WF_TRON1A/exported/play_data_<时间>_<exp_tag>.mat`,含 `meta`(load_run/ckpt/flags/load_mass_range_used/dt…)。

**统一的负载分箱(铁律)**:**每个 env 按自身真实负载 `payload_mass_ref` 分箱,箱内再求均值;绝不先跨 env 平均再算**(各 env 负载不同,跨载平均无意义)。
箱边 `[2,8,14,20,26,30.1]`,箱中心 **5/11/17/23/28 kg**。
**复算脚本**:`legged_gym/scripts/_expC_extract.py`(纯 CPU,scipy 读 .mat,不碰 GPU);绘图 `figs_ch5/{C1,C2,C3}*.m`(MATLAB)。

---

## 1. 质量估计扫描(估计精度,第4章/§7)
- **工况**:平地、行走 `--cmd_vx 0.5`、负载 `--load_mass_min 2 --load_mass_max 30`、`--load_hold`。
- **机器人数**:30(负载 2–30 沿 env 铺开,~1kg 间隔)。
- **判定**:估计质量 vs 真值的转移曲线斜率 + **逐 env RMSE**(对每 env 用其自身真值,再汇总)。
- **数据处理**:取稳态段(去前 1s 响应),`payload_mass_est` vs `payload_mass_ref`。
- **数据来源**:walk-flat 文件 `*_walk_vx0.5_*_load2-30_flat.mat`(同时充当 C-1 的 slope0 参考)。

## 2. C-0 平地搬运极限(关键反证,§7 C-0)
- **工况**:平地、`--cmd_vx 0.5`、`--load_hold`、负载 **`--load_mass_min 2 --load_mass_max 60`**(含 OOD)。脚本 `run_carry_limit.sh`。
- **机器人数**:30/变体(负载 2–60 铺开)。
- **判定条件**:**末态直立率**——末 5s 内 ≥80% 时间 `|pitch|<25°&|roll|<25°` 记该 env "站住";箱内求占比。
- **数据处理**:按负载箱(中心 6/14/22/30/38/46/54)求直立率;"搬运极限"=直立率仍≥0.8 的最大箱。
- **数据来源**:`*_walk_vx0.5_*_load2-60_flat.mat`,字段 base_pitch/roll_all、payload_mass_ref_all。
- **结论**:四方均扛到 ~54kg、无差异(平地无倾覆力矩 → 估计无用)——作为"估计只在扰动∝负载时有用"的负向对照。

## 3. C-1 斜坡存活率(主结论,§7 C-1)
- **工况**:平地+**倾斜重力**模拟坡(`--slope_deg β`,同时设 `sim.gravity=g·[sinβ,0,−cosβ]` 与 obs `gravity_vec`)、`--cmd_vx 0.5`、`--load_hold`、负载 2–30。坡度集 β∈{8,12,16,20,(22),24,(26),28}°。脚本 `run_source_plays.sh`/`run_estimate_plays.sh`(各变体一套)+ `/tmp/run_beta_refine.sh`(22/26°)。
- **机器人数**:30/(变体×坡度)。
- **判定条件**:**末态直立率**(同 C-0:末 5s ≥80% 时间在 25° 直立锥内 → 站住)。**换指标原因**:旧"全程时间占比"会把"倒地→复位重生为直立"的重生时间计入而虚高失败者;末态直立率只看末段是否稳站。
- **数据处理**:`_expC_extract.py` 的 `slope_survival()`:逐 env 算末态直立(0/1)→ 负载箱内求占比 → (变体×坡度×负载) 网格。
- **数据来源**:`*_walk_vx0.5_*_load2-30_flat_slope{β}.mat`,字段 base_pitch/roll_all、payload_mass_ref_all、dt。
- **结论**:24° 处有 QS 组(Model 0.57/Estimate 0.73)≫ 无 QS 组(Source 0.10/RL 0.04),6–15×。

## 4. β* 抗倾覆边界(派生指标,§7 统一框架)
- **工况/来源**:**复用 C-1 网格**(不另跑),坡度集含 8–28°。
- **判定/处理**:对每(变体×负载箱),取末态直立率随坡度的曲线,**线性插值出存活率跌破 0.5 的临界坡度 β\***(=该工况下的抗倾覆边界);再对负载求均值。
- **结论**:β\* = Model 24.5° / Estimate 25.2° / Source 20.5° / RL-only 21.1°(QS 组高 ~4°,几乎与负载无关)。这是"抗倾覆能力"的单一标量。

## 5. C-2 静态位置保持漂移(§7 C-2 / C-2′)
- **工况**:`--cmd_vx 0`(静止指令)、`--load_hold`、负载 2–30。**两版**:平地(C-2)与 **斜坡 15°**(C-2′,`--slope_deg 15`,脚本 `run_c2c3_slope.sh`)。
- **机器人数**:30/变体。
- **判定条件**:从 t=1s 起的**最大水平漂移** `max √(Δx²+Δy²)`(Δ 相对 t=1s 位置)。
- **数据处理**:逐 env 算最大漂移 → 负载箱内均值 + 总均值。
- **数据来源**:`*static*_load2-30_flat.mat`(平地)/ `*static*_load2-30_flat_slope15.mat`(斜坡);字段 **base_pos_x_all/base_pos_y_all**(=root_states 真位置)、payload_mass_ref_all。
- **结论**:平地差异弱;斜坡 15° 上 Model 4.32 < RL 8.27(Estimate 单 seed 偏噪)。

## 6. C-3 急停停车距离(§7 C-3 / C-3′)
- **工况**:`--estop_vx 1.5`(速度指令方波:巡航 1.5 m/s `--estop_go_s 4` → 急停 0 `--estop_stop_s 4`,反复)、`--load_hold`、负载 2–30。**两版**:平地(C-3)与 **下坡 10°**(C-3′,`--slope_deg 10`,下坡方向=+X,刹车需抗 m·g·sinβ)。
- **机器人数**:30/变体。
- **判定条件**:**停车距离** = 从"指令由>1.0 跳到<0.5"那一刻起,2.5s 窗口内机体 x 的最大前移 `max(base_pos_x − base_pos_x[归零时])`。巡航公平性:四方 go 段峰值都到 ~1.4 m/s。
- **数据处理**:逐 env 检测 go→stop 跳变、取首个;箱内均值。
- **数据来源**:`*_load2-30_flat_estop1.5.mat`(平地)/`*_estop1.5_slope10.mat`(下坡);字段 **command_x_all**(检跳变)、**base_pos_x_all**(=root_states 真位置,**非**轮子转角:轮子里程因 wrap+倒摆几何+滑移不准)。
- **结论**:平地无明显差;下坡 10° 上 QS 组(Model 0.97/Est 0.86)< 无 QS 组(Source 1.45/RL 1.43),短 ~35%。

## 7. actor-IG 输入归因(机理证据,§7 机理)
- **工况**:`analyze_actor_ig.py --headless --command_x 0.5 --load_mass_min 2 --load_mass_max 30`(负载下平地;脚本**无** --slope_deg)。
- **机器人数**:50(IG 默认),ig_samples 1024、ig_steps 32。
- **判定/处理**:对 actor 动作做积分梯度(Integrated Gradients),按输入分组汇总归因百分比(target=all, groupset=coarse)。
- **数据来源**:各 run 的 `actor_ig/actor_ig_summary_*.csv`。
- **结论**:QS 负载通道归因 Estimate 16.1% / Model 9.9% ≫ Source 1.5% / RL 2.9%;Source 对原始力矩仅 2.3%(没用上)→ 解释 Source≈RL。

---

## 8.【新设计】偏心负载实验(CoM 估计专项)
- **目的**:平地上制造**正比于负载的倾覆力矩**(∝ m·offset),专门检验 **CoM 估计头**的价值。
- **新增开关**:`play.py --load_offset_x/--load_offset_y`(把 `env.load_offset_range` 固定成 (val,val),z 保持 0.10;默认 -99=不覆盖)。tag 后缀 `_ecc{x}x{y}y`。
- **工况**:平地、`--cmd_vx 0`(静止)、`--load_hold`、负载 2–30;前偏置 `--load_offset_x 0.20` 与居中 `0` 对照。脚本 `/tmp/run_ecc20.sh`。
  - ⚠️ **有效偏置范围 = `load_offset_range`(base_task)`x∈(-0.17,0.21), y∈(-0.19,0.19)`**(即训练所用、物理有效的范围;config 里注释掉的 `load_offset_range_xy=[0.13,0.12]` 是两个月前的旧错误值,**不用**)。**x=0.30 超出平板 → 负载掉落**(实测仅 1/30 env 还挂着,作废);**x=0.20**(≈前向上限)有效(28-29/30 在板上)。
- **机器人数**:30/(变体×偏置)。
- **判定条件**:末段 5s **前倾 |pitch| 均值**(偏心力矩→前倾)+ 最大水平漂移 + 末态直立率;按负载箱。
- **数据处理**:逐 env 末段 |pitch| 均值 / 漂移 → 负载箱内均值;对比居中(x=0)与偏心,以及四变体。
- **数据来源**:`*_static_*_load2-30_flat_ecc{X}x0y.mat`;字段 base_pitch_all、base_pos_x/y_all、payload_mass_ref_all。
- **预期**:估准 CoM 的 Model/Estimate 在偏心载下前倾/漂移更小;居中(x=0)作负向对照。**(x=0.20 重跑中)**

## 9.【新设计】横向推-恢复(扰动抑制)
- **目的**:负载下施加横向力,测抗倾覆/扰动抑制能力(复用 Exp A 的持续外力设施)。
- **关键解耦**:Exp A 默认在施力时关掉真实负载(play.py:684,`ext_force_on→add_random_load=False`)。本实验新增 **`--keep_load`** 标志保留真载,使"真载+横向力"共存(不影响 Exp A 默认)。
- **工况**:平地、`--cmd_vx 0`(静止)、`--load_hold`、`--keep_load`、负载 2–30、**`--ext_force_y 40`**(世界系 +Y 持续 40N;`play.py` 检测到非零 ext_force 即置 `env.ext_force_enable=True`、`ext_force_vec`)。脚本 `/tmp/run_pushload.sh`。
  - ⚠️ **力标定**:40N=可区分工况(四方都存活);**60N 进倾翻区(roll 20–88°)噪声大、不可用**;30N(无载旧版)太弱。
- **机器人数**:30/变体。
- **判定条件**:末段 5s **峰值 |roll|**(横向力→横滚)+ 末态直立率;按负载箱。
- **数据处理**:逐 env 峰值 |roll| → 负载箱内均值;四变体对比(注:扰动≠负载本身,关联弱于斜坡/偏心)。
- **数据来源**:`*_static_*_load2-30_flat_fxyz0-40-0N.mat`(tag `fxyz{x}-{y}-{z}N`);字段 base_roll_all、payload_mass_ref_all。
- **结论**:40N 下 **RL-only 峰值 roll 最差(5.5→16°,随载快速恶化,~2–3× 于其余);Model 最稳(2→8°)**;带信息三方均优于 RL-only。外力为**持续**(非真冲量),测持续横扰下的稳态倾角。

---

### 待办 / 注意
- 偏心负载 **x=0.20 重跑**(0.30 负载掉板,数据作废)。
- push 实验结果分析中。
- 所有实验**均单 seed**;关键结论(C-1/β*)建议多 seed 复现。
- β*/C-1 用**末态直立率**(已替换旧"时间占比"指标)。
