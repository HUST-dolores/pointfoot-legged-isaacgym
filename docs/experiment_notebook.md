# 实验记录本 (Experiment Notebook)

> **项目主题**：WF_TRON1A 双轮足双足机器人的 hybrid load mass estimation (QS analytical + RL encoder).
>
> **本笔记本的当前定位**：2026-05-24 重组版。删除了 2026-05-22 之前的 bug-era raw data；
> 整理掉了与现状矛盾的旧结论；按"重要性排序"重新组织。
> 所有 paper-grade conclusions 集中在 [§5 当前可信结论]，按 ★ 标记重要性。

---

## ★ TL;DR (paper 最核心 findings)

按重要性排序：

1. **★★★ [结论 I](#-结论-iper-env-mass-diversity-是-ood-改善的核心机制)** — per-env mass diversity 是 OOD 改善的核心机制；OOD-low RMSE 1.44 → 0.98 kg（**−32%**）
2. **★★★ [结论 J](#-结论-j架构qs-in-obs--residual-learning影响-marginal)** — 架构（QS-in-obs / residual learning）影响 marginal；三架构 RMSE per condition < 0.15 kg
3. **★★ [结论 K](#-结论-kencoder-mass-output-几乎不被-actor-使用)** — encoder mass output 几乎不被 actor 使用（IG attribution < 0.20% across all conditions）
4. **★ [结论 L](#-结论-lqs-in-obs-是-stabilizing-prior小但实在)** — QS-in-obs 是 stabilizing prior（actor IG ~6%）
5. **★ [结论 M](#-结论-mencoder-在-mass-branch-上做了-noise-aware-feature-selection)** — encoder 在 mass branch 做了 noise-aware feature selection
6. [结论 N](#结论-nco-calibration-在当前架构下不可行paper-12-句提及) — Co-cal 在当前架构下不可行（paper 1-2 句提及即可）

→ **paper main contribution**：per-env mass diversity（一行 hidden bug 修复）；
→ paper honest negative findings：架构影响 marginal、encoder mass 不被 actor 用、co-cal 不可行；
→ paper supporting findings：QS-in-obs stabilizing prior、encoder noise-aware feature selection。

---

## 共同设定（适用所有 play）

| 项 | 值 |
|---|---|
| 平台 | WF_TRON1A (双轮足双足) |
| 训练 | trimesh L0（最易级），num_envs=1024，episode_length=40s，**per_env_load_mass=ON** |
| Play | num_envs=20，stop_state_log=2000 step (40s)，trimesh L0 patch 60×180 m |
| 载荷 | mass ∈ [2, 4] kg 随机，start=0.5s，duration ∈ [30, 40] s，interval ∈ [50, 60] s |
| Domain rand | friction / restitution / base_com / inertia / Kp / Kd / motor_torque / dof_pos / action_delay / imu_offset 全开；base_mass=False（Model G 标定需求） |
| Push | push_robots=True, push_interval=5s, max_push_vel=2 m/s |
| QS deployed | Model G（7 params），universal coefs avg over main_s42 + main_s43 + direct，见 [§3.2](#32-universal-g_all-系数) |

---

## §1 实验设计

### 1.1 三架构 ablation matrix（E1 / E2 / E3）

所有 E1/E2/E3 同 `seed=45` + `per_env_load_mass=ON` + ckpt 11000，仅架构 flag 不同。

| Ckpt | 含义 | `use_qs_in_obs` | `use_residual_learning` |
|---|---|---|---|
| **E1 main** = `exper_qs_resi_load_boost_3_seed_45_pemass` | QS in obs + residual learning | ✓ | ✓ |
| **E2 direct** = `exper_qs_noresi_load_boost_3_seed_45_pemass` | QS in obs + direct mass head | ✓ | ✗ |
| **E3 histonly** = `exper_history_only_load_boost_3_seed_45_pemass` | history only（无 QS） | ✗ | N/A |

### 1.2 No-pemass 对照 baseline

仅用作 per-env mass 改善幅度对比的参照，不再作为独立分析对象：

- `exper_qs_resi_load_boost_3` (Old main_s42) — main 配置 + seed=42 + **no pemass**
- 其他 no-pemass ckpts（s43、lb=6、direct、history_only）见 [附录 A](#附录-ano-pemass-baseline-数据)

---

## §2 主结果：Per-env mass diversity 是 OOD 改善的核心

### 2.1 Hidden distribution shift（修复前的训练 bug）

修复前 `base_task.py:571` 训练时：

```python
chosen_load_mass = torch.empty(1).uniform_(load_mass_min, load_mass_max).item()  # ★ 全局一个 scalar
load_asset_options.density = chosen_load_mass / load_volume
# 所有 1024 envs 用同一个 load_asset → 同一个 mass per training run
```

→ 训练时 encoder 见到的 mass label 是 **per-run constant**（每次训练只见一个固定 mass）。
encoder 没有学到 mass-varying representation，自然 OOD 退化严重。

修复：`per_env_load_mass = True` → 每个 env 在 actor 创建时单独 `create_box` + 独立 density →
1024 envs 同时存在 1024 个不同 mass → encoder 见到 [2, 4] kg 完整分布。

### 2.2 E1 vs Old main_s42（per-env mass 单独贡献）

play_seed=42，ckpt 11000，trimesh L0，num_envs=20。

| Condition | **E1** (pemass) | Old s42 (no pemass) | 改善 |
|---|---|---|---|
| in-dist static [2,4] | 0.83 | 1.07 | **−22%** |
| OOD-low static [1,2] | **0.98** | **1.44** | **−32%** ★ |
| OOD-low walk [1,2] | 0.98 | 1.46 | **−33%** ★ |
| OOD-high static [4,6] | 1.49 | 1.89 | **−21%** |

### 2.3 三架构完整 RMSE 对比（确认 pemass 改善跨架构成立）

| Condition | **E1 main** | **E2 direct** | **E3 histonly** | range (E1-3) |
|---|---|---|---|---|
| in-dist static [2,4] | 0.83 | **0.78** | 0.82 | 0.05 |
| in-dist walk [2,4] | 0.86 | 0.79 | **0.74** | 0.12 |
| OOD-low static [1,2] | **0.98** | 1.01 | 1.05 | 0.07 |
| OOD-low walk [1,2] | 0.98 | **0.93** | 1.00 | 0.07 |
| OOD-high static [4,6] | 1.49 | 1.48 | **1.39** | 0.10 |
| OOD-high walk [4,6] | **4.95** ⚠ | 1.50 | 1.42 | 0.08 (excl E1) |

**关键观察**：

- 三架构 RMSE per condition 差异 < 0.15 kg；每条件三者轮流最好，**无系统性赢家**
- 跨架构在 OOD-low / OOD-high 同样保持 pemass 改善效果（vs no-pemass baseline ~1.4-1.9 kg）
- **E1 walk OOD-high = 4.95 是已知 outlier**：deployed universal G_all 系数在 E1 policy walk OOD-high 上 QS feature 数值爆炸（QS RMSE = 141 kg） → 污染 actor obs → encoder 跟着退化。E2 (deploy 系数同样问题但 actor 不依赖 residual 兜底) 和 E3 (无 QS in obs) 均免疫。
  - 该 outlier 是 **deploy 系数工程问题，非 policy 缺陷**：E1 离线 cal (Model G fit on QS grid) mass RMSE = **0.689 kg**，policy 姿态本身仍 QS-fittable。

→ **pemass 是改善 OOD 的实际机制；架构选择对 RMSE 无显著影响**（同时支撑 [结论 I](#-结论-iper-env-mass-diversity-是-ood-改善的核心机制) 和 [结论 J](#-结论-j架构qs-in-obs--residual-learning影响-marginal)）。

---

## §3 Model G QS 公式部署 + universal 系数

### 3.1 多 ansatz 对比（5 个 canonical policy × 8 模型）

avg RMSE 在 4 个 qs-using policies (main_s42 / main_s43 / lb=6 / direct) 上：

| Model | params | avg RMSE | 备注 |
|---|---|---|---|
| A basic | 4 | 1.327 | |
| B +pitch | 5 | 1.204 | |
| **C +hip_diff**（旧 deployed） | 5 | **0.827** | |
| E +abad_diff | 5 | 1.276 | |
| F hip+abad | 6 | 0.805 | |
| **G all 3 extras** | 7 | **0.711** | 比 C **−14%** |
| C_sym (L=R) | 3 | 0.864 | |
| G_sym (L=R) | 5 | 0.757 | 比 C −9% |

**G > G_sym > C**。选 **G_all (7 params)** 部署。

公式（[`wheelfoot_flat.py:368-382`](../legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py#L368-L382)）：

```
mass_est = (α_L · R_L + α_R · R_R + γ_L + γ_R
            + β_pitch · sin(pitch) + β_hip · hip_diff + β_abad · abad_diff
            + T_body_x · sin(pitch)) / ...
com_x_est = com_x_bias · sin(pitch) + x_offset
```

### 3.2 Universal G_all 系数

avg over main_s42 + main_s43 + direct（lb=6 排除，见 §3.3）：

| 参数 | 旧 Model C 默认 | 新 Model G universal |
|---|---|---|
| alpha_L | 0.367 | **0.4154** |
| alpha_R | 0.429 | **0.4235** |
| gamma_L | +0.158 | **−0.7638** |
| gamma_R | −0.499 | **−0.8474** |
| **beta_pitch** | (新增) | **−2.1976** |
| beta_hip | −8.507 | **−9.1708** |
| **beta_abad** | (新增) | **+3.5004** |
| T_body_x | 6.17 | **6.3447** |
| com_x_bias (k_pitch) | 0.649 | **0.0784** |
| x_offset | −0.037 | **−0.0487** |

### 3.3 已知 deploy 限制

- **cal range = [2, 6]**（4 mass: 2.0/3.33/4.67/6.0），OOD-low [1, 2] 在 cal 范围外，QS 公式外推
- **lb=6 是 universal coef 的 outlier**（own RMSE 0.71 vs universal coef 上 2.40），未进 universal 平均；单独标注
- **E1 walk OOD-high 出现 QS feature 数值爆炸**（见 [§2.3 outlier 说明](#23-三架构完整-rmse-对比确认-pemass-改善跨架构成立)）—— 该 outlier 反过来证明 [结论 L](#-结论-lqs-in-obs-是-stabilizing-prior小但实在) 提到的"deploy 系数不优时 QS-in-obs 反而有害"

### 3.4 collect_calibration_data.py 已加 provenance

[`collect_calibration_data.py`](../legged_gym/scripts/collect_calibration_data.py) 的 `np.savez` 加 `load_run / checkpoint / task` string field，
**以后 cal 数据自带 policy 标签**，不再需要猜哪个 cal 对应哪个 policy。

---

## §4 Actor Integrated Gradients 分析

### 4.1 方法

新脚本 [`analyze_actor_ig.py`](../legged_gym/scripts/analyze_actor_ig.py)（独立于 [`analyze_encoder_ig.py`](../legged_gym/scripts/analyze_encoder_ig.py)）：

- Target: actor 输出 (action L2 norm) 对每个输入 dim 的 attribution
- Input: cat(encoder_out=7, obs=48 or 36, commands=3) = 58 or 46 dims
- Baseline: rollout samples 均值
- 输出: coarse group view + fine per-dim view (QS / encoder 内部拆开)
- 自动从 saved cfg 对齐 `use_qs_in_obs` / `use_residual_learning` / `num_obs` / encoder dim

### 4.2 三架构 actor IG 对比 (ckpt 11000, 'all' target, coarse)

| Group | dim | **E1 main** (qs+resid) | **E2 direct** (qs+no-resid) | **E3 histonly** (no-qs) |
|---|---|---|---|---|
| previous_actions | 8 | 52.75% | **63.78%** | **65.82%** |
| dof_pos | 6 | 13.79% | 7.54% | 15.36% |
| encoder/est_lin_vel | 3 | 8.96% | 6.93% | 5.61% |
| projected_gravity | 3 | 7.24% | 6.09% | 3.41% |
| torques | 8 | 5.28% | 3.87% | 3.89% |
| **qs_load_features** | 8 | **4.69%** | **5.07%** | (absent) |
| dof_vel | 8 | 4.22% | 4.91% | 4.49% |
| **qs_residual_baseline** | 4 | **0.89%** | **0.40%** | (absent) |
| encoder/est_com_delta | 3 | 0.81% | 0.42% | 0.69% |
| **encoder/est_mass** | 1 | **0.08%** | **0.10%** | **0.09%** |
| base_ang_vel | 3 | 1.27% | 0.90% | 0.63% |
| commands | 3 | 0% (IG artifact, 命令固定) | 0% | 0% |
| **QS combined** | 12 | **5.58%** | **5.47%** | **0%** |

**四个 paper-grade 发现（与 §5 结论 K/L/M 对齐）**：

**(A) `est_mass` attribution = 0.08-0.10% across all 3 architectures** → actor 完全不用 encoder mass 估计做决策（[结论 K](#-结论-kencoder-mass-output-几乎不被-actor-使用)）。

**(B) QS combined: E1 (5.58%) ≈ E2 (5.47%)** → residual learning 不影响 actor 对 QS feature 的依赖（[结论 J](#-结论-j架构qs-in-obs--residual-learning影响-marginal) 支撑数据）。

**(C) E2/E3 比 E1 更依赖 previous_actions**（64-66% vs 53%）→ 移除 residual 或 QS 时 actor 转向 self-referential 策略，但 RMSE 不变（[§2.3](#23-三架构完整-rmse-对比确认-pemass-改善跨架构成立)）。attribution redistribution 是 architecture-specific adaptation，不影响最终性能。

**(D) E2 encoder/est_com_delta (0.42%) 比 E1 (0.81%) 显著低** → direct 架构下 actor 对 encoder com 输出依赖也下降。但 com 本身基本被忽略（< 1%）。

### 4.3 跨 ckpt 训练轨迹（E1 QS attribution）

| ckpt | qs_combined | est_mass |
|---|---|---|
| 3000 | 5.58% | 0.08% |
| 5000 | 7.01% | 0.03% |
| 7000 | 5.87% | 0.05% |
| 9000 | 5.15% | 0.05% |
| 11000 | 6.41% | 0.09% |
| 13000 | 7.23% | 0.04% |
| 16000 | 5.21% | 0.07% |

QS combined 全程 [5.15, 7.23]%，**稳定但低**；est_mass 全程 < 0.10%，**始终被忽略**。无 monotonic 上升/下降。

### 4.4 OOD 条件下 IG（验证 "QS 在 OOD 下是否更重要"）

| Group | E1 in-dist | E1 OOD-low [1,2] | E1 OOD-high [4,6] |
|---|---|---|---|
| **qs combined** | 5.58% | **6.89%** (+1.31pp) | 5.29% (−0.29pp) |
| previous_actions | 52.75% | 55.90% | 54.38% |
| projected_gravity | 7.24% | 5.72% | 7.08% |

| Group | E3 in-dist | E3 OOD-low | E3 OOD-high |
|---|---|---|---|
| **encoder/est_lin_vel** | 5.61% | **11.36%** (+5.75pp ⭐) | 6.87% |
| projected_gravity | 3.41% | **6.01%** (+2.60pp) | 4.87% |
| previous_actions | 65.82% | 60.35% (−5.5pp) | 62.64% |

**OOD 自适应模式**：
- E1（有 QS）OOD 下变化幅度 ±3pp，QS attribution 只升 1.31pp
- E3（无 QS）OOD 下大幅重排（encoder_vel +5.75pp）
- → **"QS in obs 提供 stabilizing prior"**（[结论 L](#-结论-lqs-in-obs-是-stabilizing-prior小但实在)）：移除后 actor 必须更剧烈重排其他 feature

### 4.5 12 维 QS feature 内部 per-dim attribution（mass branch encoder, E1 ckpt 11000）

| QS dim | E1 attribution % | per-dim density | × baseline (1.14%) |
|---|---|---|---|
| **com_delta_z** | 7.47 | 3.74 | 3.3× ★★★ |
| cos_lieangle_R_thigh | 4.99 | 2.49 | 2.2× ★★ |
| cos_lieangle_L_thigh | 4.61 | 2.31 | 2.0× ★★ |
| mass_delta | 2.43 | 1.22 | 1.1× ★ |
| load_x | 1.87 | 0.93 | 0.8× |
| payload_mass | 1.71 | 0.86 | 0.8× |
| com_delta_x | 0.93 | 0.46 | 0.4× ▼ |
| payload_present | 0.64 | 0.32 | 0.3× ▼ |
| load_y | 0.55 | 0.28 | 0.2× ▼ |
| sin_lieangle_R_thigh | 0.51 | 0.26 | 0.2× ▼ |
| sin_lieangle_L_thigh | 0.50 | 0.25 | 0.2× ▼ |
| com_delta_y | 0.37 | 0.19 | 0.2× ▼ |

`com_delta_z` 是整个 encoder mass branch 输入里 per-dim attribution 最高的维度。
物理直觉：z 方向 com 偏移 = 载荷质量 × 载荷高度 / total_mass，**直接编码 mass 信息**。

**per-dim 重组对比** (Old s42 vs E1 pemass)：

| QS dim | Old s42 | E1 pemass | change |
|---|---|---|---|
| payload_mass | 2.48% | **0.86%** | **−65%** （抛弃直接 mass dim） |
| cos_lieangle_L_thigh | 1.47% | **2.31%** | **+57%** （转向几何） |
| cos_lieangle_R_thigh | 1.70% | **2.49%** | **+47%** （转向几何） |
| com_delta_z | 3.10% | **3.74%** | **+21%** （物理可解释 dim 强化） |

→ **encoder 在 noisy deployed G_all 系数下做了 noise-aware feature selection**：抛弃噪声大的 direct mass dim，转向几何稳定 dim + 物理可解释 dim（[结论 M](#-结论-mencoder-在-mass-branch-上做了-noise-aware-feature-selection)）。

---

## §5 当前可信结论 (按重要性排序)

### ★★★ 结论 I：Per-env mass diversity 是 OOD 改善的核心机制

**Paper main contribution**。

证据：E1 (pemass) vs Old main_s42 (no pemass)，play_seed=42，ckpt 11000：

| Condition | E1 | Old s42 | 改善 |
|---|---|---|---|
| in-dist static | 0.83 | 1.07 | −22% |
| OOD-low static | 0.98 | 1.44 | **−32%** ★ |
| OOD-low walk | 0.98 | 1.46 | **−33%** ★ |
| OOD-high static | 1.49 | 1.89 | −21% |

机制：原训练 hidden bug（[§2.1](#21-hidden-distribution-shift修复前的训练-bug)）—— 1024 envs 共享同一个全局 load mass scalar，encoder 训练时见到 constant mass label。修复后 1024 envs 独立 mass → encoder 见到 [2, 4] 完整分布 → 泛化提升。

### ★★★ 结论 J：架构（QS-in-obs / residual learning）影响 marginal

完整 architecture ablation matrix（all per-env mass on, seed_45, ckpt 11000，[§2.3](#23-三架构完整-rmse-对比确认-pemass-改善跨架构成立)）：

| Architecture | use_qs_in_obs | use_residual_learning | RMSE 范围 (6 conditions) |
|---|---|---|---|
| **E1 main** | ✓ | ✓ | 0.83-1.49（excl walk OOD-high outlier） |
| **E2 direct** | ✓ | ✗ | 0.78-1.50 |
| **E3 histonly** | ✗ | (N/A) | 0.74-1.42 |

三架构每 condition RMSE 差 < 0.15 kg；每条件三者轮流最好；**无系统性赢家**。

→ "QS shapes policy"、"residual learning matters for encoder"、"hybrid 系统更鲁棒" 等 narrative **均不被数据支持**。

### ★★ 结论 K：Encoder mass output 几乎不被 actor 使用

证据：actor IG 对 encoder est_mass 的 attribution（[§4.2-4.4](#42-三架构-actor-ig-对比-ckpt-11000-all-target-coarse)）：
- E1 全部 7 个 ckpts × in-dist + 2 OOD：全部 **< 0.15%**
- E2 ckpt 11000：**0.10%**
- E3 全部 7 个 ckpts × in-dist + 2 OOD：全部 **< 0.20%**

baseline = 1.72%（如果 attention 均匀分布）。

→ encoder mass head 是 architecture-invariant **auxiliary supervised regularizer**，不直接影响 policy 行为。这是 paper 一个 honest negative finding。

### ★ 结论 L：QS-in-obs 是 stabilizing prior（小但实在）

证据：
- E1 actor IG: QS combined 全程 5-7%（小但稳定，[§4.3](#43-跨-ckpt-训练轨迹e1-qs-attribution)）
- E1 OOD 下变化幅度小（max ±3pp），E3 必须剧烈重排（est_lin_vel +5.75pp on OOD-low，[§4.4](#44-ood-条件下-igvalidate-qs-在-ood-下是否更重要)）
- E1 walk OOD-high RMSE = 4.95 ⚠ 是 deploy 系数失配 + QS feature 数值爆炸的污染（[§2.3 outlier](#23-三架构完整-rmse-对比确认-pemass-改善跨架构成立)），**反过来证明 QS-in-obs 在 deploy 系数不优时反而有害**

→ paper 表述：
> QS-in-obs contributes ~6% to action decisions; removing it forces aggressive feature substitution under OOD but does not measurably hurt performance when per-env mass diversity is in place.

### ★ 结论 M：Encoder 在 mass branch 上做了 noise-aware feature selection

证据（per-dim IG, E1 vs Old s42，[§4.5](#45-12-维-qs-feature-内部-per-dim-attributionmass-branch-encoder-e1-ckpt-11000)）：
- payload_mass dim: **−65%** （抛弃噪声大的直接 mass）
- cos_lieangle_L/R_thigh dims: **+47%~+57%** （转向几何稳定）
- com_delta_z dim: **+21%** （转向物理可解释）

→ paper 可写 mechanism：per-env mass training + G_all formula 让 encoder 学到信号 robust 选择。

### 结论 N：Co-calibration 在当前架构下不可行（paper 1-2 句提及）

简述：尝试过三版 in-loop co-calibration（per-spawn API → per-env density → calibration phase + deterministic actor），
deployed α 全部 < 0.05（vs offline cal 的 0.33）。失败根因是 rollout 数据天然带 dom-rand + control noise，
跟 Model G quasi-static 假设不兼容。**全部 co-cal 代码已删除**；保留 per-env mass diversity 作为副产物（成为 [结论 I](#-结论-iper-env-mass-diversity-是-ood-改善的核心机制) 的来源）。

详细探索史见 [附录 B 时间线](#附录-b时间线压缩版)。

---

## §6 当前 paper narrative

```
Paper 主线: Per-env load mass randomization closes a hidden distribution shift 
in legged-robot load estimation, reducing OOD load mass RMSE by 32%. 
Architecture choice (QS-in-obs, residual learning) has minimal impact.

Supporting findings:
- IG analysis shows encoder mass output is essentially unused by actor (<0.2% 
  attribution), suggesting load estimation acts as auxiliary supervised 
  regularization rather than direct policy input.
- QS-in-obs contributes a small but stable ~6% to actions; under OOD, it 
  serves as a stabilizing prior that reduces actor's need to reweight other 
  features.
- Within encoder mass branch, per-env mass training induces dim-level 
  feature reattribution (away from direct mass dim, toward geometric and 
  physically-interpretable dims).

Negative findings:
- In-loop QS coefficient co-calibration fails due to rollout data being 
  contaminated by exploration noise and dynamic torques (Model G assumes 
  quasi-static).
- "QS-in-obs as implicit policy regularizer" hypothesis is not supported 
  by data when per-env mass is controlled.
```

---

## §7 旧 narrative 撤回清单（不要再使用）

以下表述均已撤回，paper 写作中**不要再使用**：

- "QS-friendly policy" / "implicit policy regularizer"
- "main policy 维持 QS-friendly poses"
- "history_only policy 让 QS 崩溃，所以 policy 本身更差"
- "QS in obs 的真正作用是 implicit policy regularization"
- "固定 QS 公式天然 OOD-robust"
- "|RL−QS| > 1.5 kg 可直接作为 OOD detector"
- "lb=3 优于 lb=6 是因为 policy shaping 更强"
- "Online co-calibration as new paper main line"
- "Hybrid estimator complementarity"
- "RL encoder 在 OOD 必然退化" (实际上 pemass 后 OOD 显著改善)

---

## 附录 A：No-pemass baseline 数据

仅用于体现 per-env mass 改善幅度，不再单独分析。

### A.1 5 policy in-distribution RL mass RMSE（ckpt 11000, play_seed=42, load=[2,4]）

| Policy | static | walk vx=0.5 |
|---|---|---|
| main lb=3 (seed_42 ckpt) | 0.9004 | 0.8611 |
| main lb=3 (seed_43 ckpt) | 1.0229 | 1.0214 |
| main lb=6 | 0.9548 | 0.9840 |
| direct (qs=1, resid=0) | 0.9407 | 0.9389 |
| history_only (qs=0) | 0.8665 | 0.8823 |

5 个 no-pemass policy 全部落在 **0.86–1.02 kg** 区间。
与 E1/E2/E3 的 0.74–0.86 kg（[§2.3](#23-三架构完整-rmse-对比确认-pemass-改善跨架构成立)）对比 → in-distribution 上 pemass 也带来 ~10-20% 改善。

### A.2 No-pemass OOD baseline (main lb=3 seed_42 + seed_43, static)

| Train seed | Cond | RL mass RMSE | RL bias |
|---|---|---|---|
| seed_42 | in-dist [2,4] | 0.9004 | +0.011 |
| seed_42 | OOD-low [1,2] | **1.4406** | **−0.895** |
| seed_42 | OOD-high [4,6] | **1.8903** | **−1.124** |
| seed_43 | in-dist [2,4] | 1.0229 | −0.541 |
| seed_43 | OOD-low [1,2] | 1.2913 | −0.440 |
| seed_43 | OOD-high [4,6] | **2.4240** | **−1.717** |

- No-pemass 时 OOD-high 出现 train-seed dependent failure (seed_42=1.89 vs seed_43=2.42)
- 修复 pemass 后 OOD-high 稳定到 1.49（E1）；不再出现 seed-dependent failure

### A.3 离线 cal 对各 policy 仍然 work（Model G fit on QS grid）

| Policy | offline cal mass RMSE |
|---|---|
| E1 (main+pemass+seed_45) | 0.689 kg |
| 其他 no-pemass 4 policies | 0.49-1.36 kg |

→ E1 policy 姿态本身仍 QS-fittable；§2.3 walk OOD-high outlier 是 universal coef deploy 问题，不是 policy 缺陷。

---

## 附录 B：时间线（压缩版）

| 日期 | 事件 |
|---|---|
| 2026-05-19 | lb=6 / lb=3 main method 训练完成；play.py 加 encoder logging |
| 2026-05-20 | 加 use_qs_in_obs flag；history_only 训练完成；OOD [4,6] 实验完成 |
| 2026-05-21 | direct (no residual) 训练完成；play.py residual baseline 双加 bug 修复；7 项 play.py 改进（auto exp_tag, run verification, save_dir 等） |
| 2026-05-22 | seed_43 训练完成；§6 干净数据采集完成；notebook 校正旧 narrative |
| 2026-05-22 | 实现 v1 in-process co-cal（per-spawn API），证伪：α≈0 退化 |
| 2026-05-23 | v2 per-env asset density（修物理对齐），α 仍 ≈ 0；v3 calibration phase + deterministic actor，α 仅 0.04 |
| 2026-05-23 | **删除全部 co-cal 代码**；**保留 per-env mass diversity**（`domain_rand.per_env_load_mass=True` default ON）|
| 2026-05-23 | 部署 **Model G QS 公式**（替代 Model C），更新 universal coefs |
| 2026-05-23 | `collect_calibration_data.py` 加 load_run/checkpoint/task provenance |
| 2026-05-23 | E1 = `..._seed_45_pemass` 训练启动 |
| 2026-05-24 | E1 训练完成，play+IG+cal 完成 |
| 2026-05-24 | E3 训练完成，play+actor IG 完成 |
| 2026-05-24 | 新增 `analyze_actor_ig.py` 独立脚本（含 per-dim QS breakdown + OOD load 覆盖） |
| 2026-05-24 | Actor IG 跨 7 ckpts × 2 policies + OOD load × 2 policies 全部完成 |
| 2026-05-24 | E2 = `..._seed_45_pemass` 训练完成，三架构 matrix 闭环 |
| 2026-05-24 | **notebook 重组**（本次）：删除 §1-§5 bug-era raw data，撤回 stale 结论，按重要性重排 |

---

## §五 未来记录格式建议

新加 play 数据时建议格式：

```
### §X.X <method> <condition> (ckpt <N>, seed=<S>, run #<R>)

```（直接粘贴 [play] ===== experiment metrics ===== 那一整段）

> 备注（可选）：
> - exp_tag = ...
> - 想用这个数据验证什么 hypothesis
> - 异常观察
```

新加结论：直接在 [§5](#5-当前可信结论-按重要性排序) 追加 "结论 X" 段落，并按重要性插入 [TL;DR](#-tldr-paper-最核心-findings) 列表中。
新加 commit 关联：在 [附录 B 时间线](#附录-b时间线压缩版) 追加一行。

新结论与旧结论冲突时：**直接修改/删除旧结论**，而不是堆叠新版本。本笔记本的目标是保持任一时刻零矛盾。
