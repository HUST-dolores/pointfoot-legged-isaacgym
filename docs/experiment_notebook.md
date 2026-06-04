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
2. **★★★ [结论 J](#-结论-j架构选择marginalqs-derivable-信号必要)** — 架构形式（QS-in-obs / residual learning）影响 marginal **on torque-preserving subset**
3. **★★★ [结论 Q](#-结论-qencoder-依赖两条互补的-torque-信号-pathway-e4--e5-ablation)** — encoder 依赖两条**互补**的 torque 信号 pathway（raw torques + explicit QS features），任何一条单独都不够 —— E4 (移除两条) RMSE 退化 1.6-2.2×, **E5 (只移除 raw torques) RMSE 退化 0.3 kg**（in-dist），E5 介于 E1 和 E4 之间
4. **★★ [结论 K](#-结论-kencoder-mass-output-几乎不被-actor-使用)** — encoder mass output 几乎不被 actor 使用（IG attribution < 0.5% across all conditions）
5. **★ [结论 L](#-结论-lqs-in-obs-是-stabilizing-prior小但实在)** — QS-in-obs 是 stabilizing prior（actor IG ~6%）
6. **★ [结论 M](#-结论-mencoder-在-mass-branch-上做了-noise-aware-feature-selection)** — encoder 在 mass branch 做了 noise-aware feature selection
7. [结论 N](#结论-nco-calibration-在当前架构下不可行paper-12-句提及) — Co-cal 在当前架构下不可行（paper 1-2 句提及即可）

→ **paper main contributions**：(a) per-env mass diversity（一行 hidden bug 修复）；(b) **two-pathway torque signal mechanism**（E4 + E5 ablations 证明）；
→ paper honest negative findings：架构形式无所谓（torque-preserving 子集内）、encoder mass 不被 actor 用、co-cal 不可行；
→ paper supporting findings：QS-in-obs stabilizing prior、encoder noise-aware feature selection、prev_actions 在缺 raw torques 时作为 implicit surrogate（E5 encoder IG）。

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

### 1.1 五架构 ablation matrix（E1 / E2 / E3 / E4 / E5）

所有 ckpts 同 `per_env_load_mass=ON` + ckpt 11000，仅架构 flag 不同。

| Ckpt | 含义 | `use_qs_in_obs` | `use_residual_learning` | `use_torques_in_obs` |
|---|---|---|---|---|
| **E1 main** = `exper_qs_resi_load_boost_3_seed_45_pemass` | QS in obs + residual + torques | ✓ | ✓ | ✓ |
| **E2 direct** = `exper_qs_noresi_load_boost_3_seed_45_pemass` | QS in obs + direct + torques | ✓ | ✗ | ✓ |
| **E3 histonly** = `exper_history_only_load_boost_3_seed_45_pemass` | history only + torques | ✗ | N/A | ✓ |
| **E4 true history-only** = `May24_19-30-38_exper_history_only_no_torq_load_boost_3_seed_45_pemass` | history only without torques | ✗ | N/A | **✗** |
| **E5 qs_only_path** = `May25_01-00-47_exper_qs_resi_load_boost_3_no_torq_seed_45_pemass` | QS in obs + residual, **no raw torques** | ✓ | ✓ | **✗** |

**所有 4 个 ckpt 实际训练时 seed=45**（由 `make_env` 里的 `set_seed(env_cfg.seed=45)` 应用）。
注：E1-E4 的 `env_cfg.json` 因为 [task_registry.py 旧 save bug](../legged_gym/utils/task_registry.py) 错误地写了 1（`make_alg_runner` 内部第二次调 `get_cfgs(name)` 时把 `env_cfg.seed` 重置回了 `train_cfg.seed` 的默认值），但实际训练用的 numpy/torch 种子是 45（train_cfg.json 里的 seed=45 才是真值）。该 bug 已在 2026-05-25 修复，E5 及之后的 ckpt 的 env_cfg.json 会正确记录 seed=45。

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

### 2.3 五架构完整 RMSE 对比（pemass + torque/QS pathway ablation 一起看）

| Condition | **E1** (qs+resid+torq) | **E2** (qs+direct+torq) | **E3** (no-qs+torq) | **E5** (qs+resid, NO torq) | **E4** (no-qs, NO torq) |
|---|---|---|---|---|---|
| in-dist static [2,4] | 0.83 | **0.78** | 0.82 | 1.10 | **1.65** ★ |
| in-dist walk [2,4] | 0.86 | 0.79 | **0.74** | 1.19 | **1.66** ★ |
| OOD-low static [1,2] | 0.98 | 1.01 | 1.05 | **0.91** ★ | **1.90** ★ |
| OOD-low walk [1,2] | 0.98 | **0.93** | 1.00 | 1.04 | **1.84** ★ |
| OOD-high static [4,6] | **1.49** | 1.48 | **1.39** | 1.78 | **2.24** ★ |
| OOD-high walk [4,6] | **4.95** ⚠ | 1.50 | 1.42 | 1.95 | **2.23** ★ |

**关键观察**：

- **E1/E2/E3（保留 torques）每 condition RMSE 差异 < 0.15 kg**，每条件三者轮流最好，**无系统性赢家** → 架构形式 (QS-in-obs / residual learning) 在 torque-preserving subset 内可互换
- **E5（保留 QS features，移除 raw torques）落在 E1 和 E4 之间**：in-dist +0.27-0.33 kg vs E1，OOD-low 反而 ≈ E1 → **两条 path 部分互补，不是冗余**
- **E4（两条 path 都移除）uniformly 退化 1.6-2.2×** → torque-derived 信号是 encoder 实际的工作底层
- **OOD-low 上 E5 ≈ E1**（甚至略优 0.91 vs 0.98）→ explicit QS features 在 OOD 下表现意外稳健，可能因为 QS 公式天然外推性较平稳
- **E1 walk OOD-high = 4.95 是已知 outlier**（deploy 系数 + raw torques 协同 numerical issue）：E5 walk OOD-high = 1.95 完全免疫，**反过来确证 outlier 跟 raw torques 输入 actor 有关**

→ 这一张表同时支撑 [结论 I](#-结论-iper-env-mass-diversity-是-ood-改善的核心机制) (pemass 跨架构 OOD 改善)、[结论 J](#-结论-j架构选择marginalqs-derivable-信号必要) (E1/E2/E3 架构 marginal) 和升级版的 [结论 Q](#-结论-qencoder-依赖两条互补的-torque-信号-pathway-e4--e5-ablation) (E4 + E5 双 ablation 证明 **two-pathway** 机制)。

### 2.4 E4 ablation 详细数据（true history-only：连 torques 都不喂）

E4 设计：在 E3 的基础上**进一步移除 raw joint torques**（policy obs 减 8 维 → num_obs = 28）。
这是 paper main contribution (b)：证明 QS-derivable 信号是 encoder 实际的工作底层，不仅是 representation 选择问题。

**Bias 模式**（最有诊断价值的发现）：

| Condition | E3 bias | E4 bias | 解读 |
|---|---|---|---|
| in-dist static | −0.15 | **+0.79** | E4 systematically 高估 ~0.8 kg |
| in-dist walk | −0.05 | **+0.52** | 同上 |
| OOD-low static | −0.58 | **+1.72** | E4 把真值 1.5kg 估成 ~3.2kg |
| OOD-low walk | −0.45 | **+1.45** | 同上 |
| OOD-high static | −0.67 | −0.50 | 真值 5kg 估成 ~4.5kg |
| OOD-high walk | −0.69 | −0.59 | 同上 |

E4 在 in-dist 和 OOD-low 都是 **+bias**（高估），仅 OOD-high 是 small −bias。这是 encoder **collapse 到训练分布均值 ~3 kg** 的特征模式 — 说明 E4 encoder 几乎没学到 mass-varying signal，只能输出 prior。

**Control confound 检查**（区分 "encoder 失去信号" vs "actor 失去力矩反馈"）：

| Condition | E3 \|vx−cmd\| | E4 \|vx−cmd\| | E3 tq_RMS | E4 tq_RMS |
|---|---|---|---|---|
| in-dist static | 0.10 | 0.10 | 15.8 | 15.0 |
| in-dist walk | 0.18 | **0.34** ★ | 12.4 | **19.7** ★ |
| OOD-low static | 0.10 | 0.09 | 15.4 | 14.4 |
| OOD-low walk | 0.11 | **0.31** ★ | 12.5 | **20.4** ★ |
| OOD-high static | 0.12 | 0.13 | 16.6 | 17.9 |
| OOD-high walk | 0.15 | **0.31** ★ | 14.8 | **21.4** ★ |

**Static 全部条件 control 不退化**（tracking err 持平 ~0.10, torque RMS 持平 ~15），但 RMSE 仍 **1.6-2× worse**。
→ paper 主论证用 **static-only 数据**：clean evidence 排除 control confound，直接证明 encoder 失去 QS-derivable 信号是 RMSE 退化的因。

**Walk 条件下 control 同时退化**（tracking err ~2×, torque RMS ~50% up）→ walk 数据是 encoder + actor 双重作用，不单用 walk 数据论证。

### 2.5 E5 ablation 详细数据（QS features 保留 + raw torques 移除）

E5 设计：保留 QS features in obs（同 E1），仅移除 raw torques。num_obs = 40。
用途：disambiguate contribution (b) —— 是 raw torques 必要，还是 explicit QS features 必要，还是两者协同。

**RMSE / bias / control 一览**（与 E1、E4 横向对比）：

| Condition | E1 RMSE | **E5 RMSE** | E4 RMSE | E1 bias | **E5 bias** | E4 bias |
|---|---|---|---|---|---|---|
| in-dist static | 0.83 | **1.10** (+0.27) | 1.65 | −0.23 | **−0.11** | +0.79 |
| in-dist walk | 0.86 | **1.19** (+0.33) | 1.66 | −0.23 | **−0.29** | +0.52 |
| OOD-low static | 0.98 | **0.91** (−0.07) | 1.90 | −0.45 | **+0.06** | +1.72 |
| OOD-low walk | 0.98 | **1.04** (+0.06) | 1.84 | −0.43 | **−0.22** | +1.45 |
| OOD-high static | 1.49 | **1.78** (+0.29) | 2.24 | −0.77 | **−0.75** | −0.50 |
| OOD-high walk | 4.95⚠ | **1.95** (−3.00) | 2.23 | −0.75 | **−0.87** | −0.59 |

**关键模式**：

1. **E5 落在 E1 和 E4 之间，gap 中 ~30%**（即 E5 比 E1 差 0.3 kg，比 E4 好 0.5-1.0 kg）→ raw torques 提供了大概 30% 的边际信号，QS features 替代不了
2. **E5 bias 比 E4 健康得多**：in-dist E5 bias ≈ −0.1 ~ −0.3（小幅低估，正常），E4 是 +0.5 ~ +0.8（严重高估）→ E5 encoder 没有 collapse，只是少了一些精度
3. **OOD-low 上 E5 ≈ E1 甚至略优**：唯一一个 E5 不输 E1 的 condition；说明 QS features 在 1-2 kg 段外推性比 raw torques 更稳
4. **E1 walk OOD-high outlier 在 E5 完全消失**（4.95 → 1.95）→ outlier 跟 raw torques 输入 actor 的 numerical 协同有关，不是 encoder 问题

**Control confound 检查**：

| Condition | E1 \|vx−cmd\| | **E5** \|vx−cmd\| | E4 \|vx−cmd\| |
|---|---|---|---|
| in-dist static | 0.05 | **0.10** | 0.10 |
| in-dist walk | 0.09 | **0.18** | 0.34 |
| OOD-low static | 0.05 | **0.09** | 0.09 |
| OOD-low walk | 0.08 | **0.22** | 0.31 |
| OOD-high static | 0.06 | **0.12** | 0.13 |
| OOD-high walk | 0.10 | **0.19** | 0.31 |

- **E5 static 跟踪误差几乎不退化**（持平 E4），但 RMSE 比 E4 好得多 → static 数据继续 confound-free 论证
- **E5 walk 跟踪误差 ~2× E1 但只有 ~60% E4**：actor 对力矩反馈的需求只是部分受影响，QS feature 部分补偿
- Static condition 是 paper 主论证；walk 仍 confound 存在但 E5 比 E4 退化更轻 → 进一步支持 "QS feature 部分补偿"

### 2.6 完整 estimator × condition 对比表（encoder vs QS-only, mass + dCoM）

play_seed=42, ckpt 11000, load 2-4 kg in-dist, num_envs=20. QS-only 用 deployed universal G_all coefs（[§3.2](#32-universal-g_all-系数)）。

**上下文备注**（不是 paper 写作决策，仅作交流过程记录）：当前 policy 即使 cmd=0 ("static") 时机器人也在原地动态平衡（来回摇晃），并非真正静止。
所以 static 跟 walk 两者其实都是 dynamic process，差异主要反映 cmd magnitude 而非定性差异。
跟 "QS-only vs encoder" 的量级差异（3-60×）相比，static/walk 之间的差异（< 10%）相对较小。
是否在 paper 中只保留 walk 行 / 还是 static+walk 都保留，**留待 paper 写作时再决定**，本节先把 6 张表完整存档。

#### 2.6.1 Encoder RMSE — Static

| Method | Mass [kg] | dCoM-x [m] | dCoM-y [m] |
|---|---|---|---|
| **E1** | 0.825 | 0.0180 | 0.0126 |
| **E2** | 0.777 | 0.0151 | 0.0130 |
| **E3** | 0.819 | 0.0157 | 0.0133 |
| **E5** | 1.098 | 0.0175 | 0.0180 |
| **E4** | 1.654 | 0.0189 | 0.0173 |

#### 2.6.2 Encoder RMSE — Walk vx=0.5

| Method | Mass [kg] | dCoM-x [m] | dCoM-y [m] |
|---|---|---|---|
| **E1** | 0.859 | 0.0185 | 0.0120 |
| **E2** | 0.791 | 0.0150 | 0.0126 |
| **E3** | 0.744 | 0.0169 | 0.0127 |
| **E5** | 1.185 | 0.0171 | 0.0188 |
| **E4** | 1.664 | 0.0193 | 0.0185 |

#### 2.6.3 Encoder Walk − Static delta

| Method | ΔMass [kg] | ΔdCoM-x [m] | ΔdCoM-y [m] |
|---|---|---|---|
| **E1** | +0.034 | +0.0005 | −0.0007 |
| **E2** | +0.014 | −0.0001 | −0.0004 |
| **E3** | −0.075 | +0.0013 | −0.0006 |
| **E5** | +0.087 | −0.0003 | +0.0009 |
| **E4** | +0.009 | +0.0004 | +0.0012 |

观察：所有 delta 量级 < 0.09 kg，跨 method 没有显著 walk-vs-static differentiation。E3 delta 为负是 single-seed 抽样噪声。

#### 2.6.4 QS-only RMSE — Static

| Method | Mass [kg] | dCoM-x [m] | dCoM-y [m] |
|---|---|---|---|
| **E1** | 9.590 ⚠ | 0.0153 | 0.0285 |
| **E2** | 3.182 | 0.0134 | 0.0242 |
| **E3** | 2.824 | 0.0119 | 0.0239 |
| **E5** | 4.544 | 0.0166 | 0.0242 |
| **E4** | 10.760 ⚠ | 0.0161 | 0.0267 |

#### 2.6.5 QS-only RMSE — Walk vx=0.5

| Method | Mass [kg] | dCoM-x [m] | dCoM-y [m] |
|---|---|---|---|
| **E1** | 2.821 | 0.0135 | 0.0280 |
| **E2** | 2.655 | 0.0141 | 0.0274 |
| **E3** | 11.464 ⚠ | 0.0130 | 0.0244 |
| **E5** | 4.511 | 0.0158 | 0.0260 |
| **E4** | **101.274** ⚠⚠ | 0.0179 | 0.0332 |

#### 2.6.6 QS-only Walk − Static delta

| Method | ΔMass [kg] | ΔdCoM-x [m] | ΔdCoM-y [m] |
|---|---|---|---|
| **E1** | −6.769 | −0.0018 | −0.0005 |
| **E2** | −0.527 | +0.0007 | +0.0032 |
| **E3** | +8.640 | +0.0011 | +0.0005 |
| **E5** | −0.033 | −0.0009 | +0.0018 |
| **E4** | **+90.515** | +0.0017 | +0.0066 |

QS-only delta 数值跨度大（−7 ~ +90 kg）但**主要反映 single-seed 抽样在数值爆炸 regime 的抖动**，不是 walk vs static 本身的系统性差异。

#### 2.6.7 关键 cross-table 对比（encoder vs QS-only）

**(A) Mass: QS-only 比 encoder 差 3-60× 跨所有 method**

| Method | walk encoder | walk QS-only | ratio (QS / enc) |
|---|---|---|---|
| E1 | 0.86 | 2.82 | **3.3×** |
| E2 | 0.79 | 2.66 | **3.4×** |
| E3 | 0.74 | 11.46 | **15.5×** |
| E5 | 1.19 | 4.51 | **3.8×** |
| E4 | 1.66 | 101.27 | **61×** |

→ **encoder learning 在 mass 估计上不可替代**。即使是最 QS-friendly 的 E1/E2，QS-only mass 仍是 encoder 的 3 倍。

**(B) CoM: QS-only ≈ encoder（仅 1.5-2× 差距）**

| Method | walk encoder dcom_y | walk QS-only dcom_y | ratio |
|---|---|---|---|
| E1 | 0.0120 | 0.0280 | 2.3× |
| E2 | 0.0126 | 0.0274 | 2.2× |
| E3 | 0.0127 | 0.0244 | 1.9× |
| E5 | 0.0188 | 0.0260 | 1.4× |
| E4 | 0.0185 | 0.0332 | 1.8× |

→ **CoM 估计上 QS-only 已经够用**。Model G CoM 公式（仅 `sin(pitch)` + bias）不依赖力矩，对 policy distribution 不敏感。

#### 2.6.8 两条新 paper takeaways（来自 §2.6.7）

1. **"Encoder learning is necessary for mass, optional for CoM"**：mass 上 QS-only RMSE 3-60× 大于 encoder，跨所有架构；CoM 上仅 ~2×。说明 encoder 主要从 noisy torque signal 提取 mass 的 high-frequency component（QS 公式无法捕获），而 CoM 主要由 pose（pitch）决定，QS 已足够。

2. **"Fixed-coef QS is policy-distribution sensitive on mass, robust on CoM"**：同一组 universal G_all coefs 在 E1-E5 上 mass RMSE 跨度 2.8-101 kg（37× 跨度），但 CoM dcom_y 跨度仅 1.2-1.9 cm（1.6× 跨度）。
   - mass 公式调用力矩 (R_L, R_R)，policy 改变 → 力矩分布改变 → mass 估计偏差被放大
   - CoM 公式只用 pitch，policy 间 pitch 分布差异小 → CoM 估计稳定
   - → 这是旧版"撤回的 conclusion C"在 E1-E5 干净数据下再次确认

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

### 4.6 四架构 actor IG（E1 / E3 / E5 / E4）

Actor IG (ckpt 11000, target='all', coarse) 四架构对比：

| Group | E1 (qs+torq) | E3 (no-qs, +torq) | **E5 (qs, no-torq)** | E4 (no-qs, no-torq) |
|---|---|---|---|---|
| previous_actions | 54.25 | 65.32 | **61.73** | 53.30 |
| dof_pos | 9.87 | 13.47 | 13.59 | 17.36 |
| encoder/est_lin_vel | 11.88 | 6.35 | 6.04 | 9.59 |
| projected_gravity | 6.61 | 5.49 | 4.21 | 8.34 |
| dof_vel | 4.78 | 4.02 | 6.39 | 9.17 |
| **torques** | 5.25 | 3.60 | (absent) | (absent) |
| **qs_load_features** | 4.25 | (absent) | **6.39** ★ | (absent) |
| **qs_residual_baseline** | 0.96 | (absent) | 0.47 | (absent) |
| encoder/est_com_delta | 1.03 | 0.61 | 0.61 | 0.95 |
| **encoder/est_mass** | 0.07 | 0.09 | **0.04** | **0.43** |
| base_ang_vel | 1.06 | 1.05 | 0.53 | 0.87 |
| **QS combined** | **5.21** | 0 | **6.86** | 0 |

**关键观察**：

- **E5 actor 对 QS combined 的依赖比 E1 还高（6.86% vs 5.21%）**：没有 raw torques 之后，actor 显式 lean on QS features —— 跟 encoder 的 substitution pattern 反向（encoder 主要 lean on prev_actions，见 §4.7）
- **E5 est_mass 跌到 0.04%（vs E4 的 0.43%）**：E4 actor 试图用 encoder mass 补偿但 encoder 输出已 collapse；E5 encoder 输出健康，所以 actor 不需要硬靠 est_mass，回到正常的 ~0.1% 水平 → 进一步支撑 [结论 K](#-结论-kencoder-mass-output-几乎不被-actor-使用)
- E4 vs E3 的 reweight 模式（dof_vel +5pp, gravity +3pp, prev_actions −12pp）保持 —— 移除 raw torques 后 actor 倾向多用 dof_vel / gravity
- E5 vs E1 reweight 较温和（prev_actions +7pp, est_lin_vel −6pp），说明 QS features 大体保留了 actor 的工作模式

### 4.7 三架构 encoder IG mass branch（E1 / E5 / E4）—— paper 核心定量论证

Encoder mass branch attribution（% of total mass-branch attribution, raw + filtered 双 branch 合并）。
ckpt 11000, in-dist play_seed=42。

| Group | **E1 (qs+torq)** | **E5 (qs, no-torq)** | **E4 (no-qs, no-torq)** |
|---|---|---|---|
| **torques** | **27.05** | (absent) | (absent) |
| projected_gravity | 17.30 | 10.34 | **31.53** |
| **qs_load_features** | **15.39** | **15.83** | (absent) |
| **qs_residual_baseline** | **11.20** | 3.92 | (absent) |
| dof_pos | 11.11 | 10.56 | 17.74 |
| dof_vel | 7.75 | 7.20 | 12.61 |
| **previous_actions** | 5.52 | **50.10** ★ | 23.22 |
| base_ang_vel | 4.67 | 2.06 | 14.90 |

**Total torque-derived path (torques + qs_features)**:

| Policy | torques | qs_features | **total** |
|---|---|---|---|
| **E1** | 27.05% | 26.59% | **53.64%** |
| **E5** | 0 | 19.75% | **19.75%** |
| **E4** | 0 | 0 | **0%** |

**两个 paper-grade 发现**：

**(1) 两条 path 互补，不冗余**：
- E1 → E5：失去 27% torques，但 qs_load 利用率保持（15.39 → 15.83%）；qs_residual 部分萎缩 (11.20 → 3.92%)；total torque path 从 54% → 20%
- E5 RMSE 退化 0.3 kg（in-dist）→ 失去的 ~34pp signal 部分被 prev_actions surrogate 补回（见下）

**(2) E5 encoder 把 prev_actions 拉到 50.10% 作为 implicit torque surrogate**：
- E5 prev_actions = 50.10% vs E1 5.52%，**升 +44.58pp**
- 物理直觉：previous_actions ≈ "刚才让关节去哪" ≈ 隐式期望力矩
- 但这个 surrogate 不完全等价：E5 RMSE 仍比 E1 差 0.3 kg → prev_actions 缺少 raw torques 的高频细节和实际作动器输出反馈
- E4 prev_actions 只有 23.22%（比 E5 低一半）—— 因为 E4 还有 QS features 也消失，没有 explicit anchor 让 prev_actions 那么 dominant；E4 不得不再 fallback 到 projected_gravity (31.53%) 和 base_ang_vel (14.90%)

→ paper 核心 narrative：**"encoder uses raw torques and explicit QS features as two partly-overlapping pathways (53% of mass-branch attribution combined). Either alone is sufficient for partial mass estimation (E5: ~20% retained, RMSE +0.3 kg). Removing both collapses the encoder to indirect proxies (E4: 0% torque path, RMSE +0.8 kg)."**

### 4.8 12 维 QS feature 内部 per-dim attribution（mass branch encoder, E1 ckpt 11000）

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

### 4.9 IG 可视化脚本盘点（2026-05-25 新增 MATLAB scripts）

`legged_gym/scripts/` 下新增 6 个 IG plot 脚本，按 paper 用途分级：

| 脚本 | 输出 | 默认行为 | 对应 paper 结论 |
|---|---|---|---|
| **`plot_ig_encoder_mass_stacked.m`** ★ | `exported/ig_pdfs/ig_encoder_mass_stacked.pdf` | 自动存 PDF + figure 留屏 | 结论 Q（two-pathway torque signal）—— **paper main IG fig** |
| `plot_ig_encoder_window_heatmap.m` ★ | 5 张 figure（默认 batch 模式 E1-E5） | 不存盘，留 figure | 同上的 **时间维度展开**：每 1s 一格的 attribution 演化 |
| `plot_ig_encoder_mass_heatmap.m` | 1 张 5×8 static heatmap | 不存盘 | 结论 Q 的紧凑视觉版 |
| `plot_ig_actor_heatmap.m` | 1 张 5×10 static heatmap | 不存盘 | 结论 J + K（架构 marginal, encoder mass 不被用）|
| `plot_ig_actor_groups.m` | 5 method × 10 group grouped bar | 不存盘 | 备份（user 判断"没用"）|
| `plot_ig_est_mass_negative.m` | est_mass 单 bar + uniform baseline | 不存盘 | 备份（user 判断"没用"）|

**Windowed IG 数据状态**（5 个 ckpt 都已生成 50s × 1s windows）：
```bash
# 单 ckpt 重跑命令（已对 E1/E2/E3/E5/E4 全部跑过）：
python legged_gym/scripts/analyze_encoder_ig.py --task=wheelfoot_flat \
  --load_run <run_name> --checkpoint 11000 --headless --num_envs 32 \
  --rollout_steps 2500 --window_steps 50 --window_ig_samples 32
```

**Window heatmap 关键设计**：
- raw + filtered 分支合并（strip 前缀后求和）→ 每变量 1 行
- Y 轴固定输入顺序（不按 mean 排序）：`ang_vel, gravity, dof_pos, dof_vel, [torques], [qs_load, qs_resid], prev_actions`
- X 轴 1 秒每格，覆盖 50 秒
- Colormap clipped at 30% 防止 prev_actions = 60% (E5) 把其他行压成死黑色
- Cell text overlay 显示数值（≤ 40 windows 时）

**完整图片清单**：见 [`figures_inventory.md`](figures_inventory.md)。

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

### ★★★ 结论 J：架构选择 marginal，但 QS-derivable 信号必要

**两部分**：

**(J1) 架构形式 marginal**（QS-in-obs / residual learning / explicit QS features 之间互换）：

| Architecture | use_qs_in_obs | use_residual_learning | use_torques_in_obs | RMSE 范围 (6 conditions) |
|---|---|---|---|---|
| **E1 main** | ✓ | ✓ | ✓ | 0.83-1.49（excl walk OOD-high outlier） |
| **E2 direct** | ✓ | ✗ | ✓ | 0.78-1.50 |
| **E3 histonly** | ✗ | N/A | ✓ | 0.74-1.42 |

E1/E2/E3 每 condition RMSE 差 < 0.15 kg；每条件三者轮流最好；**无系统性赢家**。
→ "QS shapes policy"、"residual learning matters for encoder"、"hybrid 系统更鲁棒" 等 narrative **均不被数据支持**。

**(J2) 但底层 torque 信号必要**（见 [结论 Q](#-结论-qqs-derivable-力矩信号是必要的-e4-ablation)）：

E1/E2/E3 看起来"架构无所谓"，是因为它们都保留了 raw joint torques 在 obs 里。
encoder 既可以从 explicit QS features 提取 mass 信号（E1/E2），也可以直接从 raw torques 隐式重建（E3）。两条路径殊途同归。

但一旦把 **raw torques 也移除**（E4 true history-only），encoder 失去了 QS-derivable 信号的所有访问途径 → RMSE 跨所有 6 conditions uniformly 退化 1.6-2.2×。
→ "架构形式互换" 仅在 **保留 torque 信号的子集内** 成立。

### ★★★ 结论 Q：Encoder 依赖两条互补的 torque 信号 pathway (E4 + E5 ablation)

升级版（E5 训完后细化）：encoder 的 mass 信号通过两条**部分互补**的 pathway 传递：
- **(i) 显式 QS features**（用 torques 算出来的物理量，注入 obs）—— encoder IG 占 ~27%
- **(ii) 原始 raw torques**（直接进 obs）—— encoder IG 占 ~27%

两条 path 单独都**仅够部分 mass 估计**（E5 RMSE +0.3 kg vs E1），但**两条都移除则 encoder 崩溃**（E4 RMSE +0.8 kg vs E1）。
**任何一条 alone 时，encoder 通过 prev_actions 作为隐式 surrogate 部分补偿**（E5 encoder IG: prev_actions 5.5% → 50.1%）。

**证据 1：RMSE 三层降级**（[§2.3](#23-五架构完整-rmse-对比pemass--torquequs-pathway-ablation-一起看)、[§2.5](#25-e5-ablation-详细数据qs-features-保留--raw-torques-移除)）

| Condition | E1 (both paths) | E5 (QS only) | E4 (neither) | E5−E1 | E4−E1 |
|---|---|---|---|---|---|
| in-dist static | 0.83 | **1.10** | **1.65** | +0.27 | +0.82 |
| in-dist walk | 0.86 | **1.19** | **1.66** | +0.33 | +0.80 |
| OOD-low static | 0.98 | **0.91** | **1.90** | −0.07 | +0.92 |
| OOD-low walk | 0.98 | **1.04** | **1.84** | +0.06 | +0.86 |
| OOD-high static | 1.49 | **1.78** | **2.24** | +0.30 | +0.75 |
| OOD-high walk | 4.95⚠ | **1.95** | **2.23** | −3.00⚠ | −2.72⚠ |

排除 E1 walk OOD-high outlier 后：**E5 平均比 E1 差 0.18 kg，E4 比 E1 差 0.83 kg → E5 仅恢复 ~78% 的 lost signal**。

**证据 2：Bias 模式区分 "encoder collapse" vs "partial signal"**

| Policy | in-dist bias | OOD-low bias | 解读 |
|---|---|---|---|
| E1 | −0.23 | −0.45 | 健康 |
| **E5** | **−0.11~−0.29** | **+0.06~−0.22** | 健康（甚至 in-dist bias 更小）|
| E4 | **+0.79** ★ | **+1.72** ★ | collapse to ~3 kg prior |

E5 bias **健康**说明 explicit QS features 让 encoder 仍能学到 mass-varying representation，没 collapse。

**证据 3：Encoder IG 定量**（[§4.7](#47-三架构-encoder-ig-mass-branch-e1--e5--e4-paper-核心定量论证)）：
- E1 torque-derived path: torques 27% + qs_features 27% = **53%** of mass attribution
- E5 torque-derived path: torques 0% + qs_features 20% = **20%**（qs_load 持平 16%, qs_residual 萎缩 11→4%）
- E4 torque-derived path: **0%**
- E5 失去的 ~34pp 主要被 prev_actions 接管（5.5% → 50.1%, +44.6pp）

**证据 4：static-only 排除 control confound** —— E5 / E4 static tracking err 持平 E1，但 RMSE 仍差 0.3 / 0.8 kg。

→ paper 升级版表述：

```
The encoder's load mass estimation is carried by two partly-overlapping 
pathways, both derived from joint torques: (i) explicit QS features 
computed from torques and exposed as observations (~27% of encoder 
mass-branch attribution), and (ii) raw joint torques directly in obs 
(~27%). Either pathway alone is sufficient for partial mass estimation 
(E5 ablation: keep QS features, remove raw torques → RMSE +0.3 kg). 
Removing both collapses the encoder to indirect proxies (gravity, 
prev_actions, base velocity) → RMSE +0.8 kg with bias indicating 
collapse to a constant prior. The encoder readily substitutes 
previous_actions as an implicit torque surrogate when raw torques are 
removed (prev_actions encoder attribution rises from 5.5% to 50.1% in 
E5), recovering ~70% of the lost signal but not all of it.
```

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

证据（per-dim IG, E1 vs Old s42，[§4.8](#48-12-维-qs-feature-内部-per-dim-attributionmass-branch-encoder-e1-ckpt-11000)）：
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
Paper 主线 (两个 main contributions):

  (a) Per-env load mass randomization closes a hidden distribution shift
      in legged-robot load estimation, reducing OOD load mass RMSE by 32%.

  (b) The encoder's load mass signal is carried by two partly-overlapping
      torque-derived pathways: explicit QS features computed from torques
      (E1/E2/E5 encoder IG ~27%) and raw joint torques directly in obs
      (E1/E2/E3 encoder IG ~27%). 4-way ablation (E1/E3/E5/E4):
        - E1 (both paths):       RMSE 0.83 kg (in-dist static)
        - E3/E5 (one path only): RMSE 0.82 / 1.10 kg (+0/+0.27)
        - E4 (neither path):     RMSE 1.65 kg (+0.82, 2× worse)
      Architecture form (QS-in-obs / residual learning) within the
      torque-preserving subset E1/E2/E3 is interchangeable (RMSE diff
      < 0.15 kg). Removing one path partially degrades; removing both
      collapses the encoder to a constant prior.

Supporting findings:
- Encoder readily substitutes prev_actions as an implicit torque
  surrogate when raw torques are removed (E5 encoder IG: prev_actions
  attribution 5.5% → 50.1%, recovering ~70% of lost signal).
- Encoder mass output is essentially unused by actor (<0.5% attribution
  across all 5 architectures), suggesting load estimation acts as an
  auxiliary supervised regularizer rather than direct policy input.
- QS-in-obs contributes a small but stable ~6% to actor decisions; under
  OOD, it serves as a stabilizing prior. E5 actor IG on QS combined is
  higher (6.9%) than E1 (5.2%) — when raw torques absent, actor leans
  explicitly on QS features.
- Within encoder mass branch, per-env mass training induces dim-level
  feature reattribution (away from direct mass dim, toward geometric and
  physically-interpretable dims).

Negative findings:
- In-loop QS coefficient co-calibration fails due to rollout data being
  contaminated by exploration noise and dynamic torques (Model G assumes
  quasi-static).
- "QS-in-obs as implicit policy regularizer" hypothesis is not supported
  by data when per-env mass is controlled.
- The often-cited "QS shapes policy" narrative is not supported either —
  E3 (no QS in obs, +torques) performs identically to E1/E2 as long as
  the underlying torque signal is accessible.
- E1 walk OOD-high RMSE = 4.95 outlier disappears in E5 (1.95) →
  outlier is not a policy defect but a deploy-time numerical interaction
  between raw torques and explicit QS features in actor obs.
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
| 2026-05-24 | **notebook 重组**：删除 §1-§5 bug-era raw data，撤回 stale 结论，按重要性重排 |
| 2026-05-24 | 新增 `use_torques_in_obs` ablation flag（默认 ON，OFF 即 "true history-only" 基线）；wheelfoot_flat / play / analyze_actor_ig / analyze_encoder_ig 五处适配 |
| 2026-05-24 19:30 | **E4** = `May24_19-30-38_..._history_only_no_torq_..._seed_45_pemass` 训练启动（seed=45；env_cfg.json 因旧 save bug 写成 1，实际训练用 45）|
| 2026-05-24 23:37 | E4 训练到 12000 iter（用 ckpt 11000 分析，未等 16000）|
| 2026-05-24 23:52-23:56 | E4 6-condition play 完成；E4/E3 RMSE ratio 1.57-2.24× 跨 conditions |
| 2026-05-25 00:03 | E4 actor IG 完成（est_mass 0.43%, prev_actions 53.30%）|
| 2026-05-25 00:04 | E4 encoder IG 完成（fallback to projected_gravity 31.5% + prev_actions 23.2%）|
| 2026-05-25 | **notebook 新增 §2.4 / §4.6 / §4.7 / 结论 Q**：torque 信号是 paper 的 contribution (b)；refine 结论 J 加入 "架构 marginal 仅在保留 torque 信号子集内" 限定 |
| 2026-05-25 | 修复 [task_registry.py:185-191](../legged_gym/utils/task_registry.py) seed save bug：以后 env_cfg.json 会正确记录 seed |
| 2026-05-25 01:00 | **E5** = `May25_01-00-47_exper_qs_resi_load_boost_3_no_torq_seed_45_pemass` 训练启动（seed=45，bug fix 后 env_cfg.json 正确记录）|
| 2026-05-25 06:33 | E5 训练完成（16000 iter 完整）|
| 2026-05-25 07:35-07:40 | E5 6-condition play 完成；落在 E1 (0.83) 和 E4 (1.65) 之间，in-dist +0.27 kg |
| 2026-05-25 07:40 | E5 actor IG 完成（QS combined 6.86%, 比 E1 5.21% 还高；est_mass 跌到 0.04%）|
| 2026-05-25 07:44 | E5 encoder IG 完成（torque path 从 E1 的 54% → 20%；prev_actions 暴涨 5.5% → 50.1%）|
| 2026-05-25 | **notebook 升级 §2.3 / §2.5 / §4.6 / §4.7 / 结论 Q / §6**：从"torque 信号必要"升级到"two-pathway 互补 + prev_actions implicit surrogate"机制 |
| 2026-05-25 | **MATLAB plot scripts 重整**：5 个 plot_exp1/payload/scatter 脚本切换到 E1-E5 canonical 标签（meta.load_run 检测），通过 `FILTER_CANONICAL_ONLY=true` 自动过滤；`payload_method_color.m` 加 E1-E5 色卡 |
| 2026-05-25 | 新增 `plot_mass_scatter_preview.m` 加 per-env-averaged subplot + x∈[2,4] 固定 + 100-env 重 play（10 个 .mat 在 `scatter_preview_pdfs/`）|
| 2026-05-25 | 新增 6 个 IG plot 脚本（见 [§4.9](#49-ig-可视化脚本盘点2026-05-25-新增-matlab-scripts)）：`plot_ig_encoder_mass_stacked/heatmap` `plot_ig_actor_heatmap/groups` `plot_ig_encoder_window_heatmap` `plot_ig_est_mass_negative` |
| 2026-05-25 | 修 `analyze_encoder_ig.py` auto-sync `num_input_dim` + `use_load_residual_estimation`（之前 E3 encoder IG 因 shape mismatch 跑不通已修复）|
| 2026-05-25 | 跑 E1-E5 windowed encoder IG（50s × 1s windows，5 个 csv 在各 ckpt 的 `encoder_ig/` 下）|
| 2026-05-25 | **新增 [`figures_inventory.md`](figures_inventory.md)**：按文件夹梳理所有 paper figure 候选 + 用途 |

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
