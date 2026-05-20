# 实验记录（简易索引版）

> **完整原始数据 + 对比表 + 结论请看 [experiment_notebook.md](experiment_notebook.md)**。本文件仅作快速索引。

每个实验跑完后**立即**在这里填一条。一个 entry 一行（小调参用同一行更新；重大变化新开一行）。`exp_tag` 和 .mat 文件名一致，便于事后追溯。

## 记录字段说明

| 字段 | 说明 |
|---|---|
| date | 实验时间，例如 2026-05-19 |
| exp_tag | 给 `play.py --exp_tag` 的标签；同时是 .mat 文件名后缀 |
| commit | `git rev-parse --short HEAD` 得到的 hash |
| policy | `<run_name>/model_<iter>.pt` |
| condition | 实验条件简述（速度、载荷、扰动等） |
| key_params | 跟之前不同的关键参数 |
| mat_path | 实际输出的 .mat 路径 |
| notes | 简要观察 / 备注 |

---

## 实验 1：负载估计准确性

### 1.1 三组训练 + Play 结果汇总（ckpt 11000，static & walk@0.5，trimesh L0, 1024 train envs / 20 play envs）

| 训练配置 | use_qs_in_obs | load_boost | run_name | 备注 |
|---|---|---|---|---|
| **main lb=6** | True | 6 | exper_qs_resi_load_boost_6 | 早期 pilot |
| **main lb=3** | True | 3 | exper_qs_resi_load_boost_3 | pilot 选 final |
| **history_only** | False | 3 | history_only_lb3 | 残差网络对照 |

### 1.2 关键指标对比表（ckpt 11000，单 seed）

**LOAD MASS RMSE (kg)** — 越小越好

| Policy | static [QS] | static [RL] | walk [QS] | walk [RL] |
|---|---|---|---|---|
| main lb=6 | 3.73 | 0.97 | 7.07 | 1.07 |
| **main lb=3** | 5.64 | 0.86 / 1.40 / 0.86 (3 seed) | 5.88 | 0.94 |
| **history_only** | **26.71** ↑ | **0.98** ≈ | **49.03** ↑ | **0.91** ≈ |

**CoM DELTA Y RMSE (m)** — 越小越好

| Policy | static [QS] | static [RL] | walk [QS] | walk [RL] |
|---|---|---|---|---|
| main lb=6 | 0.027 | 0.0114 | 0.039 | 0.0128 |
| **main lb=3** | 0.027 | **0.0103** | 0.039 | 0.0116 |
| **history_only** | 0.027 | 0.0120 | 0.031 | 0.0123 |

**收敛时间 conv_time (s) + reach %** — RL 都达到 100%

| Policy | static [RL] | walk [RL] |
|---|---|---|
| main lb=6 | 0.12s | 0.09s |
| main lb=3 | 0.08–0.43s (3 seed) | 0.19s |
| history_only | 0.06s | 0.03s |

### 1.3 关键发现（重要！可能改变 paper narrative）

**发现 A：RL encoder 在 main 和 history_only 几乎打平**
- 三种 policy 的 RL mass RMSE 都在 0.86–1.07 kg 范围
- 三种 policy 的 RL dcom_y 都在 0.010–0.012 m 范围
- **结论：把 QS 显式注入 obs 没有显著提高 encoder 估计精度**——encoder 自己能从原始 obs history 学到一样好

**发现 B：QS analytical 在 history_only policy 下崩溃**
- main lb=3 下 QS mass RMSE = 5.6 kg
- history_only 下 QS mass RMSE = **26.7 kg (static) / 49.0 kg (walk)**，5-8 倍恶化
- per-env std 比 mean 还大（部分 env QS 估计上百 kg）
- **解读**：main method 的 policy **隐式学会维持 QS 友好姿态**（cos_thigh 不接近 0），让 Model C 工作；history_only policy 没有这个约束，cos_thigh 偶尔趋零导致公式分母爆炸

**发现 C：play-to-play 方差对 mass 很大（同 policy）**
- 同 lb=3 ckpt 11000 static 三个 seed: mass RMSE = 0.86, 1.40, 0.86 → std=0.31 kg (~30%)
- CoM 指标方差极小（dcom_x/y std < 0.001）
- **paper 必须 mass 报 3-seed 均值±std；CoM 单 seed 可信**
- **原因**：20 envs 不是"同实验 20 次"，而是 20 个不同 (load mass / push timing / 摩擦 / 推力等) 组合，换 seed = 重新抽 20 个组合

### 1.4 对 paper 的影响（核心 narrative 调整）

| 维度 | 原 paper 假设 | 数据支持的结论 |
|---|---|---|
| Encoder 准确性 | QS 先验显著改善 | ❌ **不显著（差异在噪声内）** |
| Policy 行为 | 不区分 | ✅ **main method 维持 QS 友好姿态** |
| Hybrid 系统鲁棒性 | 不重点 | ✅ **main method 下 QS 公式仍能用作 sanity check** |

**建议 paper 重构**：
- §4.1 (原 main result) 改为 "Estimation accuracy is comparable across architectures"（honest negative finding）
- §4.2 升级为核心：**"Including QS as obs feature shapes policy behavior, keeping the analytical formula in its valid regime"**
- 加 §4.3 OOD 实验和 policy 姿态分布对比，强化"prior 影响 policy 而非 encoder"故事

类似 negative-ish 结果的 robotics paper 不少（Lee et al. 2020 Science Robotics, Margolis et al. CoRL 2022, Pinto et al. asymmetric actor critic, 整个 auxiliary loss 文献），**完全可发**。

### 1.5 详细原始数据 log

| date | exp_tag | commit | policy | condition | key_params | mat_path | notes |
|---|---|---|---|---|---|---|---|
| 2026-05-19 | lb6 static / walk_0.5 | (前版) | exper_qs_resi_load_boost_6/model_11000.pt | static, walk@0.5; trimesh L0 | load_boost=6, use_qs_in_obs=True | logs/.../play_data_*_lb6_*.mat | static [QS] mass=3.73 [RL]=0.97; walk [QS]=7.07 [RL]=1.07 |
| 2026-05-19 | lb3 static (3 seeds) / walk_0.5 | (前版) | exper_qs_resi_load_boost_3/model_11000.pt | static (3 seed) + walk@0.5 | load_boost=3, use_qs_in_obs=True | logs/.../play_data_*_lb3_*.mat | static [RL] mass=0.86 / 1.40 / 0.86 (seed=42/43/another)；walk [RL]=0.94；**Pilot 选 lb=3 final** |
| 2026-05-20 | lb3_static_compare_historyonly | 08d4c40 | history_only_lb3/model_11000.pt | static + walk@0.5; trimesh L0 | use_qs_in_obs=False, load_boost=3, seed=42 | logs/.../play_data_20260520-08*_*.mat | **关键对照**：[RL] mass=0.98 (跟 main 持平), [QS] mass=26.71 (崩溃)；揭示 QS 价值在 shape policy 而非 encoder |

## 实验 2：控制增益

| date | exp_tag | commit | policy | condition | key_params | mat_path | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## 实验 3：残差网络抗扰

| date | exp_tag | commit | policy | condition | key_params | mat_path | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## 实验 4：消融

| date | exp_tag | commit | policy | condition | key_params | mat_path | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

---

## 推荐的 git 备份流程

每次跑完一个有意义的实验：

```bash
# 1. 看下输出文件
ls -lh logs/wheelfoot_flat/<exp>/exported/play_data_*.mat

# 2. 提交（推荐把 .mat 加 .gitignore，只 commit log）
git add docs/experiment_log.md
git commit -m "exp1 静态 5kg(0.1,0): RMSE m=X.XX kg, com_x=X.XX m, com_y=X.XX m"

# 3. 如果换了 policy/config 也一起 commit
git add legged_gym/envs/...  
git commit -m "..."
```

如果 .mat 文件本身想入版本控制（小文件，单次几 MB）：
```bash
git add logs/wheelfoot_flat/<exp>/exported/play_data_<exp_tag>.mat
git commit -m "..."
```

但更推荐 .mat 不入库，只在 log 表里记 mat_path，避免仓库膨胀。
