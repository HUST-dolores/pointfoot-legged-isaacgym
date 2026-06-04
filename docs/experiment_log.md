# 实验记录（简易索引版）

> **完整数据 + 结论请看 [experiment_notebook.md](experiment_notebook.md)**。本文件是 ckpt/play 的快速索引表，不再放任何结论。
>
> **2026-05-24 重组**：删除了 1.2-1.4 的 bug-era 数字和已撤回的旧 narrative；1.5 raw data 表更新为当前 E1/E2/E3 主轴。
> Quick conclusions 不在本文件维护——任何"哪个 policy 准/不准"的结论请直接查 notebook §5。

每个实验跑完后**立即**在 [§1.5 raw data log](#15-raw-data-log) 加一行。一个 entry 一行（小调参用同一行更新；重大变化新开一行）。`exp_tag` 和 .mat 文件名一致，便于事后追溯。

---

## ★ Current canonical ckpts (paper 主轴)

| 简称 | run_name | 配置 |
|---|---|---|
| **E1 main** | `exper_qs_resi_load_boost_3_seed_45_pemass` | QS-in-obs + residual + torques + pemass + seed=45 |
| **E2 direct** | `exper_qs_noresi_load_boost_3_seed_45_pemass` | QS-in-obs + no-residual + torques + pemass + seed=45 |
| **E3 histonly** | `exper_history_only_load_boost_3_seed_45_pemass` | history only + torques + pemass + seed=45 |
| **E4 true_hist** | `May24_19-30-38_exper_history_only_no_torq_load_boost_3_seed_45_pemass` | history only **no-torques** + pemass + seed=45 |
| **E5 qs_only_path** | `May25_01-00-47_exper_qs_resi_load_boost_3_no_torq_seed_45_pemass` | QS + residual, **no raw torques** + pemass + seed=45 |

所有当前 paper 数据都来自 E1/E2/E3/E4/E5 (ckpt 11000)。其余 ckpt（no-pemass baselines、co-cal 试错版本）仅作历史参照——见 notebook 附录 A/B。

→ 数字、结论、IG 表全在 [experiment_notebook.md §2–§5](experiment_notebook.md#tldr-paper-最核心-findings)。

**所有 5 个 ckpt 实际训练时 seed=45**（由 `make_env` 里的 `set_seed(env_cfg.seed=45)` 应用）。
注：E1-E4 的 `env_cfg.json` 因为 [task_registry.py 旧 save bug](../legged_gym/utils/task_registry.py) 错误地写了 1；实际训练的 numpy/torch 种子是 45（train_cfg.json 里的 seed=45 才是真值）。该 bug 已在 2026-05-25 修复，**E5 的 env_cfg.json 正确记录 seed=45**（验证 bug fix 成功）。

---

## 记录字段说明（仅 §1.5 raw data 表用）

| 字段 | 说明 |
|---|---|
| date | 实验时间，例如 2026-05-19 |
| exp_tag | 给 `play.py --exp_tag` 的标签；同时是 .mat 文件名后缀 |
| commit | `git rev-parse --short HEAD` 得到的 hash |
| policy | `<run_name>/model_<iter>.pt` |
| condition | 实验条件简述（速度、载荷、扰动等） |
| key_params | 跟之前不同的关键参数 |
| mat_path | 实际输出的 .mat 路径 |
| notes | 简要观察 / 备注（避免写结论，结论统一放 notebook） |

---

## 实验 1：负载估计准确性

### 1.1 当前 paper 主轴 (E1 / E2 / E3 / E4 / E5, pemass on, ckpt 11000)

5 架构 in-dist + OOD-low + OOD-high 完整 RMSE / IG 数据见
[notebook §2.3 五架构完整 RMSE 对比](experiment_notebook.md#23-五架构完整-rmse-对比pemass--torquequs-pathway-ablation-一起看)、
[§2.4 E4 详细](experiment_notebook.md#24-e4-ablation-详细数据true-history-only连-torques-都不喂)、
[§2.5 E5 详细](experiment_notebook.md#25-e5-ablation-详细数据qs-features-保留--raw-torques-移除) 和
[§4 actor / encoder IG 分析](experiment_notebook.md#4-actor-integrated-gradients-分析)。

| 训练配置 | use_qs_in_obs | use_residual | use_torques_in_obs | per-env mass | run_name |
|---|---|---|---|---|---|
| **E1 main** | True | True | True | True | exper_qs_resi_load_boost_3_seed_45_pemass |
| **E2 direct** | True | False | True | True | exper_qs_noresi_load_boost_3_seed_45_pemass |
| **E3 histonly** | False | N/A | True | True | exper_history_only_load_boost_3_seed_45_pemass |
| **E4 true_hist** | False | N/A | **False** | True | May24_19-30-38_exper_history_only_no_torq_load_boost_3_seed_45_pemass |
| **E5 qs_only_path** | True | True | **False** | True | May25_01-00-47_exper_qs_resi_load_boost_3_no_torq_seed_45_pemass |

→ 主结论 [结论 I-N + Q](experiment_notebook.md#5-当前可信结论-按重要性排序)。本文件不重复贴。

**E4/E5 简表速记**（详见 notebook §2.3-2.5）：

| Condition | E1 | E5 (no torq) | E4 (no qs, no torq) | E5 vs E1 | E4 vs E1 |
|---|---|---|---|---|---|
| in-dist static | 0.83 | 1.10 | 1.65 | +0.27 | +0.82 |
| in-dist walk | 0.86 | 1.19 | 1.66 | +0.33 | +0.80 |
| OOD-low static | 0.98 | 0.91 | 1.90 | −0.07 | +0.92 |
| OOD-high static | 1.49 | 1.78 | 2.24 | +0.30 | +0.75 |

E5 落在 E1 和 E4 之间（gap 中 ~30%）→ **two-pathway 互补**：移除 raw torques 但保留 QS features，encoder 通过 prev_actions 作为 implicit surrogate 部分补偿（IG 5.5% → 50.1%），仅退化 ~0.3 kg；两条都移除则 collapse to prior，退化 ~0.8 kg。

### 1.2 No-pemass baseline ckpts（仅对照用）

| 配置 | run_name | 用途 |
|---|---|---|
| Old main_s42 | exper_qs_resi_load_boost_3 | pemass 改善对照的主参照（[notebook §2.2](experiment_notebook.md#22-e1-vs-old-main_s42per-env-mass-单独贡献)） |
| Old main_s43 | exper_qs_resi_load_boost_3_seed_43 | multi train-seed |
| Old direct | exper_qs_noresi_load_boost_3 | no-pemass direct |
| Old history_only | history_only_lb3 / exper_history_only_load_boost_3 | no-pemass history |
| Old lb=6 | exper_qs_resi_load_boost_6 | 早期 pilot；universal coef outlier |

完整 baseline 数字见 [notebook 附录 A](experiment_notebook.md#附录-ano-pemass-baseline-数据)。

### 1.3 失败的 co-cal ckpts（不再使用）

- `exper_qs_resi_load_boost_3_seed_43_cocal_buggy` — v1，per-spawn API 用错
- `exper_qs_resi_load_boost_3_seed_43_cocal_v2` — v2，per-env density 但 α≈0

→ 失败原因见 [notebook 结论 N](experiment_notebook.md#结论-nco-calibration-在当前架构下不可行paper-12-句提及)，paper 最多 1-2 句提及。

### 1.4 当前 deployed QS 公式

Model G (7 params)，universal coefs 在 [`wheelfoot_flat.py:368-382`](../legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py#L368-L382)。
完整对比 / 系数表见 [notebook §3](experiment_notebook.md#3-model-g-qs-公式部署--universal-系数)。

### 1.5 Figures 清单 + paper 候选

所有 PDF / PNG 路径、来源脚本、用途详见 **[`figures_inventory.md`](figures_inventory.md)**。

5 个 paper-grade 文件夹（都在 `logs/wheelfoot_flat/WF_TRON1A/exported/` 下）：

| 文件夹 | 数量 | 用途 |
|---|---|---|
| 顶层 `Figure_1-9_*.pdf` | 9 | 你手存的 paper 候选 |
| `ig_pdfs/` | 3 | IG 分析，主图 `ig_encoder_mass_stacked.pdf` |
| `exp1_pdfs/` | 2 | RMSE / bias bar 图（walk-only）|
| `scatter_preview_pdfs/` | 10 | encoder mass 散点（100 envs, per-timestep + per-env-avg）|
| `paper_figures_payload/` | 44 | 旧版自动产出，存档 |

新增 MATLAB plot 脚本（2026-05-25, 见 [notebook §4.9](experiment_notebook.md#49-ig-可视化脚本盘点2026-05-25-新增-matlab-scripts)）：
- `plot_ig_encoder_mass_stacked.m` ★（auto-PDF）
- `plot_ig_encoder_window_heatmap.m` ★（rollout 时间 × 输入 heatmap, 50s × 1s windows, batch 模式 5 ckpt）
- `plot_ig_encoder_mass_heatmap.m` / `plot_ig_actor_heatmap.m`（static heatmap）
- 改造 `plot_mass_scatter_preview.m` / `plot_exp1_estimation_rmse.m` / `plot_summary_condition_bars.m` 切换到 E1-E5 + walk-only

### 1.6 raw data log

新加 play 时直接在表末追加一行。**不在 notes 写结论**——结论统一放 notebook §5。

| date | exp_tag | commit | policy | condition | key_params | mat_path | notes |
|---|---|---|---|---|---|---|---|
| 2026-05-24 | E1 in-dist/OOD ×6 | 961fcd4 | E1/model_11000.pt | static + walk × in-dist/[1,2]/[4,6] | pemass=True, G_all deployed | logs/.../play_data_*_seed_45_pemass_ckpt11000_*.mat | RMSE 范围 0.83-4.95（walk OOD-high outlier）见 notebook §2.3 |
| 2026-05-24 | E1 IG × 7 ckpts | 961fcd4 | E1/model_{3-16}000.pt | in-dist + 2 OOD | analyze_actor_ig.py | logs/.../actor_ig_summary_*.csv | 跨 ckpt qs_combined 5.15-7.23%，est_mass < 0.10%。见 notebook §4.3-4.4 |
| 2026-05-24 | E3 in-dist/OOD ×6 | 961fcd4 | E3/model_11000.pt | static + walk × in-dist/[1,2]/[4,6] | history_only + pemass | logs/.../play_data_*_seed_45_pemass_*.mat | 三架构 range < 0.15 kg；见 notebook §2.3 |
| 2026-05-24 | E3 IG × 7 ckpts | 961fcd4 | E3/model_{3-16}000.pt | in-dist + 2 OOD | analyze_actor_ig.py | logs/.../actor_ig_summary_*.csv | est_lin_vel OOD-low +5.75pp；est_mass < 0.20%。见 notebook §4.4 |
| 2026-05-24 | E2 in-dist/OOD ×6 | 3610996 | E2/model_11000.pt | static + walk × in-dist/[1,2]/[4,6] | direct + pemass | logs/.../play_data_*_seed_45_pemass_*.mat | 跟 E1/E3 同档，无 walk OOD-high outlier。见 notebook §2.3 |
| 2026-05-24 | E2 IG ckpt 11000 | 3610996 | E2/model_11000.pt | in-dist | analyze_actor_ig.py | logs/.../actor_ig_summary_*.csv | qs_combined 5.47% ≈ E1 5.58%；residual 不影响 actor 对 QS 依赖。见 notebook §4.2 |
| 2026-05-24 | E1 offline cal | 961fcd4 | E1/model_11000.pt | QS grid 4 mass × 16 com | collect_calibration_data.py | logs/.../cal_data_E1_*.npz | offline cal RMSE = 0.689 kg；policy 仍 QS-fittable。见 notebook §A.3 |
| 2026-05-23 | per-env mass discovery | 016902d | v2/model_11000.pt | OOD-low static | accidental finding | logs/.../v2_cocal_*.mat | 触发 pemass narrative 的偶然实验；详见 notebook 时间线 |
| 2026-05-24 | E4 训练启动 | 7c78b7c | May24_19-30-38_..._no_torq_..._seed_45_pemass | 同 E3 配置 + use_torques_in_obs=False | num_obs=28, seed=45 (env_cfg.json 因旧 save bug 写成 1，实际训练用 45) | logs/.../model_*.pt | true history-only baseline，只到 12000 iter (paper 用 11000) |
| 2026-05-24 | E4 in-dist/OOD ×6 | 7c78b7c | E4/model_11000.pt | static + walk × in-dist/[1,2]/[4,6] | history_only + pemass + no-torques | logs/.../play_data_*_torq0_seed42_ckpt11000_load*.mat | RMSE 1.65-2.24 跨 conditions；E4/E3 ratio 1.57-2.24×。见 notebook §2.3-2.4 |
| 2026-05-25 | E4 actor IG | 7c78b7c | E4/model_11000.pt | in-dist | analyze_actor_ig.py | logs/.../actor_ig_summary_20260525_000350.csv | est_mass 0.43% (5× E3 但仍 < 1%)；prev_actions 53% (E3=66%)；fallback to dof_vel/gravity。见 notebook §4.6 |
| 2026-05-25 | E4 encoder IG | 7c78b7c | E4/model_11000.pt | in-dist | analyze_encoder_ig.py | logs/.../encoder_ig_summary_20260525_000432.csv | mass 信号 fallback 到 projected_gravity 31.5% + prev_actions 23.2%。E1 中 torques+QS = 53% 的 attribution 在 E4 完全消失。见 notebook §4.7 |
| 2026-05-25 01:00 | E5 训练启动 | 7c78b7c | May25_01-00-47_..._qs_resi_..._no_torq_..._seed_45_pemass | QS + residual + no-torques 完整跑到 16000 iter | num_obs=40, seed=45 (env_cfg.json 正确记录，task_registry seed bug fix 验证) | logs/.../model_*.pt | E5 = E1 minus raw torques but keep QS features，paper contribution (b) disambiguation |
| 2026-05-25 | E5 in-dist/OOD ×6 | 7c78b7c | E5/model_11000.pt | static + walk × in-dist/[1,2]/[4,6] | qs+resid + pemass + no-torques | logs/.../play_data_*_qs1_resid1_torq0_*_seed42_ckpt11000_*.mat | RMSE 1.10/1.19/0.91/1.04/1.78/1.95，落在 E1 (0.83) 和 E4 (1.65) 之间。E5 vs E1 ≈ +0.27 kg in-dist。见 notebook §2.3, §2.5 |
| 2026-05-25 | E5 actor IG | 7c78b7c | E5/model_11000.pt | in-dist | analyze_actor_ig.py | logs/.../actor_ig_summary_20260525_074023.csv | QS combined 6.86% (比 E1 5.21% 还高，没了 raw torques 后 actor 更依赖 QS feature)；est_mass 0.04% (vs E4 的 0.43%，回到健康水平)。见 notebook §4.6 |
| 2026-05-25 | E5 encoder IG | 7c78b7c | E5/model_11000.pt | in-dist | analyze_encoder_ig.py | logs/.../encoder_ig_summary_20260525_074423.csv | **核心发现**：torque path 从 E1 的 54% → 20%；qs_load 持平 (15.39 → 15.83%)，qs_residual 萎缩 (11.20 → 3.92%)；**prev_actions 暴涨 5.5% → 50.1%** 作为 implicit torque surrogate。见 notebook §4.7 |

> **早期 §1.5 行（lb=6 / lb=3 / history_only 各 1 行）已删除**——它们的数字是 bug-era（[notebook §2.1](experiment_notebook.md#21-hidden-distribution-shift修复前的训练-bug) 的 hidden distribution shift bug 修复前采集），不应作为当前参照。原始 .mat 仍在 `exported/` 目录下，需要时直接读。

---

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

# 2. 提交（推荐 .mat 加 .gitignore，只 commit log）
git add docs/experiment_log.md
git commit -m "exp1 <exp_tag>: <短描述>"

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

---

## 维护原则

- 本文件只放 ckpt / .mat / commit / exp_tag 索引；**不放结论、不放对比解读**。
- 任何结论性表述统一放 [experiment_notebook.md](experiment_notebook.md)，避免两份文件出现矛盾。
- raw data 表 (§1.5) 的 notes 字段允许写**一句关键现象**或 **指向 notebook section 的链接**，但不允许写"哪个 policy 好"之类的判断语。
