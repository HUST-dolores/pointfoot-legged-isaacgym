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
| **E1 main** | `exper_qs_resi_load_boost_3_seed_45_pemass` | QS-in-obs + residual + pemass + seed=45 |
| **E2 direct** | `exper_qs_noresi_load_boost_3_seed_45_pemass` | QS-in-obs + 无 residual + pemass + seed=45 |
| **E3 histonly** | `exper_history_only_load_boost_3_seed_45_pemass` | history only + pemass + seed=45 |

所有当前 paper 数据都来自 E1/E2/E3 (ckpt 11000)。其余 ckpt（no-pemass baselines、co-cal 试错版本）仅作历史参照——见 notebook 附录 A/B。

→ 数字、结论、IG 表全在 [experiment_notebook.md §2–§5](experiment_notebook.md#tldr-paper-最核心-findings)。

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

### 1.1 当前 paper 主轴 (E1 / E2 / E3, pemass on, ckpt 11000)

3 架构 in-dist + OOD-low + OOD-high 完整 RMSE / IG 数据见
[notebook §2.3 三架构完整 RMSE 对比](experiment_notebook.md#23-三架构完整-rmse-对比确认-pemass-改善跨架构成立) 和
[§4 actor IG 分析](experiment_notebook.md#4-actor-integrated-gradients-分析)。

| 训练配置 | use_qs_in_obs | use_residual | per-env mass | run_name |
|---|---|---|---|---|
| **E1 main** | True | True | True | exper_qs_resi_load_boost_3_seed_45_pemass |
| **E2 direct** | True | False | True | exper_qs_noresi_load_boost_3_seed_45_pemass |
| **E3 histonly** | False | N/A | True | exper_history_only_load_boost_3_seed_45_pemass |

→ 主结论 [结论 I-N](experiment_notebook.md#5-当前可信结论-按重要性排序)。本文件不重复贴。

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

### 1.5 raw data log

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
