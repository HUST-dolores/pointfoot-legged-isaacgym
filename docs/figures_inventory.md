# Figures Inventory

按文件夹梳理所有产出的图像，列出每张图的来源脚本、内容、主要作用。

所有路径以 `logs/wheelfoot_flat/WF_TRON1A/exported/` 为根。

---

## 1. `exported/` (顶层) — 用户手存的 paper 草图

来自 2026-05-25 11:17 ~ 13:30 你在 MATLAB GUI 里手动整理 / 美化后导出的 paper figure 候选。
这些**不是脚本自动生成**，是你定稿前的手工产出。

| 文件 | 推测内容（按文件名） | 用途 |
|---|---|---|
| `Figure_1_E1_timeseries.pdf` | E1 mass / dCoM 随时间曲线（真值 vs encoder vs QS） | paper main figure — E1 working 示例 |
| `Figure_2_E3_timeseries.pdf` | E3 同上 | E3 working 示例（无 QS 但保留 torques） |
| `Figure_3_E4_timeseries.pdf` | E4 同上 | E4 broken 示例（无 QS 也无 torques） |
| `Figure_4_E5_timeseries.pdf` | E5 同上 | E5 partial 示例（QS 在但无 raw torques） |
| `Figure_9_E2_timeseries.pdf` | E2 同上 | E2 working 示例（QS direct + torques） |
| `Figure_5_Encoder_mass_RMSE.pdf` | encoder mass RMSE bar 图（in-dist） | 5 method 静态对比 |
| `Figure_6_Encoder_mass_RMSE_OOD.pdf` | encoder mass RMSE bar 图（OOD） | OOD 鲁棒性对比 |
| `Figure_7_QS_mass_RMSE.pdf` | QS-only mass RMSE bar 图 | QS-only baseline 对比 |
| `Figure_8_QS_mass_RMSE_compare.pdf` | QS vs encoder 对比 bar 图 | encoder learning 必要性 |

---

## 2. `exported/ig_pdfs/` — Integrated Gradients 分析 PDF

来源脚本：`legged_gym/scripts/ig_*.m`。

| 文件 | 来源脚本 | 内容 | 主要作用 |
|---|---|---|---|
| **`ig_encoder_mass_stacked.pdf`** ★ | `ig_encoder_mass_stacked.m` | 5 个 ckpt（E1/E2/E3/E5/E4）的 encoder mass-branch 归因堆叠柱状图，分层显示 raw torques / QS load / QS residual / prev_actions / 其他；顶部标注 "torque pathway sum"（54% → 51% → 48% → 20% → 0%） | **paper contribution (b) 主图**：直接证明 two-pathway torque signal 机制 |
| `ig_actor_groups.pdf` | `ig_actor_groups.m` | 5 method × ~10 input group 的 actor IG grouped bar | 你判断"没用"，留作备份 |
| `ig_est_mass_negative.pdf` | `ig_est_mass_negative.m` | 5 个 ckpt 的 est_mass attribution 单 bar 图 + uniform baseline 虚线 | 你判断"没用"，留作备份 |

**还未保存为 PDF 的 IG 脚本**（默认开 figure 不存盘）：
- `ig_encoder_mass_heatmap.m` — 5×8 静态 heatmap（method × group, encoder mass）
- `ig_actor_heatmap.m` — 5×10 静态 heatmap（method × group, actor IG）
- `ig_encoder_window_heatmap.m` — **rollout 时间 × 输入 heatmap**（每 1s 一格，raw+filtered 合并，固定输入顺序）—— batch mode 一次出 5 张

---

## 3. `exported/exp1_pdfs/` — RMSE / Bias 柱状图

来源脚本：`plot_exp1_estimation_rmse.m`（自动保存 PDF）。

| 文件 | 内容 | 主要作用 |
|---|---|---|
| `exp1_rmse.pdf` | 3 subplot（mass / dCoM-x / dCoM-y RMSE），5 method × QS+Encoder 配对柱 | **paper main**：误差大小对比 |
| `exp1_bias.pdf` | 同上，但显示 signed bias（含零线） | 区分 noise-dominated vs systematic-bias |

> Walk-only 数据，load 2-4 kg, seed 42, ckpt 11000。详见 [experiment_notebook.md §2.6.2/2.6.5](experiment_notebook.md)。

---

## 4. `exported/scatter_preview_pdfs/` — encoder mass 散点图

来源脚本：`plot_mass_scatter_preview.m`。10 张 PDF（5 method × 2 condition），每张 2 个 subplot（左 per-timestep + 右 per-env-averaged）。

| 文件命名 | 内容 |
|---|---|
| `scatter_static_E1.pdf` ... `scatter_static_E5.pdf` | static 条件下 encoder mass 估计真值 vs 估计值 |
| `scatter_walk_E1.pdf` ... `scatter_walk_E5.pdf` | walk 条件下同上 |

**用途**：可视化 encoder 估计的瞬时方差 + 系统偏差。E4 的散点会看到"水平条带集中在 y≈3"的 collapse-to-prior 模式，最有视觉冲击力。

**配置**：x ∈ [2, 4]（in-dist 真值范围），y 自动，100 envs × 2000 ts → 25K 散点 downsampled。

---

## 5. `exported/paper_figures_payload/` — 旧版自动生成 paper figures

来源脚本：`plot_payload_experiments_here(true, 'all')`（自动存盘）。

**44 个文件**，包括：

| 文件 | 内容 | 状态 |
|---|---|---|
| `control_metrics_bars.pdf/.png` | 5 method 的 tracking error / torque RMS / power 对比 | 还可用，对应 §2.4/2.5 control 分析 |
| `estimation_rmse_bars.pdf/.png` | 5 method × 6 condition × {mass, dcom_x, dcom_y} RMSE 全表 | 替代版见 `exp1_pdfs/` |
| `heatmap_encoder_mass_by_load.pdf` | encoder mass RMSE 按 load range 分组 heatmap | 较老版本 |
| `heatmap_encoder_mass_rmse.pdf` | encoder mass RMSE method × condition heatmap | 较老版本 |
| `heatmap_qs_mass_rmse.pdf` | QS mass RMSE method × condition heatmap | 较老版本 |
| `payload_summary_table.csv/.mat` | 全部 summary 表的源数据 | 数据备份 |
| `scatter_mass_play_data_*.pdf` (≈8 个) | 早期单 play 散点 | 已被 `scatter_preview_pdfs/` 取代 |
| `timeseries_play_data_*.pdf` (≈8 个) | 早期单 play timeseries | 已被你手存的 `Figure_1-4,9` 取代 |

> 这些是 `plot_payload_experiments.m` 的旧 `saveOutputs=true` 模式批量产出。可以保留作存档，paper 不必再用。

---

## 6. `exported/plots_*/` — 每次 play 的诊断 PNG

来源：每次跑 `play.py`（不带 `--exit_after_save`）时 `logger.plot_states()` 自动生成。
**83 个文件夹** × 6 张 PNG = 498 张图，每个对应一次 play。

每个文件夹里的 6 张 `image*.png` 包含：
1. dof_pos / dof_pos_target vs 时间
2. dof_torque / filtered_dof_torque vs 时间
3. base_vel_x / command_x vs 时间
4. encoder vs QS vs true 的 mass / dCoM 时间序列
5. CoM 估计误差
6. push event timeline

**用途**：单 play 诊断（看某次播放具体跑得怎么样）。**paper 不用**，但调试时有用。

> 文件夹名规则：`plots_<timestamp>_lb3_qs{0|1}_resid{0|1}[_torq0]_<motion>_seed<S>_ckpt<C>_load<lo>-<hi>` —— 一眼能看出对应哪个 ckpt / 哪种 condition。

---

## 7. `exported/policies/` — 导出的 onnx / jit 模型

`play.py` 启动时自动 export 的 deployable 模型。

| 文件 | 内容 |
|---|---|
| `policy.onnx` | actor 网络（ONNX 格式，可在 robot SDK 部署） |
| `encoder.onnx` | encoder 网络 |
| `policy.pt` | actor TorchScript |

> 部署专用，paper 不直接引用。

---

# 各文件夹快速对照表

| 文件夹 | 数量 | 用途 |
|---|---|---|
| `exported/` (顶层 `Figure_*.pdf`) | 9 | **手存 paper 候选**（你自己定稿用）|
| `exported/ig_pdfs/` | 3 | **paper IG 主图**（主要是 `ig_encoder_mass_stacked`）|
| `exported/exp1_pdfs/` | 2 | **paper RMSE/bias 主图** |
| `exported/scatter_preview_pdfs/` | 10 | **paper scatter 候选 / 诊断** |
| `exported/paper_figures_payload/` | 44 | 旧版自动产出，存档 |
| `exported/plots_*/` | 83 文件夹 | **每次 play 的诊断图**，调试用，paper 不用 |
| `exported/policies/` | 3 | 部署模型 |

---

# Paper figure 候选清单（推荐先看的）

按"对 paper 的重要性"排序：

1. **`Figure_1-4,9_*_timeseries.pdf`**（你手存的）—— 时间曲线，最直观的 contribution (b) 视觉化
2. **`ig_pdfs/ig_encoder_mass_stacked.pdf`** —— two-pathway 机制的 quantitative 证据
3. **`Figure_5/6/7/8_*_RMSE.pdf`**（你手存的）—— in-dist + OOD 全量化对比
4. **`exp1_pdfs/exp1_rmse.pdf` + `exp1_bias.pdf`** —— walk-only 简洁版 RMSE/bias 对比
5. **`scatter_preview_pdfs/scatter_walk_E1/E5/E4.pdf`** —— per-env scatter 显示 collapse 模式
6. **(脚本待存)** `ig_encoder_window_heatmap` 输出 —— 时间维度的 IG 演化

剩余 `paper_figures_payload/*` 和 `plots_*` 主要是历史 / 诊断，paper 不用。
