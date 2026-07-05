# figs_chx — 第X章 Co-design 论文图

双轮足 co-design（腿长 + 电机 联合优化）的论文图。数据来自工作站
`192.168.3.23` 上的 BO 日志 + table2 eval npz（每候选/每策略一份 per-env 记录）。

## 流水线

```
工作站 npz + BO日志  ──prep_data.py──►  data/codesign_figs_data.mat  ──*.m──►  out/*.pdf,*.png
                     (python, 抽取+校验)                              (MATLAB, 出图)
```

- **prep_data.py**（python）：解析 3 个 BO 日志（run→(thigh,shank,motor,fitness)）+ 每候选
  table2 npz（per-env 的 scen/rew/fell），按 scenario 聚合成 per-scenario mean_rew，写 `.mat`。
  内置校验：每候选从 npz 重建的「4场景等权均值」应 == 日志里的 FITNESS（实测最大误差 <0.005 ✓）。
- **\*.m**（MATLAB）：读 `.mat` 出图。运行 `make_all` 一键全出到 `out/`。
- **_preview.py**：matplotlib 预览（非交付物，本机无 MATLAB 时快速看数），数值与 `.m` 一致。

## scenario id 映射（代码写死, table2_eval.py:49）
`0=obstacle(越障) 1=slope(爬坡) 2=load(负载) 3=accel(加速)`

## 图清单

| 脚本 | 图 | 数据 | 成本 |
|---|---|---|---|
| `fig1_bo_convergence.m` | BO 收敛(2D仅腿/3D腿+电机) | bo.log/bo3d_full.log | 免费 |
| `fig2_table2_bars.m` | Table-II: spatial-DR vs No-DR 逐场景+综合 | table2_final/table2_s2 npz | 免费 |
| `fig3_finetune_curve.m` | 微调效率(+400=89%,+1500追平) | finetune_val npz | 免费 |
| `fig4_scenario_vs_motor.m` | 电机 no-free-torque(综合fitness内部最优 微调vs零样本 + 逐场景机制) | 路B 电机扫描 npz | 路B(~1.1h) |
| `fig5_scenario_landscape.m` | 逐场景性能地形 vs 腿长 | 2D BO 候选 npz | 几乎免费 |
| `fig6_foot_height.m` | 抬腿峰值净高 vs 腿长 (+摔倒率) | 路B footheight npz | 路B(~8min) |
| `fig7_robustness.m` | ξ* 鲁棒性 (平坦盆地 + 权重敏感) | 3D BO 21候选 (fig7_robust.npz) | 免费(纯分析) |

路B 数据: `prep_pathB.py --pathB <scratchpad>/npz/pathB`(读 motor_k*.npz 零样本 + pathB_ft 微调 +
footheight.npz)→ 追加字段到同一 `.mat`。新增 eval 脚本 `legged_gym/scripts/foot_height_grid.py`。

## 复现

```bash
# 1) 数据（本机；npz 已 rsync 到 scratchpad）
python3 prep_data.py --npz_root <scratchpad>/npz --logs <scratchpad>/logs \
        --out data/codesign_figs_data.mat
# 2) 出图（MATLAB R2020a+）
matlab -batch "run('make_all.m')"      # 或在 MATLAB 里 run make_all
```

## 关键结论（图里讲的故事）
- **fig1**：3D(腿+电机, fit 227.6) > 2D(仅腿, fit 176.7)；BO 在 init 后快速收敛。
- **fig2**：spatial-DR 逐场景全面 ≥ No-DR，综合 +33~40（双seed 已确认方向）。
- **fig3**：generalist 微调 400 步达 specialist 89%、1500 步追平 → 框架成立，BO 每候选省 4~15×。
- **fig4**：★电机 "no free torque" —— 定腿 ξ* 扫电机，**综合 fitness 有内部最优 k≈1.0**(=3D BO ξ*电机)，
  且**仅微调后显现**(零样本饱和)。机制：质量型 obstacle/accel↓ · 扭矩型 slope/load↑ 交汇。
  弱电机差、最大电机不更好。(内部最优在综合层面; 单场景 load 在 ξ*腿长是饱和型。)
- **fig5**：各场景**均偏短腿**(低CoM稳)，负载/爬坡最敏感、越障最不敏感（净高 proxy 无"够得着"激励）
  → ξ* 落在短腿 (0.94,0.91)。
- **fig6**：抬腿峰值净高随大腿 0.8→1.2 单调升 **0.23→0.30m**(长腿够得着)，但摔倒率 3%→14%(稳定性代价)
  → proxy 漏掉的"够得着 vs 稳"权衡；解释为何短腿综合占优、长腿仅有越障运动学优势。

## 注意
- 中文若显示为方块：`set(groot,'defaultAxesFontName','<系统CJK字体>')`。
- fig5 用 12 个散点线性插值，凸包外不外推（`'linear','none'`）；仅示意地形趋势，非稠密扫描。
- fig4 是 BO 副产品（腿长有变动，故取短腿子集/定腿子集）；**干净的定腿电机扫描属"路B"**，待补。
