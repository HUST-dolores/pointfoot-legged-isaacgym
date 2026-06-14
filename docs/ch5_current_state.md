# 第5章 当前状态 / 交接文件（ch5_current_state）

> 维护人:本会话(2026-06-05 ~ 06)。配套:`experiment_notebook2.md`(详细数据)、`ch5_experiment_protocols.md`(实验流程)。
> 一句话主线:**负载估计 → 提升机器人"抗倾覆能力";且经因果消融证明,策略在运行时(经 encoder latent)使用该估计。**

---

## 0. 决策/对话脉络(为什么走到这一步)
1. 第4章已证负载估计**方法**(2–4kg 常规负载)精度 + 两通路机理;第5章要回答"负载估计有什么用"。
2. 发现 **2–4kg 在该 ~25kg 机器人上几乎不改变运动安全边界** → 第5章必须**宽负载(2–30kg)重训**(否则负载不产生显著力矩,估计无从体现价值)。
3. 训了 4 个宽基底(见 §3)。把第5章重构为 **"抗倾覆能力(anti-tipping)"** 这一统一框架(用户提出)。
4. 一系列控制实验(C 系列)+ 派生指标 β*;并把"存活"指标从"全程时间占比"换成 **末态直立率**(抗复位虚高)。
5. **因果检验(关键)**:做"测试时估计消融"。第一轮只消融 obs QS(冗余)→ 假阴性"训练时收益";加 `scope=both` 消融 encoder latent → **塌台**,反转为"运行时因果"。再用 **dose-response + shuffle** 把"非退化输入"与"负载对应"分层。
6. 发现 **IG ≠ 因果**:actor-IG 对 obs QS 归因更高,但消融证明 latent 才是因果载体 → 二者互补呈现。
7. 出图脚本统一放 `legged_gym/scripts/figs_ch5/`(编号前缀、一图一观点、**不自动保存**)。

---

## 1. 当前结论(已settled)
**A. 抗倾覆主线(均单 seed,待多 seed)**
- **β\* 抗倾覆边界**(存活跌破0.5的临界坡度):Model 24.5° / Estimate 25.2° ≫ Source 20.5° / RL-only 21.1°(**有 QS 组 +~4°**,与负载几乎无关)。
- **C1 斜坡存活**(末态直立率):24° 处 {Model 0.57, Estimate 0.73} ≫ {Source 0.10, RL 0.04}(6–15×)。分界正好在"QS 负载估计是否入 obs"。
- **C3′ 下坡10°急停**:QS 组停车短 ~35%。
- **C0 平地搬运极限(反证)**:四方都扛 ~54kg、**无差异**(平地无倾覆力矩 → 估计无用)。
- **偏心负载/横向推**:支持向证据(居中重载 QS 组前倾减半;横推40N RL-only roll 2–3× 且随载恶化)。

**B. 因果检验(本章最硬的一块)**
- **运行时因果**:同一 Model 权重,消融策略读到的负载估计(`zero_all + scope both`)→ 斜坡存活 0.96/0.83/0.61 **塌到 0.07/0/0**(=RL 水平)。
- **分层**:① 主因=latent 须非退化(zero→塌);② 次因(越陡越明显)=latent 须与当前负载对应(**shuffle** 在 24° 0.61→0.34,22° 几乎不掉)。
- **非精确 kg**:坡上 encoder 估计本就不准(早/中/末窗、仅存活 env,RMSE~14、斜率~0.19)→ latent 是**负载/倾覆风险相关表征**,不是标定 kg。
- **架构洞见**:显式 QS obs 特征运行时**冗余**(单独清零无影响);encoder latent 才是因果载体。
- **IG ≠ 因果**:D1 的 actor-IG 对 obs QS 归因(7–14%)> latent(2–3%),与消融方向相反 → 冗余下"归因≠因果必要性",**D1 与 C6 必须互补呈现**。

**C. 估计精度(四↔五章的桥)**
- 宽范围 A1:Model/Estimate 质量估计 RMSE ~1.1–1.2kg(平地),Source/RL ~3.0–3.6 → 证明方法平移到 2–30kg。

---

## 2. 新增/修改的脚本与代码
**环境/CLI(支撑所有 play 实验)**
- `legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py`:`compute_observations` 内**估计消融钩子**(`qs_corrupt_mode∈{none,zero,zero_all,scale,fixed,noise}`,默认 none 对训练无影响)。
- `legged_gym/scripts/play.py`:大量 play 开关 + `_corrupt_policy_estimate_latent`(latent 篡改:zero/scale/fixed/noise/**shuffle**)+ auto exp_tag 后缀 + 每-env `_all` 日志。
- `legged_gym/utils/helpers.py`:对应 CLI 参数(`--slope_deg/--load_hold/--flat_terrain/--no_load/--estop_*/--load_mass_min/max/--load_offset_x/y/--keep_load/--corrupt_estimate/--corrupt_val/--corrupt_estimate_scope`)。
- `legged_gym/scripts/set_variant_cfg.py`:按变体改 config 标志 + 负载范围。

**训练/批量 play(持久,在 repo)**
- `train_wide_seeds.sh`、`retrain_source.sh`(source seed1 崩→seed42 重训)。
- `run_source_plays.sh`、`run_estimate_plays.sh`、`run_carry_limit.sh`、`run_c2c3_slope.sh`。

**⚠️ 临时批量脚本(在 `/tmp/`,可能丢失——命令见 §3/protocols,必要时我移进 repo)**
- `/tmp/run_ablation.sh`、`run_ablation_strict.sh`、`run_dose.sh`、`run_shuffle.sh`、`run_eccentric.sh`/`run_ecc13.sh`/`run_ecc20.sh`、`run_pushload.sh`、`run_beta_refine.sh`。

**分析(CPU)**:`legged_gym/scripts/_expC_extract.py`(已排除 `_corr`/static/estop)。

**出图(MATLAB,`legged_gym/scripts/figs_ch5/`,一图一观点,不自动保存)**:见 §5。

**文档**:`experiment_notebook2.md`(§7 抗倾覆 + 因果检验①)、`ch5_experiment_protocols.md`(各实验工况/数量/判定/数据来源)、本文件。

---

## 3. 已跑数据(`logs/wheelfoot_flat/WF_TRON1A/`)
**四个宽负载基底(2–30kg)**
| 变体 | run 目录 | ckpt | 备注 |
|---|---|---|---|
| Model-guided | `Jun04_23-18-34_wide2-30_model_guided_seed1` | 11000 | qs/resid/torq=T/T/T |
| RL-only | `Jun05_03-01-38_wide2-30_rl_only_seed1` | 11000 | F/F/F |
| Source-guided | `Jun05_17-50-01_wide2-30_source_guided_seed42` | 16000 | F/F/T(seed1 崩,seed42 重训)|
| Estimate-guided | `Jun06_00-18-09_wide2-30_estimate_guided_seed1` | 16000 | T/F/T |

**play 数据**:`exported/play_data_<时间>_<exp_tag>.mat`,关键 tag 段:`_load2-30`(随机)/`_load26-26`(固定质量)、`_flat`、`_slopeN`、`_estop1.5`、`_static`、`fxyz0-{F}-0N`(力)、`_eccXxYy`(偏心)、`_corr{scope}{mode}{val}`(消融)、`_cycle{D}-{I}`(同步加卸载)。
**防混 seed 关键**:exp_tag 里的 `seedN` 是 play seed(恒=1),**训练 seed 只在 `meta.load_run`**。所有图脚本靠 `contains(load_run,'wide2-30')` 唯一锁定当前基线(Model/Est/RL=seed1, Source=seed42);旧 Ch4 窄基底 run(`exper_*_seed_45_pemass`)无 `wide2-30` → 自动排除(同目录共存,勿删/勿移)。备用 seed(Model-s2/Source-s2)在 `exported/_extra_seeds/`,不污染基线图。
**固定 26kg 时域图(2026-06-07 新增)**:C3b/C5b/A2 改读 `_load26-26` 数据(四方案严格同负载 26kg)。生成命令:`--load_mass_min 26 --load_mass_max 26`;C5b 推力需 `--keep_load`;A2 循环 `--load_dur 6 --load_int 10`。批处理 `/tmp/run_fixed26.sh`(4变体×{estop,push,cycle},12个,seed 与基线一致)。
失败 seed1 source 的 play 已移到 `exported/_broken_seed1_source/`。

---

## 4. 待确认 / 开放问题
1. **多 seed**(唯一硬缺口):C1/β*/消融均单 seed。用户计划两周内晚上排训练(每变体多 seed),白天 play/出图。
2. **β\*/C1 多 seed 后**给关键结论加误差棒。
3. **C2/C3 平地版**弱(已降为对照/正文一句),不进主图。
4. **重载 26–30kg** max-位移类指标受倾倒污染;若要用,改"首次倾倒时间/不可逆失败率"。
5. **C6 Panel C(fixed)**:fixed 模式同时把 CoM latent 置零,叫"fixed load-latent",勿写成纯质量匹配。
6. **/tmp 批量脚本**易失:是否移进 `legged_gym/scripts/ch5_runs/` 固化。
7. 估计器在**斜坡(倾斜重力)**上 OOD 失效是真实现象,正文需如实写"latent 携带负载/倾覆相关表征,非标定 kg"。

---

## 5. 脚本 ↔ 图 ↔ 内容 对照表

### 5.1 第五章出图 `legged_gym/scripts/figs_ch5/`(14,均 headless 跑通,不自动保存)
| 脚本 | 图内容 | 对应实验 / 支撑结论 | 拟用 |
|---|---|---|---|
| `A1_estimation_accuracy.m` | 估计值 vs 真值散点 + 各变体 RMSE | A1 宽负载估计精度(方法平移到 2–30kg)| 章首 |
| `A1b_estimation_error_vs_load.m` | 估计误差 vs 负载(全程无饱和)| A1(比 A1 紧凑)| 章首主选 |
| `A1c_com_accuracy.m` | CoM 偏置估计精度(x/y)| A1 补充(呼应 C4)| 补充 |
| `C1_slope_survival.m` | 斜坡存活率热图(4变体×坡×负载,4子图)| C1 抗倾覆全景 | 大图/补充 |
| **`C1b_survival_vs_slope.m`** | 存活率 vs 坡度线图(对负载平均,4线)| C1 主结果(QS组抗倾覆更强)| **正文主图** |
| **`C2_beta_star.m`** | β*(load) 抗倾覆边界(4线)| 抗倾覆能力单值(QS组 +4°)| **正文(紧跟C1b)** |
| `C3_downhill_estop.m` | 下坡10°急停停车距离 vs 负载 | C3 第二证据(扰动∝负载)| 正文 |
| `C3b_estop_timetrace.m` | 急停位移-时间轨迹(重载)| C3 机理(怎么停)| 补充/答辩 |
| `C4_eccentric.m` | 偏心负载前倾 vs 负载(x=0/0.20双子图)| C4 CoM 支持证据 | 补充 |
| `C5_push.m` | 横向推40N 峰值roll vs 负载 | C5 扰动抑制(RL-only最差)| 补充 |
| `C5b_push_timetrace.m` | 横向推 roll-时间轨迹(重载)| C5 恢复动态 | 补充/答辩 |
| `D1_actor_ig.m` | actor-IG 5类归因柱(Load latent/QS obs/torques/prev/other)| 机理:读了哪些通道 | 机理图(**配C6**)|
| `D1b_actor_ig_detail.m` | 全输入组归因热图 | 机理细节 | 附录 |
| **`C6_latent_ablation.m`** | A:none/shuffle/zero柱 · B:scale剂量线 · C:fixed×真实负载热图 | **运行时因果**(消融估计→塌台)| **正文核心图** |

**正文建议序**:A1b(铺垫)→ C1b + C2(抗倾覆边界,主)→ C3(第二证据)→ **C6(运行时因果,核心)** + D1(机理,注明 IG≠因果)。其余补充/答辩。
**出图约定**:不自动保存;定版后加 `exportgraphics(gcf,'X.pdf','ContentType','vector')`。
**统一配色(单一来源 `figs_ch5/ch5_colors.m`,= payload_method_color / Okabe-Ito,与第四章一致)**:Model=E1 橙 `#E69F00` / Estimate=E2 绿 `#009E73` / Source=E3 天蓝 `#56B4E9` / RL-only=E4 朱红 `#D55E00`。所有 per-variant 脚本调用 `ch5_colors()`;C6 用坡度色(22/24°)、C1/D1b 用 parula 热图,不涉变体色。

### 5.2 第四章出图(根目录,改名版,路径未动)
| 脚本 | 内容 |
|---|---|
| `ig_encoder_mass_stacked.m` ★ | encoder 质量分支 IG 堆叠柱(两通路机理,第四章主图)|
| `ig_encoder_mass_heatmap.m` / `ig_encoder_window_heatmap.m` | encoder 质量 IG 热图 / 时间窗演化 |
| `ig_actor_groups.m` / `ig_actor_heatmap.m` | actor IG 分组柱 / 热图 |
| `ig_est_mass_negative.m` | est_mass 归因单图 |
| `expA_transfer_curves.m` | 力 vs 重量 转移曲线(Exp A 主图)|
| `expA_timeseries_3kg.m` / `expA_slope_summary.m` / `expA_horizontal_force.m` | Exp A 时序 / 斜率汇总 / 水平力 |
| (helper)`expA_load_runs/pick/linfit/savefig.m`、`payload_method_color.m` | Exp A 共享数据/拟合/存图/配色 |

### 5.3 已弃用 `_archive/`(路径已失效,勿直接跑)
`plot_expB_control_stability.m`、`plot_expB_slope_load_heatmap.m`、`plot_payload_experiments.m`、`plot_wf_tron1a_zero_pose.m`、`plot_encoder_ig_windows.m`
> 已删(被 figs_ch5 取代):`plot_expC_slope_survival/static_drift/estop_distance.m`。

---

## 6. 脚本目录结构(2026-06-06 整理后)
- `legged_gym/scripts/figs_ch5/`(14):第五章出图,A1/A1b/A1c · C1/C1b/C2/C3/C3b/C4/C5/C5b/C6 · D1/D1b。**只用这套出第五章图。** 内部用 `../../../logs` 三层路径(为子目录而写)。
- `legged_gym/scripts/`(根):
  - **第四章图(改名不改路径,留原处)**:`ig_*.m`(6,IG 机理,主图 `ig_encoder_mass_stacked.m`)、`expA_*.m`(4 绘图 + 4 helper:expA_load_runs/pick/linfit/savefig)、`payload_method_color.m`。
  - 运行器/代码:`play.py`/`train.py`/`set_variant_cfg.py`/`_expC_extract.py`/`*.sh`。
- `legged_gym/scripts/_archive/`(5,弃用,**路径已失效、不要直接跑**):`plot_expB_*`(2)、`plot_payload_experiments`、`plot_wf_tron1a_zero_pose`、`plot_encoder_ig_windows`。
- 已删除(被 figs_ch5 取代):`plot_expC_slope_survival/static_drift/estop_distance.m`。
- 命名规约:第五章=`figs_ch5/{实验代号}_{用途}.m`(A 铺垫→C 实验→D 机理);第四章=根目录 `ig_*`/`expA_*`。
