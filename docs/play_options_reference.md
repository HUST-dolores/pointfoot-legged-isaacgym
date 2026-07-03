# play.py 配置选项参考

> 本文汇总 `legged_gym/scripts/play.py` 可用的全部命令行参数、含义、默认值与关键行为。
> 所有参数定义在 [`legged_gym/utils/helpers.py`](../legged_gym/utils/helpers.py) 的 `get_args()`（`custom_parameters` 列表）；
> play 特有的解析/生效逻辑在 [`legged_gym/scripts/play.py`](../legged_gym/scripts/play.py)。
> 标注 **Exp A / Exp 2 / Exp B / Ch5** 的是为博士论文补充实验新增的开关（对训练零影响，默认关闭）。

基本用法：
```bash
python legged_gym/scripts/play.py --task=wheelfoot_flat \
  --load_run <run目录> --checkpoint <iter> [其它选项]
```

---

## 0. 全局关键行为（先看这几条，避免踩坑）

| 行为 | 说明 | 代码位置 |
|---|---|---|
| **num_envs 上限 30** | play 强制 `num_envs = min(cfg, 30)`，交互观察用 `--num_envs 1`。 | play.py:659 |
| **`--headless` 自动存盘退出** | headless 或 `--exit_after_save` 时，存完 `.mat`+图立即返回，不进无限渲染循环（批处理必用）。 | play.py:1447 |
| **命令默认静止** | `cmd_vx/vy/yaw` 全默认 `0.0` → 原地站立；行走要显式给 `--cmd_vx 0.5`。 | play.py:845 |
| **外力自动关负载** | 施加 ext-force 时**默认关掉真实负载**（Exp A：力 vs 重物互斥）；要力+载共存须加 `--keep_load`。 | play.py:722 |
| **输出文件名** | `play_data_<时间戳>[_<exp_tag>].mat`；`--exp_tag` 为空时自动生成 `lb{N}_qs{0/1}_resid{0/1}_torq{0/1}_{cmd}_seed{S}_ckpt{C}[_load{lo}-{hi}]`。 | play.py:1376 |
| **"off" 哨兵值** | 质量 min/max 默认 `-1`（不覆盖）、偏置 `-99`（关）、时序 `-1`（关）、力/坡/急停 `0`（关）。 | — |

---

## 1. 标准框架参数（与 train 共用）

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--task` | str | `a1_flat` | 任务名，如 `wheelfoot_flat`。 |
| `--load_run` | str | 配置 | 加载的 run 目录名；`-1` = 最新 run。 |
| `--checkpoint` | int | 配置 | 加载的 ckpt 迭代号；`-1` = 最新 ckpt。 |
| `--num_envs` | int | 配置 | 并行环境数（play 内被压到 ≤30）。 |
| `--seed` | int | 配置 | 随机种子（play 层是**回放** seed，不改权重）。 |
| `--headless` | flag | False | 无渲染（批处理），并触发存盘即退出。 |
| `--rl_device` | str | `cuda:0` | RL 算法所在设备。 |
| `--experiment_name` / `--run_name` | str | — | 覆盖实验/ run 名。 |
| `--resume` / `--max_iterations` / `--horovod` / `--exptid` | — | — | 训练用参数，play 一般不涉及。 |

---

## 2. 速度指令

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--cmd_vx` | float | 0.0 | 前进速度指令 x (m/s)。行走常用 0.5。 |
| `--cmd_vy` | float | 0.0 | 侧向速度指令 y (m/s)。 |
| `--cmd_yaw` | float | 0.0 | 偏航角速度指令 (rad/s)。 |

> 三者全 0 = 静止定点工况（§6.3 / §9.5 定点漂移实验用）。指令在 play 里被**固定写入** `env.commands`，不随时间重采样（急停模式除外）。

---

## 3. 负载：质量与位置

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--load_mass_min` | float | -1.0 | 覆盖 `add_load_range[0]`（kg）。`<0` = 不覆盖，用存档 cfg。 |
| `--load_mass_max` | float | -1.0 | 覆盖 `add_load_range[1]`（kg）。配合 `per_env_load_mass` 时每 env 一个不同质量（linspace）。 |
| `--load_offset_x` | float | -99.0 | **Ch5 偏心负载**：固定负载质心 x 偏置（m，前+），产生 ∝ m·offset 的倾覆力矩（测 CoM 估计）。`-99` = 关（用训练随机偏置）。 |
| `--load_offset_y` | float | -99.0 | **Ch5 偏心负载**：固定负载质心 y 偏置（m，左+）。`-99` = 关。 |
| `--keep_load` | flag | False | **Ch5 推-恢复**：施加 ext-force 时**仍保留真实负载**（覆盖 Exp A 的"力则无载"默认），让负载+力共存。 |
| `--no_load` | flag | False | **Exp B 基线**：`add_random_load=False`，完全不挂负载，用于隔离"负载引起的控制退化"。 |

> 固定同质量对比（C3b/C5b/A2）：`--load_mass_min 26 --load_mass_max 26`。
> 有效偏置范围：base_task 实测 `x∈(−0.17, 0.21)`；`x≥0.3` 负载会掉板。config 里旧的 `load_offset_range_xy=[0.13,0.12]` 是弃用错值。

---

## 4. 负载：时序（同步加/卸）

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--load_dur` | float | -1.0 | 负载 ON 持续时长（s）→ `load_duration_range_s=[dur,dur]`，**所有 env 同步**加/卸。`<0` = 关。 |
| `--load_int` | float | -1.0 | 负载生成周期（s）→ `load_interval_range_s=[int,int]`；off 相 = int−dur。 |
| `--load_start` | float | -1.0 | 首次加载时刻（s）。`<0` = 用 cfg（0.5）。 |
| `--load_hold` | flag | False | **Ch5**：负载**全程恒定**（`load_duration_range_s=[1e6,1e6]`，不循环），使倾倒由负载本身决定而非瞬态。 |

> Exp A 的力/载对齐用固定阶跃：0.5s 起、开 6s、周期 10s（`--load_start 0.5 --load_dur 6 --load_int 10`）。
> 斜坡/急停/漂移这类"稳态"实验用 `--load_hold`。

---

## 5. 外力（Exp A / Exp 2：外力 vs 重物辨识）

**竖直向下力（Exp A headline，力当量按 kg 给）：**

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--ext_force_down_kg` | float | 0.0 | 单值竖直向下力，按力当量质量 kg 给（力 = kg×9.81 N），所有 env 相同。`>0` 开启外力模式。 |
| `--ext_force_down_kg_min` | float | 0.0 | **per-env 扫描**下界（kg）。与 max 配合，每 env 一个 `linspace(min,max,num_envs)` 力当量 → 一次 play 出整条转移曲线。 |
| `--ext_force_down_kg_max` | float | 0.0 | per-env 扫描上界（kg）。 |
| `--ext_force_dir` | str | `down` | 扫描方向：`down`(−Z) / `up`(+Z) / `fwd`(+X) / `back`(−X) / `left`(+Y) / `right`(−Y)。水平方向即 **Exp 2**（方向特异性）。 |

**任意方向常力（直接给 N）：**

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--ext_force_x` | float | 0.0 | 沿世界 X 的常力 (N)。 |
| `--ext_force_y` | float | 0.0 | 沿世界 Y 的常力 (N)（横向推-恢复用）。 |
| `--ext_force_z` | float | 0.0 | 沿世界 Z 的常力 (N，负=向下)。若 `--ext_force_down_kg>0` 则本项被忽略。 |

> 外力在每个 decimation 物理子步对 base 施加 **ENV_SPACE 恒定力**（竖直向下 `[0,0,−F]`，与姿态无关）。
> **默认施力则关负载**（互斥）；要真载+横推共存加 `--keep_load`（40N 可区分，60N 进倾翻区不可用）。

---

## 6. 斜坡与地形（Ch5）

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--slope_deg` | float | 0.0 | **重力前倾 β 度模拟坡度**（平地 + 倾斜重力 ≡ 坡上）。**同步**改 sim 物理重力 `sim.gravity` 与 obs 的 `env.gravity_vec`，干净隔离"力效应"与"地形几何"。 |
| `--flat_terrain` | flag | False | **Exp B**：强制平地面（`terrain.mesh_type='plane'`），使姿态指标反映控制而非地形跟随（非平地高度 std ~150mm → 平地 ~3mm）。 |

> `--slope_deg` 是"平地 + 斜重力"，**不是真实斜面**，轮子接触几何与平地相同，坡度鲁棒性结论纯关于 ∝ m·sinβ 力补偿（§9.4）。

---

## 7. 急停（Ch5 emergency-stop）

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--estop_vx` | float | 0.0 | 巡航速度方波：`go` 段以此速度巡航，`stop` 段降到 0，反复。`>0` 开启急停模式。 |
| `--estop_go_s` | float | 4.0 | `go`（巡航）段时长 (s)。 |
| `--estop_stop_s` | float | 4.0 | `stop`（vx=0）段时长 (s)。 |

> 每个 go→stop 跳变是一次急停。停车距离用**机体 x 位移**（`base_pos_x`），不是轮里程（轮转角要 `np.unwrap`×Rw=0.127，且 ≠ 机体位移）。v=2 太猛会前翻污染，干净测量用 1.5 m/s。

---

## 8. 估计消融（Ch5 因果检验：测试时篡改负载估计，不改权重）

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--corrupt_estimate` | str | `none` | 篡改方式：`none` / `zero` / `zero_all` / `scale` / `fixed` / `noise` / `shuffle`。`shuffle` = 跨 env 置换 policy 端 latent 的 mass/CoM（分布内但与真实负载错配，需 scope=latent/both）。 |
| `--corrupt_estimate_scope` | str | `obs` | 作用域：`obs`=obs 里的 QS 特征块；`latent`=policy 端 encoder mass/CoM latent；`both`=严格因果消融（两者都清）。 |
| `--corrupt_val` | float | 0.0 | 配合：scale 的倍数 / fixed 的 kg 值 / noise 的 std(kg)。 |

> **金标准因果证据**：`--corrupt_estimate zero_all --corrupt_estimate_scope both` 抹掉 latent 的 mass/CoM(idx 3:7) → 斜坡抗倾覆塌到 RL-only 水平；而只清 obs 里的 QS 特征无影响（冗余）。剂量-反应用 `scale`/`fixed`，负载对应检验用 `fixed=<kg>`，保守下限用 `shuffle`+`scope=latent`。

---

## 9. 输出与批处理

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--exp_tag` | str | `""` | 追加到输出 `.mat` 文件名，防覆盖。留空则自动生成（见 §0）。 |
| `--exit_after_save` | flag | False | 存完 `.mat`+图立即退出（headless 时自动开启），批处理避免死循环。 |

---

## 10. 单关节测试模式（调试用，与实验无关）

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `--test_joint_mode` | flag | False | 开启单关节正弦测试。 |
| `--test_joint_name` | str | — | 被测关节名，如 `hip_L_Joint`。 |
| `--test_joint_amplitude` | float | 0.3 | 转角幅值 (rad)。 |
| `--test_joint_period` | float | 4.0 | 周期 (s)。 |
| `--test_joint_offset` | float | 0.0 | 角度偏置 (rad)。 |

---

## 11. 常用配方（来自实验笔记，均已验证口径）

```bash
# Exp A —— 竖直向下力 per-env 扫描（力 vs 重物），30 env，行走 vx=0.5
python legged_gym/scripts/play.py --task=wheelfoot_flat --headless --num_envs 30 \
  --load_run <run> --checkpoint 11000 --cmd_vx 0.5 \
  --ext_force_down_kg_min 1 --ext_force_down_kg_max 6 --ext_force_dir down \
  --exp_tag fdown1-6kg
# 对照质量扫描：把上面两行 ext_force 换成 --load_mass_min 1 --load_mass_max 6

# Exp 2 —— 水平方向力（方向特异性）
#   同上，--ext_force_dir fwd  或  left

# Exp B —— 3kg 居中负载 + 0kg 基线（平地，隔离负载退化）
python legged_gym/scripts/play.py --task=wheelfoot_flat --headless --num_envs 30 \
  --load_run <run> --checkpoint 11000 --cmd_vx 0.5 --flat_terrain \
  --load_mass_min 3 --load_mass_max 3 --load_hold --exp_tag load3-3_flat
#   基线：去掉 load_* 换 --no_load，exp_tag=..._flat_noload

# Ch5 C-1 —— 斜坡抗倾覆 2D 扫描（坡度 × 负载）
python legged_gym/scripts/play.py --task=wheelfoot_flat --headless --num_envs 30 \
  --load_run <wide2-30_run> --checkpoint 16000 --cmd_vx 0.5 --flat_terrain \
  --slope_deg 24 --load_mass_min 2 --load_mass_max 30 --load_hold --exp_tag slope24

# Ch5 C-6 —— 估计消融（金标准因果，同权重只抹 latent）
python legged_gym/scripts/play.py --task=wheelfoot_flat --headless --num_envs 30 \
  --load_run <wide2-30_model_run> --checkpoint 16000 --cmd_vx 0.5 --flat_terrain \
  --slope_deg 24 --load_mass_min 2 --load_mass_max 30 --load_hold \
  --corrupt_estimate zero_all --corrupt_estimate_scope both --exp_tag slope24_ablate

# Ch5 急停 —— 巡航 1.5 → 0 方波，恒定负载
python legged_gym/scripts/play.py --task=wheelfoot_flat --headless --num_envs 30 \
  --load_run <run> --checkpoint 16000 --flat_terrain \
  --estop_vx 1.5 --estop_go_s 4 --estop_stop_s 4 \
  --load_mass_min 2 --load_mass_max 30 --load_hold --exp_tag estop15
```

> 单 env 交互观察：把 `--headless --num_envs 30` 换成 `--num_envs 1`（进游戏窗口，按 `v` 切渲染同步、`ESC` 退出）。
