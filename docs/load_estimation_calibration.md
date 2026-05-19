# 双足轮足机器人负载估计公式的自动化标定

**日期**：2026-05-18
**目标平台**：WF_TRON1A（双轮足双足）
**相关代码**：
- 数据采集：[legged_gym/scripts/collect_calibration_data.py](../legged_gym/scripts/collect_calibration_data.py)
- 拟合与残差分析：[legged_gym/scripts/fit_load_estimation.py](../legged_gym/scripts/fit_load_estimation.py)
- 一键流程：[legged_gym/scripts/run_calibration.sh](../legged_gym/scripts/run_calibration.sh)
- 公式实现：[legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py](../legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py)

---

## 1. 问题背景

机器人在 [wheelfoot_flat.py](../legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py) 中实现了基于髋/膝扭矩的解析载荷估计：每条腿独立估计承载等效质量，组合后给出总质量和负载在机体坐标系下的 (x, y)。该公式原本通过人工试凑得到形如

$$
m_\text{leg} = 0.75 \cdot \left( \frac{\tau_\text{leg}}{(l_\text{thigh}\cos\theta_\text{thigh}+\delta)\,g\,\cos\theta_\text{abad}} - m_0 \right) + b_\text{leg}
$$

的形式，其中 `m_0 = 12.08`、`b_L = −2.5`、`b_R = +0.65`、$\delta = 0.05144$ 等。

**已观察到的失效模式**：在 (x, y) = (0, 0)、m = 5 kg 工况下估计准确，但偏离原点后系统性偏差极大：
- (x=0.1, y=0)：估计质量从 5 kg 跌到 1 kg，左右质量出现 m_L = −1.26、m_R = 2.24 的严重不对称
- (x=0.1, y=0.1)：估计质量为负

**核心猜想**：公式整体结构基本正确，但其中所有标量系数都是在 (0,0) 工作点手调出来的；远离工作点时系数不再最优，且形式上缺少对位姿变化的显式补偿项。

---

## 2. 方法

### 2.1 自动化数据采集

设计一个独立的数据采集脚本，在仿真器内确定性地放置载荷并采集稳态特征。

**关键设计**：

| 维度 | 设定 |
|---|---|
| 网格 | 4 个 mass × 4 个 x × 4 个 y = 64 工况 |
| mass | 2.0, 3.33, 4.67, 6.0 kg |
| x ∈ 机体前后 | −0.17, −0.04, +0.08, +0.21 m |
| y ∈ 机体左右 | −0.19, −0.06, +0.06, +0.19 m |
| 并行 env | 1024（每工况 64 envs） |
| 稳定窗口 | 仿真启动后 5–15 s 取均值 |
| 随机化 | 全部关闭（friction、imu offset、Kp/Kd、base com、push 等） |

**确定性放置**：通过 monkey-patch `_maybe_spawn_loads`，按 env_id 把 1024 个机器人均分到 16 个 (x, y) 工况，每 env 在 t=0 一次性获得指定位置载荷。mass 固定通过 `domain_rand.add_load_range = [m, m]` 设定。每次运行只跑一个 mass，多次执行覆盖 mass 维度（脚本 [run_calibration.sh](../legged_gym/scripts/run_calibration.sh) 自动化）。

**采集字段**：每 env 对 t∈[5s, 15s] 取均值，输出 NPZ：
- 输入特征：`R_L`, `R_R`, `cos_l`, `cos_r`, `cos_abad_L`, `cos_abad_R`, `y_foot_L`, `y_foot_R`, `τ_lhip`, `τ_rhip`, `θ_*`, `pitch`, `cos_pitch`, `tan_pitch`
- 真值：`x_rel_true`, `y_rel_true`（实际稳态位置，载荷可能略微滑动），`mass_true`
- 有效性：`z_rel_true ∈ [LOAD_Z − 0.3, LOAD_Z + 0.3]`，剔除被 PhysX 弹飞的样本

### 2.2 模型与最小二乘

把原公式重写为**显式线性形式**：

$$
m_L = \alpha_L \cdot R_L + \gamma_L, \qquad m_R = \alpha_R \cdot R_R + \gamma_R
$$

四个独立参数 $(\alpha_L, \alpha_R, \gamma_L, \gamma_R)$。每个有效样本贡献两个方程：

**质量方程**（kg 量级残差）：
$$
\alpha_L R_L + \alpha_R R_R + \gamma_L + \gamma_R = m_\text{true}
$$

**y 力矩平衡方程**（kg·m 量级残差，按 1/std 缩放到 kg 量级）：
$$
\alpha_L R_L (y_{fL}-y_\text{true}) + \alpha_R R_R (y_{fR}-y_\text{true}) + \gamma_L (y_{fL}-y_\text{true}) + \gamma_R (y_{fR}-y_\text{true}) = 0
$$

x 估计单独二阶段拟合，目标：
$$
\frac{\tau_\text{lhip}-\tau_\text{rhip}-T_\text{body}}{m_\text{est}\,g\,\cos\theta_p} - k_\text{pitch}\tan\theta_p + x_\text{offset} = x_\text{true}
$$
未知数 $(T_\text{body}, k_\text{pitch}, x_\text{offset})$，其中 $m_\text{est}$ 用第一阶段拟合好的参数算出。

### 2.3 模型扩展

为了诊断结构性偏差，分别测试四个模型：

| 模型 | mass 参数 | 额外项 |
|---|---|---|
| A 基本 | 4 | — |
| B + pitch | 5 | $+\beta_p \sin(\theta_p)$（两腿对称加） |
| C + hip-diff | 5 | $+\beta_h (\theta_\text{lhip}-\theta_\text{rhip})$（两腿对称加） |
| D + 两者 | 6 | 上述两项 |

"两腿对称加"意味着该项进入 $m_\text{total}$ 但在 y 估计的左右比例中抵消。

---

## 3. 结果

### 3.1 模型 A 残差揭示结构问题

仅 4 参数的基本模型在 1024 envs × 4 mass = 4096 样本上拟合后：

| 量 | bias | RMSE |
|---|---|---|
| mass | −0.79 kg | **1.62 kg** |
| x | 0.00 m | **0.103 m** |
| y | −0.01 m | **0.081 m** |

bias 接近 0 但 RMSE 巨大，意味着各工况间互相抵消但每个工况都偏。每工况残差表（节选 m=6）：

| x_set | y_set | m_err |
|---|---|---|
| −0.17 | −0.19 | +1.04 |
| +0.21 | +0.19 | **−4.25** |

**m_err 沿 x_set 单调下降，跨度 5 kg**。这是 4 个线性参数无法吸收的结构性偏差。

### 3.2 诊断：m_err 与各特征的相关性

把 m_err 与十余个候选解释变量做皮尔逊相关：

| 特征 | corr(m_err, feat) |
|---|---|
| $\theta_\text{lhip}$ | **+0.689** |
| $\theta_\text{rhip}$ | **−0.691** |
| $\theta_\text{lhip} - \theta_\text{rhip}$ | **+0.692** |
| $\cos\theta_\text{thigh,L}$ | −0.644 |
| $\cos\theta_\text{thigh,R}$ | −0.624 |
| mass_true | −0.611 |
| x_rel_true | −0.585 |
| **pitch / sin(pitch)** | **+0.038**（≈ 0） |
| $\theta_\text{labad}$ | +0.036 |
| $\theta_\text{abad}$-diff | +0.131 |

**结论**：
- pitch 与 m_err 实际**无关**。原因是 policy 在 zero command 下通过调整髋关节角而非倾斜机体来代偿载荷偏移，pitch 跨度只有约 7°
- 真正的解释变量是**髋关节角度差** $(\theta_\text{lhip} - \theta_\text{rhip})$
- abad 角度相关性接近 0，对 m_err 不是主要病因

### 3.3 模型 B 否定了 pitch 假设

模型 B 用 $\beta \sin(\theta_p)$ 项基本无改善：

| | mass RMSE | x RMSE | y RMSE | $\beta_p$ |
|---|---|---|---|---|
| A 基本 | 1.616 | 0.103 | 0.081 | — |
| **B + pitch** | **1.615** | **0.103** | **0.081** | −0.40 |

变化仅在小数点第三位，证实最小二乘也"找不到"pitch 项来吸收偏差——和相关性分析一致。

### 3.4 模型 C 一举切中

把 pitch 替换成 hip-diff：

| | mass RMSE | x RMSE | y RMSE |
|---|---|---|---|
| A 基本 | 1.616 | 0.103 | 0.081 |
| **C + hip-diff** | **0.770** | **0.021** | **0.066** |
| 相对改善 | **−52%** | **−80%** | −18% |

x RMSE 从 103 mm 砍到 21 mm，**x 维度基本被解决**。m_err 仍存在但跨工况跨度从 5.3 kg 降到 2.6 kg。

模型 D（同时加 pitch + hip-diff）相比 C 几乎无收益：

| 模型 | mass RMSE | x RMSE | y RMSE |
|---|---|---|---|
| C | 0.770 | 0.021 | 0.066 |
| D | 0.754 | 0.020 | 0.067 |

证实 hip-diff 一项已吸收了大部分结构性偏差。

### 3.5 物理参数的合理性

模型 C 的拟合系数对比手调值：

| 参数 | 手调 | 模型 C | 备注 |
|---|---|---|---|
| $\alpha_L$ | 0.75 | 0.367 | 模型 C 左右对称程度更高（α 比 0.5 略小） |
| $\alpha_R$ | 0.75 | 0.429 | 同上 |
| $\gamma_L$ | −11.56 | +0.158 | 量级从 10 降到 1，更"干净" |
| $\gamma_R$ | −8.41 | −0.499 | 同上 |
| $\beta_h$ | — | **−8.51** | hip_diff 每多 0.1 rad，每腿少分摊 0.85 kg |
| $T_\text{body}$ | 14.0 | 6.17 | 模型 C 显著小 |
| $k_\text{pitch}$ | +0.263 | +0.649 | 模型 A 拟合曾给出 **−0.677**（符号错），模型 C 恢复正号 |
| $x_\text{offset}$ | +0.12 | −0.037 | 模型 C 接近 0 |

模型 A 的拟合结果中 $k_\text{pitch}$ 变为负号是"参数互相打架"的伪解；加入 hip-diff 修正项后该参数恢复物理合理符号，说明模型 C 的参数有真实物理意义而非数值过拟合。

### 3.6 部署

更新到 [wheelfoot_flat.py:431-446](../legged_gym/envs/wheelfoot_flat/wheelfoot_flat.py#L431-L446) 和 [wheelfoot_flat_config.py:189-198](../legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py#L189-L198)，保留旧版本作为内联注释方便回退。

---

## 4. 结论

1. **结构性误差不能通过纯线性参数调整消除**。4 参数线性模型的最优拟合（mass RMSE 1.62 kg）就是它的天花板。
2. **正确的诊断路径是残差—特征相关性分析**。直观猜测（pitch）被数据否定，hip-diff 成为正确答案。
3. **物理可解释**：policy 通过调髋角而非倾斜机体来代偿偏置载荷，因此 hip_diff 才是与 m_err 真正同步的状态变量。
4. **5 参数的模型 C 在三个维度全面优于原公式**，mass RMSE 砍半，x RMSE 砍 5 倍。

## 4.1 补充实验：abad_diff 项

为了进一步压低 y RMSE，做了模型 E/F/G 的实验：

| 模型 | 额外项 | mass RMSE | x RMSE | y RMSE | $\beta_\text{abad}$ |
|---|---|---|---|---|---|
| A 基本 | — | 1.616 | 0.103 | 0.081 | — |
| C + hip | hip_diff | **0.770** | **0.021** | 0.066 | — |
| E + abad | abad_diff | 1.604 | 0.099 | 0.081 | −1.30 |
| F hip+abad | hip + abad | 0.787 | 0.022 | **0.064** | +0.81 |
| G 三项 | pitch+hip+abad | 0.755 | 0.021 | 0.064 | +2.24 |

**结论：abad_diff 不带来有意义改善**：

1. **模型 E**：abad_diff 单独使用基本等同模型 A，与相关性分析吻合（corr(abad_diff, m_err) = +0.131，太弱）。
2. **模型 F**：在 C 之上加 abad，mass RMSE **反而变差 17 g**，y RMSE 仅微降 2 mm。$\beta_\text{abad}$ 从模型 E 的 −1.30 翻成 +0.81 ——hip_diff 已经吸收了主信号，abad_diff 只能吸收剩余噪声。
3. **模型 G**：三项同开，$\beta_\text{pitch}$ 从 −0.40 飞到 **+7.12**（符号翻、量级 ×18），这是参数过参数化耦合补偿的典型征兆。mass RMSE 仅比 C 低 0.015 kg，得不偿失。

**物理解释**：当前数据采集的 y 范围只到 ±0.19 m，policy 主要靠髋关节调姿，abad 跨度（−0.08 到 +0.11 rad）远小于 hip（0.07 到 0.45 rad）。abad_diff 信号薄弱，模型从中提取不到有效结构。

## 5. 后续工作

- **y 残差的 66 mm 是当前公式结构下的下限**。abad_diff 实验证明加项没用，下一步要么换思路（反对称修正而非对称、或更宽 y 网格采集），要么把它交给 encoder 学残差。
- **反馈闭环验证**：换用模型 C 后 policy 看到的载荷估计会变，可能引起姿态轻微改变。**建议重跑一次标定数据验证**，看模型 A 在新部署下的 RMSE 是否仍 ≈ 0.77/0.021/0.066；若漂移则迭代 1–2 轮即可收敛。
- **更激进的结构改造**：当前公式假设两腿 x 位置相同；可基于完整 FK 推导更精确的力臂表达，可能进一步压低 mass RMSE。
- **encoder 残差学习**：当前 [PPO](../legged_gym/algorithm/ppo.py) 已启用 `extra_loss_load_boost`，让神经网络学习残差。模型 C 提供更准确的先验，应能加速 encoder 收敛。

---

**附**：拟合参数（模型 C）

```
alpha_L  = 0.367
alpha_R  = 0.429
gamma_L  = +0.158
gamma_R  = −0.499
beta_hip = −8.507     # 乘以 (theta_lhip - theta_rhip)，两腿对称加
T_body_x = 6.17
com_x_bias (k_pitch) = 0.649
x_offset = −0.037
delta    = 0.05144    # 固定，未参与拟合
```
