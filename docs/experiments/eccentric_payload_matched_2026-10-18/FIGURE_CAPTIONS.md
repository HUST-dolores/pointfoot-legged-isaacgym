# 图注与写作口径

## 图22：居中与初始偏心自由负载的姿态及失效率

平地零速度命令下，两种2–30 kg训练的历史策略在8、13、18、23、26 kg自由负载下的对比。
居中放置与初始纵向偏移+0.15 m分别绘图。上排为0.50–40.00 s全部分配轨迹的机体
俯仰及横滚RMS；下排为独立判定的几何跌倒率与掉载率，事件可重叠。
每点90条轨迹，先逐环境计算，再对每评测seed的30环境取均值、对3个seed等权汇总。
阴影为3个评测seed均值的样本标准差，不是上下极值、置信区间或训练seed方差。
两种方法各仅一个训练seed1检查点，均为11000迭代，不是多训练seed显著性结论。
失败和掉载环境未删除；全窗RMS包含失稳后的实际状态，不能单独解释为持续带载稳定性。
自由箱体偏移会随运动改变，初始偏心不表示全程固定偏心。

**English.** Body attitude and event rates under initially centered and longitudinally eccentric
free payloads. The two historical policies were trained with 2–30 kg payloads and are evaluated
at zero velocity commands on flat ground. Upper panels report pitch and roll RMS over the
entire assigned post-loading window (0.50–40.00 s), including post-failure physical samples.
Lower panels report independently counted geometric falls and payload drops; events may overlap.
Each point uses 90 assigned trajectories. Lines are equally weighted means of three evaluation-seed
means, and shading denotes their sample standard deviation, not training variability or confidence
intervals. Each method uses one historical training seed and checkpoint 11000. The +0.15 m offset
is an initial condition for a freely moving box, not a maintained eccentricity.

## 图23：同一质量下放置条件与姿态、漂移的同步过程

预登记主例为13 kg、评测seed12345、环境0，未按表现挑选轨迹。左右列为RL-only和
Model-guided，颜色区分居中与初始+0.15 m偏心。由上至下为真实箱体中心在当前机体系
的纵向与横向位置、机体世界俯仰、机器人相对加载前0.48 s位置的平面位移模长。
0.5 s生成箱体，不安排主动卸载；点线为掉载开始、虚线为几何跌倒开始，均经持续时长确认，
不是触发即假定成立。失败后的实际轨迹仍保留，不用另一环境拼接。
箱体位置是仿真刚体位置，不是网络预测的等效基座质心增量。
零速度命令不等于位置闭环，因此平面位移称为漂移，而非“定点控制误差”。
该单例用于说明过程，整体比较应结合图22及全部1800条轨迹的统计表。
同一记录另给0–8 s加载局部放大；仅改变坐标范围，不替换环境或统计窗口。

**English.** Synchronized free-box position, body pitch and robot drift for the preregistered
13 kg case (evaluation seed 12345, environment 0). Columns show the two methods and colors
distinguish initial placement. The box is spawned at 0.5 s without scheduled removal. Box x and y
are actual rigid-body center coordinates in the instantaneous body frame, not the network's
effective-base CoM-increment outputs. Planar displacement is measured relative to the robot's
pre-loading XY position at 0.48 s. Dotted and dashed lines mark sustained-confirmed drop and fall
onsets, respectively. Actual post-event samples are retained. Zero velocity commands do not imply
position feedback; displacement is drift rather than position-setpoint tracking error.
An additional 0–8 s view shows the same records with local axis limits, without changing
the trials or statistical windows.

## 几何判据与统计边界

- 跌倒：机体z轴偏离世界竖直超过60°或根高度低于0.25 m，持续0.10 s。
- 掉载：曾建立支撑后，箱体保守包围球完全离开托板投影或低于板顶，且on-body为假，持续0.20 s。
- 接载失败与历史终止计数另列；历史终止不自动叫跌倒，跌倒后的掉载仍独立计数。
- 姿态RMS包含静态偏差，去均值SD与35–40 s有符号均值另存CSV；不能用RMS单独声称振动更小。
- 预定早窗0.50–2.00 s及在险窗口的统计另存；在险窗口长度不同，不能隐去时长直接作优劣排序。
- 不扩展为固定偏心、位置保持、质量扫描以外或Model-guided各质量均优的结论。
