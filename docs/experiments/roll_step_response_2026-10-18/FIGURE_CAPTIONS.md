# 图24：已有统一策略的滚动—迈步命令响应

同一冻结策略Jul03_15-59-41_/13000在无负载平地上的连续40 s响应。
0–10、10–20、20–30、30–40 s的滚动/迈步命令依次为0.5/0、0.3/0.2、0.1/0.4、0.5/0 m/s，
总前向目标始终0.5 m/s。由上到下为实际进入策略的命令、机体系实际前向速度、左右实际轮接触、
机体俯仰和横滚，点线标记10/20/30 s改命令时刻。速度按框架有限差分口径测量。
主例预登记为评测seed12345/env0，不按结果换例；另给同一记录各改命令点前后2 s放大。
若主例失效，灰色区域与曲线保留失效后物理响应，不能解释为有效迈步；接触色带只展示有效段。
改命令时不重置GRU或步态时钟；边界状态来自旧命令区间，新命令影响下一20 ms区间。

**English.** Command response of one existing unified rolling–stepping policy on flat ground
without a payload. Rolling/stepping commands are 0.5/0, 0.3/0.2, 0.1/0.4 and 0.5/0 m/s over
four successive 10 s intervals, with a constant total forward target of 0.5 m/s. Aligned panels
show the commands actually entering the actor, body-frame forward speed, measured left/right
wheel contact, body pitch and roll. Dotted lines mark command changes. The preregistered case is
evaluation seed 12345, environment 0; additional views show ±2 s around each change using the
same record. GRU history and gait phase are continuous. A state at a command boundary terminates
the preceding physical interval, whereas the new command acts during the following interval.
Any grey post-failure data remain visible but are not interpreted as valid locomotion.

## 配套接触统计表

30环境×3评测seed，共90条同策略连续试验；每个10 s物理区间各500样本。
主表按四阶段分列，展示速度RMSE、左右失接触比例、双支撑和姿态SD，并给出有效/完成分母。
补充三命令条件表将同一环境首尾纯滚动区间合并后重算，占20 s而非10 s，
不是两个独立重复，也不能据此忽略首尾纯滚动的差异。
左右“摆动”统计为Fz≤1 N的失接触代理，不必然表示正净空；独立悬空列要求失接触且胶囊轮最低净空>1 cm。
去均值SD、绝对RMS和姿态均值分别保留，不能互换。
阶段有效统计在首次原终止/几何跌倒开始前计算，全部分配环境、失效时刻与完整物理统计另表保留。
3评测seed仅覆盖评测随机性，不是多训练seed结论；一个连续试验的不同阶段也不是独立控制器比较。

**结论边界**：只描述既有策略能否随命令改变运动行为，不证明最优过渡、切换稳定性、
任意运动模式转换或滚动—跳跃切换。滚/迈速度代理也不作为任意接触条件下的精确运动学分解。
本次四阶段实际前向均速为0.1016/0.1763/0.3987/0.1037 m/s，均未准确保持0.5 m/s总目标；
不能只引用单例某段接近目标的速度，忽略全部90条的统计。

**Observed limitation.** Mean actual forward speeds across all trials are 0.1016, 0.1763,
0.3987 and 0.1037 m/s. These records demonstrate command-dependent contact behavior,
not accurate tracking of the constant 0.5 m/s total-speed target.
