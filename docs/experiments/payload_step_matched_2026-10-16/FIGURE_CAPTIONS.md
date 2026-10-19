# 图注与展示口径

## 图18：带载迈步负载感知与轮接触时序

中文图注：固定约5.93 kg自由箱体加载后，RL-only与Model-guided在带载迈步中的
负载感知与实际轮接触时序。两列采用预登记的s1检查点、评测seed=12345、env=0，
共享初始场景与公共观测通道噪声，不因估计效果重新挑选轨迹。前三行为网络监督
对应的质量及机体坐标下等效基座质心增量x/y真值和最终估计；Model-guided另示
实际进入融合支路的解析先验。监督质量依照原在体标签门控，不代表箱体固有质量变化；
质心增量不是箱体位置或逐连杆整机质心。底行为左右轮实际Fz>1 N的接触状态。
紫色虚线表示0.5 s生成箱体，灰色点划线表示首次建立在体支撑；若出现，红色实线
表示机器人终止，红色点线表示几何判据确认的掉载开始。机器人失败后的曲线不作为
有效轨迹绘制，原始记录仍完整保存。提供0--40 s全程与0--8 s固定加载窗口，
不平滑、不平移时间轴。全程图质量轴采用symlog（[-10,10] kg内线性，
外侧对数），保留近100 kg解析尖峰且不压缩掉正常载荷区间；加载放大图为线性轴。
单条时序不是总体精度或成功率的统计证据。

English caption: Load perception and measured wheel contact during stepping with a
fixed approximately 5.93 kg free box. The two columns use the preregistered s1
checkpoints, evaluation seed 12345, and environment 0, sharing the initial scene
and common-channel observation-noise innovations. Rows show the supervised load
mass, effective-base CoM shifts in body-frame x and y, and measured left/right
wheel contact (Fz > 1 N). The Model-guided column also shows the policy-facing
analytical prior. Mass supervision is gated by the original on-body label; this
is not a change in the physical box mass. CoM shifts are neither box positions
nor articulated whole-body CoM coordinates. Vertical lines mark box spawning
(purple dashed), first support establishment (gray dash-dot), and, when present,
robot termination (red solid) or confirmed geometric departure onset (red dotted).
Post-failure samples are not plotted as valid motion but remain in the raw data.
Full 0--40 s and fixed 0--8 s views are provided without smoothing or temporal
realignment. The full-view mass axis is symmetric-logarithmic, linear within
[-10,10] kg and logarithmic outside, retaining the near-100 kg analytical spike;
the loading-window mass axis is linear. An individual trace does not establish
population-level accuracy.

## 图19：空载基线与带载增量

中文图注：两种方法的空载与固定约5.93 kg自由负载运动表现及配对增量。
上排依次为前向速度对0.9 m/s目标的RMSE、机体俯仰RMS和去自身均值后的
俯仰时间标准差；下排为对应的带载减空载差值，正值表示该误差或波动增加。
两条件均使用同一配对的共同有效时间窗，从2 s至首次适用的机器人失败、
确认掉载开始或承接失败，避免带载与空载采用不同时间段，也不把掉载后的
轻载行走计作成功带载表现。空心/实心表示空载/带载，圆形/方形表示两次独立
训练，短横线为两训练重复均值。每次训练先逐环境计算，再对三个评测seed等权
汇总；环境及评测seed不冒充独立训练数。该条件化统计需结合全部计划轨迹的
失败率、有效时长和缺失计数解读，不独立证明估计精度对运动表现的因果影响。

English caption: Unloaded and loaded motion metrics and their paired increments
for RL-only and Model-guided policies. Top panels show forward-speed RMSE against
the 0.9 m/s task target, body-pitch RMS, and within-trajectory mean-centered pitch
standard deviation. Bottom panels show loaded-minus-empty differences; positive
values indicate increased error or fluctuation. Each pair uses the identical
eligible time window, beginning at 2 s and ending at the first applicable robot
failure, confirmed payload-departure onset, or failed catch. Motion after payload
loss is not credited as successful loaded locomotion. Open/filled markers denote
empty/loaded conditions; circles/squares denote the two independent training
runs. Short horizontal lines indicate their mean. Metrics are first calculated
per environment, then averaged equally across three evaluation seeds and finally
across training runs. No training-population confidence band is inferred from
environment replicates. Interpret these conditional metrics with the complete
failure, exposure-duration, and missing-data tables, not as an isolated causal
ablation of estimation quality.
