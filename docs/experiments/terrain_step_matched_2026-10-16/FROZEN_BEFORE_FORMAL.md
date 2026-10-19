# 正式评测前冻结

日期：2026-10-16，Asia/Shanghai。写入时6批预检与相机等价性已通过，
正式18批尚未启动。固定主例seed12345/env0，不按结果挑选其他环境。

| 源文件 | SHA256 |
|---|---|
| terrain_step_protocol.py | ec5f5416534f36f16df72fde0c9e721bdcb10165602dea9e61e9bff0c9c5aa78 |
| terrain_step_experiment.py | 603651d38a0446da9a7084d39380b4a931c10673c3f8b27b1010db382f080856 |
| terrain_step_launch.py | c0dc4f395b80f195c1e8f77c19a126491e8653ebb0282136afd55a40cac0a2c1 |
| terrain_step_analysis.py | 54604ba1edc63f635a6e6cce3ab83cf7ba093baddfd9abb65dd6716837241e0f |

18批 = 2既有策略 x 3真实地形 x 3评测seed，每批30环境、300个20 ms间隔。
主指标含初始化，补充1--6 s统计；不删除失败或离群样本，不替换seed/环境。
接触阈值Fz>1 N；俯仰/横滚按世界坐标；净空为实际胶囊最低点与当地三角面的竖向距离。
坡面连续+world X上升5 deg，粗糙控制间距0.4 m、三角网格0.1 m、幅值支持±0.05 m。
采图固定2.00/2.24/2.48/2.72 s，直接来自同一仿真记录的离屏碰撞模型渲染。
先评测再分析，不进行新训练，也不扩大到楼梯/台阶/任意地形结论。
