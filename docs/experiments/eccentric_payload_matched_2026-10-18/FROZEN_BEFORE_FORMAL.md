# 正式评测前冻结

2026-10-18，Asia/Shanghai。8批预检退出码均为0，逐批实际质量、加载局部坐标、
噪声/物理配对、零命令、无复位/主动卸载、组合base监督量核验通过。
写入本文件时正式60批尚未启动。

| 源文件 | SHA256 |
|---|---|
| eccentric_payload_protocol.py | 85ee4f30c8a13a8fe6fac5e03be1ff63319b348e508d5c3876e3f91157757e27 |
| eccentric_payload_experiment.py | 994f437cae6d5a5b4f27283d556fc552fee1021883480b2e7bc5d8610991e7ca |
| eccentric_payload_launch.py | 27f96a88876f4aeb7bf9269533cb5c38cc96c8af57e74e16e827f1a43dc57cd0 |
| eccentric_payload_analysis.py | b0e9b6a1813bdde818fa42cf6f000db17e5abf14fff9828804b51cc6e97951f0 |

完整方案见PROTOCOL.md。两种方法各一个原seed1、11000迭代检查点；
3评测seed不当作训练seed。60批/1800轨迹，每批30环境、40 s。
图23主例预先固定13 kg、eval12345/env0，禁止根据结果替换。
主姿态RMS和漂移使用全部分配轨迹、完整加载后物理记录，包含失败后响应；
另给早窗/在险窗口和时长、跌倒/掉载/接载失败及原终止，缺失不赋零。
不预设方法优劣，不与前两组策略或旧不同协议表混算。
