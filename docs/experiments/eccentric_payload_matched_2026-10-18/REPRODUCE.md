# 独立复现

交付包中`code/repo`是自包含代码、机器人资产、两原检查点及训练配置。
`experiment`含全部正式/预检NPZ、配对场景、逐批实际物理参数、进程退出与统计图。
分析可用本机pointfoot_legged_gym环境，不需Isaac Gym或GPU；重新仿真需工作站cdesign/Isaac Gym。

从交付包根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/code/repo" \
  MPLCONFIGDIR=/tmp/codesign_figures_mpl \
  /home/xu/anaconda3/envs/pointfoot_legged_gym/bin/python \
  -m legged_gym.scripts.eccentric_payload_analysis \
  --input "$PWD/experiment" --output /tmp/eccentric_payload_reproduction

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/code/repo" \
  /home/xu/anaconda3/envs/pointfoot_legged_gym/bin/python \
  -m unittest legged_gym.scripts.test_eccentric_payload_protocol -v

sha256sum -c --quiet SHA256SUMS
```

全部统计CSV/JSON与PNG可逐字节比较；PDF含生成日期、MAT含文件头时间，
后两者用数值/视觉内容比较，不强求文件哈希相同。
包内`experiment/analysis/fonts`保留字体；制图脚本必要时可从系统Noto CJK重新生成子集。
共享基础框架是本次评测兼容快照；任务层使用原run保存的源码，不能声称逐字复现原训练仿真版本。

评测脚本恢复每种方法自身的输入维度与解析残差设置。RL-only无原始力矩/解析特征输入，
Model-guided有这些输入，两者不是只有单一开关差别的严格消融，不人为改成同一个网络。
本组仅原训练seed1两策略，不改变第一组双训练run设置，不替代15 kg迈步或其他协同设计结果。
