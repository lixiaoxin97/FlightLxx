# 五档2500次初态实验

三个网络每档各500次、每网络2500次，共7500次有效策略试验。L1包含100次相同目标静止状态独立重复和400组小扰动；五档共2401个不同请求初态。正式seed=2026110401，独立pilot seed=2026110402（每档10组、三网150次）。

当前PID、物理模型、推力映射、权重固定。先运行pilot验证初始化、模式、日志与参数核验，不以成功率排名作为启动正式实验的门槛。初始化/通信失败同初态补做，保留原始尝试；取第一次有效策略结果，策略恢复失败不重试。

在FlightLxx根目录查看本轮：

```bash
cat sim_to_sim/sim_experiment/five_level_2500_20261104/pipeline_status.json
PYTHONNOUSERSITE=1 /usr/bin/python3 sim_to_sim/sim_experiment/run_study.py status \
  --output sim_to_sim/sim_experiment/five_level_2500_20261104/formal
```

恢复同一轮（输入源码必须未变）：

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 -u sim_to_sim/sim_experiment/run_study.py pipeline \
  --output sim_to_sim/sim_experiment/five_level_2500_20261104 --workers 12
```

报告在各阶段完成后自动生成，正式文件为 `sim_experiment/five_level_2500_20261104/formal/analysis/FIVE_LEVEL_REPORT.pdf`，附Markdown、全有效试验CSV、全部尝试CSV、配对质量与精度门槛表。未完成不能称为2500组完整结果。


最终交付状态与视频：

```bash
cat sim_to_sim/sim_experiment/five_level_2500_20261104/delivery_status.json
```

正式7500场完成后，后台交付程序会逐槽位核验第一次有效结果、请求初态、接管容差与冻结输入，然后生成详细报告 `formal/analysis/EXPERIMENT_ANALYSIS.pdf`，以及 `formal/videos/` 下的同步MP4。视频制作规则见 [VIDEO_PROTOCOL.md](VIDEO_PROTOCOL.md)。记录由Gazebo真值回放，不重新飞行。

原始路径 `logs/five_level_2500_20261104` 仅保留为兼容旧绝对路径的软链接，实际数据统一位于本目录。两个状态库也位于本轮目录，`states/` 中相应名称为链接。pilot和bootstrap独立保存，不计入正式7500场。
