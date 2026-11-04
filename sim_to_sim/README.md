# 固定 PID 的无人机初态对照实验

今后只需要更换初态列表，使用同一套 PID 对比 CTBR、CTBR+DR 和 CTBR+ID。统一入口为 `experiment.py`，下列命令均在 `~/Desktop/FlightLxx` 执行，使用系统 Python，无需预先 source ROS；启动器会为每场试验加载 ROS 环境。

```bash
cd ~/Desktop/FlightLxx
```

生成一份新的全随机1000组初态，先预览，再运行三个网络共3000次独立试验：

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 sim_to_sim/experiment.py generate \
  --seed 20261105 --count 1000 --output sim_to_sim/states/random_20261105.json

PYTHONNOUSERSITE=1 /usr/bin/python3 sim_to_sim/experiment.py run \
  --bank sim_to_sim/states/random_20261105.json \
  --output sim_to_sim/logs/random_20261105 --plan-only --max-trials 3

PYTHONNOUSERSITE=1 /usr/bin/python3 -u sim_to_sim/experiment.py run \
  --bank sim_to_sim/states/random_20261105.json \
  --output sim_to_sim/logs/random_20261105 --workers 12
```

全随机范围：位置相对于目标 `(0,0,5 m)` 各轴 `[-1,1] m`，roll/pitch 各 `[-π,π]`、yaw `[0,2π)`，世界线速度各轴 `[-1,1] m/s`、机体角速度各轴 `[-1,1] rad/s`。欧拉角分别均匀，沿用 Simulation_Experiments 的定义。无分档、筛选或失败重采样。

也可直接编辑生成文件中的 `states`。每项使用唯一 `id` 与12维 `state`：

```text
[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
```

位置存实际高度（z为4至6米）；世界坐标为 NWU，机体为 FLU，角度单位rad。请求状态为 RPY 顺序，网络观测为 YPR 顺序，运行程序负责转换。三个网络使用完全相同的初态，顺序交错，每场独立启动。

查看进度、生成分析，或继续中断的实验：

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 sim_to_sim/experiment.py status \
  --output sim_to_sim/logs/random_20261105

PYTHONNOUSERSITE=1 /usr/bin/python3 sim_to_sim/experiment.py analyze \
  --output sim_to_sim/logs/random_20261105

PYTHONNOUSERSITE=1 /usr/bin/python3 -u sim_to_sim/experiment.py run \
  --bank sim_to_sim/states/random_20261105.json \
  --output sim_to_sim/logs/random_20261105 --workers 12 --resume
```

新运行须用新目录；`--resume` 核对冻结输入，只补未执行的场次。已有未完整结束的尝试保留为失败，不自动重试。`--max-trials 3` 可先试跑三场，再以相同输入加 `--resume` 完成剩余部分。分析输出位于新实验的 `analysis/`，含统计JSON、逐次CSV、图表、Markdown和PDF；不完整实验会标记为进行中。

在当前桌面逐场打开 Gazebo，每网络看相同10组初态（共30场）：

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 -u sim_to_sim/experiment.py view \
  --bank random --count 10

# 只看ID的第1、2、3组初态
PYTHONNOUSERSITE=1 /usr/bin/python3 -u sim_to_sim/experiment.py view \
  --bank random --indices 1 2 3 --models 'CTBR+ID'
```

`--bank` 支持 `random`（保留的全随机列表）、`curriculum`（保留的五档列表）或自己的JSON路径。展示初态按固定种子抽取，不根据成功结果挑选；也可用 `--indices` 指定从1开始的序号。每场重新启动一个 Gazebo 窗口，默认间隔3秒。展示属于新实验，日志另存，不并入正式数据。

固定 PID 来自 [config/final_pid.json](config/final_pid.json)，所有网络一致，每场从 ULog 核验实际生效值：

| 轴 | P | I | D |
|---|---:|---:|---:|
| Roll | 0.097 | 0.004 | 0.00135 |
| Pitch | 0.0869 | 0.004 | 0.00135 |
| Yaw | 0.126 | 0.0018 | 0.006 |

每场使用独立 PX4/Gazebo、网络命名空间和参数副本，加载固定 PID。先保持倒推0.35秒的准备姿态并同步 EKF，然后进入 Offboard、解锁并执行连续运动准备；当前估计和动作源观测的位姿/速度误差均低于0.06后释放，由网络控制10仿真秒。非 Offboard 时不发布神经网络 CTBR。控制仍为50Hz、Prototype动力学和原有物理推力映射，未调整增益或权重。

成功标准：完整飞行10仿真秒，固定末2秒内位置误差始终 `<0.3 m`、速度始终 `<0.3 m/s`，此前失稳后恢复仍可成功。初始化失败、通信/模式中断、策略恢复失败分别统计，全部保留在成功率分母中。另报恢复时间、峰值位置误差、末段偏置、速度/角速度抖动、动作变化率和饱和比例，并用 Gazebo 真值交叉检查。

最后两轮结果与 [合并报告PDF](FINAL_PID_TWO_EXPERIMENTS_REPORT_2026-11-01.pdf)、[Markdown](FINAL_PID_TWO_EXPERIMENTS_REPORT_2026-11-01.md)、[详细数据附件](reports/final_pid_comparison_20261101/README.md) 保留：

| 实验 | CTBR | CTBR+DR | CTBR+ID |
|---|---:|---:|---:|
| 五档1000组 | 374/1000 | 480/1000 | 599/1000 |
| 全随机1000组 | 297/1000 | 419/1000 | 540/1000 |

```bash
# 只读查看两轮保留结果
PYTHONNOUSERSITE=1 /usr/bin/python3 sim_to_sim/experiment.py status

# 核验6000份结果哈希、配对初态、实际PID、报告和全部原始文件属性
PYTHONNOUSERSITE=1 /usr/bin/python3 sim_to_sim/experiment.py verify --raw-files
```

为保留原始记录中的绝对路径，两轮原始数据不搬移、不改写、不压缩：

- 五档正式数据：`logs/continued_pid_20261029_v1/rounds/r005/formal_continue_736ee51a5cf0/trials_batch/`
- 全随机正式数据：`logs/full_random_pid_20261030_v1/evaluation/`
- 全随机初始化边界验证：`logs/full_random_pid_20261030_v1/pilot/`，18次，不计入正式6000次。

两份可复用初态已集中到 `states/`。原始批次中的源码快照和运行记录保持原样；其中旧文件名仅代表当时的运行证据。历史调参程序、选参记录、开发报告、旧归档和手动调试入口已删除。最终合并报告保持原件，其旧源码引用不再作为当前运行依赖。

目录只保留当前用途：`experiment.py` 统一入口，`runtime/` 控制/初始化/记录/分析，`config/` 固定PID与现有权重推理核验，`states/` 初态列表，`gazebo/` 和 `px4_airframes/` 模型，`tools/` 安装构建工具，`tests/` 当前功能的回归检查，`logs/` 原始实验，`reports/` 最终统计附件与数据完整性索引。

当前电脑已经安装并构建运行条件。迁移或重建时才需执行：

```bash
/usr/bin/python3 sim_to_sim/tools/install_prototype_px4.py --with-recovery
/usr/bin/python3 sim_to_sim/tools/install_recovery_ekf_patch.py
bash sim_to_sim/tools/build_state_injector.sh
cd ~/PX4-Autopilot
make px4_sitl_default
cd ~/Desktop/FlightLxx
```

依赖为 Ubuntu22.04、ROS2 Humble、Gazebo Harmonic、已构建的 `~/PX4-Autopilot`、`~/px4_ros2_ws`、MicroXRCEAgent、tmux、bubblewrap，以及系统 Python 的 NumPy/pyulog/matplotlib/reportlab。运行目录请放在 `sim_to_sim/logs/`；私有命名空间会隔离 `/tmp`。`S2S_PX4_DIR` 可指定其他已准备好的 PX4 工程。

```bash
# 验证当前代码（只使用临时模型目录，不改外部PX4）
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
PYTHONNOUSERSITE=1 /usr/bin/python3 -m unittest discover -s sim_to_sim/tests -q
```
