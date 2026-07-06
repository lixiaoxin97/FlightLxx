# FlightLxx 当前项目交接总结

更新日期：2026-06-23  
目标读者：新的 Codex/agent，用于在长上下文被压缩或换窗口后继续工作。

## 最高优先级原则

用户明确要求：

> 不要试图手动干预环境策略，设计好环境策略后就让环境自行调整，如何表现不符合预期说明策略设计有问题需要重新设计，我们只控制参数的阈值。

因此后续不要通过规则强行增大/减小路径或风，不要加类似 `wind_boost` 的手工课程。应调整环境策略的观测、动作、奖励、参数上限和学习机制。

## 当前任务目标

项目从无人机竞速改为路径跟随：

1. 主策略学习稳定跟随复杂 3D 闭合路径。
2. 环境策略自动生成匹配当前主策略能力的路径和风扰。
3. 后续实物无人机希望体现双向推力优势，尤其是在强风/扰动下仍能稳定飞行。
4. 当前阶段正在从理想动力学迁移到真实电机数据 + 双向推力换向死区。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `libs/config/rl_env.yaml` | 主要配置：路径、奖励、风、动力学、域随机化、环境策略阈值 |
| `libs/src/envs/rl_env.cpp` | 主环境 step/reset、奖励计算、extra_info 日志 |
| `libs/include/envs/rl_env.hpp` | 环境状态、奖励配置、观测维度、环境策略维度 |
| `libs/src/envs/rl_env_path.cpp` | 连续路径生成、Catmull-Rom 插值、closest progress 搜索 |
| `libs/src/envs/rl_env_policy.cpp` | C++ 环境策略接口：路径/风动作应用、环境策略观测 |
| `libs/src/envs/rl_env_wind.cpp` | 风场生成：均值风、阵风、脉冲风、风预算限制 |
| `rl/lxx_baselines/envs/environment_policy.py` | Python 环境策略，SAC 风格 actor/critic/replay buffer |
| `rl/train/run_drone_control.py` | 训练/测试入口，预训练加载、checkpoint 保存、CLI 参数 |
| `rl/lxx_baselines/common/policies.py` | 主策略网络，包含历史堆叠和 `HistoryCnnPolicy` |
| `rl/lxx_baselines/ppo/ppo2.py` | PPO2 训练循环、日志输出、KL early stop、rollout info |
| `libs/src/objects/quadrotor.cpp` | 四旋翼动力学执行，包含风力和电机换向死区 |
| `libs/src/dynamics/quadrotor_dynamics.cpp` | 电机 thrust/rpm 映射、真实电机参数加载 |
| `rotor_data/` | 用户采集的真实电机数据 |
| `docs/training_checkpoint_notes.md` | 权重和训练日志总结 |
| `docs/path_following_env_policy_history.md` | 早期设计历史 |

## 主策略观测与动作

当前单帧观测维度为 25：

- `0:6`：机体系姿态矩阵前两列
- `6:9`：世界系线速度
- `9:12`：机体系角速度
- `12:16`：上一帧归一化动作
- `16:19`：最近路径点相对无人机位置
- `19:22`：前视路径点相对无人机位置
- `22:25`：路径切向方向

训练时可使用 50 帧历史，由 `HistoryCnnPolicy` 编码。这样参考了 Link-U-OS 的长历史思想，但适配为无人机路径跟随。

动作维度为 4，每个动作直接对应一个电机/桨的归一化推力命令：

```text
quad_act = normalized_action * act_std + act_mean
act_mean = 0
act_std = mass * g / 2
```

所以悬停单电机推力 `mass*g/4` 对应归一化动作约 `0.5`。最新日志 `ep_hover_action_norm_mean=0.5`，说明真实动力学后动作尺度没有跑到边界。

## 主策略奖励

当前核心奖励：

```text
reward =
  advance_reward
+ path_reward
+ body_rate_reward
+ act_reward
+ act_second_delta_reward
+ thrust_switch_reward
+ reversal_deadzone_reward
+ speed_reward
+ crash_reward
```

主要设计：

- `advance_reward`：沿连续路径前进奖励，但乘以 `track_gate`，离路径太远时前进不奖励。
- `path_reward`：连续路径横向误差惩罚。
- `body_rate_reward`：角速度惩罚，减少抖动。
- `act_reward`：动作变化惩罚，已降低，避免策略“不敢动”。
- `act_second_delta_reward`：动作二阶差分惩罚，抑制高频振荡。
- `thrust_switch_reward`：惩罚动作跨零切换。
- `reversal_deadzone_reward`：最新加入，惩罚电机换向死区真正激活的比例；不惩罚持续使用负推力。
- `speed_reward`：只惩罚超过速度上限，不奖励速度。
- `crash_reward`：坠毁惩罚。

最新配置里普通动作惩罚降低，跨零/死区惩罚更重要：

```yaml
act_coeff: 0.04
act_second_delta_coeff: 0.02
thrust_switch_coeff: 0.03
reversal_deadzone_coeff: 0.08
```

## 路径与环境策略

路径由 8 个控制点生成，插值为连续曲线，再采样成路径表。`pathPosition(s)` 表示沿路径弧长前进 `s` 米的位置，`closestPathProgress(p)` 每步搜索无人机附近的最近路径进度。

环境策略不是控制无人机，而是生成训练环境。当前生成：

- 8 个 checkpoint/control point 的 xyz 偏移
- 路径长度、半径/横向尺度、垂直尺度、波形、闭合程度、相位
- 风场参数：平均风、xyz 阵风幅值/频率/相位、脉冲风幅值/时间/持续时间

环境策略算法是 SAC 风格：

- actor 输出连续环境动作
- twin critic
- replay buffer
- target network soft update
- action smoothing / task bank 等辅助机制

后续如果表现不符合预期，应优先审查环境策略 reward 是否冗余、是否能正确表达“能力边界上的困难环境”，而不是加手工调节。

## 风扰设计

风场在动力学中通过相对气流阻力作用：

```text
relative_air_velocity = drone_velocity - wind_velocity
wind_force =
  -linear_drag * relative_air_velocity
  -quadratic_drag * |relative_air_velocity| * relative_air_velocity
```

主策略不观测风，保持与实物一致。环境策略可以生成风，因为它负责环境分布。

当前上限思路：

- 合成恒定平均风上限按 `12 m/s` 设定。
- 阵风和脉冲风是在此基础上的分布扩展。
- 后续实物条件主要能做恒定风，因此恒定风能力是重点。

## 真实电机与双向推力死区

用户添加了真实电机数据，当前采用黑桨叶正/反转数据拟合：

```yaml
motor_omega_min: -28000.0
motor_omega_max: 28500.0
motor_tau: 0.045
motor_reversal_dead_time: 0.09
motor_deadzone_omega: 1200.0
thrust_map_p: [ 1.1775e-08, 0.0, 0.0]
thrust_map_n: [-1.1083e-08, 0.0, 0.0]
kappa: 0.0118
```

物理 sanity check：

- 悬停单电机推力约 `1.91 N`
- 悬停转速约 `12746 rpm`
- 正向总最大推重比约 `5.0`
- 反向总最大推力约 `-34.8 N`
- `1200 rpm` deadzone 对应推力约 `0.017 N`

死区逻辑：

1. 期望推力转目标转速。
2. 如果目标转速和当前转速跨符号，且双方都超过 deadzone rpm，触发 `motor_reversal_dead_time`。
3. 死区计时期间目标转速置 0。
4. 计时结束后继续一阶电机响应。
5. 实际转速绝对值小于 deadzone rpm 时输出 0 推力。

死区参数已经做域随机化：

```yaml
motor_reversal_dead_time_scale: 0.40
motor_deadzone_omega_scale: 0.30
```

## 当前已知问题

1. 真实电机/死区版本从随机策略训练会全坠毁。
   - 最新短跑 `2026-06-23-22-23-06`：16 update 后仍 `ep_crashed_mean=1.0`。
   - 负推力比例约 `0.28~0.30`，死区激活比例约 `0.50~0.56`。
   - 原因：随机动作频繁跨零换向，电机反复进入无推力窗口。

2. 不能简单从头训练真实电机/死区版本。
   - 应从稳定权重加载主策略，把这个问题当作 sim-to-real 动力学迁移。

3. 上一轮 quick smoke test 里的临时 Python 脚本报过 `RuntimeError: bad conversion`。
   - 但短训练目录已成功生成 `progress.csv/log.txt`，训练入口基本可用。
   - 若继续验证，可直接用 `run_drone_control.py`，不要用那段临时 YAML dump 脚本。

4. 抗风版本仍有动作抖动。
   - 加大动作惩罚会让策略不愿动作。
   - 当前方向是惩罚高频/跨零/死区，而不是惩罚负推力本身。

## 下一步计划

第一步：从旧稳定主策略迁移真实电机/死区版本。

推荐命令：

```bash
cd /home/Data/FlightLxx/rl/train

python3 run_drone_control.py \
  --train 1 \
  --render 0 \
  --pretrained_weight /home/Data/FlightLxx/rl/train/base-7500/checkpoints/ppo_update_007500.zip \
  --pretrained_env_policy /home/Data/FlightLxx/rl/train/base-7500/checkpoints/environment_policy_update_007500 \
  --reset_pretrained_logstd 1 \
  --reset_pretrained_env_optimizer 1 \
  --env_policy_freq 25 \
  --env_policy_warmup_updates 0 \
  --checkpoint_freq_updates 500 \
  --n_steps 1000
```

第二步：观察真实电机/死区迁移是否恢复：

- `ep_crashed_mean` 是否快速下降
- `ep_completed_mean` 是否恢复
- `ep_motor_reversal_active_ratio_mean` 是否下降
- `ep_reversal_deadzone_reward_mean` 是否接近 0
- `ep_motor_negative_thrust_ratio_mean` 是否仍能在必要时使用负推力
- `policy_entropy` 是否不再在全坠毁状态下塌缩
- `approxkl/clipfrac` 是否健康

第三步：如果仍然全坠毁，不要调环境策略难度，先排查主策略迁移：

- 是否正确加载了旧主策略参数
- 是否重置了 logstd
- 当前真实电机模型下旧策略测试是否能短时间稳定悬停/前飞
- 是否需要先无风迁移，再打开环境策略风扰

## 文档索引

- 权重记录：`docs/training_checkpoint_notes.md`
- 历史设计：`docs/path_following_env_policy_history.md`
- 当前交接：`docs/current_project_handoff.md`

