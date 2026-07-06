# FlightLxx 路径跟随与环境策略设计记录

记录日期：2026-06-12  
运行环境：`flightlty` conda 环境  
主要代码位置：

- 主环境：`libs/src/envs/rl_env.cpp`
- 路径几何：`libs/src/envs/rl_env_path.cpp`
- C++ 环境策略接口：`libs/src/envs/rl_env_policy.cpp`
- 配置：`libs/config/rl_env.yaml`
- Python 环境策略：`rl/lxx_baselines/envs/environment_policy.py`
- 训练入口：`rl/train/run_drone_control.py`

## 1. 当前目标

当前任务不是竞速，而是无人机路径跟随。

核心想法来自“Environment as Policy”一类论文：把环境也看成一个可学习策略，让环境策略自动生成匹配当前主策略能力的任务。论文里环境策略主要生成赛道/门的位置，我们这里改成生成路径。当前目标是：

1. 主策略学习稳定跟随复杂 3D 闭合路径，例如 8 字轨迹。
2. 环境策略自动生成符合当前无人机能力、但有学习价值的路径。
3. 后续加入风扰时，不改变主策略观测，让环境策略同时生成路径和风场。

## 2. 当前主策略设计

### 2.1 主策略观测

当前主策略观测维度是 25：

| 索引 | 内容 |
| --- | --- |
| `0:6` | 机体系姿态矩阵前两列 |
| `6:9` | 世界系线速度 |
| `9:12` | 机体系角速度 |
| `12:16` | 上一步归一化动作 |
| `16:19` | `closest_path_pos - drone_pos` |
| `19:22` | `lookahead_path_pos - drone_pos` |
| `22:25` | 当前路径切向方向 |

这里没有参考速度误差，也没有风场信息。主策略只通过自身状态、路径相对几何和历史动作来做控制。

### 2.2 主策略动作

动作维度是 4，对应四个电机/桨的归一化单桨推力命令。

当前代码中：

```cpp
act_mean_ = 0
act_std_ = (-mass * Gz) / 2
quad_act_ = act * act_std_ + act_mean_
cmd_.thrusts = quad_act_
```

动力学配置中：

```yaml
motor_omega_min: -29400.0
motor_omega_max: 29400.0
thrust_map_p: [ 1.58536769e-08, 0.0, 0.0]
thrust_map_n: [-1.58536769e-08, 0.0, 0.0]
```

因此当前动力学支持双向推力。测试图里大部分动作表现为正推力，是策略自然学到的结果，不是因为负推力没有生效。

### 2.3 路径表示

现在已经从“离散 checkpoint 目标点”改成连续路径跟随。

路径由 8 个控制点生成，然后用 Catmull-Rom 插值形成连续曲线，再采样成离散路径表：

```text
8 个控制点 -> 插值曲线 -> sampled_path_ / sampled_s_
```

`pathPosition(s)` 表示沿路径弧长前进 `s` 米处的位置。  
`pathTangent(s)` 表示该位置处的路径切向。  
`closestPathProgress(position)` 每一步搜索无人机附近的路径段，找到当前最接近无人机的路径进度 `s`。

目前路径总长限制在 20m 以下，目标是生成复杂但仍可学习的 3D 闭合路径。

### 2.4 初始化

reset 时不再固定初始化，而是在路径起点附近做小范围随机化：

- 位置：横向和高度扰动
- 速度：前向、横向、竖直方向小扰动
- 姿态：小 roll/pitch/yaw 扰动
- 角速度：小扰动

这样避免策略只会从完全相同的初始状态起飞。

### 2.5 当前主策略奖励函数

当前主策略奖励是连续路径跟随奖励：

```text
reward =
  alive_reward
+ advance_reward
+ path_reward
+ body_rate_reward
+ act_reward
+ speed_reward
+ crash_reward
```

当前配置中的主要系数：

```yaml
alive_reward: 0.0
advance_coeff: 1.0
advance_ref_speed: 1.0
track_gate_sigma: 1.0
path_coeff: 0.5
path_error_scale: 1.0
body_rate_coeff: 0.1
body_rate_scale: 4.0
act_coeff: 0.1
act_delta_scale: 2.0
speed_limit: 2.0
speed_coeff: 0.2
crash_penalty: -30.0
terminal_penalty: -10.0
success_reward: 20.0
```

各项含义：

1. `advance_reward`

   奖励沿连续路径的进度增长，但会被路径误差门控：

   ```text
   delta_s = max(0, progress_s - previous_progress_s)
   advance_n = clamp(delta_s / (advance_ref_speed * dt), 0, 1)
   track_gate = exp(-path_error^2 / track_gate_sigma^2)
   advance_reward = advance_coeff * advance_n * track_gate
   ```

   这样可以鼓励无人机往前飞，但如果离路径太远，前进奖励会降低，避免只冲速度。

2. `path_reward`

   连续路径横向误差惩罚：

   ```text
   path_reward =
     -path_coeff * clamp((path_error / path_error_scale)^2, 0, max_penalty)
   ```

3. `body_rate_reward`

   机体角速度惩罚，用来减少抖动：

   ```text
   body_rate_reward =
     -body_rate_coeff * clamp((||omega|| / body_rate_scale)^2, 0, max_penalty)
   ```

4. `act_reward`

   动作变化惩罚，用来减少高频控制：

   ```text
   act_reward =
     -act_coeff * clamp((||act - prev_act|| / act_delta_scale)^2, 0, max_penalty)
   ```

5. `speed_reward`

   超过前向速度上限才惩罚：

   ```text
   speed_over = max(0, forward_speed - speed_limit)
   speed_reward =
     -speed_coeff * clamp((speed_over / speed_limit)^2, 0, max_penalty)
   ```

6. `crash_reward`

   高度过低时给坠毁惩罚。

终止条件包括：

- 完成整条路径且末端误差足够小
- 高度低于地面阈值
- 路径误差超过 `max_tracking_error`
- episode 时间超过 `max_t`

## 3. 当前环境策略设计

### 3.1 环境策略目标

环境策略不是直接控制无人机，而是生成训练环境。当前主要生成：

- 8 个 checkpoint/control point 的偏移
- 路径长度
- 横向尺度
- 垂直尺度
- 波形数量
- 闭合路径混合比例 `shape_mix`
- 相位

环境策略要学的是：当前主策略能力下，什么路径最有训练价值。

### 3.2 环境策略观测

环境策略观测维度当前是 37，主要包含：

- 8 个 checkpoint 的归一化位置
- 上一轮路径误差
- 上一轮 lookahead 误差
- 上一轮进度
- 是否完成
- 是否坠毁
- 环境得分
- 不可行环境计数

环境策略可以看到主策略最近表现，用这些信息判断当前无人机能力。

### 3.3 环境策略动作

环境策略动作维度当前是 30：

```text
24 维 checkpoint xyz 偏移
6 维全局路径参数
```

全局路径参数是：

```text
length
radius
vertical
wave
closed / shape_mix
phase
```

### 3.4 环境策略算法

当前 Python 环境策略是 SAC 风格实现，而不是最早的简单 REINFORCE：

- actor 输出连续环境动作
- twin critic 估计环境动作价值
- target critic 做软更新
- replay buffer 复用历史环境交互
- action smoothing 减少环境突然跳变
- task bank 保存过去有价值的环境动作

这里要注意：当前代码里还有一些 guard / boost 逻辑，用来避免环境策略长期停留在过简单或不可行路径上。它不是人工给定固定课程表，但仍属于结构性引导。后续如果追求更纯粹的环境策略，可以进一步把这些 boost 融入环境 reward 或移除。

### 3.5 当前环境策略奖励

环境策略奖励不是“路径越难越好”，而是奖励处在主策略能力边界附近的任务。

主要思想：

```text
太容易 -> 低奖励
完全飞不了 -> 低奖励
能飞一部分、有挑战、误差可学习 -> 高奖励
```

环境策略 reward 当前综合：

- `learnability`
- `tracking_score`
- `frontier_score`
- 路径复杂度
- 闭合程度
- 是否完成
- 是否坠毁
- 是否不可行
- 与目标难度/目标闭合度的距离

### 3.6 当前路径生成效果

当前环境策略已经不是只生成直线。最新训练中观察到：

- `shape_mix` 接近 1
- `checkpoint_complexity` 较高
- `checkpoint_reverse_ratio` 明显大于 0
- `checkpoint_closure_score` 较高
- 路径长度接近 18-20m

这说明路径已经接近复杂闭合/回转结构，而不只是简单前进曲线。

## 4. 按时间线记录的主要尝试

### 4.1 初始想法：仿照论文做环境策略

最开始参考论文，把“环境也作为策略”的思想迁移到无人机路径跟随任务。论文是竞速穿门，我们不做竞速，因此把“门”改成路径或虚拟目标点。

结论：思路可行，但不能照搬竞速奖励。路径跟随更关注稳定跟随、误差、平滑控制，而不是最大速度。

### 4.2 早期动态路径生成器

一开始实现过根据排名/表现动态调整路径的生成器。

效果：

- 主策略能逐渐变好
- 实现简单

问题：

- 它不是真正的环境策略
- 没有独立的环境策略网络
- 本质是手写路径 curriculum

结论：作为 baseline 有用，但不符合最终目标。

### 4.3 第一次“真正环境策略”尝试

随后改成环境策略网络生成路径。

问题：

- 环境策略更新太激进
- 生成大量当前无人机完全无法完成的路径
- 主策略梯度没有稳定优化方向
- 训练不如早期动态路径生成器

结论：环境策略必须追求“能力边界附近”的任务，而不是单纯制造困难任务。

### 4.4 从 REINFORCE 改为 SAC 风格环境策略

为了稳定环境策略训练，引入 SAC 风格：

- replay buffer
- actor/critic
- target critic
- action smoothing
- conservative update

效果：

- 比简单 REINFORCE 稳定
- 环境策略可以积累历史经验

问题：

- 如果 reward 设计不合理，仍会生成不利于主策略学习的环境

结论：算法稳定性改善了，但关键仍是环境策略 reward 和路径参数化。

### 4.5 去掉参考速度误差

曾经观测/奖励里包含类似参考速度误差的项。后来确认当前任务是路径跟随，不是轨迹跟踪。

修改：

- 去掉不必要的参考速度误差
- 主策略只关心路径几何、状态和控制历史

结论：这是正确方向。路径跟随不应该强迫无人机匹配某条时序轨迹的速度。

### 4.6 生存奖励问题

曾经设置过较大的生存奖励。

问题：

- 飞机可以通过悬停或保守动作获得生存奖励
- 如果前进奖励/路径奖励弱，策略会混生存奖励

后续：

- 生存奖励降到很小
- 最终当前版本设为 `0.0`

结论：路径跟随任务中，生存奖励不能成为主驱动力。

### 4.7 进度奖励导致超速问题

早期使用过较直接的 progress reward。

问题：

- 策略倾向于用速度刷进度
- 可能冲过路径、坠毁、抖动

后续改动：

- 降低目标速度和速度上限
- 引入 gated advance reward
- 进度奖励只在路径误差较小时有效
- 加入超速惩罚

结论：必须把“往前飞”和“贴着路径飞”绑定，否则会变成竞速。

### 4.8 离散 checkpoint / 虚拟目标点尝试

为了更接近论文穿门任务，尝试把路径改成穿过虚拟目标点。

问题：

- 8 个点难以完整表达 20m 复杂 8 字路径
- 奖励偏稀疏
- 策略可能只学会穿点，不学连续曲线
- 目标点之间可能走直线，不一定贴合弧线

结论：单纯离散 checkpoint 不适合我们最终要的连续路径跟随。

### 4.9 连续路径采样与最近路径点

后续改成连续路径：

- 8 个控制点通过插值生成曲线
- 每一步计算最近路径进度 `s`
- 观测里给最近路径点、前视点和切向
- reward 用连续路径误差和路径进度

效果：

- 比离散 checkpoint 更适合 8 字和闭合路径
- 主策略能学习连续路径跟随

结论：这是当前版本的核心结构。

### 4.10 前视点误差奖励尝试

曾尝试用前视点误差作为主要奖励，包括：

- 惩罚 `||drone_pos - lookahead_pos||`
- 只保留 0.4m 前视点
- 如果当前 step 更接近前视点就给奖励，远离则惩罚

问题：

- 前视点每一步都更新，奖励目标也在动
- 早期速度慢时单步接近量很小，奖励信号弱
- 策略 entropy 变大，动作发散
- 前视误差长期负，不利于稳定学习

结论：前视点适合做观测，不适合作为当前版本的主奖励。

### 4.11 当前奖励成型

最后形成当前奖励结构：

```text
gated advance
+ continuous path error penalty
+ body-rate penalty
+ action-delta penalty
+ overspeed penalty
+ crash / success terminal
```

有效原因：

- 有正向前进信号
- 但前进信号被路径误差门控
- 保留平滑控制约束
- 不需要参考速度误差
- 不依赖稀疏 checkpoint pass reward

结论：这是目前训练效果最稳定的一版主策略奖励。

### 4.12 n_steps 与 rollout 处理

曾经出现过 `ep_rollout_cut_mean` 很高的疑问。

澄清：

- `n_steps` 是 PPO 每次采样 rollout 长度，不等于环境 episode 必须在这里结束
- 当前 `run_drone_control.py` 默认 `n_steps=1000`
- `reset_on_rollout_end=0`，episode 可以跨 PPO rollout 继续

结论：现在不会因为 PPO rollout 截断就强制 reset，除非显式设置 `--reset_on_rollout_end 1`。

### 4.13 环境策略生成复杂闭合路径

为了解决环境策略长期只生成接近直线的问题，后来加入：

- `shape_mix`
- 闭合路径混合
- reverse ratio
- closure score
- checkpoint complexity
- 路径长度 20m 上限

效果：

- 环境策略逐渐生成接近 8 字或复杂闭合的路径
- 最新训练中主策略可以在复杂闭合路径上完成率接近 1
- 用 8 字测试可以飞完

结论：这是环境策略明显有效的一次改动。

### 4.14 平滑控制调参

在能飞完 8 字后，主要问题变成抖动。

曾调整：

```yaml
body_rate_coeff: 0.1
act_coeff: 0.1
```

效果：

- 动作曲线比之前平滑很多
- 8 字测试明显改善
- 仍存在一定抖动，但不再是完全高频乱跳

结论：提高机体角速度和动作变化惩罚有效，但不能过大，否则可能让策略不愿意做必要动作。

## 5. 有效果的尝试

1. 去掉参考速度误差。

   路径跟随不是轨迹跟踪，不需要强制匹配参考速度。

2. 生存奖励设为 0。

   避免悬停刷奖励。

3. 连续路径替代离散 checkpoint。

   对 8 字和闭合曲线明显更合理。

4. 使用 gated advance reward。

   让无人机往前飞，但只有贴近路径时前进才有高价值。

5. 加入超速惩罚。

   抑制进度奖励诱导的高速冲刺。

6. 加入动作变化和角速度惩罚。

   明显减少抖动。

7. 环境策略使用 SAC 风格而不是简单 REINFORCE。

   更新更稳定，有 replay 和 critic 支撑。

8. 环境策略路径生成加入闭合度和回转结构。

   解决了长期只生成简单直线路径的问题。

9. 训练中保存 PPO 和环境策略 checkpoint。

   可以随时测试阶段性效果，也可以用主策略参数继续训练。

10. 主策略预训练加载时重置 logstd。

    既复用已有飞行能力，又保留后续探索。

## 6. 效果不好的尝试

1. 只按排名调整随机路径。

   能训练，但不是环境策略。

2. 环境策略过激生成困难路径。

   主策略完全飞不了，梯度没有有用方向。

3. 较大的生存奖励。

   容易让策略保守悬停。

4. 未门控的进度奖励。

   容易诱导超速、冲路径、坠毁。

5. 离散 checkpoint pass reward。

   对连续 8 字路径表达不足，奖励也偏稀疏。

6. 前视点误差作为主奖励。

   因为前视点随 step 更新，早期奖励信号弱且不稳定。

7. 只提高路径难度而不考虑主策略可学习性。

   环境策略会制造不可完成任务，训练退化。

## 7. 最新训练观察

最近较好的训练目录：

```text
rl/train/2026-06-11-17-58-00/
```

代表性现象：

- 主策略单圈完成率接近或达到 1
- 路径进度接近完整路径
- 坠毁率接近 0
- `shape_mix` 接近 1
- `checkpoint_complexity` 较高
- `checkpoint_reverse_ratio` 明显存在
- `checkpoint_closure_score` 较高
- 路径长度接近 18-20m
- 8 字测试能飞完，后续调大 `body_rate_coeff` 和 `act_coeff` 后抖动明显减轻

这说明当前版本已经从“能不能飞完”进入“如何更稳、更平滑、更鲁棒”的阶段。

## 8. 当前加载与测试方式

测试 8 字轨迹：

```bash
conda run -n flightlty python rl/train/run_drone_control.py \
  --train 0 \
  --render 1 \
  --path_mode figure8 \
  -w /home/Data/FlightLxx/rl/train/2026-06-11-17-58-00/checkpoints/ppo_update_005000.zip
```

从已有主策略和环境策略继续训练：

```bash
conda run -n flightlty python rl/train/run_drone_control.py \
  --train 1 \
  --render 0 \
  --pretrained_weight /home/Data/FlightLxx/rl/train/2026-06-11-17-58-00/checkpoints/ppo_update_005000.zip \
  --pretrained_env_policy /home/Data/FlightLxx/rl/train/2026-06-11-17-58-00/checkpoints/environment_policy_update_005000 \
  --reset_pretrained_logstd 1 \
  --reset_pretrained_env_optimizer 0 \
  --n_steps 1000 \
  --env_policy_freq 100 \
  --env_policy_warmup_updates 100 \
  --checkpoint_freq_updates 1000
```

如果环境策略 action/obs 维度改变，例如后续加入风参数，旧环境策略 checkpoint 通常不能直接加载；主策略 checkpoint 仍可作为初始化，只要主策略观测和动作维度不变。

如果改了 C++ 文件，训练前需要重新编译并安装：

```bash
conda run -n flightlty python setup.py build_ext --inplace
/home/ubuntu/anaconda3/envs/flightlty/bin/pip install .
```

## 9. 下一步：加入风扰的设计计划

### 9.1 基本原则

1. 主策略观测先不变。

   实际无人机通常无法直接获得完整风场信息，因此主策略不加入风速观测。它通过状态误差、速度和路径偏差间接应对风扰。

2. 主策略奖励先不变。

   当前奖励已经能稳定完成复杂路径。加入风扰时先只改动力学和环境策略，避免同时改太多变量。

3. 环境策略重新训练。

   加入风参数会改变环境策略动作维度，旧环境策略参数不能直接复用。

4. 主策略可以加载已有 checkpoint。

   只要主策略观测和动作维度不变，已有飞行策略可以作为初始化。

### 9.2 风场由环境策略生成什么

建议第一版风场：

```text
wind(t) =
  mean_wind_xyz
+ gust_xyz(t)
+ burst_xyz(t)
```

暂时不要加入风切变，因为当前无人机低空飞行，高度变化只有几米，风切变会增加复杂度但收益不明显。

环境策略每次 reset 生成一组风场参数，而不是每一步直接给风：

```text
mean_wind_x/y/z        固定风，一局内基本不变
gust_amp_x/y/z         三轴阵风幅值
gust_freq_x/y/z        三轴阵风频率
gust_phase_x/y/z       三轴阵风相位
burst_amp_x/y/z        一次短时扰动的峰值
burst_start_ratio      扰动在 episode 的哪个比例开始
burst_duration         扰动持续多久
```

一局内风随时间连续变化：

```text
gust_i(t) = gust_amp_i * sin(2*pi*gust_freq_i*t + gust_phase_i)
```

短时扰动不是每个 step 的白噪声，而是一个平滑风包：

```text
0 -> 增强 -> 峰值 -> 衰减 -> 0
```

这样能模拟突然的一阵风，又不会把训练变成不可学习的高频随机噪声。

### 9.3 风如何作用到动力学

不建议直接：

```text
state.a += wind
```

更合理的简化模型是相对气流阻力：

```text
v_air = drone_velocity - wind_velocity
F_wind = -k_drag_linear * v_air
         -k_drag_quad * ||v_air|| * v_air
a_wind = F_wind / mass
```

然后在 `libs/src/objects/quadrotor.cpp` 中加入：

```cpp
state_.a =
  state_.q() * thrust_force / mass
+ gravity
+ wind_force / mass;
```

当前最合适的改动位置是 `Quadrotor::run()` 中计算线加速度的地方。

### 9.4 加风后的环境策略 reward 思路

环境策略不应该被奖励“风越大越好”。它应该继续追求：

```text
当前主策略能飞，但有挑战，有学习价值
```

加入风后，环境策略 difficulty 可以由两部分组成：

```text
total_difficulty =
  path_difficulty
+ wind_difficulty
```

其中 wind difficulty 来自：

- 平均风强度
- 阵风强度
- 阵风频率
- 短时扰动峰值和持续时间

但是环境策略 reward 仍然要根据主策略表现来评价：

```text
太容易 -> 低奖励
完全飞不了 -> 低奖励
能飞、有误差、有挑战 -> 高奖励
```

### 9.5 加风后的日志

建议新增日志但不参与主策略 reward：

```text
ep_wind_mean_norm
ep_wind_gust_norm
ep_wind_burst_norm
ep_wind_total_norm
ep_wind_difficulty
env_policy_wind_difficulty_mean
```

这样可以判断环境策略是否真的在生成有意义的风场。

## 10. 后续优先级

1. 保持当前无风版本作为稳定 baseline。

2. 加入风动力学，但主策略观测和奖励先不变。

3. 扩展环境策略 action，让它生成路径和风场。

4. 主策略加载当前较好 checkpoint，环境策略重新训练。

5. 先评估固定风和三轴阵风，再加入短时平滑扰动。

6. 如果加风后抖动重新变大，再微调：

   - `body_rate_coeff`
   - `act_coeff`
   - `speed_coeff`
   - wind 参数边界

7. 如果主策略在有风环境完全无法恢复，不要先手动降低环境策略难度，而是检查环境策略 reward 是否没有正确区分“可学习困难”和“不可学习困难”。

