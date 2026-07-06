# 训练权重与日志简表

更新日期：2026-06-24  
用途：记录当前工作区里仍保留的训练权重、每版相对上一版的主要改动、训练/测试效果，以及推荐用途。

## 当前原则

环境策略不做手工课程或手工加减风干预。设计好环境策略的观测、动作、奖励和参数阈值后，让环境策略自行调整路径和风；如果表现不符合预期，优先重新设计环境策略，而不是加规则强推。我们只控制参数阈值。

## 权重保存文件概览

当前主要权重位于：

- `rl/train/base-7500.zip`
- `rl/train/base-7500/checkpoints/ppo_update_*.zip`
- `rl/train/base-7500/checkpoints/environment_policy_update_*`
- `rl/train/2026-06-17-01/checkpoints/ppo_update_*.zip`
- `rl/train/2026-06-17-01/checkpoints/environment_policy_update_*`
- `rl/train/2026-06-18-01/checkpoints/ppo_update_*.zip`
- `rl/train/2026-06-18-01/checkpoints/environment_policy_update_*`
- `rl/train/2026-06-23-01/checkpoints/ppo_update_*.zip`
- `rl/train/2026-06-23-01/checkpoints/environment_policy_update_*`

说明：早期交接记录里把真实电机训练目录写成 `2026-06-23-22-23-06`；当前工作区实际目录为 `rl/train/2026-06-23-01`。

## 推荐权重

| 权重 | 推荐用途 | 理由 |
| --- | --- | --- |
| `rl/train/base-7500.zip` | 无风/弱风路径跟随初始化 | 视觉测试中 `7000~8500` 附近最稳，复杂 3D 闭合路径完成率高 |
| `rl/train/base-7500/checkpoints/ppo_update_007500.zip` | 与 `base-7500.zip` 等价的 checkpoint 形式 | `update_7500`：完成率 `1.00`，路径误差约 `0.126`，坠毁率 `0` |
| `rl/train/base-7500/checkpoints/environment_policy_update_007500` | 无风闭合路径环境策略初始化 | 已能生成较复杂闭合路径，适合作为路径环境策略基线 |
| `rl/train/2026-06-18-01/checkpoints/ppo_update_022500.zip` | 风扰训练中期测试候选 | 实测 strong 风表现较均衡，完成率高，动作仍有抖动 |
| `rl/train/2026-06-18-01/checkpoints/ppo_update_025000.zip` | 风扰后期候选，但不一定优于 22500 | 日志完成率仍高，但 `clipfrac` 偏大，路径/风分布变化后视觉效果不一定更好 |
| `rl/train/2026-06-23-01/checkpoints/ppo_update_004500.zip` | 真实电机/换向死区迁移当前候选 | `update_4500` 后完成率约 `0.90+`，换向死区激活已降到约 `1%`，但路径闭合度偏低 |

## 关键训练版本

| 训练目录 / 参数 | 相比前一版的主要改动 | 日志结果 | 结论 |
| --- | --- | --- | --- |
| `base-7500` | 从早期路径跟随训练整理出的无风 3D 闭合路径基线 | `update_7500`：`ep_completed_mean=1.00`，`ep_path_error_mean≈0.126`，`ep_crashed_mean=0`，`checkpoint_complexity≈0.94`，`closure_score≈0.92` | 当前最可靠的无风主策略初始化 |
| `base-7500` 后期 `update_25000` | 继续训练同一设定 | 仍 `completed=1.00`，但 `path_error≈0.239`，视觉测试不一定比 `7500` 更稳 | 不推荐盲目选择最后一版；`7500` 更均衡 |
| `2026-06-15-01` | 加入第一版风扰环境策略：平均风、xyz 阵风、短时脉冲风；当时仍存在手工 `wind_boost` 思路 | `update_250`：`completed≈0.98`，`path_error≈0.243`，`wind≈5.52 m/s`；尾段 `completed≈0.83`，`crashed≈0.17`，`wind≈6.01 m/s` | 早期风扰版本，可参考但不再作为最新设计依据 |
| `2026-06-17-01` | 删除手工 wind boost 后，环境策略自行调风；引入更强风扰训练 | `update_7500`：`completed=1.00`，`path_error≈0.062`，`wind≈5.91 m/s`；尾段 `completed=1.00`，`path_error≈0.108`，`wind≈5.62 m/s` | 能稳定抗中等风，但路径闭合度一段时间没起来 |
| `2026-06-18-01` | 参考 Link-U-OS 思路加入长历史/CNN、延迟建模、域随机化、高频动作惩罚等；从头训练 | 早期 `update_250` 很差：`completed≈0.09`，`crashed≈0.91`；到 `update_750` 恢复：`completed≈0.97`，`path_error≈0.328`；`update_7500`：`completed=1.00`，`path_error≈0.174`，`wind≈4.81`；尾段 `completed≈0.93`，`path_error≈0.595`，`wind≈4.39` | 后期能完成，但 `approxkl/clipfrac` 偏大，说明更新较激进；实测中 `22500` 比部分更后期权重更稳 |
| `2026-06-23-01` | 引入真实电机数据、双向推力换向死区、死区参数域随机化；新增 `reversal_deadzone_reward` 与 `hover_action_norm` 日志；从 `base-7500` 迁移继续训练 | 早期 16 update：`completed=0`、`crashed=1.0`、`path_error≈4.9`；到 `update_4637`：`completed=0.94`、`crashed=0.06`、`path_error≈0.464`、`wind≈3.49 m/s`、`motor_negative_thrust_ratio=0.025`、`motor_reversal_active_ratio=0.0075` | 迁移方向有效，已经摆脱全坠毁和高频换向死区；当前主要问题是环境策略给出的闭合路径难度偏低，PPO `early_stop_kl=1` 长期存在 |

## 最新真实电机/死区改动状态

上一轮改动已经完成到代码和安装层面：

- `libs/config/rl_env.yaml`
  - 真实电机参数：正反向推力系数、最大转速、`kappa`
  - 换向死区：`motor_reversal_dead_time=0.09`，`motor_deadzone_omega=1200`
  - 域随机化：`motor_reversal_dead_time_scale=0.40`，`motor_deadzone_omega_scale=0.30`
  - 奖励调整：降低普通动作变化惩罚，提高跨零切换惩罚，新增死区激活惩罚
- `libs/src/objects/quadrotor.cpp`
  - 电机换向时先进入无推力窗口，再按一阶响应追踪目标转速
- `libs/src/dynamics/quadrotor_dynamics.cpp`
  - 正/反转 thrust-rpm 映射
  - deadzone rpm 内输出 0 推力
- `libs/src/envs/rl_env.cpp`
  - 新增 `reversal_deadzone_reward`
  - 新增 `hover_action_norm`，当前日志为 `0.5`，说明悬停动作尺度没有跑到动作边界
- `rl/lxx_baselines/ppo/ppo2.py`
  - 控制台日志新增死区惩罚、死区激活比例、悬停动作指标

当前尚未完成：

- 真实电机/死区迁移已经从全坠毁恢复到可用区间，但仍需更长训练和多方向风测试验证。
- 当前 `2026-06-23-01` 最近 100 个 update：`completed≈0.908`，`crashed≈0.092`，`path_error≈0.638`，`progress≈0.929`，`wind_mean_norm≈3.60 m/s`。
- 当前路径难度偏中等：最近 100 个 update `checkpoint_complexity≈0.512`，但 `closure_score≈0.026`、`closure_error≈11.8 m`、`reverse_ratio≈0.063`，说明不是复杂 3D 闭合路径。
- 负推力使用已经很克制：最近 100 个 update `motor_negative_thrust_ratio≈0.042`，`motor_reversal_active_ratio≈0.012`，`reversal_deadzone_reward≈-0.00094`。
- 姿态/动作稳定性比早期好：最近 100 个 update `body_rate_reward≈-0.050`，动作一阶/二阶惩罚保持很小。
- 从随机初始化训练仍会全坠毁；真实电机/死区训练应继续从稳定权重迁移，不建议从零开始。
- 上次 quick smoke test 使用的临时 YAML 转换脚本报过 `bad conversion`，但实际训练目录 `2026-06-23-01` 已能跑出 progress.csv 和 checkpoint，说明训练入口可用。

## 最新风向对称改动

2026-06-24 已加入 episode 级风向翻转：

- 每个训练 episode 随机采样 `episode_wind_sign` 为 `+1/-1`。
- 环境策略仍生成风的强度和结构；实际施加到动力学的平均风、阵风、脉冲风整体乘以该符号。
- 不旋转路径。此改动只解决风方向偏置，不改变路径分布。
- `extra_info` 新增 `wind_sign`，同时 `wind_mean_*` 和 `wind_burst_amp_*` 记录实际施加方向。
- 非随机评估/`--test_wind strong` 默认保持固定方向，便于复现实验。
- 已执行 `python3 setup.py build_ext --inplace` 并通过；Python smoke test 确认 `wind_sign_sample [-1.0, 1.0]`。

注意：`2026-06-23-01/progress.csv` 是此改动之前的训练日志，里面没有 `wind_sign` 字段。下一轮训练日志应重点看 `ep_wind_sign_mean` 是否接近 0，以及 `wind_mean_x/y/z` 是否正负对称。

## 下一步推荐训练命令

如果继续当前真实电机/死区训练，建议从 `2026-06-23-01/checkpoints/ppo_update_004500.zip` 和对应环境策略继续；如果想重新跑一条干净的风向对称迁移曲线，则仍可从 `base-7500` 启动。

从无风稳定主策略迁移到真实电机/死区版本，环境策略每 25 个 update 更新一次，每 500 个 update 保存一次：

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

如果目标是接着 `2026-06-18-01` 的抗风训练迁移，可将主策略改成：

```bash
--pretrained_weight /home/Data/FlightLxx/rl/train/2026-06-18-01/checkpoints/ppo_update_022500.zip
```

是否加载 `2026-06-18-01` 的环境策略需要谨慎：真实电机/死区改变了主策略能力边界，旧环境策略可能会立即给过难风/路径。更稳的做法是主策略加载旧权重，环境策略从 `base-7500` 或重新初始化。

## 观察指标

真实电机/死区迁移训练应重点观察：

- `ep_completed_mean`：应从 0 快速恢复到可完成
- `ep_crashed_mean`：应明显下降
- `ep_motor_reversal_active_ratio_mean`：应下降，表示不再频繁进入无推力换向窗口
- `ep_motor_negative_thrust_ratio_mean`：不要求为 0；合理目标是负推力使用更有规律
- `ep_reversal_deadzone_reward_mean`：应接近 0，表示死区激活少
- `ep_hover_action_norm_mean`：应稳定约 `0.5`；若接近动作边界，说明动作尺度有问题
- `ep_wind_sign_mean`：加入风向翻转后应接近 `0`，说明正反风覆盖均衡
- `ep_wind_mean_x/y/z_mean`：长期不应只固定一个世界方向；实际施加方向应正负都有
- `ep_checkpoint_closure_score_mean` / `ep_checkpoint_reverse_ratio_mean`：当前真实电机训练偏低，后续应观察是否能重新推回复杂闭合路径
- `policy_entropy`：不应在全坠毁状态下快速塌缩
- `approxkl/clipfrac`：若长期偏高，说明 PPO 更新过激
