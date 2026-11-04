# 最终公共PID两轮实验报告数据附件

正式报告：[PDF](../../FINAL_PID_TWO_EXPERIMENTS_REPORT_2026-11-01.pdf)、[Markdown](../../FINAL_PID_TWO_EXPERIMENTS_REPORT_2026-11-01.md)。报告日期2026-11-01，文件均由现有日志只读生成，没有新飞行或修改原结果。

- A：五档等额1000组，CTBR/DR/ID成功374/480/599，共3000次正式尝试。
- B：各分量全随机1000组，CTBR/DR/ID成功297/419/540，共3000次正式尝试。
- 两轮公共PID相同：roll P/I/D=0.097/0.004/0.00135，pitch=0.0869/0.004/0.00135，yaw=0.126/0.0018/0.006。

|文件|内容|
|---|---|
|audit_summary.json|两轮逐模型、逐样本口径的统计、配对结果、描述性区间、初始化误差、失败分类和诊断|
|formal_6000_trials.csv|全部6000次正式试验的逐次指标、状态和原始run目录，不含开发集、pilot和Gazebo展示|
|initial_states_2000.csv|两份不同状态库共2000组初态；按dataset=A/B区分|
|quality_summary.csv|valid、complete_terminal、successful、common_success四种条件下的n、均值、中位、分位统计|
|effective_control_parameter_dump.csv|两轮参数转储一致性核对后的控制参数；命令行转储D四舍五入显示0.0014，真实值0.00135见PID JSON和ULog|
|../../config/final_pid.json|当前唯一固定PID配置|
|../integrity/|6000份原completed.json哈希、保留初态/PID/报告哈希与原始文件属性索引|
|artifact_manifest.json|新报告及图表位置、报告文件哈希、原记录/旧报告未改动的核验结果|
|*.png|报告内9张独立图，可另行引用/导出|

`formal_6000_trials.csv`中的`raw_failure_category`保留原分类，`failure_category`为按已有证据规则审计后的分类。退出阶段析构异常只在满足严格证据时修正失败类别，不改变成功数。原`completed.json`未改写。

请求数组`requested_xyz_rpy_v_w_0..11`顺序为x,y,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz。接管及动作源`handover_xyz_ypr_v_w_0..11`、`action_input_xyz_ypr_v_w_0..11`则为x,y,z,yaw,pitch,roll,vx,vy,vz,wx,wy,wz。坐标为世界NWU、机体FLU，位置z为实际目标高度附近的值，不是高度误差。

末段固定[8,10]仿真秒。缺失值保留为空/JSON null，不能当0；没有完整末段的提前终止轨迹不补造稳态指标。共同成功质量是条件子集（A128组、B70组），应与全流程成功率一同解读。

这份最终报告保留原件；原有调参/报告生成链已删除。新实验使用 `experiment.py analyze --output 新实验目录` 生成结果，具体命令见主README。

成功率、恢复与质量指标定义、全随机的Euler分布限制、A固定基准的选择背景、B准备范围扩展及统计外推限制均见正式报告，不将这些附件误作新的独立试验。
