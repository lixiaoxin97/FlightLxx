"""Mathematical methods matched to the frozen five-level experimental code."""
from pathlib import Path
import json


def sampling(doc):
    doc.title('1.1 初态集合的数学构造')
    doc.para('世界坐标记为W（NWU），机体记为B（FLU），R_WB将机体向量旋转到世界。每个请求状态有12个标量，释放目标位置为p*，默认目标水平且yaw=0。L1静止重复并不增加不同状态覆盖数；它检验同一请求下传感、估计和时序变化的重复可靠性。')
    doc.equation(r's_0=[\mathbf{p}_0^T,\phi_0,\theta_0,\psi_0,\mathbf{v}_0^T,\mathbf{\omega}_0^T]^T,\quad \mathbf{p}^*=(0,0,5)^T',
        '式(1)：请求状态按位置、roll/pitch/yaw、世界线速度和机体角速度排列。')
    doc.para('每档随机部分采用随机拉丁超立方（LHS）。若有n个状态，每个维度独立打乱n个分层编号，并在各层内均匀抖动；每一维因此恰好覆盖n个等概率区间。跨维采用不同排列，不等同于各维完全独立的IID样本。正式L1随机部分n=400，其余各档n=500。')
    doc.equation(r'u_{ij}=\frac{\pi_j(i)+\xi_{ij}}{n},\quad\xi_{ij}\sim U(0,1),\quad \pi_j:\{0,\ldots,n-1\}\to\{0,\ldots,n-1\}',
        '式(2)：pi_j为维度j的随机排列，xi_ij为层内均匀随机数。')
    doc.equation(r'e_{p,j}=P_\ell(2u_j-1),\quad v_j=V_\ell(2u_{j+3}-1),\quad\omega_j=W_\ell(2u_{j+6}-1)',
        '式(3)：P_l、V_l、W_l为第l档表中对应上界；p_0=p*+e_p。单位分别为m、m/s、rad/s。')
    doc.para('物理倾角beta定义为机身z轴与世界竖直的夹角。生成器分层cos(beta)，再采样方位alpha及独立ZYX yaw误差psi。具体变量化如下；theta和phi分别是pitch和roll。注意alpha是生成R第三行的方位参数，报告不把它宣称为世界系的均匀机身倾转方位。')
    doc.equation(r'\cos\beta=\cos\beta_{\min}+u_{10}(\cos\beta_{\max}-\cos\beta_{\min}),\quad \alpha=2\pi u_{11}',
        '式(4)：每档在对应cos倾角区间分层，不按Euler分量大小直接限制倾角。')
    doc.equation(r'\theta=\arcsin(-\sin\beta\cos\alpha),\quad\phi=\operatorname{atan2}(\sin\beta\sin\alpha,\cos\beta),\quad\psi=\Psi_\ell(2u_{12}-1)',
        '式(5)：R=R_z(psi)R_y(theta)R_x(phi)，其R33=cos(beta)。请求yaw存入[0,2pi)，物理姿态不变。')
    doc.para('上述参数化保证各档物理倾角界限和yaw误差范围，但既不是三个Euler角分别均匀，也不声称在SO(3)上Haar均匀。该设计用于可控、充分覆盖的五档组合基准；总体成功率按五档各20%权重解释。全部请求在正式运行前冻结，网络成绩不参与采样。')


def platform_and_coordinates(doc,manifest):
    doc.title('2.1 仿真对象、网络接口与坐标数学')
    doc.table(['对象','固定值或约定'],[
        ['机型','Prototype Recovery；总质量0.78kg，臂长0.125m；目标惯量diag(0.0023,0.0023,0.0036) kg*m^2'],
        ['电机','四个；T=k_f*Omega^2，k_f=1.4131562457e-5 N/(rad/s)^2；升降时间常数0.033s'],
        ['角速度/推力范围','roll/pitch +/-2pi，yaw +/-pi rad/s；比推力0至19.62m/s^2'],
        ['物理/控制周期','Gazebo步长0.004s，目标250Hz；网络目标50Hz；计分10仿真秒'],
        ['重力','Gazebo(0,0,-9.8)m/s^2；动作缩放保留训练约定9.81'],
        ['网络','三个现有12-64-64-4 tanh策略，确定性推理，不重新训练或修改权重'],
        ['推理核对','已保留4096个输入的原PPO与NumPy单样本路径核验，三模型最大绝对动作差均<7e-7'],
        ['数据来源','策略输入来自PX4 VehicleOdometry；Gazebo真值用于传感链、初始化及独立交叉检查'],
    ],[1,3.6])
    doc.para('三网络使用完全相同的物理平台。自由飞行阶段的简化刚体结构为下式；实际仿真还包含电机一阶响应、分配限幅、PX4滤波/积分器和估计时序，不能把下式当作理想指令被瞬时执行。')
    doc.equation(r'\dot{\mathbf{p}}=\mathbf{v},\quad m\dot{\mathbf{v}}=R_{WB}\mathbf{e}_3\sum_{i=1}^4 T_i-mg\mathbf{e}_3',
        '式(6)：e3=(0,0,1)^T；世界系重力向下，比推力沿机体正z。')
    doc.equation(r'\dot R_{WB}=R_{WB}[\mathbf{\omega}]_\times,\quad J\dot{\mathbf{\omega}}=\mathbf{\tau}-\mathbf{\omega}\times(J\mathbf{\omega}),\quad\dot\Omega_i=(\Omega_{i,d}-\Omega_i)/\tau_m',
        '式(7)：J为惯量，tau_m=0.033s；Omega_i为旋翼角速度。')
    doc.para('令D=diag(1,-1,-1)。以下变换在位置、速度和姿态上保持一致；w是机体系量，仅做FLU/FRD变换。')
    doc.equation(r'\mathbf{p}_N=D\mathbf{p}_W,\quad\mathbf{v}_N=D\mathbf{v}_W,\quad R_{ND}=D R_{WB}D,\quad \mathbf{\omega}_D=D\mathbf{\omega}_B',
        '式(8)：N=PX4 NED，D下标表示机体FRD；D矩阵自身等于Rx(pi)。')
    doc.equation(r'\mathbf{p}_E=R_z(\pi/2)\mathbf{p}_W=(-p_y,p_x,p_z)^T,\quad R_{EB}=R_z(\pi/2)R_{WB}',
        '式(9)：E=Gazebo ENU。本轮世界yaw对齐角为0，不将随机初始yaw重置为目标yaw。')
    doc.para('策略观测为o=[p^T,yaw,pitch,roll,v^T,w^T]^T，位置输入为实际位置而非预先减目标的误差。BODY_FRD速度须先按姿态转到世界；NED速度则直接按式(8)转换。部署使用principal_zyx；请求与实际姿态用四元数旋转距离校验，避免等价Euler分支造成伪误差。训练/部署姿态编码的差异属于现有接口条件，本轮未作新的编码消融。')


def action_mapping(doc,manifest):
    doc.title('2.2 神经网络输出、比推力返归一化与公共底层')
    doc.equation(r'\mathbf{\omega}_c=(2\pi a_0,2\pi a_1,\pi a_2)^T,\quad c=9.81(a_3+1),\quad \mathbf{a}=\pi_\vartheta(\mathbf{o})\in[-1,1]^4',
        '式(10)：c为F/m，单位m/s^2，不是扣除重力后的净加速度；源码pi常量为3.1415926。')
    doc.equation(r'F=mc,\quad\Omega^*=\sqrt{mc/(4k_f)},\quad x=\operatorname{clip}\!\left(\frac{\Omega^*-\Omega_{\min}}{\Omega_{\max}-\Omega_{\min}},0,1\right)',
        '式(11)：先按物理质量和旋翼推力系数反算集体等推力参考速度，再映射到电机归一化区间。')
    doc.equation(r'u=\operatorname{clip}((1-f)x+fx^2,0,1),\quad \mathbf{t}_D=(0,0,-u)^T,\quad \mathbf{\omega}_{c,D}=D\mathbf{\omega}_c',
        '式(12)：f=THR_MDL_FAC=0；Omega_min=38、Omega_max=1000rad/s。PX4向上推力在机体FRD中指向负z。')
    doc.para('m=0.78kg，四电机，k_f=1.4131562456735816e-5。悬停比推力9.81m/s^2约映射到u=0.342955；最低转速产生的非零怠速推力按映射饱和记录。该式是集体参考，实际每个旋翼还承担姿态力矩，不能把CA_ROTOR*_CT替代Gazebo物理k_f。')
    doc.para('PX4角速度控制器接收网络w_c与集体推力u，分配器产生旋翼差速；位置和姿态外环不替策略修正目标位置。Offboard需要状态和控制模式双确认，非Offboard不发布网络CTBR；推理线程单任务无积压，控制周期零阶保持最近有效动作。')
    values=manifest['fixed_pid']
    doc.table(['固定公共配置','P','I','D'],[[axis]+[values['MC_'+key+'RATE_'+term] for term in ('P','I','D')]
        for axis,key in (('Roll','ROLL'),('Pitch','PITCH'),('Yaw','YAW'))])
    doc.para('该表仅供复现当前公共底层条件。本轮不分析或进行PID搜索，三个网络使用同一组值。所有接受的7500场均由ULog核对实际参数，而非只读默认airframe。网络权重SHA256见manifest与最终审计。')


def initialization(doc,gate_stats):
    doc.title('2.3 连续初态准备、释放与同步门槛')
    doc.para('直接在静止机体上突变速度会造成物理状态和EKF速度不同步。本流程将计分释放时刻设为t=0，先在[-T,0]执行一致位姿轨迹；T=0.35s。恒定世界速度与恒定机体角速度对应以下运动学式。')
    doc.equation(r'\mathbf{p}(t)=\mathbf{p}_0+t\mathbf{v}_0,\quad q(t)=q_0\otimes\exp_q(t\mathbf{\omega}_0/2),\quad -0.35\leq t\leq0',
        '式(13)：右乘体轴角增量；exp_q产生相应单位四元数。准备位置p(-T)=p0-T*v0是所选恒速轨迹的精确结果。')
    doc.para('先在p(-T),q(-T)处零自由速度保持，外部视觉与EKF通过2秒连续稳定、至少80样本检查；随后Offboard心跳、正常解锁，执行式(13)运动准备。仅受约束阶段为EV/IMU提供与强制轨迹导数一致的速度参考；释放后保留自由物理传感与状态，原始物理读数另存。')
    doc.equation(r'e_p=\|\hat{\mathbf{p}}-\mathbf{p}_0\|_2,\quad e_v=\|\hat{\mathbf{v}}-\mathbf{v}_0\|_2,\quad e_\omega=\|\hat{\mathbf{\omega}}-\mathbf{\omega}_0\|_2',
        '式(14)：当前位置/运动状态与请求状态的欧氏误差。')
    doc.equation(r'e_R=2\arccos\!\left(\min(1,|\langle\hat q,q_0\rangle|)\right),\quad (e_p,e_R,e_v,e_\omega)<(0.06,0.06,0.06,0.06)',
        '式(15)：四元数内积取绝对值消除q与-q等价；四个分量各自严格小于门槛，单位分别m/rad/m/s/rad/s。')
    doc.para('上述门槛同时作用于当前PX4估计与生成被采用动作的源观测；两者reset_counter相同，动作年龄和源观测年龄均<=0.1s。连续运动中至少3个匹配样本，释放请求距目标时刻remaining位于[-0.004,0.025]s；晚于目标0.04s则标记初始化失败。释放前动作已发布，第一自由物理步约4ms。')
    doc.table(['网络','接管位置最大m','姿态最大rad','速度最大m/s','角速度最大rad/s'],[[model]+[f"{row[k]:.6f}" for k in ('position_m','attitude_rad','velocity_m_s','rate_rad_s')] for model,row in gate_stats.items()],[1.1,1,1,1,1.2])
    doc.para('表中是所有接受试验的“当前观测”最大误差，动作源观测也逐场通过相同门槛。目标边界[-1,1]与v各轴[-1,1]倒推可到[-1.35,1.35]，只有准备位姿验证允许这个确定包络；释放请求的原始边界不放宽。')


def metrics(doc):
    doc.title('3.1 成功、精确悬停与稳定时间的数学定义')
    doc.equation(r'e_i(t)=\|\mathbf{p}_i(t)-\mathbf{p}^*\|_2,\quad v_i(t)=\|\mathbf{v}_i(t)\|_2,\quad B_i(t;\epsilon_p,\epsilon_v)=\mathbf{1}[e_i(t)<\epsilon_p\ \wedge\ v_i(t)<\epsilon_v]',
        '式(16)：位置误差与速度都用三维欧氏范数；不是逐轴门槛。')
    doc.equation(r'S_i(\epsilon_p,\epsilon_v)=A_i H_i\prod_{k\in\mathcal{K}_{[8,10]}} B_i(t_k;\epsilon_p,\epsilon_v)',
        '式(17)：A_i表示初始化、通信、模式和参数有效且控制流程正常结束；H_i表示完整末段覆盖。')
    doc.para('T=10s、末段[8,10]s固定不随提前结束移动。H要求最后样本>=9.975s、首样本不晚于8s、末段相邻样本最大间隔<=0.1s。包含8s边界前最近零阶保持样本；正常结束还须完成请求10秒。全体样本严格小于阈值。基本/精确/高精度阈值分别(0.3m,0.3m/s)、(0.1m,0.1m/s)、(0.05m,0.1m/s)。')
    doc.equation(r'\hat P_{\ell,m}=\frac{1}{500}\sum_{i\in\mathcal{I}_\ell}S_{i,m},\quad \hat P_m=\frac{1}{5}\sum_{\ell=1}^5\hat P_{\ell,m}',
        '式(18)：每网络每档500有效试验，整体按五档等权；目标静止重复是原计划的重复子组。')
    doc.equation(r't_i^{\mathrm{stable}}=\inf\{t:\ B_i(s;0.3,0.3)=1\ \forall s\in[t,10]\},\quad C_m(t)=\frac{1}{N}\sum_i S_{i,m}\mathbf{1}[t_i^{\mathrm{stable}}\leq t]',
        '式(19)：稳定进入时间按采样实现；失败没有伪造时间。C(t)始终使用全部有效槽位作为分母。')
    doc.para('首次连续0.3s满足基本阈值的确认时间另作诊断；复发不否决最终成功。L1初态本来在阈值内的稳定进入时间可能接近0，故增加排除L1的L2-L5速度分析，避免将保持任务混称为大偏差恢复。')
    doc.equation(r'\mathcal{C}=\{i:S_{i,\mathrm{CTBR}}=S_{i,\mathrm{DR}}=S_{i,\mathrm{ID}}=1\},\quad \Delta_{i,M}=M_{i,\mathrm{ID}}-M_{i,\mathrm{base}}',
        '式(20)：共同成功配对集及质量差值；Delta<0说明ID该质量指标更小。该条件集不替代总体成功率。')


def metric_details(doc):
    doc.title('3.2 误差、抖动、平顺性与统计量')
    doc.equation(r'w_k=|[t_k,t_{k+1})\cap[8,10]|,\quad\langle x\rangle_w=\frac{\sum_k w_kx_k}{\sum_k w_k},\quad E_p=\langle e\rangle_w,\quad E_{p,\mathrm{RMS}}=\sqrt{\langle e^2\rangle_w}',
        '式(21)：零阶保持时间权重，末样本只覆盖到已知结束时刻。采用时间权重避免采样抖动改变样本影响。')
    doc.equation(r'\bar{\mathbf{e}}_p=\langle\mathbf{p}-\mathbf{p}^*\rangle_w,\quad b_p=\|\bar{\mathbf{e}}_p\|_2,\quad J_x=\sqrt{\langle\|\mathbf{x}-\langle\mathbf{x}\rangle_w\|_2^2\rangle_w}',
        '式(22)：位置偏置范数b_p与位置误差均值E_p不同；x分别取世界线速度与机体角速度得到J_v和J_omega。')
    doc.equation(r'D_{a,j}=\sqrt{\frac{\sum_k \delta_k((a_{k+1,j}-a_{k,j})/(t_{k+1}-t_k))^2}{\sum_k\delta_k}},\quad TV_{a,j}=\frac{\sum_k\delta_k |\dot a_{k,j}|}{\sum_k\delta_k}',
        '式(23)：动作变化率RMS与每秒总变差，delta为有效相邻区间和分析窗口的交长；大于0.1s的间隔不用于导数。')
    doc.equation(r'\rho_{\mathrm{sat}}=\langle\mathbf{1}[\max_j|a_j|\geq0.98]\rangle,\quad IAE_p=\sum_k e_k\delta_k,\quad E_{p,\max}=\max_k e_k',
        '式(24)：任一动作边界时间比例、位置误差积分与峰值；四通道单位为归一化动作/s，不合并为能耗。')
    doc.para('角速度跟踪误差使用末段sqrt(<||w-w_cmd||^2>)，包含闭环和传输时序差异。提前越界的试验只有观察到的全程积分，不伪造剩余时间或完整末段；完整末段质量应与失败原因一起解释。谱功率诊断在50Hz重采样、去均值与Hann窗后计算，不作为成功门槛。')
    doc.equation(r'\overline{\Delta}_M=\frac{1}{|\mathcal{C}|}\sum_{i\in\mathcal{C}}\Delta_{i,M},\quad r_{\mathrm{lower}}=\frac{1}{|\mathcal{C}|}\sum_{i\in\mathcal{C}}\mathbf{1}[\Delta_{i,M}<0]',
        '式(25)：报告配对均值差、ID较小比例及固定seed=2026110404的2000次配对bootstrap描述性区间。')
    doc.para('Wilson95%区间用于描述二项成功比例。LHS各点是设计分层且互相关联，100次目标重复共用初态，多质量指标又来自成功条件集，因此这些区间和bootstrap仅描述当前设计数据，不作为任意状态总体或不同训练种子的显著性保证。报告另外去掉目标重复、去掉L1核对主优势。')
