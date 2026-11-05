"""Audited five-level recovery results and an ID-focused Chinese PDF report."""
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from xml.sax.saxutils import escape

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak

from result_io import save,digest,audit_cleanup_warning
from five_level_study import collect,admissible,MODELS,PRECISION_BANDS

COLORS={'CTBR':'#3973AC','CTBR+DR':'#D48927','CTBR+ID':'#23836F'}
QUALITY=('final_stabilization_entry_s','peak_position_error_m','whole_position_iae_m_s',
    'tail_position_error_mean_m','tail_position_bias_norm_m','tail_speed_rms_m_s',
    'tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s','tail_rate_tracking_error_rms_rad_s',
    'action_saturation_fraction','whole_outside_position_speed_band_fraction')
ACTION_KEYS=('whole_action_derivative_rms_per_s_xyz_thrust','tail_action_derivative_rms_per_s_xyz_thrust')


def describe(values):
    a=[float(v) for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(a),mean=statistics.mean(a) if a else None,median=statistics.median(a) if a else None,
        std=statistics.stdev(a) if len(a)>1 else None,p95=float(np.quantile(a,.95)) if a else None)


def band_pass(row,position,speed):
    m=row.get('metrics',{})
    return bool(row.get('success') and m.get('final_window_complete') and
        m.get('tail_position_max_m',math.inf)<position and m.get('tail_speed_max_m_s',math.inf)<speed)


def group_summary(rows):
    complete=[r for r in rows if r['metrics'].get('final_window_complete')]
    successful=[r for r in rows if r['success']]
    n=len(rows);k=len(successful)
    z=1.95996398454
    if n:
        p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        interval=[max(0,center-half),min(1,center+half)]
    else:interval=None
    return dict(n=n,successes=k,success_fraction=k/n if n else None,descriptive_wilson95=interval,
        precision={name:dict(successes=sum(band_pass(r,p,v) for r in rows),
            fraction=sum(band_pass(r,p,v) for r in rows)/n if n else None) for name,p,v in PRECISION_BANDS},
        quality_complete={key:describe([r['metrics'].get(key) for r in complete]) for key in QUALITY},
        quality_success={key:describe([r['metrics'].get(key) for r in successful]) for key in QUALITY},
        complete_terminal_n=len(complete),policy_failure_reasons=dict(Counter(r['termination_reason'] for r in rows if not r['success'])))


def flat_rows(rows):
    flat=[]
    for row in rows:
        x={k:row.get(k) for k in ('trial_id','model','state_id','success','failure_category','termination_reason',
            'initialization_valid','effective_pid_verified','accepted_attempt_number','infrastructure_makeups_before_acceptance','run_directory')}
        x.update(level=row['spec']['level'],category=row['spec'].get('category'))
        for k,v in row['metrics'].items():
            if isinstance(v,list):x.update({k+'_'+str(i):item for i,item in enumerate(v)})
            elif not isinstance(v,dict):x[k]=v
        for prefix,key in (('requested','requested_state'),('handover','actual_handover_observation'),('action_source','actual_action_input_observation')):
            x.update({prefix+'_'+str(i):v for i,v in enumerate(row.get(key) or [])})
        for key,p,v in PRECISION_BANDS:x[key+'_success']=band_pass(row,p,v)
        flat.append(x)
    return flat


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(rows)


def build_report(root):
    root=Path(root).resolve();manifest,rows=collect(root)
    out=root/'analysis';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    records={r['state_id']+':'+r['model']:r for r in rows}
    planned_states=manifest['state_bank']['states']
    groups={str(level):{m:group_summary([r for r in rows if r['spec']['level']==level and r['model']==m]) for m in MODELS}
        for level in range(1,6)}
    total={m:group_summary([r for r in rows if r['model']==m]) for m in MODELS}
    categories={category:{m:group_summary([r for r in rows if r['model']==m and r['spec'].get('category')==category]) for m in MODELS}
        for category in ('exact_goal','near_goal','combined_recovery')}
    common={};paired=[];heat=[];common_rows=[]
    for level in range(1,6):
        states=[s for s in planned_states if s['level']==level]
        shared=[s['id'] for s in states if all(s['id']+':'+m in records for m in MODELS)]
        wins=[s for s in shared if all(records[s+':'+m]['success'] for m in MODELS)]
        cg={m:[records[s+':'+m] for s in wins] for m in MODELS}
        common[str(level)]=dict(n=len(wins),state_ids=wins,quality={m:{
            key:describe([r['metrics'].get(key) for r in cg[m]]) for key in QUALITY} for m in MODELS})
        for m in MODELS:
            for key in QUALITY:
                common_rows.append(dict(level=level,model=m,metric=key,subset='all_three_success',**common[str(level)]['quality'][m][key]))
        for baseline in ('CTBR','CTBR+DR'):
            pair=[(records[s+':CTBR+ID'],records[s+':'+baseline]) for s in shared]
            paired.append(dict(level=level,baseline=baseline,n=len(pair),
                id_only_success=sum(a['success'] and not b['success'] for a,b in pair),
                baseline_only_success=sum(b['success'] and not a['success'] for a,b in pair),
                both_success=sum(a['success'] and b['success'] for a,b in pair),
                neither_success=sum(not a['success'] and not b['success'] for a,b in pair)))
    # First-attempt counts remain visible even though only valid attempts fill the requested slots.
    attempts=[]
    for path in sorted((root/'attempts/trials').glob('*/completed.json')):
        row=audit_cleanup_warning(json.loads(path.read_text()))[0]
        attempts.append(row)
    first=[r for r in attempts if r['spec'].get('attempt_number')==1]
    infrastructure=[r for r in attempts if not admissible(r)]
    attempt_summary={m:dict(first_attempts=sum(r['model']==m for r in first),
        first_attempt_successes=sum(r['model']==m and r['success'] for r in first),
        total_attempts=sum(r['model']==m for r in attempts),
        infrastructure_failures=dict(Counter(r.get('failure_category') or 'unverified' for r in infrastructure if r['model']==m))) for m in MODELS}
    for m in MODELS:
        own=[r for r in rows if r['model']==m]
        for p in (.03,.05,.10,.15,.20,.30):
            for v in (.05,.10,.15,.20,.30):
                heat.append(dict(model=m,position_threshold_m=p,speed_threshold_m_s=v,
                    passed=sum(band_pass(r,p,v) for r in own),n=len(own)))
    summary=dict(completed=len(rows),planned=len(manifest['schedule']),partial=len(rows)!=len(manifest['schedule']),
        protocol=manifest['protocol'],precision_bands=PRECISION_BANDS,by_level=groups,by_model=total,
        category=categories,common_success=common,paired_success=paired,attempts=attempt_summary,
        attempt_count=len(attempts),infrastructure_attempts=len(infrastructure),
        original_model_weight_hashes={m:manifest['numpy_validation'][m]['checkpoint_sha256'] for m in MODELS},
        primary_denominator='First valid policy attempt per original state/model slot, failures included',
        limitations='Fixed current shared controller; five-level benchmark is an equal-weight designed distribution. Goal repeats are repeated nominal state, not distinct states. Common-success quality is conditional. No policy failure is retried.')
    save(out/'summary.json',summary)
    write_csv(out/'trials.csv',flat_rows(rows))
    write_csv(out/'all_attempts.csv',flat_rows(attempts))
    write_csv(out/'common_success_quality.csv',common_rows)
    write_csv(out/'paired_success.csv',paired)
    write_csv(out/'precision_thresholds.csv',heat)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.2})
    def savefig(fig,name):
        fig.savefig(figs/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(figs/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    x=np.arange(5);width=.23
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    for ax,(band,_,_) in zip(axes,PRECISION_BANDS):
        for index,m in enumerate(MODELS):
            y=[100*(groups[str(level)][m]['precision'][band]['fraction'] or 0) for level in range(1,6)]
            ax.bar(x+(index-1)*width,y,width,color=COLORS[m],label=m)
        ax.set(xticks=x,xticklabels=['L1','L2','L3','L4','L5'],ylim=(0,105),title=band.replace('_',' '),ylabel='Success / valid slots (%)')
    axes[0].legend(fontsize=8);savefig(fig,'01_success')
    fig,axes=plt.subplots(1,5,figsize=(15,3.2),sharey=True,layout='constrained')
    for level,ax in enumerate(axes,1):
        for m in MODELS:
            own=[r for r in rows if r['model']==m and r['spec']['level']==level]
            ts=np.linspace(0,10,251)
            counts=[100*sum(r['success'] and r['metrics'].get('final_stabilization_entry_s',math.inf)<=t for r in own)/len(own) if own else 0 for t in ts]
            ax.plot(ts,counts,color=COLORS[m],label=m,lw=1.8)
        ax.set(title=f'L{level}',xlabel='Entry time (s)',ylim=(0,105))
    axes[0].set_ylabel('Entered & stays stable through 10s (%)');axes[0].legend(fontsize=7);savefig(fig,'02_stabilization')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,m in zip(axes,MODELS):
        own=[r for r in rows if r['model']==m];ps=[.03,.05,.10,.15,.20,.30];vs=[.05,.10,.15,.20,.30]
        grid=np.array([[100*sum(band_pass(r,p,v) for r in own)/len(own) if own else 0 for v in vs] for p in ps])
        im=ax.imshow(grid,vmin=0,vmax=100,cmap='YlGnBu',origin='lower',aspect='auto')
        for i in range(len(ps)):
            for j in range(len(vs)):ax.text(j,i,f'{grid[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if grid[i,j]>55 else 'black')
        ax.set(title=m,xticks=range(len(vs)),xticklabels=vs,yticks=range(len(ps)),yticklabels=ps,xlabel='Speed tolerance (m/s)',ylabel='Position tolerance (m)')
    fig.colorbar(im,ax=axes,label='All-level valid-slot success (%)');savefig(fig,'03_precision')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    metrics=('tail_position_error_mean_m','tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s')
    for ax,key in zip(axes,metrics):
        for m in MODELS:
            y=[common[str(level)]['quality'][m][key]['median'] for level in range(1,6)]
            ax.plot(range(1,6),[v if v is not None else np.nan for v in y],'o-',color=COLORS[m],label=m)
        ax.set(title=key.replace('tail_','').replace('_',' '),xticks=range(1,6),xlabel='Level')
    axes[0].legend(fontsize=8);savefig(fig,'04_matched_quality')
    # Simple hover is a separate diagnostic population, including failures with complete windows.
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,key in zip(axes,metrics):
        vectors=[[r['metrics'][key] for r in rows if r['model']==m and r['spec'].get('category')=='exact_goal'
            and r['metrics'].get(key) is not None] for m in MODELS]
        if all(vectors):
            boxes=ax.boxplot(vectors,labels=MODELS,patch_artist=True)
            for box,m in zip(boxes['boxes'],MODELS):box.set_facecolor(COLORS[m]);box.set_alpha(.55)
        ax.set_title(key.replace('tail_','').replace('_',' '))
    fig.suptitle('Exact goal repeats: all complete terminal windows, including failures');savefig(fig,'05_goal_hold')
    # Chinese report: ID-led narrative, with fully visible unfavorable outcomes.
    pdfmetrics.registerFont(TTFont('StudyCJK','/usr/share/fonts/truetype/arphic/uming.ttc',subfontIndex=0))
    style=ParagraphStyle('body',fontName='StudyCJK',fontSize=9,leading=15,spaceAfter=9,wordWrap='CJK')
    heading=ParagraphStyle('head',parent=style,fontSize=16,leading=23,spaceAfter=14)
    cell=ParagraphStyle('cell',parent=style,fontSize=7.5,leading=11)
    story=[];md=[]
    def title(text):
        if story:story.append(PageBreak())
        story.append(Paragraph(escape(text),heading));md.extend(['# '+text,''])
    def para(text):story.append(Paragraph(escape(text),style));md.extend([text,''])
    def table(headers,values):
        md.extend(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|'])
        md.extend('|'+'|'.join(map(str,row))+'|' for row in values);md.append('')
        t=Table([[Paragraph(escape(str(v)),cell) for v in row] for row in [headers]+values],
            colWidths=[511/len(headers)]*len(headers),repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EFF3')),
            ('GRID',(0,0),(-1,-1),.25,colors.HexColor('#CAD4DD')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,9)])
    def figure(name,caption):
        image=Image(str(figs/(name+'.png')));scale=505/image.imageWidth;image.drawWidth=505;image.drawHeight*=scale
        story.extend([image,Paragraph(escape(caption),style)]);md.extend([f'![{caption}](figures/{name}.png)','',caption,''])
    def fmt(v):return '-' if v is None else f'{v:.4f}'
    title('ID方法五档初态恢复与悬停控制实验')
    para(f"状态：{'已完成' if not summary['partial'] else '进行中'}；有效试验 {len(rows)}/{summary['planned']}。每网络每档目标{manifest['state_bank']['states_per_level']}次，初态共用；全部使用当前同一公共角速度控制配置、Prototype模型、既有权重与推力映射。本报告重点评价ID的控制优势，所有网络结果和失败同时保留。")
    table(['网络','成功/有效试验','基本成功率','精确悬停率','高精度悬停率'],[[m,
        f"{total[m]['successes']}/{total[m]['n']}",fmt(total[m]['success_fraction']),
        fmt(total[m]['precision']['precise']['fraction']),fmt(total[m]['precision']['high_precision']['fraction'])] for m in MODELS])
    if all(total[m]['n'] for m in MODELS):
        order=sorted(MODELS,key=lambda m:total[m]['success_fraction'],reverse=True)
        para('本轮实际总体排序：'+' > '.join(order)+'。ID相对于CTBR、DR的基本成功率差分别为'+
            f"{100*(total['CTBR+ID']['success_fraction']-total['CTBR']['success_fraction']):.2f}、{100*(total['CTBR+ID']['success_fraction']-total['CTBR+DR']['success_fraction']):.2f}个百分点。")
    para('主成功定义：完整10仿真秒，固定末2秒位置误差始终<0.3m、速度始终<0.3m/s；此前复发允许恢复。补充精度等级为0.1m/0.1m/s与0.05m/0.1m/s，均在实验前确定。')
    title('1. 五档设计与冻结的公共条件')
    table(['档位','位置各轴m','机身倾角deg','yaw范围deg','速度/角速度各轴'],[
        ['L'+str(c['level']),str(c['position']),str(c['tilt_deg']),str(c['yaw_deg']),str(c['velocity'])+' / '+str(c['rate'])]
        for c in manifest['state_bank']['levels']])
    para('目标为(0,0,5m)。L1中20%为同一目标静止状态独立重复，其余为附近小扰动；正式列表为100次目标重复+400次小扰动。其余每档500组。位置存绝对高度，线速度为NWU世界系、角速度为FLU机体系。物理倾角按cos(theta)在档内分层，ZYX yaw单独采样，不使用欧拉角范数代替倾角。')
    para('每档12维随机拉丁超立方采样；随机种子'+str(manifest['state_bank']['seed'])+'。每五个状态覆盖五档且顺序打乱，三网络顺序按六种排列轮换。全部请求初态在运行前生成，不根据网络结果修改或替换。初始化误差门槛位置/姿态/速度/角速度均0.06，动作与输入新鲜度0.1秒。')
    para('公共条件仅作为实验控制变量：50Hz网络控制、0.35秒连续准备、4ms物理步长、10秒自由控制。当前PID固定，全程从ULog核验，不在此轮调参。每场独立SITL，实例间参数和网络隔离。')
    title('2. 成功率随组合难度变化')
    figure('01_success','图1：五档基本/精确/高精度成功率；分母为每档第一次有效策略试验。')
    table(['档位','CTBR成功/次数','DR成功/次数','ID成功/次数'],[
        ['L'+str(level)]+[f"{groups[str(level)][m]['successes']}/{groups[str(level)][m]['n']}" for m in MODELS] for level in range(1,6)])
    para('五档是位置、姿态与运动状态共同增加的组合难度；不能把档间变化仅归因于倾角。总体为五档等权基准的结果；不与旧实验混合。目标点重复与小扰动另外单列，避免把重复稳定性等同于不同初态覆盖能力。')
    title('3. 精细悬停与进入稳定的速度')
    figure('02_stabilization','图2：截至某时刻已进入并持续保持至10秒的比例；分母包含所有有效策略失败。')
    figure('03_precision','图3：位置-速度容差网格的末段成功比例；所有门槛同时展示，不改变主判定。')
    para('最终稳定进入时间为最后一次离开基本阈值后首次持续在阈值内直到结束的时刻。L1初始已在阈值内时可能接近零，因此L1侧重保持能力和末段质量，不把其接近零的时间混称为困难状态的恢复速度。')
    title('4. 同初态配对的控制精度与抖动')
    figure('04_matched_quality','图4：每档三网络共同成功初态的末段质量中位数。每档样本数见表；失败不补造末段数据。')
    table(['档位','共同成功n','CTBR位置误差m','DR位置误差m','ID位置误差m'],[
        ['L'+str(level),str(common[str(level)]['n'])]+[fmt(common[str(level)]['quality'][m]['tail_position_error_mean_m']['mean']) for m in MODELS] for level in range(1,6)])
    para('质量指标包含位置误差、偏置、速度RMS、去均值速度/角速度抖动、指令变化率和饱和比例。同初态共同成功子集用于公平配对，同时all_trials.csv/summary保留完整末段含失败与各自成功子集，避免只凭成功样本评价总体可靠性。共同成功是条件子集，不能替代全部初态成功率。')
    title('5. 目标点保持：网络是否引入偏差与振荡')
    figure('05_goal_hold','图5：精确目标点重复试验，包含具有完整末段的未稳定失败；用于区分保持精度与动态抖动。')
    table(['组别','网络','成功/有效次数','末段位置误差均值m','离开基本稳定区时间比例'],[
        [category,m,f"{categories[category][m]['successes']}/{categories[category][m]['n']}",
            fmt(categories[category][m]['quality_complete']['tail_position_error_mean_m']['mean']),
            fmt(categories[category][m]['quality_complete']['whole_outside_position_speed_band_fraction']['mean'])]
        for category in ('exact_goal','near_goal') for m in MODELS])
    para('同一目标状态的重复主要评价噪声、估计与运行时序下的重复可靠性；不能当作100个独立初态。稳态误差与抖动分别统计，动作变化率按四个归一化通道分别输出，不把混合单位合成能耗。')
    title('6. 补做、失败与初始化审计')
    table(['网络','首次尝试成功/次数','全部尝试','基础设施失败分类'],[[m,
        f"{attempt_summary[m]['first_attempt_successes']}/{attempt_summary[m]['first_attempts']}",
        str(attempt_summary[m]['total_attempts']),str(attempt_summary[m]['infrastructure_failures'])] for m in MODELS])
    para('初始化/通信/模式/参数核验失效在同一请求初态重新独立启动；原始尝试目录、日志和失败全部保留。每个状态-网络取第一次有效策略试验，恢复失败不重试。主结果是基础设施有效条件下的策略成功率；首次尝试全流程成功率作为补充，补做开销与失败单列。发生持续无法初始化的状态不会替换为更容易样本。')
    for level in range(1,6):
        para('L'+str(level)+'策略失败原因：'+str({m:groups[str(level)][m]['policy_failure_reasons'] for m in MODELS}))
    title('7. ID方法的证据与结论范围')
    para('方法比较从三个层面展开：恢复范围以全有效初态成功率衡量；恢复效率以稳定进入比例曲线衡量；精细控制以位置误差、抖动和指令变化率衡量。只有实际数据领先的指标写入ID优势，动作饱和、峰值偏移等取舍完整保留。现有控制配置与权重的结果不等价于训练机制在任意平台、任意参数下的普遍最优。')
    para('每档500组的比例区间为描述性Wilson95%；分层设计总体按五档等权解释。目标重复具有共同初态，属于重复运行而非不同状态。共同成功质量存在条件化；各网有效尝试的隐藏噪声、积分器与线程时序并非逐值相同。初态生成与评分冻结，pilot仅用于工作流门槛，不以网络排序决定是否开展正式试验。')
    para('已保存trials.csv、all_attempts.csv、common_success_quality.csv、paired_success.csv、precision_thresholds.csv和summary.json。每次试验原始ULog、ROS记录、控制trace、请求/实际接管初态与参数核验保留。报告生成只读原始结果，不改变成功判定。')
    (out/'FIVE_LEVEL_REPORT.md').write_text('\n'.join(md),encoding='utf-8')
    def footer(canvas,doc):
        canvas.setFont('StudyCJK',7);canvas.drawString(42,22,'FlightLxx | Five-level recovery study | ID / DR / CTBR');canvas.drawRightString(553,22,str(doc.page))
    SimpleDocTemplate(str(out/'FIVE_LEVEL_REPORT.pdf'),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=39,bottomMargin=40,
        title='ID方法五档初态控制实验报告').build(story,onFirstPage=footer,onLaterPages=footer)
    save(out/'artifact_manifest.json',dict(input_manifest_sha256=digest(root/'manifest.json'),
        completed_result_sha256={r['trial_id']:digest(root/'trials'/r['trial_id']/'completed.json') for r in rows},
        outputs_sha256={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}))
    print('REPORT',out/'FIVE_LEVEL_REPORT.pdf',flush=True)
    return summary
