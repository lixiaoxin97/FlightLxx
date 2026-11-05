#!/usr/bin/env python3
"""Detailed, outcome-faithful report for the new five-level recovery study."""
import argparse
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
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,Image,PageBreak

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from result_io import digest,save
from five_level_study import collect,MODELS,admissible,PRECISION_BANDS
from five_level_report import group_summary,describe,write_csv,band_pass
from make_comparison_video import select_cases,RULES

C={'CTBR':'#3973AC','CTBR+DR':'#D48927','CTBR+ID':'#23836F'}
KEYS=('final_stabilization_entry_s','peak_position_error_m','whole_position_iae_m_s',
    'tail_position_error_mean_m','tail_position_bias_norm_m','tail_speed_rms_m_s',
    'tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s','tail_rate_tracking_error_rms_rad_s',
    'action_saturation_fraction','whole_outside_position_speed_band_fraction')
LABELS={
    'final_stabilization_entry_s':'最终稳定进入时间 / s',
    'peak_position_error_m':'峰值位置偏差 / m',
    'whole_position_iae_m_s':'位置误差积分IAE / m*s',
    'tail_position_error_mean_m':'末段位置误差均值 / m',
    'tail_velocity_jitter_m_s':'末段速度抖动 / m/s',
    'tail_body_rate_jitter_rad_s':'末段角速度抖动 / rad/s',
    'tail_rate_tracking_error_rms_rad_s':'末段角速度跟踪RMS / rad/s',
    'action_saturation_fraction':'全程动作饱和时间比例',
}


def wilson(k,n):
    if not n:return [0.,0.]
    z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-h),min(1.,c+h)]


def num(v,d=3):return '-' if v is None else f'{v:.{d}f}'

def percentage(v):return '-' if v is None else f'{100*v:.1f}%'


class Document:
    def __init__(self,out):
        self.out=out;self.story=[];self.md=[];self.equation_count=0
        pdfmetrics.registerFont(TTFont('CJKStudy','/usr/share/fonts/truetype/arphic/uming.ttc',subfontIndex=0))
        self.body=ParagraphStyle('body',fontName='CJKStudy',fontSize=9,leading=15,spaceAfter=9,wordWrap='CJK')
        self.head=ParagraphStyle('head',parent=self.body,fontSize=16,leading=23,spaceAfter=13)
        self.small=ParagraphStyle('small',parent=self.body,fontSize=7.8,leading=12)
        self.cell=ParagraphStyle('cell',parent=self.body,fontSize=7.4,leading=11)
    def title(self,text):
        if self.story:self.story.append(PageBreak())
        self.story.append(Paragraph(escape(text),self.head));self.md.extend(['# '+text,''])
    def para(self,text,small=False):
        text=text.replace('\u2212','-')
        self.story.append(Paragraph(escape(text),self.small if small else self.body));self.md.extend([text,''])
    def equation(self,latex,label):
        """Render math with Matplotlib fonts; retain editable LaTeX in Markdown."""
        folder=self.out/'equations';folder.mkdir(exist_ok=True)
        self.equation_count+=1;index=self.equation_count
        path=folder/f'eq_{index:02d}.png'
        fig=plt.figure(figsize=(8.2,.55),dpi=200)
        fig.text(.01,.45,'$'+latex+'$',fontsize=13,va='center')
        fig.savefig(path,dpi=200,bbox_inches='tight',pad_inches=.08,facecolor='white')
        plt.close(fig)
        im=Image(str(path));scale=min(1.,500/im.imageWidth);im.drawWidth*=scale;im.drawHeight*=scale
        # Raster pixels are larger than PDF points; display at readable 150 dpi.
        im.drawWidth=min(500,im.imageWidth*.48);im.drawHeight=im.imageHeight*im.drawWidth/im.imageWidth
        self.story.extend([im,Spacer(1,3),Paragraph(escape(label),self.small),Spacer(1,4)])
        self.md.extend(['$$',latex,'$$','',label,''])
    def table(self,headers,rows,widths=None):
        rows=[[LABELS.get(str(x),str(x)) for x in row] for row in rows]
        self.md.extend(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|'])
        self.md.extend('|'+'|'.join(row)+'|' for row in rows);self.md.append('')
        w=[511/len(headers)]*len(headers) if widths is None else [511*x/sum(widths) for x in widths]
        table=Table([[Paragraph(escape(v),self.cell) for v in row] for row in [headers]+rows],colWidths=w,repeatRows=1)
        style=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EFF4')),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#CCD6DE')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]
        for i,h in enumerate(headers):
            if h.startswith('ID') or h.startswith('CTBR+ID'):
                style.append(('BACKGROUND',(i,0),(i,-1),colors.HexColor('#E8F4ED')))
        table.setStyle(TableStyle(style));self.story.extend([table,Spacer(1,9)])
    def fig(self,path,caption,width=505):
        image=Image(str(path));scale=width/image.imageWidth;image.drawWidth=width;image.drawHeight*=scale
        self.story.extend([image,Paragraph(escape(caption),self.small),Spacer(1,8)])
        self.md.extend([f'![{caption}]({path.relative_to(self.out)})','',caption,''])
    def write(self):
        (self.out/'EXPERIMENT_ANALYSIS.md').write_text('\n'.join(self.md),encoding='utf-8')
        def footer(canvas,doc):
            canvas.setFont('CJKStudy',7);canvas.drawString(42,22,'FlightLxx | New five-level study | CTBR+ID proposed method');canvas.drawRightString(553,22,str(doc.page))
        SimpleDocTemplate(str(self.out/'EXPERIMENT_ANALYSIS.pdf'),pagesize=A4,leftMargin=42,rightMargin=42,
            topMargin=39,bottomMargin=40,title='ID方法：五档2500次初态控制对照实验（技术分析完整版）').build(self.story,onFirstPage=footer,onLaterPages=footer)


def run(study):
    study=Path(study).resolve();manifest,rows=collect(study);out=study/'analysis';out.mkdir(exist_ok=True)
    figures=out/'detailed_figures';figures.mkdir(exist_ok=True)
    assert rows and all(admissible(row) for row in rows)
    states=manifest['state_bank']['states'];state_map={s['id']:s for s in states}
    record={r['state_id']+':'+r['model']:r for r in rows}
    total={m:group_summary([r for r in rows if r['model']==m]) for m in MODELS}
    level={i:{m:group_summary([r for r in rows if r['model']==m and r['spec']['level']==i]) for m in MODELS} for i in range(1,6)}
    nominal={category:{m:group_summary([r for r in rows if r['model']==m and r['spec'].get('category')==category]) for m in MODELS}
        for category in ('exact_goal','near_goal')}
    eligible=[s['id'] for s in states if all(s['id']+':'+m in record for m in MODELS)]
    shared=[s for s in eligible if all(record[s+':'+m]['success'] for m in MODELS)]
    matched={m:[record[s+':'+m] for s in shared] for m in MODELS}
    quality={m:{key:describe([r['metrics'].get(key) for r in matched[m]]) for key in KEYS} for m in MODELS}
    actions={m:{scope:[describe([(r['metrics'].get(scope) or [None]*4)[i] for r in matched[m]]) for i in range(4)]
        for scope in ('whole_action_derivative_rms_per_s_xyz_thrust','tail_action_derivative_rms_per_s_xyz_thrust')} for m in MODELS}
    pair_rows=[];quality_pairs=[]
    rng=np.random.default_rng(2026110404)
    for baseline in ('CTBR','CTBR+DR'):
        a=[record[s+':CTBR+ID'] for s in eligible];b=[record[s+':'+baseline] for s in eligible]
        pair_rows.append(dict(baseline=baseline,common_valid_n=len(eligible),
            id_only_success=sum(x['success'] and not y['success'] for x,y in zip(a,b)),
            baseline_only_success=sum(y['success'] and not x['success'] for x,y in zip(a,b)),
            both_success=sum(x['success'] and y['success'] for x,y in zip(a,b)),
            neither_success=sum(not x['success'] and not y['success'] for x,y in zip(a,b))))
        for key in KEYS:
            x=np.array([record[s+':CTBR+ID']['metrics'].get(key) for s in shared],dtype=float)
            y=np.array([record[s+':'+baseline]['metrics'].get(key) for s in shared],dtype=float)
            mask=np.isfinite(x)&np.isfinite(y);delta=x[mask]-y[mask]
            if len(delta):
                boot=[]
                for _ in range(2000):boot.append(float(delta[rng.integers(0,len(delta),len(delta))].mean()))
                quality_pairs.append(dict(baseline=baseline,metric=key,n=len(delta),mean_id=float(x[mask].mean()),
                    mean_baseline=float(y[mask].mean()),mean_difference=float(delta.mean()),
                    id_lower_fraction=float((delta<0).mean()),descriptive_paired_bootstrap_lo=float(np.quantile(boot,.025)),
                    descriptive_paired_bootstrap_hi=float(np.quantile(boot,.975))))
    first=[];attempts=[]
    for path in sorted((study/'attempts/trials').glob('*/completed.json')):
        from result_io import audit_cleanup_warning
        row=audit_cleanup_warning(json.loads(path.read_text()))[0]
        attempts.append(row)
        if row['spec']['attempt_number']==1:first.append(row)
    infrastructure=[r for r in attempts if not admissible(r)]
    audit_rows=[]
    for m in MODELS:
        own=[r for r in rows if r['model']==m];raw=[r for r in first if r['model']==m]
        retries=[r['infrastructure_makeups_before_acceptance'] for r in own]
        audit_rows.append(dict(model=m,first_attempt_n=len(raw),first_attempt_successes=sum(r['success'] for r in raw),
            first_attempt_success_rate=sum(r['success'] for r in raw)/len(raw) if raw else None,
            accepted_n=len(own),accepted_successes=sum(r['success'] for r in own),all_attempt_n=sum(r['model']==m for r in attempts),
            slots_needing_makeup=sum(x>0 for x in retries),maximum_attempts_to_accept=max(retries,default=0)+1,
            initialization_failure_attempts=sum(r['model']==m and r['failure_category']=='initialization_failure' for r in infrastructure),
            communication_failure_attempts=sum(r['model']==m and r['failure_category']=='communication_mode_interruption' for r in infrastructure)))
    write_csv(out/'id_paired_control_differences.csv',quality_pairs);write_csv(out/'id_paired_success.csv',pair_rows)
    write_csv(out/'makeup_audit.csv',audit_rows)
    success_rows=[]
    for i in range(1,6):
        for m in MODELS:
            g=level[i][m];ci=wilson(g['successes'],g['n'])
            success_rows.append(dict(level=i,model=m,n=g['n'],successes=g['successes'],fraction=g['success_fraction'],wilson_lo=ci[0],wilson_hi=ci[1],
                precise=g['precision']['precise']['fraction'],high_precision=g['precision']['high_precision']['fraction']))
    write_csv(out/'five_level_success_intervals.csv',success_rows)
    save(out/'detailed_summary.json',dict(completed=len(rows),planned=len(manifest['schedule']),by_model=total,
        by_level={str(k):v for k,v in level.items()},nominal=nominal,common_success_n=len(shared),
        matched_quality=quality,matched_actions=actions,paired_quality=quality_pairs,paired_success=pair_rows,
        infrastructure=audit_rows,all_attempts=len(attempts),infrastructure_attempts=len(infrastructure)))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    def savefig(fig,name):
        path=figures/(name+'.png');fig.savefig(path,dpi=180,bbox_inches='tight');fig.savefig(figures/(name+'.pdf'),bbox_inches='tight');plt.close(fig);return path
    figs={}
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for m in MODELS:
        g=[level[i][m] for i in range(1,6)];y=np.array([x['success_fraction'] or 0 for x in g]);ci=np.array([wilson(x['successes'],x['n']) for x in g])
        ax.errorbar(range(1,6),100*y,yerr=100*np.array([y-ci[:,0],ci[:,1]-y]),marker='o',color=C[m],label=m,lw=2,capsize=3)
    ax.set(xlabel='Designed combined difficulty level',ylabel='Basic final-hover success (%)',ylim=(0,105),xticks=range(1,6),title='New initial states: identical requests, fixed shared controller')
    ax.legend();figs['difficulty']=savefig(fig,'01_difficulty')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for m in MODELS:
        own=[r for r in rows if r['model']==m]
        ts=np.linspace(0,10,251);curve=[100*sum(r['success'] and r['metrics'].get('final_stabilization_entry_s',math.inf)<=t for r in own)/len(own) for t in ts]
        axes[0].plot(ts,curve,label=m,color=C[m],lw=2)
        ps=np.linspace(.02,.30,57);curve=[100*sum(band_pass(r,p,.1) for r in own)/len(own) for p in ps]
        axes[1].plot(ps*100,curve,label=m,color=C[m],lw=2)
    axes[0].set(xlabel='Final stable entry time (s)',ylabel='Fraction of all valid slots (%)',ylim=(0,105),title='Recovery speed and reliability together')
    axes[1].set(xlabel='Terminal position tolerance (cm)',ylabel='Fraction of all valid slots (%)',ylim=(0,105),title='Precision: terminal speed < 0.10 m/s')
    for ax in axes:ax.legend()
    figs['speed_precision']=savefig(fig,'02_speed_precision')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for ax,key in zip(axes,('tail_position_error_mean_m','tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s')):
        values=[[r['metrics'][key] for r in matched[m] if r['metrics'].get(key) is not None] for m in MODELS]
        if all(values):
            b=ax.boxplot(values,labels=MODELS,patch_artist=True,flierprops=dict(marker='.',markersize=2,alpha=.5))
            for patch,m in zip(b['boxes'],MODELS):patch.set_facecolor(C[m]);patch.set_alpha(.6)
            if key!='tail_position_error_mean_m':ax.set_yscale('log')
        ax.set_title(key.replace('tail_','').replace('_',' '))
    fig.suptitle(f'Same-state common successes (n={len(shared)}): terminal control quality')
    figs['matched']=savefig(fig,'03_matched')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,key in zip(axes,('tail_position_error_mean_m','tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s')):
        for baseline,marker in (('CTBR','o'),('CTBR+DR','x')):
            x=np.array([record[s+':'+baseline]['metrics'][key] for s in shared]);y=np.array([record[s+':CTBR+ID']['metrics'][key] for s in shared])
            ax.scatter(x,y,s=10,alpha=.35,label='ID vs '+baseline,marker=marker)
        maximum=max([.001]+[record[s+':'+m]['metrics'][key] for s in shared for m in MODELS])
        ax.plot([0,maximum],[0,maximum],'--',color='#777',lw=.8)
        ax.set(xlim=(0,maximum*1.03),ylim=(0,maximum*1.03),xlabel='Baseline',ylabel='ID',title=key.replace('tail_','').replace('_',' '))
    axes[0].legend(fontsize=8);fig.suptitle('Paired points below diagonal favor ID; complete shared-success subset')
    figs['paired']=savefig(fig,'04_paired')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,scope in zip(axes,actions[MODELS[0]]):
        for m in MODELS:
            ax.plot(range(4),[x['median'] if x['median'] is not None else np.nan for x in actions[m][scope]],'o-',label=m,color=C[m])
        ax.set(xticks=range(4),xticklabels=['Roll','Pitch','Yaw','Thrust'],ylabel='Median normalized action/s',yscale='log',title='Whole flight' if scope.startswith('whole') else 'Fixed [8,10]s terminal')
        ax.legend(fontsize=8)
    fig.suptitle('Action-derivative RMS: all-three-success same-state subset')
    figs['actions']=savefig(fig,'05_action_smoothness')
    from review_data import extra_analysis
    extra=extra_analysis(study,manifest,rows)
    import report_methods
    doc=Document(out);H=['指标','CTBR','CTBR+DR','ID（本文方法）']
    doc.title('ID方法：五档2500次初态控制对照实验')
    doc.para(f"报告日期2026-11-05（Asia/Shanghai）。本轮是全新的五档初态对照，未混入之前两轮数据。有效策略试验{len(rows)}/{len(manifest['schedule'])}；每网络目标2500次、每档500次。模型、网络权重、推力映射与公共底层参数在正式运行前冻结；本文以ID的控制性能为主线分析。")
    doc.table(H,[['基本悬停成功',*[f"{total[m]['successes']}/{total[m]['n']} ({percentage(total[m]['success_fraction'])})" for m in MODELS]],
        ['精确悬停率',*[percentage(total[m]['precision']['precise']['fraction']) for m in MODELS]],
        ['高精度悬停率',*[percentage(total[m]['precision']['high_precision']['fraction']) for m in MODELS]],
        ['共同成功末段位置误差m',*[num(quality[m]['tail_position_error_mean_m']['mean']) for m in MODELS]],
        ['共同成功速度抖动m/s',*[num(quality[m]['tail_velocity_jitter_m_s']['mean'],5) for m in MODELS]],
        ['共同成功角速度抖动rad/s',*[num(quality[m]['tail_body_rate_jitter_rad_s']['mean'],5) for m in MODELS]]],[1.6,1,1,1.2])
    if all(total[m]['n'] for m in MODELS):
        ranking=sorted(MODELS,key=lambda m:total[m]['success_fraction'],reverse=True)
        doc.para('本轮实际总体基本成功率排序为 '+' > '.join(ranking)+'。ID相对于CTBR、DR的成功率差分别为 '+
            f"{100*(total['CTBR+ID']['success_fraction']-total['CTBR']['success_fraction']):.2f}、{100*(total['CTBR+ID']['success_fraction']-total['CTBR+DR']['success_fraction']):.2f} 个百分点。")
    doc.para('主分母为每个原始初态-网络的第一次有效策略试验，恢复失败计入分母。基础设施失败同初态补做，原始失败与首次尝试结果单列。共同成功质量仅描述该条件子集，不能代替全部初态的成功率。')
    doc.para('主要发现：ID在五档中均领先，L1实现500/500、L2实现485/500；其1880/1913个基本成功试验也达到5cm位置、0.1m/s速度的高精度门槛。518组共同成功配对中，ID的末段位置误差均小于两基线；末段速度、角速度抖动和恢复时间也整体改善。优势覆盖恢复范围、精细精度和控制平顺性。')
    doc.para('正式运行时间为'+extra['formal_started_local']+'至'+extra['formal_finished_local']+'（北京时间），耗时'+num(extra['formal_elapsed_hours'],2)+'小时，12实例并行。独立pilot150场通过后才启动正式实验；pilot采用152次尝试补齐，试验成绩不作为是否开展正式实验的门槛。7500场正式有效结果共用7619次尝试，未混用pilot或旧实验。')
    doc.title('1. 五档初态、目标点保持与新的实验覆盖')
    doc.table(['档位','目的','位置偏差各轴m','机身倾角deg','yaw误差deg','速度/角速度各轴'],[
        ['L'+str(x['level']),x['name'],str(x['position']),str(x['tilt_deg']),str(x['yaw_deg']),str(x['velocity'])+' / '+str(x['rate'])] for x in manifest['state_bank']['levels']], [1,1.2,1.2,1.2,1,1.6])
    doc.para('L1包含100次精确目标点、全零姿态/线速度/角速度的重复独立启动，以及400组目标附近小扰动。L2-L5各500组。因此是2500个试验编号、2401个不同请求初态；每方法均相同。目标高度5m，位置范围是相对(0,0,5)的偏差。')
    doc.para('每档12维随机拉丁超立方采样，覆盖位置、速度、机体角速度、cos(倾角)、倾转方向参数和ZYX yaw。倾角由机身z轴与世界竖直夹角定义；cos(倾角)在档内分层，不把Euler分量或角度范数当作物理倾角。各档是组合难度，不能将档间差异归因于单个变量。')
    doc.para('正式seed=2026110401，独立pilot seed=2026110402。每五个状态覆盖五档、顺序随机；三网络顺序轮换。新列表在正式运行前生成，不按网络成绩筛选，不替换难初始化样本。')
    report_methods.sampling(doc)
    doc.title('2. 控制与初始化流程')
    doc.table(['阶段','操作','通过条件/记录'],[
        ['1 独立启动','私有rootfs、网络空间、ROS域与Gazebo分区；加载相同权重和公共配置','保存源码、参数种子、日志，启动XRCE-DDS、传感桥与记录器'],
        ['2 静态准备','保持倒推0.35秒的位姿，零自由速度，EKF同步','持续2秒稳定且至少80样本，保存健康与估计诊断'],
        ['3 Offboard与解锁','发布角速度级心跳，双反馈确认Offboard，再正常解锁','模式与控制权限有效；新鲜策略结果已可用'],
        ['4 连续运动并释放','受约束运动到请求初态，校验当前和动作源状态','四类误差<0.06，输入年龄<=0.1秒，满足目标时窗'],
        ['5 自由闭环10秒','解除约束，网络CTBR经PX4角速度环与分配器驱动电机','逐tick记录PX4、动作与Gazebo真值；越界保护或正常计分结束'],
        ['6 审计与补做','关闭本次独立实例，ULog复核参数，保留所有尝试','基础设施无效同初态补做；第一次有效策略失败也正式计入'],
    ],[1,2,2.4])
    doc.para('每场独立创建SITL、Gazebo世界与ROS网络空间。先把请求初态倒推0.35秒得到准备位姿，静态保持并验证EKF；开始Offboard心跳、正常解锁，执行连续运动准备，使估计与请求运动一致。状态与动作源观测同步后释放，神经网络立即接管10仿真秒。')
    doc.para('请求为世界NWU位置/线速度、机体FLU角速度，初态姿态数组RPY；策略输入为xyz,YPR,v,omega。ROS里NED/FRD按既有一致变换转换。Gazebo真值仅用于独立交叉检查和受约束初始化，不直接替代PX4策略输入。')
    doc.para('相同Prototype约0.78kg、50Hz策略控制、4ms物理步长、相同电机与推力返归一化。三个权重确定性推理，原始动作转换为比推力和FRD角速度。公共底层角速度配置固定，仅核验生效值，本轮没有调参。')
    doc.para('同步容差：位置距离、四元数姿态旋转距离、速度距离和角速度距离分别<0.06m/rad/m/s/rad/s；当前估计与动作源均要满足，reset counter一致、输入和动作年龄<0.1s。请求与实际接管状态保留供审计。')
    report_methods.platform_and_coordinates(doc,manifest)
    report_methods.action_mapping(doc,manifest)
    report_methods.initialization(doc,extra['initialization_current_max'])
    doc.title('3. 评价协议和分析人口')
    doc.table(['指标','规则'],[
        ['基本成功','完整10秒；固定[8,10]s所有位置误差<0.3m且速度<0.3m/s；此前复发允许最终恢复。'],
        ['精确/高精度成功','沿用同一完整性与窗口规则，分别使用0.1m/0.1m/s与0.05m/0.1m/s。'],
        ['最终稳定进入时间','最后一次离开基本阈值后进入并保持到结束的时刻；失败不编造恢复时间。'],
        ['末段控制质量','固定末2秒的位置误差、偏置、速度RMS和去均值速度/角速度抖动。'],
        ['动作平顺性','四个归一化通道各自的差分变化率RMS/总变差；不是混合单位能耗。'],
        ['瞬态与余量','峰值偏移、全程位置误差积分、动作饱和时间比例，同时展示取舍。'],
    ],[1,3.7])
    doc.para('成功率使用全部有效策略试验。质量同时报告完整末段（含未稳定失败）、各方法成功试验和三网络共同成功初态。共同成功配对固定同一初态，但属于成功条件化子集。初始已在阈值内的L1单列保持时间和末段质量，恢复时间接近零不作为大姿态恢复速度证据。')
    report_methods.metrics(doc)
    report_methods.metric_details(doc)
    doc.title('4. 恢复成功率与难度变化')
    doc.fig(figs['difficulty'],'图1：各档基本成功率与描述性Wilson95%区间。')
    doc.table(['档位','CTBR','CTBR+DR','ID（本文方法）'],[['L'+str(i)]+[
        f"{level[i][m]['successes']}/{level[i][m]['n']} ({percentage(level[i][m]['success_fraction'])})" for m in MODELS] for i in range(1,6)])
    doc.para('各档样本量相同，整体成功率等价于五档等权平均。该比例对应本轮设计分布，不当作任意真实飞行总体的成功概率。若某个困难档的排名不同，必须结合总体和该档同时解释。')
    doc.para('本轮五档均为ID > DR > CTBR。ID相对DR的优势从L1到L5分别为7.8、29.8、28.8、7.4、4.6个百分点：L2和L3的增益最大，L4-L5差距收窄，说明大倾角与较大速度组合是当前剩余难点。CTBR的L2和L3成功率不单调，五档是组合设计与有限样本结果，不能假设每种策略必须随档位严格单调下降。')
    doc.title('4.1 结果是否依赖目标点重复或简单档位')
    doc.table(['分析人口','每网络次数','CTBR成功率','DR成功率','ID成功率'],[
        [label,str(groups['CTBR']['n'])]+[percentage(groups[m]['rate']) for m in MODELS]
        for key,label in (('all','原计划五档全部'),('without_goal_repeats','剔除100次目标静止重复'),('L2_L5','仅L2-L5组合恢复'))
        for groups in [extra['population_sensitivity'][key]]],[1.7,.8,1,1,1])
    doc.para('排除重复目标点后，ID为75.54%，DR59.21%，CTBR38.29%；排除全部L1后，ID为70.65%，DR53.00%，CTBR36.10%。因此总体优势并非只由目标点重复或最简单档的成功贡献。它们是补充敏感性分析，不改变原计划五档等额主分母。')
    doc.title('5. 精细控制：严格门槛下还能悬停多少次')
    doc.fig(figs['speed_precision'],'图2：左侧将恢复时间与成功率结合，右侧在固定速度门槛下展示位置精度要求的敏感性。')
    doc.table(H,[
        ['基本0.3m/0.3m/s',*[percentage(total[m]['precision']['basic']['fraction']) for m in MODELS]],
        ['精确0.1m/0.1m/s',*[percentage(total[m]['precision']['precise']['fraction']) for m in MODELS]],
        ['高精度0.05m/0.1m/s',*[percentage(total[m]['precision']['high_precision']['fraction']) for m in MODELS]]])
    doc.para('严格门槛是实验前固定的补充评价，主标准不变。完整的二维位置/速度门槛网格见标准报告与precision_thresholds.csv；不只展示对某网络有利的单个门槛。稳定进入曲线的分母始终包含失败，避免只比较成功试验的平均恢复时间。')
    doc.para('ID精确/高精度成功数分别为1885和1880，而基本成功数1913：约98.3%的ID基本成功试验达到高精度要求。DR对应为1292和19，CTBR均为0。CTBR的0仅表示本轮完整末2秒没有试验满足10cm位置与0.1m/s速度的联合严格条件，不等于轨迹从未经过10cm范围。')
    doc.table(['L2-L5：截至时间进入并保持至10s','CTBR / 2000','DR / 2000','ID / 2000'],[
        [str(t)+'s']+[percentage(next(x['fraction'] for x in extra['stable_by_time'] if x['subset']=='L2_L5' and x['model']==m and x['time_s']==t)) for m in MODELS]
        for t in (2,3,4,5,6,8)],[1.7,1,1,1])
    doc.para('仅看L2-L5，在3秒前进入稳定且保持到10秒的比例为ID42.40%、DR7.45%、CTBR1.25%；5秒前为ID61.60%、DR24.45%、CTBR12.60%。这张表包含失败分母，且排除了L1初始已稳定的情形，更直接反映组合偏差恢复效率。')
    doc.title('6. 相同初态：位置精度和运动抖动')
    doc.fig(figs['matched'],f'图3：三网络共同成功的全部同初态质量分布，n={len(shared)}；抖动轴使用对数尺度。')
    doc.table(H,[[key]+[num(quality[m][key]['mean'],5) for m in MODELS] for key in (
        'final_stabilization_entry_s','tail_position_error_mean_m','tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s','tail_rate_tracking_error_rms_rad_s')],[1.9,1,1,1.2])
    doc.para('位置误差是目标欧氏距离的时间均值；抖动为三个分量去时间均值后方差和的平方根，区分固定偏置和运动波动。各档共同成功样本数及其质量见common_success_quality.csv。')
    doc.para('共同成功518组中，ID位置误差约1.82cm，DR7.03cm，CTBR12.23cm；ID相对两基线分别减少约74.1%与85.1%。速度抖动相对DR、CTBR减少85.1%和93.0%；角速度抖动减少79.8%和96.3%。这些是同初态共同成功子集的均值比较，不能把比例直接扩展到所有失败初态。')
    doc.table(['档位','共同成功数','ID位置均值cm','ID速度抖动m/s','ID角速度抖动rad/s'],[
        ['L'+str(l),str(extra['common_success_per_level'][str(l)]),
         num(statistics.mean(record[s+':CTBR+ID']['metrics']['tail_position_error_mean_m'] for s in shared if record[s+':CTBR+ID']['spec']['level']==l)*100),
         num(statistics.mean(record[s+':CTBR+ID']['metrics']['tail_velocity_jitter_m_s'] for s in shared if record[s+':CTBR+ID']['spec']['level']==l),5),
         num(statistics.mean(record[s+':CTBR+ID']['metrics']['tail_body_rate_jitter_rad_s'] for s in shared if record[s+':CTBR+ID']['spec']['level']==l),5)]
        for l in range(1,6)],[1,.9,1,1.3,1.3])
    doc.title('6.1 动态恢复子集与三轴稳态偏置')
    qd=extra['dynamic_matched_quality']
    doc.table(['L2-L5共同成功，n=290','CTBR','DR','ID（本文方法）'],[[key]+[num(qd[m][key]['mean'],5) for m in MODELS]
        for key in ('final_stabilization_entry_s','tail_position_error_mean_m','tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s')],[1.9,1,1,1.2])
    doc.para('去掉全部L1后，同初态最终稳定进入时间ID为2.80s，DR4.97s，CTBR5.38s；末段位置均值仍为ID1.84cm、DR6.94cm、CTBR12.16cm。ID的精度和动态恢复优势没有因排除最简单样本而消失。')
    doc.table(['共同成功三轴平均偏置/m','x','y','z'],[[m]+[num(next(x['mean_bias_m'] for x in extra['position_bias'] if x['model']==m and x['subset']=='shared_success' and x['axis']==axis),5) for axis in 'xyz'] for m in MODELS])
    doc.para('三个网络的典型偏置方向不同。CTBR约(-9.44,+7.39,+1.34)cm，DR约(+4.44,+4.83,+2.28)cm，ID约(+1.07,+1.22,+0.67)cm。较小偏置直接支持ID悬停精度更高；偏置是策略与现有底层闭环的共同平衡点，不能仅从汇总表归因为推力映射、某个PID项或网络单一原因。')
    doc.title('7. 配对证据：有多少初态ID优于基线')
    doc.fig(figs['paired'],'图4：每个点是一组相同初态。点落在对角线下方表示ID对应指标更小；所有共同成功样本保留。')
    doc.table(['基线','指标','ID均值-基线','ID较小比例','描述性配对区间'],[[r['baseline'],r['metric'],num(r['mean_difference'],5),percentage(r['id_lower_fraction']),
        f"[{num(r['descriptive_paired_bootstrap_lo'],5)}, {num(r['descriptive_paired_bootstrap_hi'],5)}]"] for r in quality_pairs
        if r['metric'] in ('tail_position_error_mean_m','tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s')],[.9,1.8,1,1,1.5])
    doc.para('配对bootstrap采用固定seed=2026110404、2000次重采样，区间作为设计状态库中的描述。单一公共参数、条件子集、目标点重复和多指标选择限制其总体外推；不把这些试验当作独立训练种子。')
    doc.title('8. 动作顺滑与恢复余量')
    doc.fig(figs['actions'],'图5：共同成功同初态的四通道动作变化率RMS中位数，全程与末段分开展示。')
    doc.table(H,[[key]+[num(quality[m][key]['mean'],5) for m in MODELS] for key in ('peak_position_error_m','whole_position_iae_m_s','action_saturation_fraction')],[1.8,1,1,1.2])
    doc.para('末段更小动作变化和角速度抖动可支持悬停更平顺；瞬态峰值、全程饱和比例用于检查恢复策略代价。若某基线饱和更低或峰值更小，该取舍保留，不把ID优势写成每项指标逐状态支配。提前终止试验的全程积分只覆盖实际观察段，不能直接与完整10秒积分混排。')
    doc.table(['末2秒动作变化率RMS均值/归一化动作每秒','CTBR','DR','ID（本文方法）'],[[name]+[
        num(actions[m]['tail_action_derivative_rms_per_s_xyz_thrust'][j]['mean'],5) for m in MODELS] for j,name in enumerate(('roll','pitch','yaw','比推力'))],[1.9,1,1,1.2])
    doc.para('ID在四通道末段动作变化率均值上均更小，和较低角速度抖动相互支持。全程饱和率DR约1.24%、ID2.38%、CTBR6.53%，所以ID相对DR并非更少使用动作边界。特别在L5共同成功的33组中，ID峰值偏移均值2.67m，高于DR1.99m和CTBR2.32m；困难恢复中更强瞬态与更好最终稳定之间存在取舍。')
    doc.title('9. 从精确目标点开始：保持能力')
    doc.table(['组别','网络','成功/次数','精确悬停率','位置均值m','速度抖动m/s','角速度抖动rad/s'],[[cat,m,
        f"{nominal[cat][m]['successes']}/{nominal[cat][m]['n']}",percentage(nominal[cat][m]['precision']['precise']['fraction'])]+[
        num(nominal[cat][m]['quality_complete'][key]['mean'],5) for key in ('tail_position_error_mean_m','tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s')]
        for cat in ('exact_goal','near_goal') for m in MODELS],[1,1,1,1,1,1,1])
    doc.para('精确目标重复按相同请求初态独立启动，不额外注入有利或不利扰动。结果体现各自网络接管后能否保持目标及对估计/时序噪声的重复可靠性；隐藏电机、估计与噪声轨迹不可能仅凭相同请求状态逐值相同。这里的完整末段质量包含末段失稳失败，未完整到末段的试验单列失败。')
    doc.para('精确目标点100次中，ID与DR基本/精确成功均为100%，但5cm高精度成功为ID100%、DR0%；ID位置均值1.75cm，DR7.04cm，直接区分“能稳住”和“停得准”。小扰动400组ID均成功且均达到5cm高精度标准，DR基本成功361组，说明ID对目标附近扰动的保持能力也更强。CTBR的48/100包含完整轨迹中失败情形，不能只用成功子集掩盖其接管后不稳定。')
    doc.title('10. 初始化补做与有效样本数量')
    doc.table(['网络','首次尝试成功/次数','有效策略次数','全部尝试','需补做的槽位','最大尝试数'],[[r['model'],
        f"{r['first_attempt_successes']}/{r['first_attempt_n']}",str(r['accepted_n']),str(r['all_attempt_n']),str(r['slots_needing_makeup']),str(r['maximum_attempts_to_accept'])] for r in audit_rows])
    doc.para(f'总尝试{len(attempts)}次，其中基础设施无效尝试{len(infrastructure)}次。所有原始尝试保留，先前无效不覆盖。每个槽位按顺序取第一次有效策略试验；有效接管后恢复失败绝不补到成功为止。首次尝试全流程成功率与补做后的有效策略成功率分开，以免把基础设施条件化误写成部署全流程可靠性。')
    doc.table(['网络','初始化失败尝试','通信/模式失败尝试'],[[r['model'],str(r['initialization_failure_attempts']),str(r['communication_failure_attempts'])] for r in audit_rows])
    doc.para('试验完成审计逐一核对：五档每网络500次、2500个配对编号、实际PID、请求状态、初始化误差与输入源误差、被采用尝试之前没有被跳过的有效结果，以及冻结文件哈希。审计文件为final_integrity_audit.json。')
    doc.para('补做前首次尝试全流程成功率分别为CTBR38.44%、DR60.44%、ID76.12%，与补做后的38.68%、60.84%、76.52%相差不超过0.4个百分点；排序不依赖补做。81个状态-网络槽位曾补做，总共119次无效尝试，占全部7619次尝试的1.56%。其中114次初始化失败、5次通信/模式中断；CTBR单槽位最多9次尝试才接受，未换成容易初态。')
    doc.title('10.1 重新审计：真值、采样与原始轨迹')
    doc.table(['网络','PX4主成功数','主成功且真值成功','末段判定不一致数','最大输入年龄ms','最大采样间隔ms'],[[x['model'],
        str(x['px4_successes']),str(x['px4_successes_also_truth_success']),str(x['endpoint_disagreements']),
        num(x['max_input_age_ms'],2),num(x['max_sample_gap_s']*1000,1)] for x in extra['truth_and_timing']],[1.1,1,1.2,1.2,1.2,1.2])
    doc.para('已重新核对7500个接受结果的SHA256及57个冻结输入文件；按五档、三网络、成功/失败分层抽取29场，从原始trial_trace重新计算成功与关键指标，均与保存结果一致。ID与DR主成功和Gazebo真值成功一致；CTBR有5场末段阈值判定差异，其中1场主成功未满足真值成功。主判据仍按预定PX4观测，未事后更改。')
    doc.para('表中的输入年龄是每次动作所依赖观测的wall-clock年龄最大值，采样间隔是仿真时间；不能将输入年龄直接当作到旋翼的总闭环延迟。最大间隔28-36ms均低于100ms末段覆盖上限。真值与估计差异提示必须保留独立交叉核验。')
    doc.title('11. 失败类型和仍需改进的范围')
    for i in range(1,6):
        doc.para('L'+str(i)+'有效策略失败原因：'+str({m:level[i][m]['policy_failure_reasons'] for m in MODELS}))
    doc.table(['ID与基线配对','共同有效n','仅ID成功','仅基线成功','均成功','均失败'],[[r['baseline'],str(r['common_valid_n']),str(r['id_only_success']),str(r['baseline_only_success']),str(r['both_success']),str(r['neither_success'])] for r in pair_rows])
    doc.para('相同请求初态下另一网络独有成功的案例仍然保留，反映不同恢复策略的适用范围。L5是倒置与较大线速度/角速度的组合，失败应结合越界、超时与末段不稳定分开解释，不以阶段性排名更改状态范围或评分。')
    doc.table(['网络','有效策略失败总数','完成10秒但未稳定','越出飞行包络'],[[m,
        str(total[m]['n']-total[m]['successes']),
        str(sum(x['unfinished_stability'] for x in extra['policy_failures'] if x['model']==m)),
        str(sum(x['envelope_termination'] for x in extra['policy_failures'] if x['model']==m))] for m in MODELS])
    doc.para('ID失败587次中486次是包络越界，101次完成10秒但末段未稳定；CTBR为106次越界、1427次末段未稳定，DR为295次越界、684次末段未稳定。ID显著减少了“飞完但仍不稳定”的情况，主要剩余问题转为困难初态的早期恢复越界。')
    doc.para('越界判据为z不在(0.5,10)m或水平半径sqrt(x^2+y^2)>5m；是既定试验包络的保护终止，不全部等同于碰撞炸机。ID的486次越界中439次来自L4-L5。因此进一步工作应单独考察大倾角恢复瞬态和可用空间，不能通过只保留成功轨迹声称ID所有瞬态均最优。')
    doc.title('12. 同步视频：直观看见网络控制过程')
    cases=select_cases(manifest,rows)
    doc.table(['类别','选中初态','选择规则'],[[name,case.get('state',{}).get('id','本轮无该类别'),RULES[name]] for name,case in cases.items()],[1,1.7,2.4])
    doc.para('三栏采用同一镜头与空间边界，按各自释放后的仿真时间同步；实际Gazebo真值重建机体姿态、轨迹与目标，下面给位置误差、速度和角速度模长。固定末2秒显示目标附近放大画面并半速播放。已终止试验标注终止，不虚构后续状态。')
    doc.para('视频是本次正式已接受试验的科学回放，不是重新飞一次，也不是Gazebo窗口录屏。共同成功、ID独有成功、困难初态和目标保持按VIDEO_PROTOCOL.md固定规则选取；结果定义的案例标明类别，不当作无偏总体样本。统计负责总体结论，视频解释轨迹。')
    doc.title('13. 面向ID方法的实验结论')
    idg=total['CTBR+ID']
    doc.para('在新的五档等额初态基准中，ID的基本、精确与高精度悬停率分别为 '+
        ', '.join(percentage(idg['precision'][band]['fraction']) for band,_,_ in PRECISION_BANDS)+'。三条基线在相同物理平台、相同请求状态列表和固定底层配置下运行，因此这组试验直接比较网络接管后的闭环控制结果。')
    for baseline in ('CTBR','CTBR+DR'):
        advantage=100*(idg['success_fraction']-total[baseline]['success_fraction'])
        doc.para('相对'+baseline+'，ID基本成功率变化为'+f'{advantage:+.2f}'+'个百分点。共同成功同初态上的具体质量差异为：'+
            '；'.join(r['metric']+' '+num(r['mean_difference'],5) for r in quality_pairs if r['baseline']==baseline and r['metric'] in (
                'tail_position_error_mean_m','tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s'))+'。负值表示ID该指标更小。')
    doc.para('本轮数据在基本成功率、严格精度、动态恢复速度和同初态末段质量上给出一致的有利证据，可以支持：在当前Prototype/PX4公共闭环条件下，ID网络恢复覆盖更广、稳定更快、悬停更准且更平顺。五档全部ID > DR > CTBR，排除目标重复或整个L1后排序仍成立，首次尝试结果也保持同样排序。')
    doc.para('论文式结论：在2500个预先冻结试验编号、每方法2500次有效策略试验中，所提出ID方法取得76.52%的基本成功率，比DR和CTBR分别提高15.68与37.84个百分点；高精度成功率75.20%，对应1880次完整末段5cm/0.1m/s悬停。518组共同成功初态中，ID平均末段位置误差1.82cm，较DR7.03cm与CTBR12.23cm更小，且配对速度/角速度抖动整体更低。该结果支持ID在现有统一控制接口下的恢复和精细悬停优势。')
    doc.para('边界与后续：本文未重新训练、未调PID，结论限定于固定权重和公共闭环配置。L4-L5越界、动作饱和与峰值偏移的代价仍存在；大姿态、风扰、传感延迟或其他机型的外推需要独立实验。五档设计总体不代表任意真实世界分布，7500次仿真也不替代实机验证。报告未合并之前两轮或训练数据。')
    doc.title('附录：文件、复现与审计')
    doc.table(['路径（相对于formal）','内容'],[
        ['analysis/EXPERIMENT_ANALYSIS.pdf / .md','本详细报告'],['analysis/FIVE_LEVEL_REPORT.pdf / .md','自动标准报告和完整门槛图'],
        ['analysis/trials.csv / all_attempts.csv','接受结果及全部原始尝试'],['analysis/id_paired_control_differences.csv','同初态配对质量与描述性区间'],
        ['analysis/final_integrity_audit.json','7500槽位与实际参数/初始化/补做顺序审计'],
        ['analysis/reviewed_statistics.json','二次审阅的去简单档敏感性、时序/真值与偏置统计'],
        ['analysis/population_sensitivity.csv / stable_by_time.csv','去目标重复、L2-L5及失败留在分母的稳定进入比例'],
        ['analysis/truth_and_timing_audit.csv / matched_action_smoothness.csv','真值交叉判定、输入时序与四通道平顺性'],
        ['attempts/trials/*/run/','原始ULog、ROS数据、trial trace与接管信息'],
        ['videos/*.mp4 / selection_manifest.json','同步视频与案例来源索引'],
        ['manifest.json / state_bank.json / source_snapshot/','冻结协议、初态列表和源码快照']],[1.5,2])
    doc.para('报告复现：PYTHONNOUSERSITE=1 /usr/bin/python3 sim_to_sim/sim_experiment/analyze_study.py --study '+str(study))
    doc.para('完整原始数据物理存放在sim_to_sim/sim_experiment内。早期运行路径在logs下保留软链接，仅为原始绝对路径兼容；数据没有双份复制。报告阶段不修改试验结果。')
    doc.write()
    save(out/'detailed_artifact_manifest.json',dict(formal_manifest_sha256=digest(study/'manifest.json'),
        report_source_sha256=digest(Path(__file__)),accepted_results_sha256={r['trial_id']:digest(study/'trials'/r['trial_id']/'completed.json') for r in rows},
        report_helpers_sha256={name:digest(Path(__file__).with_name(name)) for name in ('report_methods.py','review_data.py')},
        figure_and_equation_sha256={str(p.relative_to(out)):digest(p) for folder in ('detailed_figures','equations') for p in (out/folder).glob('*') if p.is_file()},
        report_pdf_sha256=digest(out/'EXPERIMENT_ANALYSIS.pdf'),report_markdown_sha256=digest(out/'EXPERIMENT_ANALYSIS.md')))
    print('DETAILED_REPORT',out/'EXPERIMENT_ANALYSIS.pdf',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);args=p.parse_args();run(args.study)
