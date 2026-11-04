#!/usr/bin/env python3
"""Live/final CSV, JSON, figures and PDF for terminal-hover comparisons."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from xml.sax.saxutils import escape
from result_io import save, audit_cleanup_warning

MODELS=('CTBR','CTBR+DR','CTBR+ID')
QUALITY=('final_stabilization_entry_s','recovery_time_s','peak_position_error_m',
    'whole_position_iae_m_s','tail_position_error_mean_m','tail_position_error_rms_m',
    'tail_position_bias_norm_m','tail_speed_rms_m_s','tail_velocity_jitter_m_s',
    'tail_body_rate_jitter_rad_s','tail_rate_tracking_error_rms_rad_s',
    'tail_rate_power_3_10hz_fraction','action_saturation_fraction')


def stats(values):
    a=[float(v) for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(a),mean=statistics.mean(a) if a else None,median=statistics.median(a) if a else None,
        p95=sorted(a)[min(len(a)-1,math.ceil(.95*len(a))-1)] if a else None)


def interval(k,n):
    if not n:return None
    z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-h),min(1.,c+h)]


def summarize(rows):
    attempted=len(rows);valid=[r for r in rows if r.get('valid_policy_trial')]
    success=[r for r in rows if r.get('success')]
    full=[r for r in valid if r.get('metrics',{}).get('final_window_complete')]
    smooth={}
    for key,size in [('whole_action_derivative_rms_per_s_xyz_thrust',4),('tail_action_derivative_rms_per_s_xyz_thrust',4),
                     ('whole_action_total_variation_per_s_xyz_thrust',4),('tail_attitude_jitter_xyz_rad',3)]:
        smooth[key]=[stats([(r.get('metrics',{}).get(key) or [None]*size)[i] for r in valid]) for i in range(size)]
    return dict(attempted=attempted,valid_policy_trials=len(valid),successes=len(success),
        success_fraction_attempted=len(success)/attempted if attempted else None,
        success_fraction_valid=len(success)/len(valid) if valid else None,
        descriptive_wilson95=interval(len(success),attempted),
        failure_counts=dict(Counter(r.get('failure_category') or 'success' for r in rows)),
        truth_successes=sum(bool(r.get('truth_final_success')) for r in rows),
        endpoint_disagreements=sum(bool(r.get('endpoint_disagreement')) for r in rows),
        complete_terminal_quality_trials=len(full),
        all_complete_tail_quality={k:stats([r.get('metrics',{}).get(k) for r in full]) for k in QUALITY},
        successful_tail_quality={k:stats([r.get('metrics',{}).get(k) for r in success]) for k in QUALITY},
        all_attempt_quality={k:stats([r.get('metrics',{}).get(k) for r in valid]) for k in QUALITY if not k.startswith('tail_')},
        per_axis_quality=smooth,
        successful_state_ids=[r['state_id'] for r in success])


def collect(folder):
    m=json.loads((folder/'manifest.json').read_text());rows=[]
    for spec in m['schedule']:
        p=folder/'trials'/spec['trial_id']/'completed.json'
        if p.exists():rows.append(audit_cleanup_warning(json.loads(p.read_text()))[0])
    return m,rows


def build(folder,pdf=True):
    folder=Path(folder).resolve();m,rows=collect(folder);out=folder/'analysis';out.mkdir(exist_ok=True)
    groups={p:{model:summarize([r for r in rows if r['model']==model and r['profile']==p]) for model in MODELS}
        for p in m['profiles']}
    strata=sorted({s.get('stratum') for s in m['state_bank']['states'] if s.get('stratum')})
    by_stratum={p:{s:{model:summarize([r for r in rows if r['profile']==p and r['stratum']==s and r['model']==model])
        for model in MODELS} for s in strata} for p in m['profiles']}
    pairs={}
    for profile in m['profiles']:
        pairs[profile]={}
        for a,b in [('CTBR+ID','CTBR+DR'),('CTBR+DR','CTBR'),('CTBR+ID','CTBR')]:
            left={r['state_id']:r for r in rows if r['profile']==profile and r['model']==a}
            right={r['state_id']:r for r in rows if r['profile']==profile and r['model']==b}
            shared=set(left)&set(right);valid={s for s in shared if left[s].get('valid_policy_trial') and right[s].get('valid_policy_trial')}
            pairs[profile][a+' vs '+b]=dict(common_attempts=len(shared),common_valid_states=len(valid),
                only_a_success=sum(bool(left[s]['success']) and not right[s]['success'] for s in shared),
                only_b_success=sum(bool(right[s]['success']) and not left[s]['success'] for s in shared),
                both_success=sum(bool(left[s]['success']) and bool(right[s]['success']) for s in shared),
                both_failure=sum(not left[s]['success'] and not right[s]['success'] for s in shared))
    summary=dict(version=m['protocol']['version'],partial=len(rows)<len(m['schedule']),completed=len(rows),
        planned=len(m['schedule']),profiles=groups,by_stratum=by_stratum,paired=pairs,
        primary_success='Complete 10s flight; every sample in [8,10]s has position<0.3m and speed<0.3m/s. Prior relapse is diagnostic only.',
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        quality_note='Report every complete terminal window and successful trials separately; crashes have no invented steady-state metrics.')
    save(out/'summary.json',summary)
    flat=[]
    for r in rows:
        row={k:r.get(k) for k in ('trial_id','model','profile','state_id','stratum','success','valid_policy_trial',
            'failure_category','termination_reason','truth_final_success','endpoint_disagreement','baseline_restored','effective_pid_verified','wall_duration_s')}
        for k,v in r.get('metrics',{}).items():
            if isinstance(v,list):row.update({k+'_'+str(i):x for i,x in enumerate(v)})
            else:row[k]=v
        for prefix,key in [('requested','requested_state'),('handover_ypr','actual_handover_observation'),('action_source_ypr','actual_action_input_observation')]:
            row.update({prefix+'_'+str(i):v for i,v in enumerate(r.get(key) or [])})
        flat.append(row)
    if flat:
        with (out/'trials.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,list(dict.fromkeys(k for r in flat for k in r)))
            writer.writeheader();writer.writerows(flat)
    figures=[]
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        # Each frozen shared profile is compared separately; failures remain
        # in the success denominator, terminal quality never invents crash tails.
        for profile,g in groups.items():
            fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
            for ax,key,title in zip(axes,('success_fraction_attempted','tail_position_error_mean_m','tail_body_rate_jitter_rad_s'),
                ('Final hover success / all attempts','Complete-window position error (m)','Complete-window rate jitter (rad/s)')):
                values=[g[model][key] if key=='success_fraction_attempted' else g[model]['all_complete_tail_quality'][key]['mean'] for model in MODELS]
                ax.bar(range(3),[v if v is not None else 0 for v in values],color=['#3972ad','#d1882b','#26816f'])
                ax.set(xticks=range(3),xticklabels=MODELS,title=title)
                if key=='success_fraction_attempted':ax.set_ylim(0,1.05)
            name='quality_'+profile+'.png';fig.savefig(out/name,dpi=150);plt.close(fig);figures.append(out/name)
    except ImportError:pass
    def f(v):return '—' if v is None else f'{v:.4f}'
    text=['# 最终稳定悬停对照试验','',f'状态：{"进行中" if summary["partial"] else "已完成"}，已完成 {len(rows)}/{len(m["schedule"])} 次。',
        '',summary['primary_success'],'','复发不再否定成功。位置/速度主判定沿用PX4观测，独立记录Gazebo真值判定和差异。',
        '','| 公共PID | 网络 | 成功/已尝试 | 有效策略次数 | 全流程成功率 | 有效策略成功率 |','|---|---|---:|---:|---:|---:|']
    for p,g in groups.items():
        for model,x in g.items():text.append(f"|{p}|{model}|{x['successes']}/{x['attempted']}|{x['valid_policy_trials']}|{f(x['success_fraction_attempted'])}|{f(x['success_fraction_valid'])}|")
    text+=['','主成功率使用全部尝试作为分母；初始化、通信和策略失败分类保留。进行中报告仅反映已尝试样本，不能当成完整1000组结论。Wilson区间是描述性区间；初态分布以state_bank.json为准，分层初态另列层内结果，同时报告配对结果。',
        '','## 控制质量','',
        '| PID | 网络 | 完整末段n | 末段位置均值m | 速度抖动m/s | 角速度抖动rad/s |','|---|---|---:|---:|---:|---:|']
    for p,g in groups.items():
        for model,x in g.items():
            q=x['all_complete_tail_quality'];text.append(f"|{p}|{model}|{x['complete_terminal_quality_trials']}|{f(q['tail_position_error_mean_m']['mean'])}|{f(q['tail_velocity_jitter_m_s']['mean'])}|{f(q['tail_body_rate_jitter_rad_s']['mean'])}|")
    text+=['','CSV另含每轴稳态偏置、姿态波动、位置积分误差、最终收敛时间、指令变化率RMS与总变差、饱和比例、角速度跟踪误差及3–10Hz频段功率占比。动作指标分通道保存，单位是归一化动作/秒；不把混合单位的动作范数当作能耗。成功条件化质量另列JSON，避免只展示成功样本。',
        '','## 各初态层的成功率','',
        '| 公共PID | 初态层 | CTBR成功/尝试 | DR成功/尝试 | ID成功/尝试 |','|---|---|---:|---:|---:|']
    for profile,layer in by_stratum.items():
        for stratum,g in layer.items():
            text.append('|'+ '|'.join([profile,stratum]+[f"{g[model]['successes']}/{g[model]['attempted']}" for model in MODELS])+'|')
    text+=['','## 同初态配对比较','',
        '| PID | 比较A/B | 共同尝试 | 仅A成功 | 仅B成功 | 均成功 | 均失败 |','|---|---|---:|---:|---:|---:|---:|']
    for profile,group in pairs.items():
        for pair,x in group.items():text.append('|'+ '|'.join(map(str,[profile,pair,x['common_attempts'],x['only_a_success'],x['only_b_success'],x['both_success'],x['both_failure']]))+'|')
    text+=[
        '','## 协议与数据','',f"初态种子：{m['state_bank'].get('seed')}，分组：{m['state_bank'].get('split')}。三个网络共用相同初态、公共PID、模型物理与控制流程。全部网络共用config/final_pid.json的固定参数。",
        '',f'数据目录：{folder}','',f'记录目录：{out}。summary.json含逐层和配对比较；trials.csv含逐次数据。']
    markdown='\n'.join(text);(out/'FINAL_HOVER_REPORT.md').write_text(markdown,encoding='utf-8')
    if pdf:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,Image
        from reportlab.lib import colors
        pdfmetrics.registerFont(TTFont('FinalHoverCJK','/usr/share/fonts/truetype/arphic/uming.ttc',subfontIndex=0))
        style=ParagraphStyle('body',fontName='FinalHoverCJK',fontSize=8.5,leading=13,wordWrap='CJK',spaceAfter=7)
        cell=ParagraphStyle('cell',fontName='FinalHoverCJK',fontSize=7,leading=10,wordWrap='CJK')
        story=[];i=0
        while i<len(text):
            line=text[i]
            if line.startswith('|'):
                table=[]
                while i<len(text) and text[i].startswith('|'):
                    line=text[i]
                    if not set(line.replace('|','').replace('-','').replace(':','').replace(' ',''))==set():
                        table.append([Paragraph(escape(c),cell) for c in line.strip('|').split('|')])
                    i+=1
                t=Table(table,colWidths=[511/len(table[0])]*len(table[0]),repeatRows=1)
                t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.25,colors.HexColor('#cad5df')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f6')),('VALIGN',(0,0),(-1,-1),'TOP')]))
                story.extend([t,Spacer(1,8)]);continue
            if line:story.append(Paragraph(escape(line.lstrip('# ')),style))
            i+=1
        for p in figures:
            im=Image(str(p));ratio=511/im.imageWidth;im.drawWidth=511;im.drawHeight*=ratio
            story.extend([Spacer(1,10),im])
        SimpleDocTemplate(str(out/'FINAL_HOVER_REPORT.pdf'),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=42,bottomMargin=42).build(story)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    parser.add_argument('--no-pdf',action='store_true');args=parser.parse_args()
    s=build(args.batch,not args.no_pdf);print(json.dumps(dict(completed=s['completed'],planned=s['planned'],partial=s['partial'],profiles=s['profiles']),indent=2))
