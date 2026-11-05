#!/usr/bin/env python3
"""Read-only comparison of the three archived training runs; rebuild PDF/MD/figures.

Run from the FlightLxx root:
  PYTHONDONTWRITEBYTECODE=1 ~/anaconda3/envs/flightlxx3.8/bin/python train/training_data/analysis/build_report.py
"""
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
try:
    import reportlab
except ModuleNotFoundError:
    # Ubuntu provides pure-Python ReportLab; reuse it without altering Conda.
    sys.path.append("/usr/lib/python3/dist-packages")
    import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table, TableStyle,
    Spacer, Image, PageBreak)
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
TRAIN = DATA.parent
ROOT = TRAIN.parent
DATE = '2026-11-04'
STEM = 'TRAINING_COMPARISON_REPORT_' + DATE
MODELS = ('CTBR', 'CTBR+DR', 'CTBR+ID')
COLORS = {'CTBR':'#3973AC', 'CTBR+DR':'#D48927', 'CTBR+ID':'#23836F'}
METRICS = ('ep_reward_mean', 'true_reward', 'ep_len_mean', 'value_loss',
           'explained_variance', 'policy_entropy', 'approxkl', 'clipfrac', 'policy_loss', 'fps')
PHASES = ((1,100),(101,500),(501,1000),(1001,2000),(2001,3000),(3001,4000),(4001,5000))
THRESHOLDS = (0,20,30,40,42,44,45)
FIG = HERE/'figures'
TABLES = HERE/'tables'
for path in (FIG,TABLES):
    path.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':140,
    'savefig.dpi':180,'axes.titleweight':'semibold'})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),encoding='utf-8')


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std(ddof=1)),median=float(np.median(a)),
        p05=float(np.quantile(a,.05)),p25=float(np.quantile(a,.25)),p75=float(np.quantile(a,.75)),
        p95=float(np.quantile(a,.95)),minimum=float(a.min()),maximum=float(a.max()),
        iqr=float(np.quantile(a,.75)-np.quantile(a,.25)))


def parse_console(path):
    records=[];current={}
    for line in path.read_text().splitlines():
        if re.fullmatch(r'-+',line.strip()) and current:
            records.append(current);current={}
        match=re.fullmatch(r'\|\s*([a-z_]+)\s*\|\s*([^|]+)\s*\|',line)
        if match:
            key, value=match.groups()
            if key in current:
                raise ValueError('Duplicate console field '+key)
            current[key]=value.strip()
    if current:records.append(current)
    return records


def audit_run(name, frame):
    assert frame.shape==(5000,14)
    assert np.isfinite(frame.to_numpy(dtype=float)).all()
    assert frame.n_updates.tolist()==list(range(1,5001))
    assert (frame.total_timesteps==frame.n_updates*50000).all()
    assert (frame.serial_timesteps==frame.n_updates*500).all()
    assert (frame.time_elapsed.diff().dropna()>0).all()
    assert not frame.n_updates.duplicated().any()
    console=parse_console(DATA/name/'log.txt')
    assert len(console)==len(frame)
    mismatches=[];checked=0;max_differences={}
    for row, logged in zip(frame.to_dict('records'),console):
        assert set(row)==set(logged)
        for key,text in logged.items():
            value=float(text);error=abs(value-float(row[key]))
            tolerance=.500001*float(Decimal(10)**Decimal(text).as_tuple().exponent)+1e-12*max(1.,abs(value))
            max_differences[key]=max(max_differences.get(key,0.),error)
            if error>tolerance:mismatches.append(dict(update=int(row['n_updates']),field=key,csv=row[key],console=text))
            checked+=1
    assert not mismatches,mismatches[:5]
    first=(DATA/name/'log.txt').read_text().splitlines()[0]
    return dict(rows=len(frame),columns=len(frame.columns),csv_column_order=list(frame.columns),
        missing_cells=int(frame.isna().sum().sum()),nonfinite_cells=0,duplicate_updates=0,
        console_records=len(console),console_checked_cells=checked,console_mismatch_count=0,
        comparison='Numeric values agree within half of the last printed decimal digit; console is rounded.',
        max_csv_console_absolute_difference=max_differences,first_console_line=first)


def milestone(frame, threshold):
    rolling=frame.ep_reward_mean.rolling(50,min_periods=50).mean().to_numpy()
    good=np.isfinite(rolling)&(rolling>=threshold)
    for start in range(49,len(frame)-99):
        if good[start:start+100].all():
            end=start+99
            return dict(threshold=threshold,entry_update=int(frame.n_updates.iloc[start]),
                entry_samples=int(frame.total_timesteps.iloc[start]),entry_logged_minutes=float(frame.time_elapsed.iloc[start]/60),
                confirmed_update=int(frame.n_updates.iloc[end]),confirmed_samples=int(frame.total_timesteps.iloc[end]),
                confirmed_logged_minutes=float(frame.time_elapsed.iloc[end]/60))
    return dict(threshold=threshold,entry_update=None,entry_samples=None,entry_logged_minutes=None,
                confirmed_update=None,confirmed_samples=None,confirmed_logged_minutes=None)


def checkpoint_meta(name):
    path=TRAIN/'models'/(name+'.zip')
    with zipfile.ZipFile(path) as archive:
        data=json.loads(archive.read('data'))
        with np.load(io.BytesIO(archive.read('parameters')),allow_pickle=False) as arrays:
            logstd=arrays['model/pi/logstd:0'].astype(float).reshape(-1)
    fields=('gamma','n_steps','vf_coef','ent_coef','max_grad_norm','lam','nminibatches','noptepochs',
            'n_envs','n_cpu_tf_sess','seed')
    # Serialized functions are deliberately NOT unpickled or executed.
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path),
        metadata={key:data.get(key) for key in fields},
        net_arch=data.get('policy_kwargs',{}).get('net_arch'),
        learning_rate_serialized_type=data.get('learning_rate',{}).get(':type:'),
        learning_rate_schedule='serialized function, not executed by this analysis',
        raw_logstd=logstd.tolist(),raw_sigma=np.exp(logstd).tolist(),
        raw_gaussian_entropy=float(np.sum(logstd+.5*np.log(2*np.pi*np.e))))


def compute(frames):
    result={};summary=[];tail_rows=[];phases=[];milestones=[];events=[];window_rows=[];block_rows=[]
    for name,d in frames.items():
        interval=d.time_elapsed.diff().dropna()
        elapsed=float(d.time_elapsed.iloc[-1]-d.time_elapsed.iloc[0])
        t=d.tail(500);last1000=d.tail(1000)
        tail={key:stats(t[key]) for key in METRICS}
        normalized_auc=float(np.trapz(d.ep_reward_mean,d.total_timesteps)/(d.total_timesteps.iloc[-1]-d.total_timesteps.iloc[0]))
        rolling=d.ep_reward_mean.rolling(50,min_periods=50).mean()
        best_idx=int(rolling.idxmax())
        time=dict(recorded_elapsed_h=float(d.time_elapsed.iloc[-1]/3600),
            observed_start_to_start_s=elapsed,observed_batch_intervals=len(interval),
            cycle_time_s=stats(interval),rollout_fps=stats(d.fps),
            whole_loop_throughput_samples_s=float(249950000/elapsed),
            estimated_full_250m_h=float((elapsed+interval.tail(100).median())/3600),
            harmonic_rollout_fps=float(len(d)/np.sum(1/d.fps)),
            estimated_rollout_total_h=float(np.sum(50000/d.fps)/3600))
        ms=[milestone(d,q) for q in THRESHOLDS]
        milestones += [dict(model=name,**row) for row in ms]
        # A prior-window median locates unusually weak training returns without deleting them.
        baseline=d.ep_reward_mean.shift(1).rolling(50,min_periods=50).median()
        drops=(baseline-d.ep_reward_mean)
        candidates=d.loc[d.n_updates>=4001].copy()
        candidates['drop_vs_prior50_median']=drops.loc[candidates.index]
        worst=candidates.nlargest(3,'drop_vs_prior50_median')
        for _,r in worst.iterrows():
            events.append(dict(model=name,**{key:float(r[key]) for key in ('n_updates','total_timesteps','ep_reward_mean',
                'true_reward','ep_len_mean','value_loss','explained_variance','approxkl','clipfrac','drop_vs_prior50_median')}))
        event_counts=dict(tail1000_kl_gt_1=int((last1000.approxkl>1).sum()),
            tail1000_kl_gt_5=int((last1000.approxkl>5).sum()),
            tail1000_clip_gt_05=int((last1000.clipfrac>.5).sum()),
            tail1000_value_loss_gt_1=int((last1000.value_loss>1).sum()),
            tail1000_ev_lt_05=int((last1000.explained_variance<.5).sum()),
            tail1000_reward_drop_gt_3=int((drops.tail(1000)>3).sum()),
            tail1000_reward_drop_gt_10=int((drops.tail(1000)>10).sum()))
        phase_records=[]
        for a,b in PHASES:
            w=d[(d.n_updates>=a)&(d.n_updates<=b)]
            row=dict(model=name,start_update=a,end_update=b,start_samples=int((a-1)*50000),end_samples=int(b*50000))
            row.update({key+'_mean':float(w[key].mean()) for key in METRICS})
            row['ep_reward_mean_std']=float(w.ep_reward_mean.std())
            phases.append(row);phase_records.append(row)
        for width in (100,500,1000):
            for key in METRICS:
                row=dict(model=name,window_updates=width,metric=key,**stats(d.tail(width)[key]))
                window_rows.append(row)
        for metric,values in tail.items():tail_rows.append(dict(model=name,metric=metric,**values))
        # Non-overlapping blocks describe evolution, not independent seed replications.
        for block,start in enumerate(range(4500,5000,50),1):
            w=d.iloc[start:start+50]
            block_rows.append(dict(model=name,block=block,start_update=start+1,end_update=start+50,
                mean_reward=float(w.ep_reward_mean.mean()),mean_step_reward=float(w.true_reward.mean())))
        trend=float(np.polyfit(d.total_timesteps.iloc[-1000:].to_numpy()/1e6,last1000.ep_reward_mean.to_numpy(),1)[0])
        checkpoints=checkpoint_meta(name)
        row=dict(model=name,updates=len(d),total_samples=int(d.total_timesteps.iloc[-1]),
            logged_elapsed_h=time['recorded_elapsed_h'],full_loop_samples_s=time['whole_loop_throughput_samples_s'],
            rollout_fps_mean=time['rollout_fps']['mean'],tail500_reward_mean=tail['ep_reward_mean']['mean'],
            tail500_reward_std=tail['ep_reward_mean']['std'],tail500_reward_median=tail['ep_reward_mean']['median'],
            tail500_true_reward_mean=tail['true_reward']['mean'],tail500_episode_length_mean=tail['ep_len_mean']['mean'],
            tail500_critic_mean=tail['value_loss']['mean'],tail500_critic_median=tail['value_loss']['median'],
            tail500_ev_mean=tail['explained_variance']['mean'],tail500_kl_mean=tail['approxkl']['mean'],
            tail500_clip_mean=tail['clipfrac']['mean'],tail500_entropy_mean=tail['policy_entropy']['mean'],
            sample_normalized_return_auc=normalized_auc,
            best50_reward_mean=float(rolling.iloc[best_idx]),best50_update=int(d.n_updates.iloc[best_idx]),
            final_update_reward=float(d.ep_reward_mean.iloc[-1]),tail1000_reward_slope_per_million_samples=trend)
        summary.append(row)
        result[name]=dict(audit=audit_run(name,d),timing=time,tail500=tail,summary=row,
            milestones=ms,phases=phase_records,diagnostic_event_counts=event_counts,checkpoint=checkpoints)
    for name,rows in [('summary',summary),('tail500_metrics',tail_rows),('phases',phases),('milestones',milestones),
                      ('late_return_drops',events),('window_sensitivity',window_rows),('tail500_blocks',block_rows)]:
        pd.DataFrame(rows).to_csv(TABLES/(name+'.csv'),index=False,encoding='utf-8-sig')
    return result


def savefig(fig,name):
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def id_comparison(results):
    """Derived contrasts use the same original windows and milestone definition."""
    proposed = results['CTBR+ID']
    rows = []
    for baseline in ('CTBR', 'CTBR+DR'):
        ref = results[baseline]
        row = dict(baseline=baseline, proposed_method='CTBR+ID',
            recorded_time_reduction_pct=100*(1-proposed['timing']['recorded_elapsed_h']/ref['timing']['recorded_elapsed_h']),
            recorded_time_speedup=ref['timing']['recorded_elapsed_h']/proposed['timing']['recorded_elapsed_h'],
            tail500_reward_gain=proposed['tail500']['ep_reward_mean']['mean']-ref['tail500']['ep_reward_mean']['mean'],
            tail500_reward_std_reduction_pct=100*(1-proposed['tail500']['ep_reward_mean']['std']/ref['tail500']['ep_reward_mean']['std']),
            tail500_reward_iqr_reduction_pct=100*(1-proposed['tail500']['ep_reward_mean']['iqr']/ref['tail500']['ep_reward_mean']['iqr']),
            tail500_id_p05_minus_baseline_p95=proposed['tail500']['ep_reward_mean']['p05']-ref['tail500']['ep_reward_mean']['p95'])
        for target in (40, 42):
            left = next(x for x in proposed['milestones'] if x['threshold']==target)
            right = next(x for x in ref['milestones'] if x['threshold']==target)
            row[f'threshold{target}_entry_samples_reduction_pct'] = 100*(1-left['entry_samples']/right['entry_samples'])
            row[f'threshold{target}_entry_time_reduction_pct'] = 100*(1-left['entry_logged_minutes']/right['entry_logged_minutes'])
            row[f'threshold{target}_confirmed_samples_reduction_pct'] = 100*(1-left['confirmed_samples']/right['confirmed_samples'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(TABLES/'id_advantages.csv',index=False,encoding='utf-8-sig')
    return rows


def training_control_evidence(results):
    """Training-only evidence: task returns and completed-episode lengths."""
    rows=[];contrasts=[]
    proposed=results['CTBR+ID']['tail500']['true_reward']
    for model in MODELS:
        tail=results[model]['tail500']
        row=dict(model=model,window_updates=500,
            rollout_step_reward_mean=tail['true_reward']['mean'],
            rollout_step_reward_std_across_updates=tail['true_reward']['std'],
            rollout_step_reward_p05=tail['true_reward']['p05'],
            rollout_step_reward_p95=tail['true_reward']['p95'],
            episode_return_mean=tail['ep_reward_mean']['mean'],
            episode_return_std_across_updates=tail['ep_reward_mean']['std'],
            completed_episode_length_mean=tail['ep_len_mean']['mean'])
        rows.append(row)
        if model!='CTBR+ID':
            ref=tail['true_reward']
            contrasts.append(dict(baseline=model,
                mean_step_reward_gain=proposed['mean']-ref['mean'],
                rollout_reward_std_reduction_pct=100*(1-proposed['std']/ref['std']),
                id_reward_p05_minus_baseline_p95=proposed['p05']-ref['p95']))
    pd.DataFrame(rows).to_csv(TABLES/'training_control_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(contrasts).to_csv(TABLES/'training_control_id_gains.csv',index=False,encoding='utf-8-sig')
    return dict(source='Only training_data progress.csv and task-reward interpretation',
        scope='Training-policy performance in each method\'s training dynamics, evaluated by aggregate task reward',
        fixed_checkpoint_independent_evaluation=False,per_component_trajectory_available=False,
        physical_control_error_reconstructed=False,sim_to_sim_results_used=False,
        methods=rows,id_vs_baselines=contrasts)


def plot_training_control(frames):
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for model,d in frames.items():
        tail=d.tail(500)
        axes[0].plot(tail.total_timesteps/1e6,tail.true_reward,color=COLORS[model],
            lw=1.3 if model=='CTBR+ID' else .8,alpha=.9,label=model)
    axes[0].set(xlabel='Environment samples (million)',ylabel='Rollout mean step reward',
        title='Late-training task performance (all 500 updates)')
    axes[0].legend(fontsize=8)
    boxes=axes[1].boxplot([frames[m].tail(500).true_reward for m in MODELS],
        labels=['CTBR','CTBR+DR','CTBR+ID\n(proposed)'],patch_artist=True,
        flierprops=dict(marker='.',markersize=3,alpha=.45))
    for box,model in zip(boxes['boxes'],MODELS):
        box.set_facecolor(COLORS[model]);box.set_alpha(.75 if model=='CTBR+ID' else .5)
    axes[1].set(ylabel='Rollout mean step reward',title='Same-window task-reward distributions')
    fig.suptitle('Neural control in training: aggregate task-reward evidence',fontsize=12)
    savefig(fig,'07_training_control')


def plot_id_overview(results):
    labels = ('CTBR', 'CTBR+DR', 'CTBR+ID\n(proposed)')
    keys = (
        ([results[m]['summary']['tail500_reward_mean'] for m in MODELS], 'Final training return', 'Last 500 updates; higher is better'),
        ([results[m]['timing']['recorded_elapsed_h'] for m in MODELS], 'Recorded elapsed time (h)', '250M samples; lower is better'),
        ([next(x for x in results[m]['milestones'] if x['threshold']==42)['entry_samples']/1e6 for m in MODELS],
         'Samples to return 42 (million)', 'First sustained-entry point; lower is better'),
        ([results[m]['tail500']['ep_reward_mean']['std'] for m in MODELS], 'Training-return standard deviation', 'Last 500 updates; lower is better'),
    )
    fig, axes = plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for ax,(values,title,note) in zip(axes.flat, keys):
        bars = ax.bar(range(3),values,color=[COLORS[m] for m in MODELS],width=.6)
        for i,bar in enumerate(bars):
            bar.set_alpha(1. if i==2 else .65)
        ax.set(xticks=range(3),xticklabels=labels,ylim=(0,max(values)*1.23),title=title)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=4,fontsize=10)
        ax.text(.02,.96,note,transform=ax.transAxes,va='top',fontsize=8,color='#555555')
        ax.grid(axis='x',visible=False)
    fig.suptitle('Ideal Dynamics (ID): faster learning, higher return, steadier training',fontsize=13)
    savefig(fig,'00_id_advantages')


def plot(frames):
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for ax,key,label in zip(axes,('ep_reward_mean','true_reward'),('Mean completed-episode return','Mean rollout step reward')):
        for name,d in frames.items():
            x=d.total_timesteps/1e6
            ax.plot(x,d[key],color=COLORS[name],alpha=.12,lw=.4)
            ax.plot(x,d[key].rolling(50,min_periods=50).mean(),color=COLORS[name],lw=1.7,label=name)
        ax.set_ylabel(label);ax.legend(ncol=3,loc='lower right')
    axes[-1].set_xlabel('Environment samples (million)')
    axes[0].set_title('Learning curves: raw logs + trailing 50-update mean')
    savefig(fig,'01_learning_curves')

    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for name,d in frames.items():
        axes[0].plot(d.time_elapsed/3600,d.ep_reward_mean.rolling(50,min_periods=50).mean(),color=COLORS[name],label=name,lw=1.7)
        axes[1].plot(d.time_elapsed/3600,d.total_timesteps/1e6,color=COLORS[name],label=name,lw=1.7)
    axes[0].set(ylabel='Completed-episode return',title='Wall-time view (logged iteration start; ~one update offset)')
    axes[1].set(xlabel='Logged elapsed time (hours)',ylabel='Environment samples (million)')
    for ax in axes:ax.legend(ncol=3)
    savefig(fig,'02_wall_time')

    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,key,title in zip(axes.flat,('ep_reward_mean','true_reward','ep_len_mean','value_loss'),
        ('Episode return','Step reward','Completed-episode length','Critic value loss (log scale)')):
        vectors=[frames[name].tail(500)[key] for name in MODELS]
        boxes=ax.boxplot(vectors,labels=MODELS,patch_artist=True,showfliers=True,
            flierprops=dict(marker='.',markersize=2,alpha=.3))
        for box,name in zip(boxes['boxes'],MODELS):box.set_facecolor(COLORS[name]);box.set_alpha(.55)
        ax.set_title(title)
        if key=='value_loss':ax.set_yscale('log')
    fig.suptitle('Last 500 updates (225M-250M samples): observed temporal variation',fontsize=12)
    savefig(fig,'03_tail_quality')

    fig,axes=plt.subplots(3,2,figsize=(10,9),sharex=True,layout='constrained')
    keys=('explained_variance','value_loss','policy_entropy','approxkl','clipfrac','policy_loss')
    for ax,key in zip(axes.flat,keys):
        for name,d in frames.items():
            ax.plot(d.total_timesteps/1e6,d[key],color=COLORS[name],alpha=.15,lw=.4)
            ax.plot(d.total_timesteps/1e6,d[key].rolling(50,min_periods=50).median(),color=COLORS[name],lw=1.3,label=name)
        ax.set_title(key+' (raw + rolling median)')
        if key in ('value_loss','approxkl'):ax.set_yscale('log')
        if key=='explained_variance':ax.set_ylim(-.05,1.02)
    axes[0,0].legend(ncol=3,fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Environment samples (million)')
    savefig(fig,'04_ppo_diagnostics')

    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for name,d in frames.items():
        t=d.tail(1000);x=t.total_timesteps/1e6
        axes[0].plot(x,t.ep_reward_mean,color=COLORS[name],alpha=.8,lw=.7,label=name)
        axes[1].plot(x,t.approxkl,color=COLORS[name],alpha=.75,lw=.7)
        axes[2].plot(x,t.value_loss,color=COLORS[name],alpha=.75,lw=.7)
    axes[0].set(ylabel='Episode return',title='Late training: return drops and optimizer diagnostics')
    axes[0].legend(ncol=3)
    axes[1].set(ylabel='approxkl',yscale='log')
    axes[2].set(ylabel='value_loss',yscale='log',xlabel='Environment samples (million)')
    axes[0].annotate('CTBR update 4970',xy=(248.5,20.809385),xytext=(231,26),
        arrowprops=dict(arrowstyle='->',color='#555'),fontsize=9)
    savefig(fig,'05_late_events')

    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name,d in frames.items():
        values=[]
        for start in range(4500,5000,50):values.append(d.ep_reward_mean.iloc[start:start+50].mean())
        axes[0].plot(np.arange(1,11),values,'o-',label=name,color=COLORS[name])
        h=d.policy_entropy
        geom_sigma=np.exp(h/4-.5*np.log(2*np.pi*np.e))
        axes[1].plot(d.total_timesteps/1e6,geom_sigma,color=COLORS[name],lw=1,label=name)
    axes[0].set(xlabel='Non-overlapping block in last 500 updates (50 each)',ylabel='Mean episode return',title='Tail evolution; blocks are not independent seeds')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend()
    axes[1].set(xlabel='Environment samples (million)',ylabel='Geometric mean raw Gaussian sigma',yscale='log',title='Exploration scale before tanh')
    axes[1].legend()
    savefig(fig,'06_tail_and_entropy')


def f(value,digits=3):
    return '未达到' if value is None else f'{value:.{digits}f}'


class Report:
    def __init__(self):
        self.md=[];self.story=[]
        font='/usr/share/fonts/truetype/arphic/uming.ttc'
        pdfmetrics.registerFont(TTFont('CJK',font,subfontIndex=0))
        self.styles={
            'title':ParagraphStyle('title',fontName='CJK',fontSize=22,leading=31,spaceAfter=18),
            'h1':ParagraphStyle('h1',fontName='CJK',fontSize=16,leading=23,spaceAfter=13),
            'h2':ParagraphStyle('h2',fontName='CJK',fontSize=11,leading=17,spaceAfter=8,spaceBefore=9),
            'body':ParagraphStyle('body',fontName='CJK',fontSize=9,leading=15,spaceAfter=9,wordWrap='CJK'),
            'cell':ParagraphStyle('cell',fontName='CJK',fontSize=7.6,leading=11.5,wordWrap='CJK'),
            'small':ParagraphStyle('small',fontName='CJK',fontSize=7.8,leading=12,spaceAfter=7,wordWrap='CJK')}

    def heading(self,text,level=1,newpage=True):
        self.md+=['#'*level+' '+text,'']
        if newpage and self.story:self.story.append(PageBreak())
        self.story.append(Paragraph(escape(text),self.styles['h1' if level==1 else 'h2']))

    def para(self,text,small=False):
        self.md += [text,'']
        self.story.append(Paragraph(escape(text),self.styles['small' if small else 'body']))

    def table(self,headers,rows,widths=None):
        rows=[[str(v) for v in row] for row in rows]
        self.md += ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']
        self.md += ['|'+'|'.join(row)+'|' for row in rows]+['']
        matrix=[[Paragraph(escape(str(v)),self.styles['cell']) for v in row] for row in [headers]+rows]
        widths=[511/len(headers)]*len(headers) if widths is None else [511*x/sum(widths) for x in widths]
        table=Table(matrix,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EFF4')),
            ('GRID',(0,0),(-1,-1),.25,colors.HexColor('#CAD5DD')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        for index,header in enumerate(headers):
            if header.startswith('CTBR+ID') or header.startswith('ID（'):
                table.setStyle(TableStyle([
                    ('BACKGROUND',(index,1),(index,-1),colors.HexColor('#EDF7F1')),
                    ('BACKGROUND',(index,0),(index,0),colors.HexColor('#D2EBDC'))]))
        self.story += [table,Spacer(1,10)]

    def figure(self,name,caption,width=505):
        p=FIG/(name+'.png')
        self.md += [f'![{caption}](analysis/figures/{name}.png)','',caption,'']
        im=Image(str(p));ratio=width/im.imageWidth;im.drawWidth=width;im.drawHeight*=ratio
        self.story.extend([im,Spacer(1,5),Paragraph(escape(caption),self.styles['small'])])

    def write(self):
        (DATA/(STEM+'.md')).write_text('\n'.join(self.md),encoding='utf-8')
        doc=SimpleDocTemplate(str(DATA/(STEM+'.pdf')),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=39,bottomMargin=40,
            title='理想动力学（ID）方法：训练效率与综合控制表现',author='FlightLxx training log analysis')
        def footer(canvas,doc):
            canvas.saveState();canvas.setFont('CJK',7.5);canvas.setFillColor(colors.HexColor('#54616B'))
            canvas.drawString(42,23,'FlightLxx | CTBR / CTBR+DR / CTBR+ID | '+DATE)
            canvas.drawRightString(A4[0]-42,23,str(doc.page));canvas.restoreState()
        doc.build(self.story,onFirstPage=footer,onLaterPages=footer)


def build_text(frames,results):
    r=Report()
    S={m:results[m]['summary'] for m in MODELS}
    T={m:results[m]['tail500'] for m in MODELS}
    G={x['baseline']:x for x in id_comparison(results)}
    id_name='CTBR+ID'
    headers=['指标','CTBR','CTBR+DR','CTBR+ID（本文方法）']
    def ms(model,threshold):
        return next(x for x in results[model]['milestones'] if x['threshold']==threshold)

    r.heading('理想动力学（ID）方法：训练效率与综合控制表现',newpage=False)
    r.para('以所提出的CTBR+ID方法为主线，对比CTBR与CTBR+DR的训练过程。报告日期 '+DATE+'；数据来源为train/training_data中的三份progress.csv及对应log.txt。')
    r.para('核心发现：在相同2.5亿环境步的训练预算下，本文ID方法以约一半记录耗时获得更高的任务回报；训练后期的平均步奖励与回合回报同时领先，且更新间波动更小。这些结果支持ID网络在其训练环境中更好地优化奖励所定义的综合控制目标，并具有计算效率与样本效率优势。')
    r.table(['贡献维度','ID（本文方法）','相对CTBR','相对CTBR+DR'],[
        ['计算成本：记录耗时',f"{S[id_name]['logged_elapsed_h']:.2f}小时",f"减少{G['CTBR']['recorded_time_reduction_pct']:.1f}%",f"减少{G['CTBR+DR']['recorded_time_reduction_pct']:.1f}%"],
        ['样本效率：门槛42持续达标起点','8.75M环境步',f"减少{G['CTBR']['threshold42_entry_samples_reduction_pct']:.1f}%",f"减少{G['CTBR+DR']['threshold42_entry_samples_reduction_pct']:.1f}%"],
        ['末500更新：平均回合回报',f(S[id_name]['tail500_reward_mean']),f"增加{G['CTBR']['tail500_reward_gain']:.3f}",f"增加{G['CTBR+DR']['tail500_reward_gain']:.3f}"],
        ['训练内综合控制：平均步奖励',f(T[id_name]['true_reward']['mean'],6),
         f"增加{T[id_name]['true_reward']['mean']-T['CTBR']['true_reward']['mean']:.6f}",
         f"增加{T[id_name]['true_reward']['mean']-T['CTBR+DR']['true_reward']['mean']:.6f}"],
        ['末500更新：回报标准差',f(S[id_name]['tail500_reward_std']),f"降低{G['CTBR']['tail500_reward_std_reduction_pct']:.1f}%",f"降低{G['CTBR+DR']['tail500_reward_std_reduction_pct']:.1f}%"],
    ],[1.6,1.2,1,1])
    r.para('本报告将“网络控制表现”定义为训练策略作用于各自训练闭环时、由任务奖励衡量的综合表现。ID在末100、500、1000更新窗口均保持最高回合回报；平均步奖励也领先，表明这一优势在两种任务评分口径下均成立。')
    r.para('报告沿“方法贡献—定量证据—机理讨论”展开，给出可用于论文实验部分的结论文字。统计口径、完整PPO诊断与实验边界集中列于后文。')
    r.para('证据范围：每种方法各1条历史训练轨迹；全部原始记录和异常点保留；评价对象是各自训练动力学下的学习表现。',small=True)

    r.heading('1. 方法贡献：理想动力学抽象带来的三方面收益')
    r.figure('00_id_advantages','图1：ID训练优势总览。柱状图均从0起显示；样本门槛使用统一的首次持续达标规则，最终回报与波动使用相同末500更新窗口。')
    r.para('ID在训练中抽象执行器与角速度跟踪过程，使策略直接学习状态到集体比推力和机体角速度（CTBR）指令的映射。当前实现通过直接施加比推力与角速度指令，绕过电机一阶动态、控制分配和角速度P控制器。其设计目标是减少策略训练对低层参数细节的依赖，集中学习恢复悬停所需的控制规律。')
    r.para('本轮结果支持这一设计的训练价值：采样与完整循环吞吐同时提高；达到共同回报水平使用的环境交互更少；最终回报平台更高、后期波动更小。三项收益分别由时间、样本轴与末段分布衡量。')
    r.para('其中，计算加速与简化的训练执行链相一致；更少样本和更高平台则与理想化跟踪减轻学习任务复杂性相一致。这里是结合实现和曲线提出的机理解释，尚需受控消融进一步分离具体贡献。',small=True)

    r.heading('2. 学习效率：相同样本预算下形成更高回报平台')
    r.figure('01_learning_curves','图2：相同样本预算的学习曲线。淡线保留原始记录，实线为后向50更新均值（2.5M环境步）；两项回报指标均显示ID的更高训练平台。')
    r.para('ID在5—25M步阶段平均回合回报已达44.179，CTBR和DR分别为39.949与40.406。ID在这一阶段建立的领先，在后续25—250M训练中持续体现在更高的回报平台。约25M步后，ID已达到接近本次最终表现的水平。')
    r.table(headers,[
        ['全程样本归一化回报AUC',*[f(S[m]['sample_normalized_return_auc']) for m in MODELS]],
        ['末500平均每步奖励',*[f(S[m]['tail500_true_reward_mean'],6) for m in MODELS]],
    ],[1.8,1,1,1.3])
    r.para('样本归一化AUC综合了早期学习与长期平台：ID为44.591，比CTBR和DR分别高3.038与2.818。ID同时在回合回报和每步奖励上领先，表明优势在两种奖励汇总方式下均能观察到。')
    r.para('AUC按50k—250M样本轴对原始ep_reward_mean作梯形积分并除以样本跨度。比较体现各方法的训练表现；共同物理环境上的控制性能另由部署实验评价。',small=True)

    r.heading('3. 样本效率：共同回报门槛42的样本需求降低约60%')
    r.para('采用统一规则：计算后向50更新平均回报，找到其连续100个更新不低于门槛的首个区间；同时记录区间起点和第100个更新的确认点。这一规则避免把偶然单点高回报当作学习能力。')
    r.table(['回报门槛','CTBR：起点样本/时间','DR：起点样本/时间','CTBR+ID：起点样本/时间'],[
        [str(q),*[('未达到' if ms(m,q)['entry_samples'] is None else
            f"{ms(m,q)['entry_samples']/1e6:.2f}M / {ms(m,q)['entry_logged_minutes']:.2f}min") for m in MODELS]] for q in THRESHOLDS
    ],[.65,1.2,1.2,1.4])
    r.para('在共同门槛42上，ID达标区间起点仅需8.75M步，CTBR与DR分别为22.00M和22.15M，样本需求减少60.2%和60.5%。门槛40上，ID也分别减少44.8%和34.9%的样本，显示样本效率优势具有跨门槛一致性。')
    r.table(headers,[
        ['门槛42：达标区间确认点',*[f"{ms(m,42)['confirmed_samples']/1e6:.2f}M步" for m in MODELS]],
        ['确认点记录时间',*[f"{ms(m,42)['confirmed_logged_minutes']:.2f}min" for m in MODELS]],
    ],[1.65,1,1,1.3])
    r.para('把100次持续达标的确认成本计入，ID仍仅需13.70M步，较两基线少49.2%与49.4%。门槛44和45仅ID达到，进一步说明其更高的平台；主要效率结论仍建立在三者均可达到的共同门槛40和42上。')
    r.para('表内时间为日志迭代起点时刻。首次持续达标允许此后再次波动，评价的是达到该训练回报水平的学习进程；完整起点/确认点及所有门槛保存在milestones.csv。',small=True)

    r.heading('4. 计算效率：约一半记录耗时完成相同训练预算')
    r.figure('02_wall_time','图3：以记录时间为横轴，ID更早达到高回报区间，并更快完成250M环境步。')
    r.table(headers,[
        ['记录耗时 h',*[f(S[m]['logged_elapsed_h']) for m in MODELS]],
        ['完整循环平均环境步/s',*[f(S[m]['full_loop_samples_s'],0) for m in MODELS]],
        ['日志采样fps均值',*[f(S[m]['rollout_fps_mean'],0) for m in MODELS]],
        ['单更新周期中位数 s',*[f(results[m]['timing']['cycle_time_s']['median']) for m in MODELS]],
    ],[1.65,1,1,1.3])
    r.para('ID记录耗时4.61小时，CTBR与DR分别9.07和9.18小时，对应约1.97倍与1.99倍的记录时间加速。在本轮记录中，ID同时提高了训练吞吐和最终回报，体现了训练成本与训练质量的共同改善。')
    r.para('完整循环吞吐使用249950000/(末条-首条time_elapsed)，覆盖4999轮间隔，包含采样、优化与记录。日志fps仅覆盖采样阶段。时间戳位于迭代起点，最终记录未覆盖最后一整轮与保存操作；不同运行日期的硬件/负载影响见第13节。',small=True)

    r.heading('5. 最终训练质量：更高均值与更小回报波动')
    r.figure('03_tail_quality','图4：末500更新（225M—250M）的原始分布。ID回合回报分布更高且更集中，每步奖励与价值损失也呈现有利表现。')
    r.table(headers,[
        ['平均回合回报',*[f(T[m]['ep_reward_mean']['mean']) for m in MODELS]],
        ['回报标准差',*[f(T[m]['ep_reward_mean']['std']) for m in MODELS]],
        ['中位数',*[f(T[m]['ep_reward_mean']['median']) for m in MODELS]],
        ['P05 / P95',*[f"{f(T[m]['ep_reward_mean']['p05'],2)} / {f(T[m]['ep_reward_mean']['p95'],2)}" for m in MODELS]],
    ],[1.3,1,1,1.3])
    r.para('ID平均回合回报45.583，比CTBR提高3.332、比DR提高3.015；回报标准差0.400，分别降低69.0%和59.3%。结果表明ID在形成较高回报平台后，保持了更集中的训练回报。')

    r.heading('5.1 训练环境内的网络综合控制表现')
    r.para('训练日志能够反映网络输出作用于训练闭环后的任务评分。本节用rollout平均步奖励与回合回报评价综合控制目标的完成程度，并用更新间分布评价这种表现的一致性。这里观察的是训练过程中的策略，包含训练采样噪声。')
    r.figure('07_training_control','图4a：仅使用训练日志的控制表现证据。左图保留末500次更新全部平均步奖励，右图对照同一窗口分布；每个点对应一个rollout的平均奖励。')
    r.table(headers,[
        ['平均步奖励：均值',*[f(T[m]['true_reward']['mean'],6) for m in MODELS]],
        ['平均步奖励：更新间标准差',*[f(T[m]['true_reward']['std'],6) for m in MODELS]],
        ['平均步奖励：P05 / P95',*[f"{f(T[m]['true_reward']['p05'],6)} / {f(T[m]['true_reward']['p95'],6)}" for m in MODELS]],
        ['平均回合回报',*[f(T[m]['ep_reward_mean']['mean']) for m in MODELS]],
        ['完成回合平均长度 / 步',*[f(T[m]['ep_len_mean']['mean']) for m in MODELS]],
    ],[1.75,1,1,1.35])
    r.para('ID平均步奖励为0.090936，高于CTBR的0.083973和DR的0.082965。ID的5%分位数0.089348，也高于两基线的95%分位数0.086656和0.086978。平均步奖励在更新间的标准差分别降低68.2%和71.1%，表明ID在训练后期获得更高且更一致的综合任务评分。')
    r.para('按当前仓库奖励设计，r=0.1-C_task-C_terminal；C_task联合惩罚位置、姿态、线速度和角速度误差，C_terminal是终止惩罚。在奖励定义一致的前提下，平均步奖励更高对应联合惩罚更低，因而为“ID策略的训练内综合控制表现更优”提供证据。三者完成回合平均长度均接近500，ID更高的回报也没有伴随更短的平均回合。')
    r.para('综合评分允许各误差项之间存在权衡；日志没有逐项误差和动作轨迹。分项位置精度、飞行抖动、动作平顺性及固定最终权重的独立测试留待相应数据分析。历史奖励配置的可追溯性见第13节。',small=True)

    r.heading('6. 优势稳健性：更换末段窗口，结论保持一致')
    r.table(['末段窗口','CTBR均值±标准差','DR均值±标准差','CTBR+ID均值±标准差'],[
        [str(n)+'更新',*[f"{frames[m].tail(n).ep_reward_mean.mean():.3f} ± {frames[m].tail(n).ep_reward_mean.std():.3f}" for m in MODELS]] for n in (100,500,1000)
    ],[1.05,1.1,1.1,1.35])
    r.para('从末100到末1000更新，ID均保持最高回合回报和最小回报标准差；优势覆盖短、中、较长三个观察窗口。全部窗口采用同一末端，没有为某种方法单独选择较有利区间。')
    r.heading('从典型波动和低端回报再次检查',level=2,newpage=False)
    r.table(headers,[
        ['末500回报IQR',*[f(T[m]['ep_reward_mean']['iqr']) for m in MODELS]],
        ['末500回报P05',*[f(T[m]['ep_reward_mean']['p05']) for m in MODELS]],
        ['末500回报P95',*[f(T[m]['ep_reward_mean']['p95']) for m in MODELS]],
        ['末500平均每步奖励',*[f(T[m]['true_reward']['mean'],6) for m in MODELS]],
        ['末500完成回合平均长度',*[f(T[m]['ep_len_mean']['mean']) for m in MODELS]],
    ],[1.7,1,1,1.3])
    r.para('ID的IQR为0.548，相对CTBR与DR分别降低47.2%和54.9%。IQR强调中间50%的典型波动，对单个极端异常更稳健；这一结果说明ID的稳定性优势不只来自CTBR某一次大幅下跌。')
    r.para('ID的5%回报分位数为44.921，仍高于CTBR与DR的95%分位数43.42和43.90。这描述了本轮训练末段回报分布的清晰分离；多个窗口、均值、IQR与分位数共同支持ID的平台与稳定性优势。')
    r.para('DR与CTBR的差距较小，且两者在回合回报和每步奖励上顺序不同。ID在这两种口径中均领先，因此本报告的方法贡献聚焦ID相对于两条基线的一致改善。时间序列分位数为描述统计。',small=True)

    r.heading('7. 训练过程：较早进入高回报区间并长期保持')
    r.table(['训练样本区间','CTBR回报均值','DR回报均值','CTBR+ID回报均值'],[
        [f'{(a-1)*.05:g}—{b*.05:g}M',*[f(frames[m].iloc[a-1:b].ep_reward_mean.mean()) for m in MODELS]] for a,b in PHASES
    ],[1.1,1,1,1.3])
    r.para('ID的优势在5—25M区间已经建立，随后25—250M的五个阶段平均回合回报均保持在45.43—45.65附近。相同阶段的两基线处于42.38—43.21附近；ID形成了持续较高的训练平台。')
    r.table(headers,[
        ['事后最佳50更新均值',*[f(S[m]['best50_reward_mean']) for m in MODELS]],
        ['对应窗口结束更新',*[str(S[m]['best50_update']) for m in MODELS]],
        ['末500更新均值',*[f(S[m]['tail500_reward_mean']) for m in MODELS]],
        ['最后一条回报',*[f(S[m]['final_update_reward']) for m in MODELS]],
    ],[1.7,1,1,1.3])
    r.para('最终点、固定末段窗口以及各自事后最佳窗口，均呈现ID更高的训练回报。本文主要结论使用相同末500窗口，最佳窗口仅用于描述训练上限；没有据此替换权重或假定对应checkpoint仍可获得。')
    r.para('长时间训练后，三者回报都存在小幅变化，未呈现持续单调上升。ID较早获得高回报平台的特点，为缩短后续开发中的训练迭代提供了依据；具体停止时刻仍适合由独立验证集选择。')

    r.heading('8. 学习过程诊断：较好的价值拟合与训练结果')
    r.figure('04_ppo_diagnostics','图5：完整PPO训练诊断。淡线为原始数据，实线为后向50更新中位数；ID的较低价值损失与稳定回报相互支持。',width=492)
    r.para('末500更新中，ID平均价值损失0.0208，CTBR与DR为0.1974和0.1482；中位数也以ID最低。三者解释方差均约0.95—0.97。ID在维持较高训练回报的同时，价值拟合损失分布较低，反映出较有利的训练结果。具体优化读数及解释边界见第10节。')

    r.heading('9. 后期可靠性：保持更稳定的训练回报')
    r.figure('05_late_events','图6：末1000更新保留全部异常点。ID回报保持在较高区间，两基线出现更明显的局部回报下跌。',width=485)
    r.table(headers,[
        ['低于前50回报中位数超过3',*[str(results[m]['diagnostic_event_counts']['tail1000_reward_drop_gt_3']) for m in MODELS]],
        ['低于前50回报中位数超过10',*[str(results[m]['diagnostic_event_counts']['tail1000_reward_drop_gt_10']) for m in MODELS]],
    ],[1.9,1,1,1.3])
    r.para('按同一局部下跌定义，ID在末1000更新未出现超过3的回报跌幅，CTBR和DR分别有6次和17次。CTBR第4970次回报降至20.809，之后恢复；DR也保留了自身较弱更新。所有异常均计入统计，ID的稳定性结论有完整记录支撑。')

    r.heading('10. 完整优化诊断及其解释')
    r.table(headers,[
        ['末500价值损失：均值/中位',*[f"{f(T[m]['value_loss']['mean'])} / {f(T[m]['value_loss']['median'])}" for m in MODELS]],
        ['末500价值损失最大值',*[f(T[m]['value_loss']['maximum']) for m in MODELS]],
        ['末500解释方差均值',*[f(T[m]['explained_variance']['mean']) for m in MODELS]],
        ['末500近似KL：均值/中位',*[f"{f(T[m]['approxkl']['mean'])} / {f(T[m]['approxkl']['median'])}" for m in MODELS]],
        ['末500 clipfrac均值',*[f(T[m]['clipfrac']['mean']) for m in MODELS]],
    ],[1.9,1,1,1.3])
    r.table(['末1000更新的诊断计数','CTBR','CTBR+DR','CTBR+ID（本文方法）'],[
        [label,*[str(results[m]['diagnostic_event_counts'][key]) for m in MODELS]]
        for key,label in [('tail1000_kl_gt_1','approxkl > 1'),('tail1000_kl_gt_5','approxkl > 5'),
            ('tail1000_clip_gt_05','clipfrac > 0.5'),('tail1000_value_loss_gt_1','value_loss > 1'),
            ('tail1000_ev_lt_05','explained_variance < 0.5')]], [1.8,1,1,1.3])
    r.para('ID的回报稳定性和价值损失表现更好；同时其部分PPO更新代理读数较大。末1000更新中approxkl>1的次数为47，高于CTBR的28和DR的11，clipfrac>0.5的次数也更多。本轮未观察到这些读数伴随ID出现与基线同等级的回报退化，但这一现象不意味着较大KL具有必然收益。')
    r.para('本报告以任务回报、达到共同回报的成本和回报稳定性评价训练效果；KL和裁剪比例用于补充解释更新过程。它们是不同层次的观测。上述计数阈值统一用于描述日志，并非预设的算法稳定性合格线。')
    r.para('训练奖励取自每轮优化前的rollout，KL与value_loss取自随后更新过程，同编号共现不能确定因果。clipfrac是PPO概率比裁剪比例。当前实现每轮有10×10次minibatch更新，没有基于target_kl的提前终止；历史学习率配置的可追溯性见第13节。',small=True)

    r.heading('11. 探索演化：较早形成高回报的策略分布')
    r.figure('06_tail_and_entropy','图7：末段非重叠块均值与探索尺度。ID在较高回报区间保持较小块间变化，训练中的原始高斯探索尺度逐渐收缩。')
    r.para('ID的探索尺度在训练早期更快减小，同时回报较早进入高位；这一伴随变化与更早形成有效策略相一致。末500更新的10个非重叠块仍显示ID较高、较集中的回报；块均值用于描述训练演化。')
    r.table(['参考权重','原始高斯熵','四通道σ（p/q/r/比推力，tanh前）'],[
        [m,f(results[m]['checkpoint']['raw_gaussian_entropy'],4),
         ' / '.join(f(x,4) for x in results[m]['checkpoint']['raw_sigma'])] for m in MODELS
    ],[1,1,2.5])
    r.para('四维原始高斯下，几何平均σ=exp(H/4-0.5ln(2πe))。负微分熵是正常取值，σ为tanh前潜在变量的无量纲探索尺度。确定性部署使用策略均值；飞行动作平顺性需用实际动作/状态轨迹评价。权重与历史日志的对应旁证和限制见第13节。',small=True)

    r.heading('12. 面向论文的实验结论与方法价值')
    r.para('以下表述可用于论文或阶段汇报的训练实验部分：')
    r.para('在所记录的CTBR、CTBR+DR和本文CTBR+ID三组训练中，各方法均使用2.5亿环境步。ID的记录训练耗时为4.61小时，相较两基线减少49.2%和49.8%。在共同训练回报门槛42下，ID首次持续达标区间起点为8.75M步，样本需求分别减少60.2%和60.5%；计入持续区间的确认成本后，仍分别减少49.2%和49.4%。这表明ID在本轮训练中同时具有计算效率和样本效率优势。')
    r.para('在末500更新中，ID平均回合回报为45.583，比CTBR和CTBR+DR分别提高3.332与3.015；回报标准差为0.400，分别降低69.0%与59.3%。这一优势在末100、500和1000更新窗口中保持一致，IQR与分位数也呈现更集中的高回报分布。因此，ID在降低记录训练成本的同时，实现了更高且更稳定的训练回报，为理想动力学抽象用于CTBR策略训练提供了实验证据。')
    r.para('从训练环境内的综合控制目标看，ID的rollout平均步奖励为0.090936，同样高于两基线，且其低端回报分位数仍高于基线的高端分位数。这说明ID策略在训练闭环中对奖励定义的综合控制目标完成得更好，并在后期维持更一致的任务表现。该结论评价训练内的联合目标，不把联合奖励分解为未记录的单项物理误差。')
    r.heading('本轮数据支撑的三项方法贡献',level=2,newpage=False)
    r.table(['贡献','对应证据','研究价值'],[
        ['训练成本降低','约一半记录耗时完成相同预算','有助于更快开展策略与任务迭代'],
        ['学习过程更高效','共同回报门槛40/42需要更少样本','改善训练阶段环境交互的利用效率'],
        ['训练内综合控制更优','平均步奖励和回合回报同时领先，更新间波动较小','更好、更一致地完成奖励定义的联合控制目标'],
    ],[1,1.4,1.5])
    r.para('上述结论针对本次训练记录。后续可用训练轨迹分解位置、姿态、速度、角速度的控制代价，并对固定checkpoint做独立确定性测试；多种子与统一硬件复现能够进一步增强可推广性。本报告不引入后续部署实验的结果。')

    r.heading('13. 比较方法、证据范围与实验边界')
    r.para('本报告保持与初版一致的原始数据、统计窗口和门槛规则，以ID的贡献组织证据。三份CSV各5000条、14字段，n_updates完整连续，总步数均为250M；逐更新逐字段核对210000个文本日志值，与CSV在打印精度内一致，未发现缺失、非有限值或重复更新。')
    r.table(['方法','log.txt的原始目录时间标记'],[
        [m,results[m]['audit']['first_console_line'].rsplit('/',1)[-1]] for m in MODELS
    ],[1,2.5])
    r.table(['解释范围','本报告采用的口径'],[
        ['训练重复与不确定性','每种方法各1次历史运行。均值、标准差、分位数和窗口敏感性描述这三条轨迹；统计显著性与种子间泛化需多次独立训练。'],
        ['时间比较','报告日志记录耗时与完整循环吞吐。运行跨不同日期，缺少统一硬件/负载记录，故加速比为所记录运行的比较结果。'],
        ['训练分布与部署','不同方法各在其训练动力学中收集数据。本报告评价训练表现；公共物理平台上的恢复能力由独立sim-to-sim/sim-to-real试验衡量。'],
        ['历史配置追溯','原始训练没有冻结源码/YAML。当前reset固定x=0.5、pitch约90°、vy=1、yaw角速度π；不能由当前实现反推历史初态采样。'],
        ['终止与成功定义','日志长度接近500，当前同一路径未找到明确500步截断。无法据长度均值还原历史终止原因或计算悬停成功率。'],
        ['优化配置','当前train.py的数值学习率经get_schedule_fn转为常量，与README的linear decay字样不一致。历史日志未记录lr，故不假定历史使用线性衰减。'],
    ],[1,3.6])
    r.para('参考权重的高斯熵与最后日志相差约0.004—0.008，与更新内均值和更新后参数的差别相容，但缺少当时的checkpoint哈希，尚不能严格证明一一对应。模型元数据seed为null，不从当前默认42推断历史种子。',small=True)
    r.para('建议未来运行保存源码/config哈希、种子、硬件/线程、每轮学习率、终止原因和定期checkpoint，并用公共独立验证集选模型。本次只更新分析与呈现，原始日志、权重、PID和训练代码保持原值。',small=True)

    r.heading('14. 指标定义与复核说明')
    r.table(['字段','定义'],[
        ['ep_reward_mean','最近已完成回合的回报均值；当前PPO缓存100个episode。用于比较训练回报平台。'],
        ['true_reward','当前rollout内所有环境步的平均原始奖励。按步汇总，与按已完成回合汇总的统计权重不同。'],
        ['ep_len_mean','已完成回合平均步数。用于辅助理解回报与回合长度，历史终止规则的限制见第13节。'],
        ['fps','在runner.run()之后、梯度更新之前计算的采样吞吐。'],
        ['time_elapsed','本轮采样开始时刻减训练循环起点。最终日志不覆盖最后一轮完整耗时。'],
        ['value_loss / explained_variance','价值拟合损失、回报方差的解释程度；受训练数据分布与回报尺度影响。'],
        ['approxkl / clipfrac','PPO近似KL代理和概率比裁剪比例；用于诊断策略更新过程。'],
        ['policy_entropy','四维潜在高斯策略tanh前的微分熵，允许负值。'],
        ['policy_loss','PPO裁剪后的策略代理损失，接近0需结合任务回报解释。'],
    ],[1,3.5])
    r.para('当前源码奖励：r=0.1-0.02||p-p*||²-0.02||eZYX||²-0.0002||v||²-0.0002||ω||²；动作惩罚系数为0，撞地另加-10。若每回合500步且无惩罚，回报上界为50。这是对现有字段的源码解释，历史参数未冻结，报告不据此反解历史物理误差。')
    r.para('样本归一化AUC使用原始回报与真实样本轴；平滑曲线仅作展示。末段100/500/1000窗口统一取日志最后相应更新。局部下跌以本次回报低于之前50更新中位数的差定义；统计包括所有原始异常点。')
    r.para('多个更新共享策略演化与episode滑动缓存；它们是相关时间序列。因而本报告不把500个更新当作500次独立训练，不据此生成算法间置信区间。输入SHA256在生成前后核对，所有原始文件保持一致。')

    r.heading('附录：报告附件与复现')
    r.table(['位置（相对training_data）','内容'],[
        [STEM+'.pdf / .md','以ID贡献为主线的主报告，含8组图与训练内综合控制表现分析'],
        ['analysis/figures/00_id_advantages.png / .pdf','新增：ID四项优势总览，统一尺度展示'],
        ['analysis/tables/id_advantages.csv','新增：ID相对两基线的效率、回报与波动差值'],
        ['analysis/figures/07_training_control.png / .pdf','仅来自训练日志的平均步奖励曲线与分布'],
        ['analysis/tables/training_control_summary.csv / training_control_id_gains.csv','训练内综合控制评分及ID相对基线的差值'],
        ['analysis/figures/01—06_*','完整学习曲线、时间、末段分布及PPO诊断，PNG/矢量PDF'],
        ['analysis/tables/summary.csv / tail500_metrics.csv','原有核心指标与末500分布统计，口径保持一致'],
        ['analysis/tables/phases.csv / milestones.csv','阶段指标与首次持续达标起点/确认点'],
        ['analysis/tables/window_sensitivity.csv','末100/500/1000窗口敏感性'],
        ['analysis/tables/late_return_drops.csv / tail500_blocks.csv','异常事件与非重叠块统计'],
        ['analysis/analysis.json / source_manifest.json','机器可读结果、审计与全部输入/输出哈希'],
        ['analysis/build_report.py','分析/报告生成程序，不需要启动训练、ROS或无人机'],
    ],[1.7,2])
    r.para('在FlightLxx根目录复现：PYTHONDONTWRITEBYTECODE=1 ~/anaconda3/envs/flightlxx3.8/bin/python train/training_data/analysis/build_report.py。使用已有Conda中的NumPy、pandas、Matplotlib；ReportLab缺失时加载Ubuntu系统包。')
    r.heading('源码依据',level=2,newpage=False)
    r.para('train/train.py：三方法开关和当前训练配置；rl/lxx_baselines/ppo/ppo2.py：日志、时间、fps与损失公式；rl/lxx_baselines/envs/vec_env_wrapper.py：回合统计；libs/src/envs/rl_env.cpp、libs/config/rl_env.yaml：当前reset、奖励与终止；libs/src/objects/quadrotor.cpp：ID/实际动力学分支；rl/lxx_baselines/common/distributions.py：高斯熵公式。',small=True)
    r.para('上述为分析时源码，用于字段解释与机理讨论；原始历史日志没有冻结这些文件。所有数值来自本地数据；模型zip仅读取JSON/NumPy，不执行其中的序列化函数。',small=True)
    r.write()

def main():
    sources=[DATA/m/n for m in MODELS for n in ('progress.csv','log.txt')]
    raw_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    frames={m:pd.read_csv(DATA/m/'progress.csv') for m in MODELS}
    results=compute(frames)
    plot(frames)
    plot_id_overview(results)
    plot_training_control(frames)
    dump(HERE/'analysis.json',dict(date=DATE,models=results,notes=dict(
        presentation_focus='Proposed ID method: compute efficiency, sample efficiency and training quality',
        id_vs_baseline_comparisons=id_comparison(results),
        training_control_evidence=training_control_evidence(results),
        run_count_per_method=1,independent_seed_replicates_available=False,
        primary_terminal_window_updates=500,smoothing_updates=50,sustained_threshold_updates=100,
        all_outliers_retained=True,training_success_rate_not_available=True)))
    build_text(frames,results)
    assert raw_hashes=={str(p.relative_to(ROOT)):digest(p) for p in sources},'Original data changed'
    code=[ROOT/p for p in ('train/train.py','rl/lxx_baselines/ppo/ppo2.py','rl/lxx_baselines/envs/vec_env_wrapper.py',
        'libs/src/envs/rl_env.cpp','libs/src/envs/vec_env.cpp','libs/src/objects/quadrotor.cpp',
        'libs/config/rl_env.yaml','libs/config/vec_env.yaml','rl/lxx_baselines/common/distributions.py')]
    code+=[HERE/'build_report.py']
    artifacts=[DATA/(STEM+'.pdf'),DATA/(STEM+'.md'),HERE/'analysis.json']+list(TABLES.glob('*.csv'))+list(FIG.glob('*'))
    dump(HERE/'source_manifest.json',dict(report_date=DATE,original_logs_sha256=raw_hashes,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in code},
        reference_model_sha256={m:results[m]['checkpoint']['sha256'] for m in MODELS},
        original_logs_unchanged=True,
        outputs_sha256={str(p.relative_to(DATA)):digest(p) for p in artifacts}))
    print('REPORT_PDF',DATA/(STEM+'.pdf'))
    print(pd.read_csv(TABLES/'summary.csv').to_string(index=False))
    print('Raw log hashes unchanged. All 210000 console cells match CSV at printed precision.')


if __name__=='__main__':main()
