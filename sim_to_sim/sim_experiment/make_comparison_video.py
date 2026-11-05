#!/usr/bin/env python3
"""Synchronized three-policy videos reconstructed from recorded Gazebo truth.

This is a scientific replay of the actual accepted trials, not a fresh flight
or a Gazebo screen recording. All panels use simulation time since release.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from result_io import save,digest
from five_level_study import collect,MODELS
from recovery_state import quaternion_multiply,rotate_vector

COLORS={'CTBR':'#3973AC','CTBR+DR':'#D48927','CTBR+ID':'#23836F'}
RULES={
    'common_success':'All three valid trials succeed; select request nearest median normalized initial severity, ties by state ID.',
    'id_only_success':'Only ID succeeds, CTBR and DR both fail; select nearest median initial severity in this explicit outcome-defined category.',
    'difficult':'All three valid trials available in L5; select severity nearest the 90th percentile of ALL L5 requests, without using policy success.',
    'goal_hold':'Exact goal repeats; choose smallest SHA256(seed=2026110403 + state_id), independent of policy outcomes.'}


def severity(state):
    x=state['state'];p=np.linalg.norm([x[0],x[1],x[2]-5])/math.sqrt(3)
    return float((p+state['requested_body_z_tilt_deg']/180+np.linalg.norm(x[6:9])/math.sqrt(3)+np.linalg.norm(x[9:])/math.sqrt(3))/4)


def select_cases(manifest,rows):
    by_state={}
    for row in rows:by_state.setdefault(row['state_id'],{})[row['model']]=row
    states=manifest['state_bank']['states']
    complete=[s for s in states if set(by_state.get(s['id'],{}))==set(MODELS)]
    target90=float(np.quantile([severity(s) for s in states if s['level']==5],.9))
    groups={
        'common_success':[s for s in complete if all(by_state[s['id']][m]['success'] for m in MODELS)],
        'id_only_success':[s for s in complete if by_state[s['id']]['CTBR+ID']['success'] and
            not by_state[s['id']]['CTBR']['success'] and not by_state[s['id']]['CTBR+DR']['success']],
        'difficult':[s for s in complete if s['level']==5],
        'goal_hold':[s for s in complete if s.get('category')=='exact_goal']}
    cases={}
    for category,candidates in groups.items():
        if not candidates:
            cases[category]=dict(available=False,category=category,rule=RULES[category]);continue
        if category=='goal_hold':
            state=min(candidates,key=lambda s:hashlib.sha256(('2026110403:'+s['id']).encode()).hexdigest())
        else:
            target=target90 if category=='difficult' else float(np.median([severity(s) for s in candidates]))
            state=min(candidates,key=lambda s:(abs(severity(s)-target),s['id']))
        group=by_state[state['id']]
        cases[category]=dict(available=True,category=category,rule=RULES[category],candidate_count=len(candidates),
            state=state,severity=severity(state),results={m:dict(success=group[m]['success'],
                trial_id=group[m]['trial_id'],run_directory=group[m]['run_directory'],
                termination_reason=group[m]['termination_reason']) for m in MODELS})
    return cases


def load_trace(run):
    run=Path(run);report=json.loads((run/'trial_result.json').read_text())
    release=report.get('release_sim_time_s');rows=[]
    for line in (run/'trial_trace.jsonl').open():
        row=json.loads(line)
        if row.get('event')!='tick' or row.get('phase')!='flight' or row['truth']['holding']:continue
        truth=row['truth'];origin=release if release is not None else truth.get('released_sim_time_s')
        if origin is None:continue
        t=truth['sim_time_s']-origin
        if -1e-7<=t<=10.04 and (not rows or t>rows[-1]['t']+1e-9):
            enu=truth['position_enu'];v=truth['velocity_enu'];w=truth['angular_velocity_world_enu']
            q=quaternion_multiply((math.sqrt(.5),0,0,-math.sqrt(.5)),truth['quaternion_enu_flu_wxyz'])
            rows.append(dict(t=max(0.,min(10.,t)),p=[enu[1],-enu[0],enu[2]],
                q=list(q),speed=float(np.linalg.norm(v)),rate=float(np.linalg.norm(w))))
    if not rows:raise RuntimeError('No free-flight truth trace: '+str(run))
    t=np.array([r['t'] for r in rows]);p=np.array([r['p'] for r in rows]);q=np.array([r['q'] for r in rows])
    for i in range(1,len(q)):
        if np.dot(q[i-1],q[i])<0:q[i]*=-1
    return dict(t=t,p=p,q=q,error=np.linalg.norm(p-[0,0,5],axis=1),
        speed=np.array([r['speed'] for r in rows]),rate=np.array([r['rate'] for r in rows]),
        end=float(t[-1]),reason=report['reason'],trace_sha256=digest(run/'trial_trace.jsonl'))


def render(case,path,fps=24,width=1920,height=1080,preview=False):
    traces={m:load_trace(case['results'][m]['run_directory']) for m in MODELS}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(width/120,height/120),dpi=120,facecolor='#F5F7FA')
    grid=fig.add_gridspec(5,3,height_ratios=[3.4,1.5,1,1,1],hspace=.42,wspace=.25,
        left=.055,right=.985,top=.89,bottom=.055)
    dynamic={};camera=(24.,-60.)
    allp=np.vstack([x['p'] for x in traces.values()]+[np.array([[0,0,5]])])
    lo=allp.min(axis=0)-.45;hi=allp.max(axis=0)+.45
    center=(lo+hi)/2;span=max(1.4,float((hi-lo).max()))
    lo=center-span/2;hi=center+span/2
    # Same viewing volume and camera for all networks; no per-policy zoom.
    metric_limits={key:max(.35,float(max(x[key].max() for x in traces.values()))*1.12) for key in ('error','speed','rate')}
    rotor_offsets=np.array([[.0884,.0884,0],[-.0884,-.0884,0],[.0884,-.0884,0],[-.0884,.0884,0]])
    for column,model in enumerate(MODELS):
        color=COLORS[model];data=traces[model]
        ax=fig.add_subplot(grid[0,column],projection='3d');ax.view_init(*camera)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),
            xlabel='NWU x (m)',ylabel='NWU y (m)',zlabel='z (m)')
        ax.set_box_aspect((1,1,1))
        ax.set_title(model+(' - proposed' if model=='CTBR+ID' else ''),color=color,fontweight='bold',fontsize=13,pad=3)
        ax.scatter([0],[0],[5],c='#239B56',marker='*',s=90,label='Goal')
        circle=np.linspace(0,2*np.pi,80)
        for axis in (0,1,2):
            points=np.zeros((80,3));points[:,2]=5;others=[i for i in range(3) if i!=axis]
            points[:,others[0]]+=.3*np.cos(circle);points[:,others[1]]+=.3*np.sin(circle)
            ax.plot(*points.T,color='#239B56',alpha=.22,lw=.65)
        trail,=ax.plot([],[],[],color=color,lw=1.6,alpha=.85)
        arms=[ax.plot([],[],[],color=color,lw=2.5)[0] for _ in range(2)]
        rotors=ax.plot([],[],[],'o',color=color,ms=4)[0]
        status=ax.text2D(.01,.96,'',transform=ax.transAxes,color=color,fontsize=8)
        zoom=fig.add_subplot(grid[1,column]);zoom.set_aspect('equal',adjustable='box')
        zoom.set(xlim=(-.20,.20),ylim=(-.20,.20),xlabel='x error (m)',ylabel='y error (m)')
        zoom.add_patch(Circle((0,0),.10,fill=False,ec='#999999',ls='--',lw=.8))
        zoom.add_patch(Circle((0,0),.05,fill=False,ec='#239B56',ls=':',lw=1))
        zoom.plot(0,0,'*',color='#239B56',ms=8);zoom.grid(alpha=.2)
        zoom.set_title('Target detail: fixed +/- 0.20 m',fontsize=9)
        zoomline,=zoom.plot([],[],color=color,lw=1);zoompoint,=zoom.plot([],[],'o',color=color,ms=4)
        zoomtext=zoom.text(.01,.92,'Appears at t = 8 s',transform=zoom.transAxes,fontsize=7)
        cursors=[]
        for row,(key,label) in enumerate((('error','Position error (m)'),('speed','Speed (m/s)'),('rate','Body-rate norm (rad/s)')),2):
            curve=fig.add_subplot(grid[row,column]);curve.plot(data['t'],data[key],color=color,lw=1)
            curve.set(xlim=(0,10),ylim=(0,metric_limits[key]),ylabel=label)
            curve.axvspan(8,10,color='#239B56',alpha=.07)
            if key in ('error','speed'):curve.axhline(.3,color='#777777',ls='--',lw=.7)
            curve.grid(alpha=.2);cursor=curve.axvline(0,color='#111111',lw=.8);cursors.append(cursor)
            if row==4:curve.set_xlabel('Simulation time since release (s)')
        dynamic[model]=(trail,arms,rotors,status,zoomline,zoompoint,zoomtext,cursors)
    title=fig.suptitle('',y=.98,fontsize=14,fontweight='bold')
    fig.text(.5,.935,'Recorded Gazebo truth replay | same requested state, camera and time | curves include full recorded flight',ha='center',fontsize=10)
    fig.text(.5,.912,case['category']+' | '+case['state']['id']+' | terminal 8-10 s at 0.5x',ha='center',fontsize=9)
    if preview:times=np.linspace(0,10,31)
    else:times=np.r_[np.arange(0,8,1/fps),np.linspace(8,10,4*fps+1)]
    writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=5000,
        extra_args=['-pix_fmt','yuv420p','-preset','fast','-threads','2'],metadata={'title':'Three-network synchronized Gazebo truth replay'})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with writer.saving(fig,str(path),dpi=120):
        for t in times:
            title.set_text(f"Same-state recovery | simulation t = {t:5.2f} s | level L{case['state']['level']}")
            for model in MODELS:
                d=traces[model];trail,arms,rotors,status,zl,zp,zt,cursors=dynamic[model]
                time=min(t,d['end']);pos=np.array([np.interp(time,d['t'],d['p'][:,i]) for i in range(3)])
                q=np.array([np.interp(time,d['t'],d['q'][:,i]) for i in range(4)]);q/=np.linalg.norm(q)
                visible=d['t']<=time+1e-9;trail.set_data_3d(*d['p'][visible].T)
                ends=np.array([pos+rotate_vector(q,offset) for offset in rotor_offsets])
                for line,indices in zip(arms,((0,1),(2,3))):line.set_data_3d(*ends[list(indices)].T)
                rotors.set_data_3d(*ends.T)
                terminated=t>d['end']+.04
                status.set_text(('TERMINATED '+f"at {d['end']:.2f}s: " if terminated else '')+
                    (d['reason'] if terminated else f"error {np.linalg.norm(pos-[0,0,5]):.3f} m"))
                status.set_color('#B42318' if terminated else COLORS[model])
                if t>=8 and not terminated:
                    ix=(d['t']>=8)&visible;zl.set_data(d['p'][ix,0],d['p'][ix,1]);zp.set_data([pos[0]],[pos[1]])
                    zt.set_text('Outside detail view' if max(abs(pos[0]),abs(pos[1]))>.2 else f'z error {pos[2]-5:+.3f} m')
                elif t>=8:
                    zl.set_data([],[]);zp.set_data([],[]);zt.set_text('Trial ended; no invented terminal data')
                for cursor in cursors:cursor.set_xdata([t,t])
            writer.grab_frame()
    plt.close(fig)
    return dict(path=str(path),sha256=digest(path),frames=len(times),fps=fps,
        trace_sha256={m:traces[m]['trace_sha256'] for m in MODELS},
        display='Gazebo truth scientific replay, not renderer screen capture',
        alignment='time since each accepted trial release; no trajectory time warping',
        termination_handling='freeze last measured geometry, label termination, omit unavailable terminal zoom')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--category',choices=tuple(RULES),nargs='+',default=list(RULES))
    parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args();manifest,rows=collect(args.study)
    cases=select_cases(manifest,rows)
    if args.list:print(json.dumps(cases,indent=2,ensure_ascii=False));return
    out=(args.output or args.study/'videos').resolve();out.mkdir(parents=True,exist_ok=True)
    save(out/'selection_manifest.json',dict(rules=RULES,cases=cases,
        selection_seed=2026110403,not_used_for_estimation=True,original_manifest_sha256=digest(args.study/'manifest.json')))
    outputs={}
    for name in args.category:
        if not cases[name]['available']:
            outputs[name]=dict(available=False);continue
        outputs[name]=render(cases[name],out/(name+('_preview' if args.preview else '')+'.mp4'),fps=args.fps,preview=args.preview)
        save(out/'video_manifest.json',outputs)
        print('VIDEO',name,outputs[name]['path'],flush=True)


if __name__=='__main__':main()
