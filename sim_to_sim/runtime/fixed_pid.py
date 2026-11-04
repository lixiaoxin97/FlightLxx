#!/usr/bin/env python3
"""Apply and audit the one fixed PID in an isolated SITL process."""

import argparse
import json
import re
from pathlib import Path
import subprocess
import site
import sys

# Airframe values restored only inside the disposable private SITL rootfs.
BASELINE={
    "MC_ROLLRATE_P":"0.0165","MC_ROLLRATE_I":"0.020","MC_ROLLRATE_D":"0.0003",
    "MC_PITCHRATE_P":"0.0165","MC_PITCHRATE_I":"0.020","MC_PITCHRATE_D":"0.0003",
    "MC_YAWRATE_P":"0.018","MC_YAWRATE_I":"0.009","MC_YAWRATE_D":"0.0",
}
CONFIG = Path(__file__).resolve().parents[1] / "config/final_pid.json"
PROFILES = json.loads(CONFIG.read_text())["profiles"]
if len(PROFILES) != 1 or set(next(iter(PROFILES.values()))) != set(BASELINE):
    raise ValueError("Expected exactly one fixed nine-gain PID configuration")
PROFILE = next(iter(PROFILES))

try:
    from pyulog import ULog
except ImportError:
    sys.path.append(site.getusersitepackages())
    from pyulog import ULog


def effective_values(profile):
    if profile not in PROFILES:raise ValueError("Unknown PID profile: "+profile)
    return dict(PROFILES[profile])


def read_parameter(binary,name):
    output=subprocess.check_output([str(binary),'show','-q',name],text=True,timeout=12)
    match=re.search(r'[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?',output)
    if not match:raise RuntimeError('Cannot read '+name+': '+output)
    return float(match.group(0))


def set_verified(binary,values):
    for name,value in values.items():
        if name not in BASELINE:raise ValueError('Unexpected parameter '+name)
        subprocess.run([str(binary),'set',name,str(value)],check=True,timeout=12)
    for name,value in values.items():
        actual=read_parameter(binary,name)
        # PX4 `param show -q` prints four decimal places. A D gain of 0.00045
        # therefore reads back as 0.0005; the post-run ULog verifies the full
        # float value captured when logging started.
        if abs(actual-float(value))>5.1e-5:
            raise RuntimeError(f'Parameter readback mismatch {name}: {actual} vs {value}')


def parameters_at(ulog_path,timestamp_us):
    ulog=ULog(str(ulog_path))
    values=dict(ulog.initial_parameters)
    for stamp,name,value in sorted(ulog.changed_parameters,key=lambda row:row[0]):
        if stamp<=timestamp_us:values[name]=value
    return values


def verify_ulog_profile(ulog_path,profile,elapsed_s,tolerance=2e-6):
    values=parameters_at(ulog_path,float(elapsed_s)*1e6)
    expected={key:float(value) for key,value in effective_values(profile).items()}
    mismatches={key:(values.get(key),value) for key,value in expected.items()
                if values.get(key) is None or abs(float(values[key])-value)>tolerance}
    if mismatches:raise RuntimeError('ULog rate PID does not match profile '+profile+': '+repr(mismatches))
    return {key:float(values[key]) for key in expected}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('profile',choices=tuple(PROFILES));parser.add_argument('--shell',action='store_true')
    parser.add_argument('--px4-bin',type=Path);parser.add_argument('--record',type=Path);parser.add_argument('--restore',action='store_true')
    args=parser.parse_args()
    if args.px4_bin is None:
        values=PROFILES[args.profile] if args.shell else effective_values(args.profile)
        for name,value in values.items():print(name+'='+value)
        return
    if args.record is None:parser.error('--record required with --px4-bin')
    binary=args.px4_bin/'px4-param'
    if args.restore:
        data=json.loads(args.record.read_text())
        restore_target=data.get('restore_target',BASELINE)
        set_verified(binary,restore_target)
        subprocess.run([str(binary),'save'],check=True,timeout=20)
        verified={name:read_parameter(binary,name) for name in restore_target}
        for name,value in restore_target.items():
            if abs(verified[name]-float(value))>5.1e-5:raise RuntimeError('Saved PID baseline readback mismatch: '+name)
        data['restored_values']=verified;data['restored_verified']=True;data['baseline_saved_to_sitl_rootfs']=True
        args.record.write_text(json.dumps(data,indent=2));return
    if args.record.exists():raise RuntimeError('Refusing to overwrite PID original-parameter record')
    original={name:read_parameter(binary,name) for name in BASELINE}
    data=dict(profile=args.profile,original_at_process_start=original,restore_target=BASELINE,
              requested=effective_values(args.profile),restored_verified=False)
    args.record.write_text(json.dumps(data,indent=2))
    set_verified(binary,data['requested']);data['applied_verified']=True
    args.record.write_text(json.dumps(data,indent=2))


if __name__=='__main__':main()
