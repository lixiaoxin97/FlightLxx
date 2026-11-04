"""Fixed-PID paired experiments; fresh isolated simulator for every attempt."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

from analyze_final_hover import analyze
from final_hover_metrics import PROTOCOL
from fixed_pid import BASELINE, PROFILE, PROFILES, verify_ulog_profile
from sitl_runtime import prepare_slot, command, environment, gui_config
from result_io import save, digest, audit_cleanup_warning
from recovery_state import validate_training_state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ("CTBR", "CTBR+DR", "CTBR+ID")
ORDERS = ((0,1,2),(1,2,0),(2,0,1),(2,1,0),(0,2,1),(1,0,2))


def schedule(bank, models=MODELS):
    states = bank["states"]
    if not states or len({s["id"] for s in states}) != len(states):
        raise ValueError("Initial-state list must be nonempty with unique IDs")
    if any(m not in MODELS for m in models) or not models or len(set(models)) != len(models):
        raise ValueError("Unknown or duplicate policy")
    rows = []
    for block, state in enumerate(states):
        if not state["id"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("Invalid state ID")
        validate_training_state(state["state"])
        for index in ORDERS[block % len(ORDERS)]:
            model = MODELS[index]
            if model not in models:
                continue
            rows.append(dict(trial_id=f'{len(rows)+1:05d}_{state["id"]}_{model.replace("+","_")}_{PROFILE}',
                ordinal=len(rows)+1, state_id=state["id"], state=state["state"], model=model,
                profile=PROFILE, stratum=state.get("stratum"), category=state.get("category"), level=state.get("level"),
                initializer="trajectory", trajectory_lead_s=.35))
    return rows


def hashes(files):
    return {str(p): digest(p) for p in files}


def frozen_files(px4, bank_path, seed):
    files = list(HERE.glob("*.py")) + [HERE/"launch_trial.sh", ROOT/"experiment.py",
        ROOT/"config/final_pid.json", ROOT/"config/policy_equivalence.json", Path(bank_path).resolve(),
        ROOT.parent/"Simulation_Experiments/recovery_metrics.py", ROOT.parent/"libs/config/prototype.yaml",
        px4/"build/px4_sitl_default/bin/px4", ROOT/"build/state_injector/libflightlxx_state_injector.so",
        ROOT/"build/state_injector/flightlxx_state_injector_cli",
        px4/"src/modules/ekf2/EKF/aid_sources/external_vision/ev_vel_control.cpp",
        px4/"Tools/simulation/gz/worlds/default.sdf", px4/"ROMFS/px4fmu_common/init.d/rc.mc_defaults"]
    files += list((ROOT/"gazebo").rglob("*.sdf")) + list((ROOT/"gazebo/plugins").glob("*.cpp"))
    files += list((ROOT/"px4_airframes").glob("*_gz_*"))
    files += [ROOT.parent/"Simulation_Experiments/models"/(m+".zip") for m in MODELS]
    files += list(seed.glob("parameters*.bson"))
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise RuntimeError("Required runtime files missing: " + str(missing))
    return list(dict.fromkeys(files))


def run_one(output,px4,spec,gui=None):
    d=output/'trials'/spec['trial_id'];done=d/'completed.json'
    if done.exists():return json.loads(done.read_text())
    if d.exists():
        # An interrupted attempt is retained, never overwritten or silently retried.
        result=analyze(d/'run',spec)
        result.update(spec=spec,returncode=None,reconstructed_after_interruption=True)
        result.update(success=False,final_success=False,valid_policy_trial=False,failure_category='communication_mode_interruption')
        save(done,result);return result
    d.mkdir(parents=True);save(d/'requested_trial.json',spec)
    runtime=prepare_slot(d/'runtime',px4,output/'parameter_seed',spec.get('initializer','trajectory'),gui=gui)
    start=time.monotonic();timeout=False
    with (d/'launcher.log').open('w') as log:
        p=subprocess.Popen(command(runtime,px4,spec['state'],gui=bool(gui)),env=environment(d/'run',spec,60+(spec['ordinal']%100),gui=gui),
            cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    result=analyze(d/'run',spec)
    if timeout:result.update(success=False,final_success=False,valid_policy_trial=False,failure_category='communication_mode_interruption')
    report_path=d/'run/trial_result.json';report=json.loads(report_path.read_text()) if report_path.exists() else {}
    release=report.get('release_sim_time_s');ulog=next((d/'run/recording').rglob('*.ulg'),None)
    if release is not None and ulog:
        try:
            result['effective_pid']=verify_ulog_profile(ulog,spec['profile'],release+.1)
            verify_ulog_profile(ulog,spec['profile'],release+min(8.,result['metrics'].get('observed_duration_s',8.)*.9))
            result['effective_pid_verified']=True
        except RuntimeError as exc:
            result.update(effective_pid_verified=False,pid_audit_error=str(exc),success=False,final_success=False,
                          valid_policy_trial=False,failure_category='communication_mode_interruption')
    else:
        result['effective_pid_verified']=False
        if release is not None:
            result.update(success=False,final_success=False,valid_policy_trial=False,
                          failure_category='communication_mode_interruption',pid_audit_error='flight_ulog_missing')
    rec=d/'run/pid_override_record.json';restore=json.loads(rec.read_text()) if rec.exists() else {}
    result['baseline_restored']=bool(restore.get('restored_verified') and restore.get('baseline_saved_to_sitl_rootfs') and
        restore.get('restored_values')=={k:float(v) for k,v in BASELINE.items()})
    result.update(spec=spec,returncode=code,watchdog_timeout=timeout,wall_duration_s=time.monotonic()-start,
        runtime_isolation='private PID/network/mount namespace, private rootfs and /tmp',postflight_landing_scored=False)
    save(done,result)
    # ULogs and ROS/trace data are saved under run/. Delete only the duplicate
    # rootfs logs produced by this disposable, stopped simulator.
    for p in (runtime/'rootfs/fs/log',runtime/'rootfs/log'):
        if p.exists():shutil.rmtree(p)
    if not report and result['wall_duration_s']<5.:
        raise RuntimeError('Simulator bootstrap failed before flight; stop this batch: '+str(d/'launcher.log'))
    return result


def run_batch(bank_path, output, *, workers=12, models=MODELS, max_trials=None,
              parameter_seed=None, resume=False, gui=False, pause=3.):
    bank_path, output = Path(bank_path).resolve(), Path(output).resolve()
    if output.is_relative_to(Path("/tmp")):
        raise ValueError("Use an output directory under sim_to_sim/logs; isolated /tmp hides host /tmp files")
    bank = json.loads(bank_path.read_text())
    plan = schedule(bank, models)
    px4 = Path(os.environ.get("S2S_PX4_DIR", Path.home()/"PX4-Autopilot")).resolve()
    if not 1 <= workers <= 12 or (gui and workers != 1):
        raise ValueError("Use 1..12 workers; Gazebo viewing requires one worker")
    prepared_view = gui and output.is_dir() and {p.name for p in output.iterdir()} == {"view_states.json"}
    if output.exists() and not resume and not prepared_view:
        raise ValueError("Output exists; choose a new directory or explicitly --resume")
    if resume and not (output/"manifest.json").is_file():
        raise ValueError("Cannot resume a directory without its experiment manifest")
    check = json.loads((ROOT/"config/policy_equivalence.json").read_text())
    for model in models:
        path = ROOT.parent/"Simulation_Experiments/models"/(model+".zip")
        if digest(path) != check[model]["checkpoint_sha256"] or check[model]["max_absolute_output_error"] >= 1e-5:
            raise RuntimeError("Checkpoint no longer matches validated NumPy inference: " + model)
    output.mkdir(parents=True, exist_ok=True)
    seed = output/"parameter_seed"
    if not seed.exists():
        seed.mkdir()
        original = Path(parameter_seed) if parameter_seed else px4/"build/px4_sitl_default/rootfs/fs"
        for path in original.glob("parameters*.bson"):
            shutil.copy2(path, seed/path.name)
        if not list(seed.glob("parameters*.bson")):
            raise RuntimeError("No SITL parameter seed found")
    manifest = output/"manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        if m["schedule"] != plan or m.get("gui", False) != gui:
            raise RuntimeError("State list, policies or execution mode changed; use a fresh directory")
        if hashes([Path(k) for k in m["frozen_sha256"]]) != m["frozen_sha256"]:
            raise RuntimeError("Frozen inputs changed; use a fresh directory")
    else:
        files = frozen_files(px4, bank_path, seed)
        m = dict(created_utc=datetime.now(timezone.utc).isoformat(), state_bank=bank, schedule=plan,
                 frozen_sha256=hashes(files), protocol=PROTOCOL, worker_count=workers,
                 profiles=[PROFILE], fixed_pid=PROFILES[PROFILE], all_failures_retained=True,
                 automatic_retries=False, numpy_validation=check, simulator_speed_factor=1., gui=gui,
                 postflight_cleanup="discard simulator after complete scored flight")
        save(manifest, m)
        save(output/"state_bank.json", bank)
        for index, path in enumerate(files):
            dest = output/"source_snapshot"/f"{index:02d}_{path.name}"
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(path, dest)
    gui_path = output/"gazebo_gui.config" if gui else None
    if gui:
        gui_config(gui_path)
    small = {k:v for k,v in m["frozen_sha256"].items() if Path(k).stat().st_size < 2_000_000}
    selected = plan[:max_trials] if max_trials else plan
    results, pending = [], []
    for spec in selected:
        path = output/"trials"/spec["trial_id"]/"completed.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            pending.append(spec)
    started = time.monotonic()
    lock, live = threading.Lock(), set()

    def progress():
        with lock:
            active = sorted(live)
        save(output/"progress.json", dict(completed=len(results), selected_total=len(selected), total=len(plan),
             running=active, counts=dict(Counter(r.get("failure_category") or "success" for r in results)),
             workers=workers, finished=len(results)==len(plan), selection_finished=len(results)==len(selected),
             wall_s=time.monotonic()-started))

    def invoke(spec):
        if hashes([Path(k) for k in small]) != small:
            raise RuntimeError("Frozen inputs changed during experiment")
        with lock:
            live.add(spec["trial_id"])
        try:
            return run_one(output, px4, spec, gui=gui_path)
        finally:
            with lock:
                live.remove(spec["trial_id"])

    progress()
    try:
        # GUI runs sequentially. Headless runs use the same isolated per-flight workflow.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            iterator, futures = iter(pending), {}
            for _ in range(workers):
                spec = next(iterator, None)
                if spec:
                    futures[pool.submit(invoke, spec)] = spec
            while futures:
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    futures.pop(future)
                    result = audit_cleanup_warning(future.result())[0]
                    results.append(result)
                    progress()
                    print(f'[{len(results)}/{len(selected)}] {result["trial_id"]}: '
                          f'{result.get("failure_category") or "success"}', flush=True)
                    if gui:
                        log = Path(result["run_directory"])/"gazebo_gui.txt"
                        if not log.exists():
                            raise RuntimeError("Gazebo GUI failed to start; inspect launcher.log")
                        if pause:
                            time.sleep(pause)
                    spec = next(iterator, None)
                    if spec:
                        futures[pool.submit(invoke, spec)] = spec
    finally:
        save(output/"results.json", sorted(results, key=lambda r:r["spec"]["ordinal"]))
        progress()
    if hashes([Path(k) for k in m["frozen_sha256"]]) != m["frozen_sha256"]:
        raise RuntimeError("Frozen inputs changed during experiment")
    print("EXPERIMENT_COMPLETE", len(results), "wall_s", time.monotonic()-started, flush=True)

