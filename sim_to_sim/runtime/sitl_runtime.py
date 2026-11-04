"""Private PX4 rootfs, PID/network namespaces and optional Gazebo window."""
import os
from pathlib import Path
import shlex
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def prepare_slot(folder, px4, parameter_seed=None, initializer="trajectory", gui=None):
    if initializer != "trajectory":
        raise ValueError("The maintained workflow uses trajectory initialization")
    folder = Path(folder).resolve()
    source = Path(px4).resolve() / "build/px4_sitl_default/rootfs"
    root = folder / "rootfs"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("etc", "test_data"):
        (root / name).symlink_to((source / name).resolve())
    fs = root / "fs"
    fs.mkdir()
    seed = Path(parameter_seed) if parameter_seed is not None else source / "fs"
    for path in seed.glob("parameters*.bson"):
        shutil.copy2(path, fs / path.name)
    (folder / "tmp").mkdir()
    program = (HERE / "launch_trial.sh").read_text()
    program = "\n".join("SCRIPT_DIR=" + shlex.quote(str(HERE)) if line.startswith("SCRIPT_DIR=") else line
                        for line in program.splitlines()) + "\n"
    if gui is not None:
        (folder / "tmp/.X11-unix").mkdir()
        ready = "(( PX4_READY )) || die 'PX4 command client did not become ready'"
        assert program.count(ready) == 1
        program = program.replace(ready, ready + '''
gz sim -g --gui-config "$S2S_VIEW_GUI_CONFIG" > "$RUN_DIR/gazebo_gui.txt" 2>&1 &
VIEW_GUI_PID=$!
sleep 4
kill -0 "$VIEW_GUI_PID" 2>/dev/null || die 'Gazebo GUI exited; inspect gazebo_gui.txt'
timeout 5 gz topic -t /gui/track -m gz.msgs.CameraTrack \\
    -p "track_mode: FOLLOW, follow_target: {name: 'prototype_recovery_0'}, follow_offset: {x: -1.8, y: -1.8, z: 1.0}, follow_pgain: 1.0, track_pgain: 1.0" \\
    > "$RUN_DIR/camera_follow.txt" 2>&1 || true
''')
    (folder / "launcher.sh").write_text(program)
    return folder


def command(folder, px4, state, gui=False):
    folder, px4 = Path(folder).resolve(), Path(px4).resolve()
    args = ["bwrap", "--unshare-user", "--unshare-pid", "--unshare-net", "--die-with-parent",
            "--dev-bind", "/", "/", "--bind", str(folder / "rootfs"),
            str(px4 / "build/px4_sitl_default/rootfs"), "--bind", str(folder / "tmp"), "/tmp"]
    if gui:
        args += ["--bind", "/tmp/.X11-unix", "/tmp/.X11-unix"]
    return args + ["--tmpfs", "/dev/shm", "--proc", "/proc", "--", "bash", str(folder / "launcher.sh")] + [
        format(value, ".17f") for value in state]


def environment(run, spec, domain=120, gui=None):
    env = os.environ.copy()
    env.update(S2S_RUN_DIR=str(Path(run).resolve()), S2S_TRIAL_ID=spec["trial_id"],
               S2S_PX4_DIR=str(Path(os.environ.get("S2S_PX4_DIR", Path.home() / "PX4-Autopilot")).resolve()),
               S2S_HEADLESS="1", HEADLESS="1", S2S_TRIAL_DURATION="10", S2S_POLICY_PORT="5801",
               S2S_POLICY_PYTHON="/usr/bin/python3",
               S2S_MODEL=str(ROOT.parent / "train/models" / (spec["model"] + ".zip")),
               S2S_ATTITUDE_ENCODING="principal_zyx", S2S_PID_PROFILE=spec["profile"],
               S2S_BATCH_DISCARD_SIMULATOR="1", S2S_TRAJECTORY_LEAD_S="0.35",
               ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY="1", GZ_IP="127.0.0.1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if gui:
        env.update(S2S_VIEW_GUI_CONFIG=str(gui), QT_X11_NO_MITSHM="1", QT_QPA_PLATFORM="xcb", PX4_GZ_NO_FOLLOW="1")
    for name in ("PX4_SIM_SPEED_FACTOR", "TMUX", "TMUX_PANE", "TMUX_TMPDIR"):
        env.pop(name, None)
    return env


def gui_config(path):
    source = Path("/usr/share/gz/gz-sim8/gui/gui.config").read_text()
    source = source.replace('show_again="true"', 'show_again="false"')
    source = source.replace("<width>1000</width>", "<width>1280</width>", 1)
    source = source.replace("<height>845</height>", "<height>900</height>", 1)
    source = source.replace("<dialog_on_exit>true</dialog_on_exit>", "<dialog_on_exit>false</dialog_on_exit>")
    source = source.replace("<camera_pose>-6 0 6 0 0.5 0</camera_pose>",
                            "<camera_pose>-3 -3 7 0 0.44 0.785398</camera_pose>")
    Path(path).write_text(source)
