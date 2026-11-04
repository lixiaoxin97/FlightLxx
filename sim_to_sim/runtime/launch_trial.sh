#!/usr/bin/env bash
# Private fixed-PID neural recovery trial, prepared by sitl_runtime.py.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${S2S_PX4_DIR:-$HOME/PX4-Autopilot}"
PX4_BIN="$PX4_DIR/build/px4_sitl_default/bin"
ROS_SETUP="${S2S_ROS_SETUP:-/opt/ros/humble/setup.bash}"
ROS_WS_SETUP="${S2S_ROS2_WS_SETUP:-$HOME/px4_ros2_ws/install/setup.bash}"
ROS_PYTHON="${S2S_ROS_PYTHON:-/usr/bin/python3}"
STATE=("$@")
read -r -a SPAWN_STATE <<< "$("$ROS_PYTHON" "$SCRIPT_DIR/full_random_initialization.py" spawn --lead 0.35 --state "${STATE[@]}" --shell)"
ATTITUDE_ENCODING="${S2S_ATTITUDE_ENCODING:-principal_zyx}"
case "$ATTITUDE_ENCODING" in eigen_zyx|principal_zyx) ;; *) printf 'Invalid attitude encoding\n' >&2; exit 2 ;; esac
SESSION="flightlxx_init_probe_$$"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
RUN_DIR="${S2S_RUN_DIR:-$SCRIPT_DIR/../logs/recovery_$RUN_ID}"
REPORT="$RUN_DIR/initial_state.json"
SESSION_CREATED=0
AGENT_PID=""
RECORDER_PID=""
OBSERVER_PID=""
VISION_PID=""
POLICY_PID=""
PX4_READY=0
PASS=0
export GZ_PARTITION="flightlxx_recovery_$RUN_ID"

usage() { printf 'Internal launcher: x y z roll pitch yaw vx vy vz wx wy wz\n'; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
[[ $# -eq 12 ]] || { usage >&2; exit 2; }
[[ -r "$ROS_SETUP" && -r "$ROS_WS_SETUP" ]] || die 'ROS setup files are missing'
[[ -x "$PX4_BIN/px4" ]] || die 'PX4 SITL build is missing'
if pgrep -x px4 >/dev/null 2>&1; then die 'Stop the existing PX4 SITL before this probe'; fi
set +u
# shellcheck source=/dev/null
source "$ROS_SETUP"
# shellcheck source=/dev/null
source "$ROS_WS_SETUP"
set -u
command -v tmux >/dev/null || die 'tmux not found'
command -v MicroXRCEAgent >/dev/null || die 'MicroXRCEAgent not found'
# Validate all state values before starting processes.
POSE_LINE="$("$ROS_PYTHON" "$SCRIPT_DIR/full_random_initialization.py" pose --state "${SPAWN_STATE[@]}" --shell | sed -n 's/^PX4_GZ_MODEL_POSE=//p')"
[[ "$POSE_LINE" =~ ^[-0-9.,eE+]+$ ]] || die 'Invalid initial state or spawn pose'
mkdir -p "$RUN_DIR"
COMMAND_TOPIC=/world/default/model/prototype_recovery_0/flightlxx/set_state
ACK_TOPIC=/world/default/model/prototype_recovery_0/flightlxx/state_ack

px4_cmd() {
    local module="$1"
    shift
    printf '\n>>> %s %s\n' "$module" "$*" | tee -a "$RUN_DIR/px4_commands.txt"
    timeout 12 "$PX4_BIN/px4-$module" "$@" 2>&1 | tee -a "$RUN_DIR/px4_commands.txt"
}
stop_child() {
    local pid="$1"
    [[ -n "$pid" ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        kill -INT "$pid" 2>/dev/null || true
        for _ in $(seq 1 200); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.1
        done
        kill -TERM "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    # Capture unexpected process exits before deliberately stopping children.
    local child_name child_pid child_rc
    local -a unexpected=()
    for child_name in vision policy recorder agent; do
        case "$child_name" in
            vision) child_pid="$VISION_PID" ;;
            policy) child_pid="$POLICY_PID" ;;
            recorder) child_pid="$RECORDER_PID" ;;
            agent) child_pid="$AGENT_PID" ;;
        esac
        if [[ -n "$child_pid" ]] && ! kill -0 "$child_pid" 2>/dev/null; then
            child_rc=0
            wait "$child_pid" || child_rc=$?
            unexpected+=("$child_name" "$child_rc")
            printf 'Unexpected child exit: %s status=%s\n' "$child_name" "$child_rc" >&2
        fi
    done
    if [[ -d "$RUN_DIR" ]]; then
        "$ROS_PYTHON" - "$RUN_DIR/process_health.json" "${unexpected[@]}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"unexpected_exits": [
    {"process": name, "returncode": int(code)}
    for name, code in zip(sys.argv[2::2], sys.argv[3::2])
]}, indent=2))
PY
    fi
    stop_child "$OBSERVER_PID"
    stop_child "$VISION_PID"
    stop_child "$POLICY_PID"
    if (( PX4_READY )) && [[ -f "$RUN_DIR/pid_override_record.json" ]]; then
        "$ROS_PYTHON" "$SCRIPT_DIR/fixed_pid.py" "${S2S_PID_PROFILE}" \
            --px4-bin "$PX4_BIN" --record "$RUN_DIR/pid_override_record.json" --restore \
            > "$RUN_DIR/pid_restore.txt" 2>&1 || exit_code=1
    fi
    if (( PX4_READY )); then
        px4_cmd logger off || true
        px4_cmd logger status || true
    fi
    if (( SESSION_CREATED )); then
        tmux capture-pane -pt "$SESSION:px4" -S -3000 > "$RUN_DIR/px4_console.txt" 2>/dev/null || true
        tmux send-keys -t "$SESSION:px4" C-c 2>/dev/null || true
        for _ in $(seq 1 100); do
            if ! timeout 1 "$PX4_BIN/px4-commander" status >/dev/null 2>&1; then break; fi
            sleep 0.1
        done
        tmux kill-session -t "$SESSION" 2>/dev/null || true
    fi
    # Finalize the recorder after PX4 closes its ULog.
    stop_child "$RECORDER_PID"
    stop_child "$AGENT_PID"
    if (( exit_code != 0 || !PASS )); then
        printf 'Probe incomplete. Inspect %s\n' "$RUN_DIR" >&2
        (( exit_code != 0 )) || exit_code=1
    fi
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! pgrep -x MicroXRCEAgent >/dev/null 2>&1; then
    # Restore SIGINT ignored by noninteractive Bash background jobs.
    "$ROS_PYTHON" -c 'import os,signal; signal.signal(signal.SIGINT,signal.SIG_DFL); os.execvp("MicroXRCEAgent", ["MicroXRCEAgent","udp4","-p","8888"])' \
        > "$RUN_DIR/dds_agent.txt" 2>&1 &
    AGENT_PID=$!
fi
"$ROS_PYTHON" -c 'import os,signal,sys; signal.signal(signal.SIGINT,signal.SIG_DFL); os.execv(sys.argv[1],sys.argv[1:])' \
    "$ROS_PYTHON" "$SCRIPT_DIR/recovery_vision_bridge.py" > "$RUN_DIR/vision_bridge.txt" 2>&1 &
VISION_PID=$!
PX4_CMD="export GZ_PARTITION=$(printf '%q' "$GZ_PARTITION") PX4_GZ_MODEL_POSE=$(printf '%q' "$POSE_LINE") && export GZ_SIM_SYSTEM_PLUGIN_PATH=$(printf '%q' "$SCRIPT_DIR/../build/state_injector"):\${GZ_SIM_SYSTEM_PLUGIN_PATH:-} && cd $(printf '%q' "$PX4_DIR") && env PX4_SIM_MODEL=gz_prototype_recovery GZ_IP=127.0.0.1 $(printf '%q' "$PX4_BIN/px4") -d"
if [[ "${S2S_HEADLESS:-0}" == 1 ]]; then PX4_CMD="export HEADLESS=1 && $PX4_CMD"; fi
PX4_CMD="export ROS_DOMAIN_ID=$(printf '%q' "${ROS_DOMAIN_ID:-0}") && $PX4_CMD"
tmux new-session -d -s "$SESSION" -n px4 'bash --noprofile --norc -i'
SESSION_CREATED=1
tmux set-option -t "$SESSION" history-limit 15000
tmux send-keys -t "$SESSION:px4" -l "$PX4_CMD"
tmux send-keys -t "$SESSION:px4" Enter
printf 'Recovery probe: %s\nRequested state: %s\nSpawn pose: %s\nReports: %s\n' \
    "$SESSION" "${STATE[*]}" "$POSE_LINE" "$RUN_DIR"
printf 'Waiting for the PX4 command client...\n'
for _ in $(seq 1 240); do
    if timeout 2 "$PX4_BIN/px4-commander" status >/dev/null 2>&1 &&
            timeout 2 "$PX4_BIN/px4-logger" status >/dev/null 2>&1; then PX4_READY=1; break; fi
    sleep 1
done
(( PX4_READY )) || die 'PX4 command client did not become ready'
px4_cmd param set NAV_DLL_ACT 0
PID_PROFILE="${S2S_PID_PROFILE}"
if [[ -n "${S2S_PID_PROFILE:-}" ]]; then
    "$ROS_PYTHON" "$SCRIPT_DIR/fixed_pid.py" "$PID_PROFILE" --px4-bin "$PX4_BIN" \
        --record "$RUN_DIR/pid_override_record.json"
fi
printf '%s\n' "$PID_PROFILE" > "$RUN_DIR/pid_profile.txt"
for parameter in SYS_AUTOSTART EKF2_EV_CTRL EKF2_HGT_REF EKF2_GPS_CTRL COM_ARM_WO_GPS NAV_DLL_ACT; do
    px4_cmd param show "$parameter"
done
# Save effective controller/allocation/motor values, including saved overrides.
for parameter in 'MC_*' MPC_THR_HOVER THR_MDL_FAC 'SIM_GZ_EC_*' 'CA_ROTOR*'; do
    timeout 12 "$PX4_BIN/px4-param" show "$parameter"
done > "$RUN_DIR/effective_control_parameters.txt" 2>&1
"$ROS_PYTHON" "$SCRIPT_DIR/state_injector.py" \
    --command-topic "$COMMAND_TOPIC" --ack-topic "$ACK_TOPIC" --log "$RUN_DIR/state_injection.jsonl" \
    anchor --physical-anchor "${SPAWN_STATE[0]}" "${SPAWN_STATE[1]}" "${SPAWN_STATE[2]}" \
    --physical-hover 0 0 5 --physical-yaw-reference 0 --nn-goal 0 0 5 --nn-yaw-goal 0
px4_cmd logger on
"$ROS_PYTHON" -c 'import os,signal,sys; signal.signal(signal.SIGINT,signal.SIG_DFL); os.execv(sys.argv[1],sys.argv[1:])' \
    "$ROS_PYTHON" "$SCRIPT_DIR/recorder.py" \
    --output-dir "$RUN_DIR/recording" --with-ulog --px4-dir "$PX4_DIR" \
    > "$RUN_DIR/recorder.txt" 2>&1 &
RECORDER_PID=$!
EXTRA=()
if [[ "${S2S_REQUIRE_PREFLIGHT:-0}" == 1 ]]; then EXTRA+=(--require-preflight); fi
printf 'Checking held PX4 pose, zero twist, validity and reset stability...\n'
if "$ROS_PYTHON" "$SCRIPT_DIR/full_random_initialization.py" probe \
    --state "${SPAWN_STATE[@]}" --attitude-encoding "$ATTITUDE_ENCODING" --timeout 45 \
    --stable-duration 2 --minimum-samples 80 "${EXTRA[@]}" --report "$REPORT"; then
    HOLD_OK=1
else
    HOLD_OK=0
fi
px4_cmd commander check || true
sleep 0.3
for topic in vehicle_status failsafe_flags health_report vehicle_local_position estimator_status_flags estimator_status sensor_combined; do
    px4_cmd listener "$topic" -n 1 > "$RUN_DIR/diagnostic_$topic.txt" 2>&1 || true
done
px4_cmd ekf2 status > "$RUN_DIR/diagnostic_ekf2.txt" 2>&1 || true
(( HOLD_OK )) || die 'Held state did not pass; model was not released'

    MODEL="${S2S_MODEL}"
    POLICY_PYTHON="${S2S_POLICY_PYTHON:-/usr/bin/python3}"
    POLICY_PORT="${S2S_POLICY_PORT:-5801}"
    TRIAL_PROGRAM="$SCRIPT_DIR/trajectory_controller.py"
    TRIAL_EXTRA=(--attitude-encoding "$ATTITUDE_ENCODING")
    [[ -f "$MODEL" && -x "$POLICY_PYTHON" ]] || die 'Policy model or Python not found'
    "$POLICY_PYTHON" -c 'import os,signal,sys; signal.signal(signal.SIGINT,signal.SIG_DFL); os.execv(sys.argv[1],sys.argv[1:])' \
        "$POLICY_PYTHON" -u "$SCRIPT_DIR/policy_server.py" --model "$MODEL" \
        --host 127.0.0.1 --port "$POLICY_PORT" > "$RUN_DIR/policy_server.txt" 2>&1 &
    POLICY_PID=$!
    POLICY_READY=0
    for _ in $(seq 1 60); do
        if "$ROS_PYTHON" "$SCRIPT_DIR/policy_client.py" --port "$POLICY_PORT" --timeout .2 \
                > "$RUN_DIR/policy_ping.txt" 2>&1; then POLICY_READY=1; break; fi
        kill -0 "$POLICY_PID" 2>/dev/null || die 'Policy server stopped'
        sleep .5
    done
    (( POLICY_READY )) || die 'Policy server did not become ready'
    if "$ROS_PYTHON" "$TRIAL_PROGRAM" --state "${STATE[@]}" \
            --output "$RUN_DIR" --model "$MODEL" --port "$POLICY_PORT" \
            --duration "${S2S_TRIAL_DURATION:-10}" --timeout 60 "${TRIAL_EXTRA[@]}"; then
        PASS=1
    else
        die 'Recovery trial failed; inspect trial_result.json'
    fi
    printf 'NEURAL RECOVERY TRIAL PASSED: %s\n' "$RUN_DIR"
    exit 0
