#!/usr/bin/env bash

# Launch and supervise one PX4/Gazebo x500 sim-to-sim experiment.
# The dry-run and Offboard handover intentionally require operator confirmation.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FLIGHTLXX_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PX4_DIR="${S2S_PX4_DIR:-$HOME/PX4-Autopilot}"
ROS_SETUP="${S2S_ROS_SETUP:-/opt/ros/humble/setup.bash}"
ROS2_WS_SETUP="${S2S_ROS2_WS_SETUP:-$HOME/px4_ros2_ws/install/setup.bash}"
ROS_PYTHON="${S2S_ROS_PYTHON:-/usr/bin/python3}"
POLICY_PYTHON="${S2S_POLICY_PYTHON:-$HOME/anaconda3/envs/flightlxx3.8/bin/python}"
MODEL_PATH="${S2S_MODEL:-$FLIGHTLXX_ROOT/Simulation_Experiments/models/CTBR+ID.zip}"
POLICY_PORT="${S2S_POLICY_PORT:-5800}"
TAKEOFF_ALT_M="${S2S_TAKEOFF_ALT_M:-5}"
SESSION="flightlxx_s2s_$$"

SESSION_CREATED=0
TAKEOFF_REQUESTED=0
LANDED=0
PRESERVE_SESSION=0
AGENT_OWNED=0
LOCAL_TOPIC=""
STATUS_TOPIC=""
LAND_TOPIC=""

usage() {
    cat <<'EOF'
Usage: sim_to_sim/run_experiment.sh

Starts the x500 SITL experiment in a tmux session. The launcher waits for a
stable takeoff, runs the policy in dry-run, and asks before enabling policy
control. Press Enter at the final prompt to request PX4 landing.

Optional environment overrides:
  S2S_PX4_DIR           PX4-Autopilot path (default: ~/PX4-Autopilot)
  S2S_ROS_SETUP         ROS setup file (default: /opt/ros/humble/setup.bash)
  S2S_ROS2_WS_SETUP     PX4 ROS 2 workspace setup file
  S2S_ROS_PYTHON        ROS Python (default: /usr/bin/python3)
  S2S_POLICY_PYTHON     Policy Python environment
  S2S_MODEL             PPO model path (default: CTBR+ID.zip)
  S2S_POLICY_PORT       Local UDP policy port (default: 5800)
  S2S_TAKEOFF_ALT_M     PX4 takeoff altitude in meters (default: 5)
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
[[ $# -eq 0 ]] || die "unexpected argument: $1 (see --help)"
[[ -t 0 ]] || die "run this script from an interactive terminal"

for required in tmux timeout awk grep pgrep seq sed tail; do
    command -v "$required" >/dev/null 2>&1 || die "required command not found: $required"
done
[[ -r "$ROS_SETUP" ]] || die "ROS setup file not found: $ROS_SETUP"
[[ -r "$ROS2_WS_SETUP" ]] || die "PX4 ROS 2 workspace setup file not found: $ROS2_WS_SETUP"
[[ -x "$ROS_PYTHON" ]] || die "ROS Python is not executable: $ROS_PYTHON"
[[ -x "$POLICY_PYTHON" ]] || die "policy Python is not executable: $POLICY_PYTHON"
[[ -f "$MODEL_PATH" ]] || die "policy model not found: $MODEL_PATH"
[[ -d "$PX4_DIR/build/px4_sitl_default" ]] || die "PX4 SITL build not found under $PX4_DIR"
[[ -f "$FLIGHTLXX_ROOT/sim_to_sim/step09_px4_offboard_rates.py" ]] || die "sim-to-sim files are missing"
[[ "$POLICY_PORT" =~ ^[0-9]+$ ]] || die "S2S_POLICY_PORT must be an integer"
(( POLICY_PORT > 0 && POLICY_PORT < 65536 )) || die "S2S_POLICY_PORT must be in 1..65535"
[[ "$TAKEOFF_ALT_M" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "S2S_TAKEOFF_ALT_M must be a positive number"
awk -v h="$TAKEOFF_ALT_M" 'BEGIN { exit !(h > 0 && h <= 100) }' || die "S2S_TAKEOFF_ALT_M must be in (0, 100]"

# ROS setup scripts read optional variables without default values. Temporarily
# disable nounset while sourcing them, then restore the launcher's strict mode.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$ROS2_WS_SETUP"
set -u

for required in ros2 make; do
    command -v "$required" >/dev/null 2>&1 || die "required command not found after sourcing ROS: $required"
done
if ! pgrep -x MicroXRCEAgent >/dev/null 2>&1; then
    command -v MicroXRCEAgent >/dev/null 2>&1 || die "MicroXRCEAgent not found after sourcing ROS workspace"
fi

# Fail before takeoff if the PX4 message libraries are not available in this
# environment. The controller window sources the same setup files separately.
if ! "$ROS_PYTHON" - <<'PY'
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleOdometry, VehicleRatesSetpoint

rclpy.init()
node = Node("flightlxx_px4_typesupport_preflight")
qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                 durability=DurabilityPolicy.VOLATILE)
node.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry",
                         lambda message: None, qos)
node.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", 1)
node.create_publisher(VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", 1)
node.destroy_node()
rclpy.shutdown()
print("PX4 ROS message type support: ready")
PY
then
    die "PX4 ROS message type support check failed; source/rebuild px4_msgs before takeoff"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    die "tmux session already exists: $SESSION"
fi

tmux new-session -d -s "$SESSION" -n console
SESSION_CREATED=1

new_window() {
    local name="$1"
    local command="$2"
    tmux new-window -d -t "$SESSION" -n "$name"
    tmux send-keys -t "$SESSION:$name" -l "$command"
    tmux send-keys -t "$SESSION:$name" Enter
}

send_px4() {
    local command="$1"
    tmux send-keys -t "$SESSION:px4" -l "$command"
    tmux send-keys -t "$SESSION:px4" Enter
}

stop_window() {
    local name="$1"
    tmux send-keys -t "$SESSION:$name" C-c 2>/dev/null || true
}

topic_once() {
    local topic="$1"
    timeout 4s ros2 topic echo "$topic" --once 2>/dev/null || true
}

discover_px4_topics() {
    local topics
    topics="$(ros2 topic list 2>/dev/null || true)"
    LOCAL_TOPIC="$(printf '%s\n' "$topics" | grep -E '^/fmu/out/vehicle_local_position(_v[0-9]+)?$' | head -n 1 || true)"
    STATUS_TOPIC="$(printf '%s\n' "$topics" | grep -E '^/fmu/out/vehicle_status(_v[0-9]+)?$' | head -n 1 || true)"
    LAND_TOPIC="$(printf '%s\n' "$topics" | grep -E '^/fmu/out/vehicle_land_detected(_v[0-9]+)?$' | head -n 1 || true)"
}

wait_for_px4_topics() {
    local timeout_s="${1:-90}"
    local deadline=$((SECONDS + timeout_s))
    while (( SECONDS < deadline )); do
        discover_px4_topics
        if [[ -n "$LOCAL_TOPIC" && -n "$STATUS_TOPIC" && -n "$LAND_TOPIC" ]]; then
            printf 'PX4 topics detected:\n  local position: %s\n  vehicle status: %s\n  land detected: %s\n' \
                "$LOCAL_TOPIC" "$STATUS_TOPIC" "$LAND_TOPIC"
            return 0
        fi
        sleep 1
    done
    discover_px4_topics
    printf 'Timed out waiting for PX4 topics. Current matches: local=%s status=%s landed=%s\n' \
        "${LOCAL_TOPIC:-none}" "${STATUS_TOPIC:-none}" "${LAND_TOPIC:-none}" >&2
    return 1
}

wait_for_px4_prompt() {
    local timeout_s="${1:-180}"
    local deadline=$((SECONDS + timeout_s))
    while (( SECONDS < deadline )); do
        if tmux capture-pane -p -t "$SESSION:px4" -S -100 2>/dev/null | grep -q 'pxh>'; then
            return 0
        fi
        sleep 1
    done
    printf 'PX4 did not reach the pxh> prompt. See tmux window "px4".\n' >&2
    return 1
}

wait_for_text() {
    local window="$1"
    local text="$2"
    local timeout_s="${3:-45}"
    local deadline=$((SECONDS + timeout_s))
    while (( SECONDS < deadline )); do
        if tmux capture-pane -p -t "$SESSION:$window" -S -100 2>/dev/null | grep -Fq -- "$text"; then
            return 0
        fi
        sleep 1
    done
    printf 'Timed out waiting for "%s" in tmux window "%s".\n' "$text" "$window" >&2
    tmux capture-pane -p -t "$SESSION:$window" -S -40 >&2 || true
    return 1
}

wait_for_hover() {
    local timeout_s="${1:-150}"
    local deadline=$((SECONDS + timeout_s))
    local good_samples=0
    local last_state="not received"
    printf 'Waiting for stable hover near (x,y)=(0,0), NED z=-%.1f m, position error <0.3 m, and speed <0.3 m/s.\n' "$TAKEOFF_ALT_M"
    while (( SECONDS < deadline )); do
        local output values x y z vx vy vz
        output="$(topic_once "$LOCAL_TOPIC")"
        values="$(printf '%s\n' "$output" | awk -F': ' '
            /^xy_valid: / { xy_valid=$2 }
            /^z_valid: /  { z_valid=$2 }
            /^x: /  { x=$2 }
            /^y: /  { y=$2 }
            /^z: /  { z=$2 }
            /^vx: / { vx=$2 }
            /^vy: / { vy=$2 }
            /^vz: / { vz=$2 }
            END { if (xy_valid == "true" && z_valid == "true" && x != "" && y != "" && z != "" && vx != "" && vy != "" && vz != "") print x, y, z, vx, vy, vz }
        ')"
        if [[ -n "$values" ]]; then
            read -r x y z vx vy vz <<<"$values"
            last_state="position=($x, $y, $z) m, velocity=($vx, $vy, $vz) m/s"
            if awk -v x="$x" -v y="$y" -v z="$z" -v vx="$vx" -v vy="$vy" -v vz="$vz" -v h="$TAKEOFF_ALT_M" \
                'BEGIN { position_error=sqrt(x*x+y*y); speed=sqrt(vx*vx+vy*vy+vz*vz); exit !(z >= -h-0.35 && z <= -h+0.35 && position_error < 0.3 && speed < 0.3) }'; then
                ((good_samples += 1))
                if (( good_samples >= 4 )); then
                    printf 'Stable hover confirmed: %s\n' "$last_state"
                    return 0
                fi
            else
                good_samples=0
            fi
        else
            good_samples=0
        fi
        sleep 0.2
    done
    printf 'Timed out waiting for stable hover; latest state: %s\n' "$last_state" >&2
    return 1
}

wait_for_offboard() {
    local timeout_s="${1:-20}"
    local deadline=$((SECONDS + timeout_s))
    while (( SECONDS < deadline )); do
        local output nav_state arming_state
        output="$(topic_once "$STATUS_TOPIC")"
        nav_state="$(printf '%s\n' "$output" | awk -F': ' '/^nav_state: / {print $2; exit}')"
        arming_state="$(printf '%s\n' "$output" | awk -F': ' '/^arming_state: / {print $2; exit}')"
        if [[ "$nav_state" == "14" && "$arming_state" == "2" ]]; then
            printf 'PX4 confirms Offboard and armed (nav_state=14, arming_state=2).\n'
            return 0
        fi
        sleep 0.5
    done
    printf 'PX4 did not confirm Offboard while armed. Latest nav_state=%s arming_state=%s\n' \
        "${nav_state:-unknown}" "${arming_state:-unknown}" >&2
    return 1
}

wait_for_landed() {
    local timeout_s="${1:-180}"
    local deadline=$((SECONDS + timeout_s))
    local landed_samples=0
    printf 'Waiting for PX4 land detection on %s.\n' "$LAND_TOPIC"
    while (( SECONDS < deadline )); do
        local output landed
        output="$(topic_once "$LAND_TOPIC")"
        landed="$(printf '%s\n' "$output" | awk -F': ' '/^landed: / {print $2; exit}')"
        if [[ "$landed" == "true" ]]; then
            ((landed_samples += 1))
            if (( landed_samples >= 3 )); then
                printf 'PX4 confirms the vehicle is landed.\n'
                return 0
            fi
        else
            landed_samples=0
        fi
        sleep 0.3
    done
    printf 'Timed out waiting for land detection. Keep the PX4/Gazebo session open and inspect it.\n' >&2
    return 1
}

status_field_once() {
    local field="$1"
    topic_once "$STATUS_TOPIC" | awk -F': ' -v field="$field" '$1 == field {print $2; exit}'
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    if (( SESSION_CREATED )) && tmux has-session -t "$SESSION" 2>/dev/null; then
        if (( TAKEOFF_REQUESTED && !LANDED )); then
            printf '\nA takeoff was requested but landing is not confirmed. Requesting PX4 land and leaving the services running until it is safe to stop.\n' >&2
            send_px4 'commander land' || true
            if wait_for_landed 180; then
                LANDED=1
            else
                PRESERVE_SESSION=1
                printf 'Safety hold: tmux session %s remains active. Attach with: tmux attach -t %s\n' "$SESSION" "$SESSION" >&2
            fi
        fi

        if (( !PRESERVE_SESSION )); then
            stop_window controller || true
            sleep 1
            stop_window px4 || true
            sleep 2
            stop_window recorder || true
            sleep 3
            stop_window policy || true
            if (( AGENT_OWNED )); then
                stop_window agent || true
            fi
            sleep 1
            tmux kill-session -t "$SESSION" 2>/dev/null || true
            printf 'Experiment processes stopped. Recorder output is under %s/sim_to_sim/logs/.\n' "$FLIGHTLXX_ROOT"
        fi
    fi
    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Starting FlightLxx sim-to-sim experiment.\n'
printf 'PX4: %s\nModel: %s\nTmux session: %s\n' "$PX4_DIR" "$MODEL_PATH" "$SESSION"

if pgrep -x MicroXRCEAgent >/dev/null 2>&1; then
    printf 'An existing MicroXRCEAgent process was found; reusing it.\n'
    new_window agent 'echo "Using the existing MicroXRCEAgent process"; exec bash'
else
    AGENT_OWNED=1
    new_window agent "source $(printf '%q' "$ROS_SETUP") && source $(printf '%q' "$ROS2_WS_SETUP") && exec MicroXRCEAgent udp4 -p 8888"
fi

new_window policy "cd $(printf '%q' "$FLIGHTLXX_ROOT") && exec $(printf '%q' "$POLICY_PYTHON") $(printf '%q' "$FLIGHTLXX_ROOT/sim_to_sim/step05_policy_server.py") --model $(printf '%q' "$MODEL_PATH") --host 127.0.0.1 --port $(printf '%q' "$POLICY_PORT")"
new_window px4 "cd $(printf '%q' "$PX4_DIR") && make px4_sitl gz_x500"

if [[ -n "${DISPLAY:-}" ]] && command -v gnome-terminal >/dev/null 2>&1; then
    tmux select-window -t "$SESSION:px4"
    gnome-terminal --title="FlightLxx sim-to-sim ($SESSION)" -- tmux attach-session -t "$SESSION" >/dev/null 2>&1 &
    disown || true
else
    printf 'No desktop terminal detected. View process windows with: tmux attach -t %s\n' "$SESSION"
fi

printf 'Waiting for PX4 startup and DDS topic discovery...\n'
wait_for_px4_prompt 180 || die "PX4 startup failed; inspect tmux window px4"
wait_for_px4_topics 90 || die "PX4 topics are unavailable; check the Agent and PX4 DDS synchronization"

new_window recorder "source $(printf '%q' "$ROS_SETUP") && source $(printf '%q' "$ROS2_WS_SETUP") && cd $(printf '%q' "$FLIGHTLXX_ROOT") && exec $(printf '%q' "$ROS_PYTHON") $(printf '%q' "$FLIGHTLXX_ROOT/sim_to_sim/step15_record_experiment.py") --with-ulog --px4-dir $(printf '%q' "$PX4_DIR")"

printf 'Waiting for the policy server...\n'
policy_ready=0
for _ in $(seq 1 45); do
    if "$ROS_PYTHON" "$FLIGHTLXX_ROOT/sim_to_sim/step06_policy_client.py" \
        --host 127.0.0.1 --port "$POLICY_PORT" >/dev/null 2>&1; then
        policy_ready=1
        break
    fi
    sleep 1
done
(( policy_ready )) || die "policy server did not answer on UDP port $POLICY_PORT"
printf 'Policy server and recorder are ready.\n'

if ! wait_for_px4_prompt 2; then
    die 'PX4 shell is no longer available'
fi
printf 'Setting MIS_TAKEOFF_ALT to %s m and requesting PX4 takeoff.\n' "$TAKEOFF_ALT_M"
send_px4 "param set MIS_TAKEOFF_ALT $TAKEOFF_ALT_M"
sleep 1
send_px4 'param show MIS_TAKEOFF_ALT'
sleep 1
tmux capture-pane -p -t "$SESSION:px4" -S -20 | tail -n 8
send_px4 'commander takeoff'
TAKEOFF_REQUESTED=1
wait_for_hover 150 || die 'takeoff did not reach a stable 5 m hover'

printf 'Calibrating yaw from the live PX4 odometry...\n'
if ! yaw_output="$("$ROS_PYTHON" "$FLIGHTLXX_ROOT/sim_to_sim/step12_yaw_calibration.py" --samples 10 --timeout 10 2>&1)"; then
    printf '%s\n' "$yaw_output" >&2
    die 'yaw calibration failed'
fi
printf '%s\n' "$yaw_output"
YAW_REFERENCE="$(printf '%s\n' "$yaw_output" | sed -n 's/^physical_yaw_reference_rad: *//p' | tail -n 1)"
[[ -n "$YAW_REFERENCE" ]] || die 'could not read the calibrated yaw reference'

CONTROLLER_BASE="source $(printf '%q' "$ROS_SETUP") && source $(printf '%q' "$ROS2_WS_SETUP") && cd $(printf '%q' "$FLIGHTLXX_ROOT") && $(printf '%q' "$ROS_PYTHON") $(printf '%q' "$FLIGHTLXX_ROOT/sim_to_sim/step09_px4_offboard_rates.py") --physical-hover 0 0 $TAKEOFF_ALT_M --nn-goal 0 0 $TAKEOFF_ALT_M --physical-yaw-reference $(printf '%q' "$YAW_REFERENCE")"
new_window controller "$CONTROLLER_BASE"
printf '\nDry-run controller started with calibrated yaw %.6f rad. Check its request/action output in tmux window "controller".\n' "$YAW_REFERENCE"
wait_for_text controller 'request[' 60 || die 'dry-run controller did not produce policy output'
read -r -p 'If dry-run actions are finite and thrust is not unexpectedly saturated, press Enter to start publishing (Ctrl+C aborts and requests landing): ' _

stop_window controller
sleep 1
CONTROLLER_LIVE="$CONTROLLER_BASE --publish"
tmux send-keys -t "$SESSION:controller" -l "$CONTROLLER_LIVE"
tmux send-keys -t "$SESSION:controller" Enter
wait_for_text controller 'request[' 60 || die 'publishing controller did not produce policy output'
printf 'Publishing controller is active. It must remain running before Offboard is requested.\n'
read -r -p 'When the x500 is still stable and controller output looks healthy, press Enter to request Offboard: ' _

printf 'Requesting Offboard mode...\n'
"$ROS_PYTHON" "$FLIGHTLXX_ROOT/sim_to_sim/step10_px4_offboard_command.py" --send --repeat 5 --period 0.1
wait_for_offboard 20 || die 'PX4 did not accept Offboard; landing will be requested'

printf '\nExperiment is running in Offboard. Keep the controller and recorder active.\n'
read -r -p 'Press Enter when you want PX4 to land and finish the experiment (Ctrl+C also requests landing): ' _
send_px4 'commander land'
if ! wait_for_landed 180; then
    while true; do
        read -r -p 'Landing is not confirmed. Inspect PX4/Gazebo; press Enter to retry commander land, or keep this terminal open: ' _
        send_px4 'commander land'
        wait_for_landed 90 && break
    done
fi
LANDED=1

arming_state="$(status_field_once arming_state)"
if [[ "$arming_state" == "2" ]]; then
    printf 'Vehicle is landed but still armed; requesting disarm now.\n'
    "$ROS_PYTHON" "$FLIGHTLXX_ROOT/sim_to_sim/step11_px4_arm_command.py" --disarm --send --repeats 3 --period 0.1
fi

printf 'Landing confirmed. Finalizing recorder and PX4 ULog.\n'
exit 0
