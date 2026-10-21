# Sim-to-sim: world frames and CTBR thrust mapping

## World-frame contract (2026-10-21 correction)

The controller and ROS policy bridge both use
`step02_state_contract.vehicle_odometry_to_observation()`. The corrected
contract is `world_yaw_rotation_eigen_zyx_v2`. After converting PX4 world NED
and body FRD to FlightLxx world/body FLU, define the fixed angle
`alpha = nn_yaw_goal - physical_yaw_reference` and apply:

```text
p_nn = Rz(alpha) * (p_flu - physical_hover) + nn_goal
v_nn = Rz(alpha) * v_flu
R_nn = Rz(alpha) * R_flu
omega_nn = omega_flu = [omega_frd.x, -omega_frd.y, -omega_frd.z]
```

`physical_hover` is the target position in the original physical world FLU
axes. `nn_goal` is its coordinate in the network world frame. Subtract the
physical target before rotating and add the network goal afterwards. This is
a fixed frame for the whole run, not a yaw-following body frame; keep the
calibrated reference constant during flight. Body-rate commands still use
only FLU→FRD; a world yaw alignment does not rotate their axes again.

For example, physical yaw `-pi/2` and network yaw `0` give `alpha=+pi/2`:
horizontal vectors `(x,y)` become `(-y,x)`. This applies to position error,
velocity, and attitude together. Earlier code changed yaw alone, leaving
these world quantities inconsistent; restart the controller to load the fix.
The thrust mapping and the policy weights are unchanged.

Policy attitude inputs now match the training environment in
`libs/src/envs/rl_env.cpp`: `R_nn.eulerAngles(2,1,0)` (Eigen ZYX radians).
This branch has yaw in `[0,pi]`. A small negative principal yaw can therefore
be encoded as yaw near `pi`, with alternate pitch/roll near `+/-pi`; the three
angles together still represent the same physical attitude. Do not normalize
the yaw element alone or interpret those individual pitch/roll values as
evidence of a physical flip. For calibration, `step12_yaw_calibration.py`
continues to use signed principal yaw in `[-pi,pi]`, which is a different
representation from the policy's training Euler inputs.

Offline verification (no vehicle commands):

```bash
/usr/bin/python3 sim_to_sim/step02_state_contract.py --self-test
/usr/bin/python3 sim_to_sim/step12_yaw_calibration.py --self-test
```

The frame regressions check translation/rotation order, 90-degree calibration,
nonzero network goals, arbitrary attitudes, body-rate invariance, velocity as
the derivative of transformed position, and gravity/thrust consistency. When
`g++` and Eigen headers are installed, they also compare 1,084 attitudes with
the actual C++ training Euler expression. Controller startup and
`policy_trace.config` record the contract version, rotation angle, and source
hash so the next experiment can be distinguished from old runs.

## Control ownership: PX4 native modes versus neural Offboard

`--publish` now enables a mode-gated publisher. The controller continues
inference for diagnostics while PX4 owns takeoff, native Hold/Loiter, and
landing. Its startup command and independent terminal windows stay the same.

| Controller/vehicle state | Offboard heartbeat | Neural CTBR setpoint |
|---|---|---|
| Dry run (without `--publish`) | No | No |
| PX4 takeoff, native hover, landing | Yes, when inference/state/feedback are fresh | No |
| Armed Offboard with body-rate control confirmed | Yes | Yes |
| Missing/stale mode feedback or stale inference/odometry | No | No |
| Offboard while disarmed or waiting for controller flags | Yes, when inputs are fresh | No |

The node subscribes to `VehicleStatus` and `VehicleControlMode`. It requires
`nav_state=OFFBOARD`, armed state, and confirmed Offboard rate control with
the internal attitude controller disabled before sending
`/fmu/in/vehicle_rates_setpoint`. Either feedback reporting exit from this
mode stops CTBR on the following control tick. Detection follows received
PX4 feedback; DDS transport and the control timer add transition latency.
The default feedback timeout is 1.5 s (PX4 normally publishes these topics
about twice per second). Missing feedback stops output; it never falls back
to publishing neutral/zero CTBR during native PX4 control.

Topic names follow the installed `px4_msgs` message version automatically,
including `/fmu/out/vehicle_status_v4` on this machine. Override them with
`--vehicle-status-topic` / `--vehicle-control-mode-topic` if namespaced.

Only `OffboardControlMode` is streamed before handover, so you can still
request Offboard after the heartbeat has been active for at least two seconds.
PX4's ROS 2 heartbeat and actual control setpoints are separate; external
setpoints must only be sent during Offboard to avoid overwriting internal
setpoints. See the [PX4 Offboard control documentation](https://docs.px4.io/main/en/flight_modes/offboard#ros-2-offboard-control).

In the controller window, `output=heartbeat_only` means no neural CTBR is sent;
`output=published` means neural CTBR was actually sent. `CTBR_hz` should be
zero during native PX4 modes, while `heartbeat_hz` remains near 50 Hz.
`rates_frd` and `thrust_body` in inference logs are candidate actions; read the
`output` field to determine whether they were sent. `policy.csv` now records
heartbeat publication, mode authority, PX4 nav/arming state, and feedback
ages separately from `published` (actual CTBR publication).

Restart the old controller process to load this change. For PX4 landing,
use `commander land` in the PX4 window as before; the controller can remain
running and will gate off CTBR after receiving the mode transition.

Offline regressions (no ROS messages sent):

```bash
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
/usr/bin/python3 sim_to_sim/test_offboard_authority.py
```

## CTBR thrust mapping

`step08_px4_rates_contract.py` maps FlightLxx's fourth policy output to the
PX4 normalized collective thrust used by `VehicleRatesSetpoint.thrust_body`.
The default parameters describe the PX4 Gazebo Harmonic x500 in this
workspace. `step09_px4_offboard_rates.py` uses the same conversion and records
the model parameters, commanded rotor speed, and saturation flags in its
`policy_trace`. `step15_record_experiment.py` also exports these values to
`policy.csv`.

FlightLxx commands mass-normalized thrust, in m/s²:

```text
a_cmd = 9.81 * (action[3] + 1)
```

The x500 motor model is quadratic in rotor speed. The conversion therefore
uses the vehicle's all-up mass, motor count, Gazebo `motorConstant`, and the
PX4 motor output speed range to recover a rotor speed for `a_cmd`. It then maps
that speed through PX4's `THR_MDL_FAC` motor curve to get the normalized PX4
thrust request. PX4 body-Z points down, so upward thrust is published as
`thrust_body[2] = -u`.

The built-in x500 defaults are:

| Parameter | Default | Source |
|---|---:|---|
| all-up mass | 2.0643077 kg | x500 base plus four rotor-link masses |
| rotor count | 4 | x500 airframe |
| thrust coefficient | 8.54858e-6 N/(rad/s)² | x500 Gazebo `motorConstant` |
| minimum output speed | 150 rad/s | `SIM_GZ_EC_MIN` |
| maximum output speed | 1000 rad/s | `SIM_GZ_EC_MAX`; no greater than SDF `maxRotVelocity` |
| `THR_MDL_FAC` | 0 | recorded x500 SITL parameter |

With these values, maximum modeled thrust acceleration is about 16.56 m/s².
The FlightLxx `action[3] == 0` hover-level request (9.81 m/s²) maps to a PX4
thrust fraction of about 0.729. The 2 g policy endpoint is above x500 authority
and saturates at 1.0. Since the default armed motor minimum is 150 rad/s, the
very-low-thrust region cannot reproduce exact zero thrust; the converter marks
that region with `saturated_low`. Set the SITL output minimum to zero if exact
zero-thrust behavior is required, and pass the effective min/max speed values
to the controller.

Override the model when using another PX4/Gazebo vehicle:

```bash
python3 sim_to_sim/step09_px4_offboard_rates.py \
  --vehicle-mass-kg 2.0643077 \
  --motor-count 4 \
  --motor-thrust-coefficient 8.54858e-6 \
  --motor-omega-min 150 \
  --motor-omega-max 1000 \
  --thrust-model-factor 0
```

Use Gazebo's physical `motorConstant` for the thrust coefficient. PX4's
`CA_ROTOR*_CT` values describe control-allocation effectiveness and are not
the same coefficient. The motor count, mass, speed limits, and PX4 thrust model
must match the active SITL airframe. Rotor asymmetry or different per-motor
limits require extending the equal-rotor model. The reported speed is an
equivalent collective speed, not a measurement of the four motors. This static
conversion is exact for equal motor commands under the quadratic model;
rate-controller torque demands, allocator saturation, and motor lag can change
the total thrust during maneuvers. In particular, it does not compensate for
the increase in sum-of-squares of rotor speeds when the mixer adds differential
motor commands. The x500 defaults assume conventional normalized allocation
and disabled PX4 battery scaling, as in the inspected SITL log.

Run the conversion checks with:

```bash
python3 sim_to_sim/step08_px4_rates_contract.py --self-test
```

## Run one x500 sim-to-sim experiment

### One-command launcher

On the Ubuntu desktop, start the full experiment from the repository root with:

```bash
bash sim_to_sim/run_experiment.sh
```

The launcher opens the DDS Agent, PX4/Gazebo, policy server, recorder, and
controller in a `tmux` session (and opens a GNOME Terminal to view that
session). It waits for PX4 DDS topics, sets `MIS_TAKEOFF_ALT` to 5 m, requests
PX4 takeoff, and waits until local position is within 0.3 m of the takeoff
origin, NED altitude is near -5 m, and speed is below 0.3 m/s. It then measures the current yaw and starts the controller in
dry-run mode. Inspect the dry-run output and press Enter before the launcher
enables publishing; it asks again before requesting Offboard.

At the end, press Enter in the launcher terminal to request `commander land`.
It waits for PX4 land detection before stopping the controller and recorder,
then copies the completed ULog. Ctrl+C also requests landing. If landing cannot
be confirmed, the launcher leaves the tmux session and PX4/Gazebo processes
running so they can be inspected; attach with the session name printed by the
script (`tmux attach -t SESSION_NAME`). Switch process views with `Ctrl+B`
followed by the window number (`1` Agent, `2` policy, `3` PX4, `4` recorder,
`5` controller).

Defaults match the current machine. Override them when needed, for example:

```bash
S2S_MODEL="$PWD/Simulation_Experiments/models/CTBR+DR.zip" \
  bash sim_to_sim/run_experiment.sh
```

Optional variables are documented by `bash sim_to_sim/run_experiment.sh --help`.
The launcher requires `tmux`, ROS 2 Humble, the sourced PX4 ROS 2 workspace,
the `flightlxx3.8` policy environment, and a built `gz_x500` SITL target.

### Manual six-terminal procedure

Use **six terminals**. This procedure starts PX4/Gazebo and its DDS bridge,
starts the policy and recorder, takes off to the policy's 5 m training height
using PX4, then hands control to the FlightLxx policy in Offboard mode.

Before starting, check that `~/PX4-Autopilot/build/px4_sitl_default/` exists
and that the `flightlxx3.8` Conda environment can load the model. All commands
below assume the repositories are in their current `~/Desktop` and `~/` paths.

### Terminal 1 — Micro XRCE-DDS Agent

```bash
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
MicroXRCEAgent udp4 -p 8888
```

Leave it running. PX4 SITL's `uxrce_dds_client` connects to this agent on UDP
port 8888.

### Terminal 2 — PX4 SITL and Gazebo Harmonic

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500
```

PX4 launches Gazebo and the x500. Wait for the PX4 `pxh>` prompt and its
preflight-ready message. Leave it waiting at `pxh>` until the policy server and
recorder in Terminals 3 and 4 are running. Then, at the `pxh>` prompt, set the
takeoff altitude to the policy's 5 m reference and use PX4's normal takeoff
mode to get there:

```text
param set MIS_TAKEOFF_ALT 5
param show MIS_TAKEOFF_ALT
commander takeoff
```

Run all three lines at the `pxh>` prompt. `param set` alone is incomplete and
returns “not enough arguments”; verify the value is `5.0` before takeoff.

`commander takeoff` changes to PX4's takeoff mode and arms the vehicle. Wait
for the takeoff to finish and the vehicle to hold near 5 m before handing over
to the neural policy. The policy was trained around z=5 m; do not hand it
control while the vehicle is still on the ground.

### Terminal 3 — FlightLxx policy server

```bash
cd ~/Desktop/FlightLxx
~/anaconda3/envs/flightlxx3.8/bin/python \
  sim_to_sim/step05_policy_server.py \
  --model Simulation_Experiments/models/CTBR+ID.zip \
  --host 127.0.0.1 --port 5800
```

The command uses the installed PPO2/TensorFlow 1.15 environment. Replace the
model path with `CTBR+DR.zip` or `CTBR+ID.zip` to test another policy. Keep
the server running while the controller is active.

### Terminal 4 — Experiment recorder

```bash
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
cd ~/Desktop/FlightLxx
/usr/bin/python3 sim_to_sim/step15_record_experiment.py \
  --with-ulog --px4-dir ~/PX4-Autopilot
```

Start recording before the takeoff/controller portion you want to analyze.
The recorder discovers PX4 topics and `/flightlxx/sim_to_sim/policy_trace` as
they appear. It prints the output directory; after shutdown, that directory
and a ZIP archive are saved under `sim_to_sim/logs/`. It records PX4 local
position in `local_position.csv` as well as the odometry topic in
`odometry.csv`.

### Terminal 5 — State check, dry run, then policy controller

In a ROS 2 terminal, source the same workspaces before running these commands:

```bash
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
cd ~/Desktop/FlightLxx
```

First check that the policy server answers and PX4 odometry is arriving:

```bash
/usr/bin/python3 sim_to_sim/step06_policy_client.py \
  --host 127.0.0.1 --port 5800
ros2 topic hz /fmu/out/vehicle_odometry
```

Stop `ros2 topic hz` with Ctrl+C. Now execute the `param set` and
`commander takeoff` commands in Terminal 2 and wait for the takeoff to finish.
Then check that PX4 reports an NED `z` position near `-5`:

```bash
LOCAL_POSITION_TOPIC="$(ros2 topic list | grep -E '^/fmu/out/vehicle_local_position(_v[0-9]+)?$' | head -n 1)"
ros2 topic echo "$LOCAL_POSITION_TOPIC" --once
```

Capture the current heading so that the policy sees its training yaw zero:

```bash
/usr/bin/python3 sim_to_sim/step12_yaw_calibration.py \
  --samples 10 --timeout 10
```

Copy the printed `physical_yaw_reference_rad` into both controller commands
below by setting this variable in Terminal 5. Do not assume the x500 startup
reference is zero. In the recorded run the FlightLxx-frame yaw at the 5 m
hover was about `-1.58 rad`; using `0.0` made the network see a 90-degree yaw
error:

```bash
S2S_YAW_REFERENCE_RAD=-1.5708  # replace with this run's calibration output
```

Start the controller in **dry-run mode first**. It will run policy inference
and print the converted thrust without sending PX4 control messages:

```bash
/usr/bin/python3 sim_to_sim/step09_px4_offboard_rates.py \
  --physical-hover 0 0 5 --nn-goal 0 0 5 \
  --physical-yaw-reference "$S2S_YAW_REFERENCE_RAD"
```

Confirm that observations and raw actions are finite and the physical heading
still holds the calibrated reference. Use the signed calibration yaw or the
rotation reconstructed from all three policy Euler angles for this check:
the training Eigen branch can encode a small negative aligned yaw near `pi`
with alternate pitch/roll, rather than as a yaw near zero. For the
recorded CTBR+ID model at this state, the yaw-aligned thrust was about
`thrust_body=[0, 0, -0.74]`; other model/state combinations can differ. Check for
`thrust_sat=(low:True ...)` or `(... high:True)`; either flag means the
requested policy thrust is outside the configured motor range. For the first
hover handover, also wait until the latest dry-run observation is settled near
the network target: horizontal position error and total speed below about
0.3 m / 0.3 m/s, and roll/pitch close to level. The second recorded attempt
switched to Offboard with about 1.6 m/s horizontal speed; it then tipped and
reached the ground in under two seconds. Stop the dry-run with Ctrl+C, then
start the publishing controller with the same arguments plus `--publish`:

```bash
/usr/bin/python3 sim_to_sim/step09_px4_offboard_rates.py \
  --physical-hover 0 0 5 --nn-goal 0 0 5 \
  --physical-yaw-reference "$S2S_YAW_REFERENCE_RAD" --publish
```

Set `S2S_YAW_REFERENCE_RAD` to the printed value if it differs from zero.
Leave this node running. Wait at least two seconds and check that its timing
output shows fresh inference and heartbeat near 50 Hz with `CTBR_hz=0` and
`output=heartbeat_only` before
requesting Offboard mode.

### Terminal 6 — PX4 ROS checks and Offboard handover

```bash
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
cd ~/Desktop/FlightLxx
ros2 topic list | grep -E '^/fmu/(in|out)/(vehicle_odometry|vehicle_rates_setpoint|offboard_control_mode|vehicle_command|vehicle_status(_v[0-9]+)?)$'
STATUS_TOPIC="$(ros2 topic list | grep -E '^/fmu/out/vehicle_status(_v[0-9]+)?$' | head -n 1)"
ros2 topic echo "$STATUS_TOPIC" --once
```

With Terminal 5 already running with `--publish`, request Offboard mode:

```bash
/usr/bin/python3 sim_to_sim/step10_px4_offboard_command.py \
  --send --repeat 5 --period 0.1
ros2 topic echo "$STATUS_TOPIC" --once
```

The vehicle was armed by `commander takeoff`; do not send a second arm
command. After the mode request, check that `$STATUS_TOPIC` reports
Offboard (`nav_state: 14`) and armed (`arming_state: 2`), and watch the vehicle
while the controller runs. `step14_px4_takeoff_mode.py` is not part of this
handover; PX4 takeoff and the policy Offboard controller are separate modes.

### Stop and save the experiment

Keep Terminal 5 and Terminal 4 open until the experiment is over. **Do not
disarm while the vehicle is airborne.** Request a PX4 land in Terminal 2:

```text
commander land
```

Keep the policy node running while PX4 takes over landing. In Terminal 6,
check `$STATUS_TOPIC` and `/fmu/out/vehicle_land_detected`; wait for
the vehicle to be landed and disarmed. In the recorded run, the disarm command
was rejected with PX4 ACK result `1` while the vehicle was still airborne.
If PX4 remains armed after landing, then send a disarm command:

```bash
/usr/bin/python3 sim_to_sim/step11_px4_arm_command.py \
  --disarm --send --repeats 3 --period 0.1
```

After PX4 confirms landing/disarming, stop the remaining processes in this
order:

1. Stop the controller in Terminal 5 with Ctrl+C.
2. Stop PX4/Gazebo in Terminal 2 with Ctrl+C so PX4 closes its ULog.
3. Stop the recorder in Terminal 4 with Ctrl+C. It will finalize CSV/JSONL,
   collect the completed ULog, and create the ZIP archive.
4. Stop the policy server in Terminal 3 and the DDS Agent in Terminal 1.

The experiment directory contains `metadata.json`, `summary.json`,
`events.jsonl`, `policy.csv`, PX4 topic CSV files, and—when a new ULog was
available—the copied PX4 log. Check `summary.json` for dropped recorder events
and whether the controller policy trace was captured.

If ROS 2 does not show `/fmu/out/vehicle_odometry`, check that the DDS Agent is
running and PX4 reports `uxrce_dds_client` synchronized. If policy requests
fail, confirm Terminal 3 is listening on `127.0.0.1:5800`. If Offboard is
rejected, ensure Terminal 5 shows `output=heartbeat_only` and has been sending
fresh Offboard heartbeats for at least two seconds, both mode feedback topics
are arriving, and the vehicle has completed takeoff before retrying.

PX4's generated DDS topic names include message-version suffixes on this
workspace (`vehicle_local_position_v1` and `vehicle_status_v4`). Discover the
active names instead of assuming the unsuffixed forms. The ROS 2 flag is
`--once` (not `--oncece`):

```bash
ros2 topic list | grep -E '^/fmu/out/(vehicle_local_position|vehicle_status)(_v[0-9]+)?$'
LOCAL_POSITION_TOPIC="$(ros2 topic list | grep -E '^/fmu/out/vehicle_local_position(_v[0-9]+)?$' | head -n 1)"
STATUS_TOPIC="$(ros2 topic list | grep -E '^/fmu/out/vehicle_status(_v[0-9]+)?$' | head -n 1)"
ros2 topic echo "$LOCAL_POSITION_TOPIC" --once
ros2 topic echo "$STATUS_TOPIC" --once
```

The latest recorded run showed that `vehicle_status_v4` and
`vehicle_local_position_v1` were published and recorded. If the versioned
topics do not appear, wait until the Micro XRCE-DDS Agent is running and PX4
reports DDS synchronization, then retry.
