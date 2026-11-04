"""Asynchronous inference and Offboard message construction."""
from dataclasses import dataclass
import threading
import time
from thrust_mapping import DEFAULT_THRUST_MODEL, ThrustMapping, convert_raw_action_to_px4

@dataclass(frozen=True)
class PolicyResult:
    sequence: int
    raw_action: tuple
    rates_frd: tuple
    thrust_body_frd: tuple
    collective_thrust: float
    observation_time: float
    started_at: float
    completed_at: float
    observation: tuple = ()
    source_odometry: dict = None
    thrust_mapping: ThrustMapping = None

def result_is_fresh(result, now, max_action_age, max_odometry_age):
    """Check both reply age and the age of the state used for that reply."""
    return (
        result is not None
        and 0.0 <= now - result.completed_at <= max_action_age
        and 0.0 <= now - result.observation_time <= max_odometry_age
    )

class PolicyWorker:
    """Own the UDP client on one thread; reject submissions while busy."""

    def __init__(self, client, client_factory=None, thrust_model=None):
        self._client = client
        self._client_factory = client_factory
        self._thrust_model = dict(thrust_model or DEFAULT_THRUST_MODEL)
        self._timeout = client.timeout
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._busy = False
        self._job = None
        self._sequence = 0
        self._result = None
        self._error = None
        self._success_count = 0
        self._thread = threading.Thread(
            target=self._run, name="flightlxx_policy_worker", daemon=True
        )
        self._thread.start()

    def submit(self, observation, observation_time, source_odometry=None):
        with self._lock:
            if self._stop.is_set() or self._busy:
                return False
            self._busy = True
            self._job = (
                tuple(observation), float(observation_time),
                dict(source_odometry or {}),
            )
            self._wake.set()
            return True

    def snapshot(self):
        with self._lock:
            return self._result, self._error, self._success_count

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=max(2.0, self._timeout + 0.5))
        return not self._thread.is_alive()

    def _run(self):
        try:
            while not self._stop.is_set():
                self._wake.wait()
                self._wake.clear()
                if self._stop.is_set():
                    break
                with self._lock:
                    job = self._job
                    self._job = None
                    self._sequence += 1
                    sequence = self._sequence
                if job is None:
                    continue
                observation, observation_time, source_odometry = job
                started_at = time.monotonic()
                try:
                    action = tuple(
                        self._client.infer(observation, sequence=sequence)
                    )
                    rates, thrust, collective, thrust_mapping = (
                        convert_raw_action_to_px4(
                            action, **self._thrust_model
                        )
                    )
                    result = PolicyResult(
                        sequence=sequence,
                        raw_action=action,
                        rates_frd=tuple(rates),
                        thrust_body_frd=tuple(thrust),
                        collective_thrust=float(collective),
                        observation_time=observation_time,
                        started_at=started_at,
                        completed_at=time.monotonic(),
                        observation=observation,
                        source_odometry=source_odometry,
                        thrust_mapping=thrust_mapping,
                    )
                    with self._lock:
                        self._result = result
                        self._success_count += 1
                        self._error = None
                except (OSError, TypeError, ValueError, RuntimeError) as exc:
                    with self._lock:
                        self._error = "policy/action conversion failed: %s" % exc
                    # A new socket discards late responses from a timed-out
                    # request instead of shifting every subsequent sequence.
                    if self._client_factory is not None:
                        self._client.close()
                        try:
                            self._client = self._client_factory()
                        except OSError as reopen_error:
                            with self._lock:
                                self._error = "policy socket reopen failed: %s" % (
                                    reopen_error,
                                )
                finally:
                    with self._lock:
                        self._busy = False
        finally:
            self._client.close()

def fill_px4_messages(
    offboard_message,
    rates_message,
    timestamp_us,
    rates_frd,
    thrust_body_frd,
):
    """Preserve the original CTBR message contract."""
    fill_offboard_heartbeat(offboard_message, timestamp_us)
    fill_rates_setpoint(rates_message, timestamp_us, rates_frd, thrust_body_frd)

def fill_offboard_heartbeat(offboard_message, timestamp_us):
    """Declare the intended control level without sending actuator demands."""
    offboard_message.timestamp = int(timestamp_us)
    offboard_message.position = False
    offboard_message.velocity = False
    offboard_message.acceleration = False
    offboard_message.attitude = False
    offboard_message.body_rate = True
    offboard_message.thrust_and_torque = False
    offboard_message.direct_actuator = False

def fill_rates_setpoint(rates_message, timestamp_us, rates_frd, thrust_body_frd):
    """Fill CTBR; callers must first verify PX4 granted Offboard control."""
    rates_message.timestamp = int(timestamp_us)
    rates_message.roll = float(rates_frd[0])
    rates_message.pitch = float(rates_frd[1])
    rates_message.yaw = float(rates_frd[2])
    rates_message.thrust_body = [float(value) for value in thrust_body_frd]
    rates_message.reset_integral = False
