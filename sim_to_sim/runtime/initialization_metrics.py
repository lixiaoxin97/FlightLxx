"""Initialization tolerances and infrastructure failure evidence."""
import json
import math
from pathlib import Path
from recovery_state import quaternion_from_zyx
INIT_TOLERANCES = dict(position_m=.06, attitude_rad=.06, velocity_m_s=.06, rate_rad_s=.06)

def initial_errors(actual, expected):
    if actual is None or len(actual) != 12:
        return None
    qa = quaternion_from_zyx(actual[5], actual[4], actual[3])
    qe = quaternion_from_zyx(expected[5], expected[4], expected[3])
    angle = 2 * math.acos(min(1.0, abs(sum(a * b for a, b in zip(qa, qe)))))
    return {
        "position_m": math.dist(actual[:3], expected[:3]),
        "attitude_rad": angle,
        "velocity_m_s": math.dist(actual[6:9], expected[6:9]),
        "rate_rad_s": math.dist(actual[9:12], expected[9:12]),
    }

def infrastructure_evidence(directory):
    """Separate a dead data/control process from a populated EKF mismatch."""
    directory = Path(directory)
    evidence = []
    health = directory / "process_health.json"
    if health.exists():
        for item in json.loads(health.read_text()).get("unexpected_exits", []):
            evidence.append(item)
    for name in ("vision_bridge.txt", "policy_server.txt", "dds_agent.txt"):
        path = directory / name
        if not path.exists():
            continue
        log = path.read_text(errors="replace")
        if "Traceback (most recent call last):" in log:
            evidence.append({"process_log": name, "reason": "unhandled_exception"})
    initial = directory / "initial_state.json"
    if initial.exists():
        data = json.loads(initial.read_text())
        if data.get("sample_count") == 0:
            evidence.append({"process_log": initial.name, "reason": "no_state_samples"})
    return evidence
