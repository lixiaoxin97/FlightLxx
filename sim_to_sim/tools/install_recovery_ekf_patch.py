#!/usr/bin/env python3
"""Apply the narrow EV position+velocity reset fix used by Recovery SITL."""

import argparse
from pathlib import Path

OLD = "if ((ev_reset && isOnlyActiveSourceOfHorizontalAiding(_control_status.flags.ev_vel)) || yaw_alignment_changed) {"
NEW = """// FlightLxx: EV position and EV velocity belong to the same reset source.
			// A reset from that source is valid while both are fused, provided no
			// independent horizontal aiding source is active. No change without EV reset.
			const bool only_ev_aiding = isOnlyActiveSourceOfHorizontalAiding(_control_status.flags.ev_vel)
						    || (_control_status.flags.ev_vel && _control_status.flags.ev_pos
							&& getNumberOfActiveHorizontalAidingSources() == 2);

			if ((ev_reset && only_ev_aiding) || yaw_alignment_changed) {"""


def install(root):
    path = (
        Path(root)
        / "src/modules/ekf2/EKF/aid_sources/external_vision/ev_vel_control.cpp"
    )
    content = path.read_text()
    if NEW in content:
        print("Recovery EKF reset patch already installed:", path)
        return
    if content.count(OLD) != 1:
        raise RuntimeError("Unexpected PX4 source; refusing to overwrite: " + str(path))
    path.write_text(content.replace(OLD, NEW))
    print("Installed recovery EKF reset patch:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-dir", type=Path, default=Path.home() / "PX4-Autopilot")
    install(parser.parse_args().px4_dir)
