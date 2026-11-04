#!/usr/bin/env python3
"""Install FlightLxx's local Gazebo model/airframe into PX4 (idempotently)."""

import argparse
from pathlib import Path
import re

AIRFRAME = "22001_gz_prototype"
GENERATED_MARKER = "FlightLxx Prototype SITL integration"


def install(px4_dir, flightlxx_dir, with_recovery=False):
    px4_dir = Path(px4_dir).expanduser().resolve()
    flightlxx_dir = Path(flightlxx_dir).expanduser().resolve()
    source_models = flightlxx_dir / "sim_to_sim" / "gazebo" / "models"
    target_models = px4_dir / "Tools/simulation/gz/models"
    airframe_names = (AIRFRAME, "22002_gz_prototype_recovery") if with_recovery else (AIRFRAME,)
    model_names = ("prototype", "prototype_base", "prototype_recovery",
                   "prototype_recovery_base") if with_recovery else ("prototype", "prototype_base")
    airframe_target_dir = px4_dir / "ROMFS/px4fmu_common/init.d-posix/airframes"
    cmake_file = airframe_target_dir / "CMakeLists.txt"
    if not (px4_dir / "build/px4_sitl_default").is_dir():
        raise FileNotFoundError("PX4 SITL build is missing under %s" % px4_dir)
    for model in model_names:
        if not (source_models / model / "model.sdf").is_file():
            raise FileNotFoundError("missing local Gazebo model %s" % model)
    if not cmake_file.is_file() or not all(
        (flightlxx_dir/"sim_to_sim/px4_airframes"/name).is_file()
        for name in airframe_names
    ):
        raise FileNotFoundError("PX4 airframe integration inputs are missing")

    for model in model_names:
        source = (source_models / model).resolve()
        target = target_models / model
        if target.is_symlink():
            if target.resolve() != source:
                raise FileExistsError("refusing to replace unrelated model link: %s" % target)
        elif target.exists():
            raise FileExistsError("refusing to overwrite an existing PX4 model: %s" % target)
        else:
            target.symlink_to(source, target_is_directory=True)

    airframe_targets = []
    for name in airframe_names:
        airframe_source = flightlxx_dir/"sim_to_sim/px4_airframes"/name
        airframe_target = airframe_target_dir/name
        source_text = airframe_source.read_text()
        if airframe_target.exists():
            installed_text = airframe_target.read_text()
            if (installed_text != source_text and GENERATED_MARKER not in installed_text
                    and "FlightLxx recovery SITL integration" not in installed_text):
                raise FileExistsError("refusing to overwrite existing PX4 airframe: %s" % airframe_target)
            if installed_text != source_text:
                airframe_target.write_text(source_text)
        else:
            airframe_target.write_text(source_text)
        airframe_targets.append(airframe_target)
        cmake = cmake_file.read_text()
        if name not in [line.strip() for line in cmake.splitlines()]:
            pattern = re.compile(r"(?m)^([ \t]*)4021_gz_x500_flow[ \t]*$")
            if not pattern.search(cmake):
                raise ValueError("could not find PX4 Gazebo airframe insertion point")
            cmake_file.write_text(pattern.sub(
                lambda match: match.group(0) + "\n\t" + name, cmake, count=1))
    return tuple(target_models/name for name in model_names) + tuple(airframe_targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-dir", type=Path,
                        default=Path.home() / "PX4-Autopilot")
    parser.add_argument("--flightlxx-dir", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--with-recovery", action="store_true",
                        help="also install isolated state-initialization variant")
    args = parser.parse_args()
    paths = install(args.px4_dir, args.flightlxx_dir, args.with_recovery)
    for path in paths:
        print("PX4 Prototype integration:", path)
    print("PX4 build will regenerate its airframe and gz_prototype targets on the next make.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
