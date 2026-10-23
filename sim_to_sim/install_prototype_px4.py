#!/usr/bin/env python3
"""Install FlightLxx's local Gazebo model/airframe into PX4 (idempotently)."""

import argparse
from pathlib import Path
import re

AIRFRAME = "22001_gz_prototype"
GENERATED_MARKER = "FlightLxx Prototype SITL integration"


def install(px4_dir, flightlxx_dir):
    px4_dir = Path(px4_dir).expanduser().resolve()
    flightlxx_dir = Path(flightlxx_dir).expanduser().resolve()
    source_models = flightlxx_dir / "sim_to_sim" / "gazebo" / "models"
    target_models = px4_dir / "Tools/simulation/gz/models"
    airframe_source = flightlxx_dir / "sim_to_sim" / "px4_airframes" / AIRFRAME
    airframe_target_dir = px4_dir / "ROMFS/px4fmu_common/init.d-posix/airframes"
    cmake_file = airframe_target_dir / "CMakeLists.txt"
    if not (px4_dir / "build/px4_sitl_default").is_dir():
        raise FileNotFoundError("PX4 SITL build is missing under %s" % px4_dir)
    for model in ("prototype", "prototype_base"):
        if not (source_models / model / "model.sdf").is_file():
            raise FileNotFoundError("missing local Gazebo model %s" % model)
    if not airframe_source.is_file() or not cmake_file.is_file():
        raise FileNotFoundError("PX4 airframe integration inputs are missing")

    for model in ("prototype", "prototype_base"):
        source = (source_models / model).resolve()
        target = target_models / model
        if target.is_symlink():
            if target.resolve() != source:
                raise FileExistsError("refusing to replace unrelated model link: %s" % target)
        elif target.exists():
            raise FileExistsError("refusing to overwrite an existing PX4 model: %s" % target)
        else:
            target.symlink_to(source, target_is_directory=True)

    airframe_target = airframe_target_dir / AIRFRAME
    if airframe_target.exists():
        installed_text = airframe_target.read_text()
        source_text = airframe_source.read_text()
        if installed_text != source_text and GENERATED_MARKER not in installed_text:
            raise FileExistsError("refusing to overwrite existing PX4 airframe: %s" % airframe_target)
        if installed_text != source_text:
            airframe_target.write_text(source_text)
    else:
        airframe_target.write_text(airframe_source.read_text())

    cmake = cmake_file.read_text()
    entries = [line.strip() for line in cmake.splitlines()]
    if AIRFRAME not in entries:
        pattern = re.compile(r"(?m)^([ \t]*)4021_gz_x500_flow[ \t]*$")
        if not pattern.search(cmake):
            raise ValueError("could not find PX4 Gazebo airframe insertion point")
        cmake_file.write_text(
            pattern.sub(lambda match: match.group(0) + "\n\t" + AIRFRAME, cmake, count=1)
        )
    return target_models / "prototype", target_models / "prototype_base", airframe_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-dir", type=Path,
                        default=Path.home() / "PX4-Autopilot")
    parser.add_argument("--flightlxx-dir", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    paths = install(args.px4_dir, args.flightlxx_dir)
    for path in paths:
        print("PX4 Prototype integration:", path)
    print("PX4 build will regenerate its airframe and gz_prototype targets on the next make.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
