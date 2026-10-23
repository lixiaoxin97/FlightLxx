#!/usr/bin/env python3
"""Structural/parameter checks for the local PX4 Gazebo Prototype model."""

from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import install_prototype_px4
from step08_px4_rates_contract import (
    DEFAULT_VEHICLE_PROFILE,
    PROTOTYPE_THRUST_MODEL,
    add_thrust_model_arguments,
    map_collective_acceleration,
    thrust_model_from_args,
)


MODEL_DIR = HERE / "gazebo/models"


class PrototypeModelTests(unittest.TestCase):
    def test_controller_cli_defaults_to_matching_prototype_thrust_model(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_thrust_model_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(DEFAULT_VEHICLE_PROFILE, "prototype")
        self.assertEqual(args.vehicle_profile, "prototype")
        self.assertEqual(thrust_model_from_args(args), PROTOTYPE_THRUST_MODEL)

    def test_flightlxx_prototype_profile_hover_and_motor_limits(self):
        model = PROTOTYPE_THRUST_MODEL
        self.assertEqual(model["vehicle_mass_kg"], 0.78)
        self.assertEqual(model["motor_omega_min"], 38.0)
        self.assertEqual(model["motor_omega_max"], 1000.0)
        hover = map_collective_acceleration(9.81, **model)
        self.assertAlmostEqual(hover.px4_thrust_fraction, 0.3429551081, places=8)
        self.assertFalse(hover.saturated_high)

    def test_sdf_body_inertia_sensors_and_motor_system(self):
        base = ET.parse(MODEL_DIR / "prototype_base/model.sdf").getroot()
        base_link = base.find(".//link[@name='base_link']")
        self.assertIsNotNone(base_link)
        self.assertAlmostEqual(float(base_link.findtext("./inertial/mass")), 0.779996)
        self.assertAlmostEqual(float(base_link.findtext("./inertial/inertia/ixx")), 0.00229997)
        self.assertAlmostEqual(float(base_link.findtext("./inertial/inertia/izz")), 0.00359994)
        self.assertEqual({x.get("type") for x in base_link.findall("./sensor")},
                         {"air_pressure", "magnetometer", "imu", "navsat"})
        for i in range(4):
            link = base.find(".//link[@name='rotor_%d']" % i)
            self.assertIsNotNone(link)
            self.assertAlmostEqual(float(link.findtext("./inertial/mass")), 1e-6)
            self.assertIsNotNone(base.find(".//joint[@name='rotor_%d_joint']" % i))

        sdf = ET.parse(MODEL_DIR / "prototype/model.sdf").getroot()
        self.assertEqual(len(sdf.findall(".//plugin[@filename='gz-sim-multicopter-motor-model-system']")), 4)
        for plugin in sdf.findall(".//plugin[@filename='gz-sim-multicopter-motor-model-system']"):
            self.assertAlmostEqual(float(plugin.findtext("motorConstant")),
                                   PROTOTYPE_THRUST_MODEL["motor_thrust_coefficient"])
            self.assertEqual(float(plugin.findtext("maxRotVelocity")),
                             PROTOTYPE_THRUST_MODEL["motor_omega_max"])
            self.assertEqual(float(plugin.findtext("timeConstantUp")), .033)
            self.assertEqual(float(plugin.findtext("timeConstantDown")), .033)

    def test_px4_airframe_matches_flightlxx_motor_geometry(self):
        airframe = (HERE / "px4_airframes/22001_gz_prototype").read_text()
        expected = [
            ("0.08838835", "-0.08838835", "-0.01"),
            ("0.08838835", "0.08838835", "0.01"),
            ("-0.08838835", "0.08838835", "-0.01"),
            ("-0.08838835", "-0.08838835", "0.01"),
        ]
        self.assertIn("PX4_SIM_MODEL=${PX4_SIM_MODEL:=gz_prototype}", airframe)
        self.assertIn("param set-default NAV_DLL_ACT 0", airframe)
        # Prototype's much smaller inertia needs a conservative inner-loop
        # baseline instead of PX4's x500-oriented default rate gains.
        for name, value in (
            ("MC_ROLLRATE_P", "0.0165"), ("MC_ROLLRATE_I", "0.020"),
            ("MC_ROLLRATE_D", "0.0003"),
            ("MC_PITCHRATE_P", "0.0165"), ("MC_PITCHRATE_I", "0.020"),
            ("MC_PITCHRATE_D", "0.0003"),
            ("MC_YAWRATE_P", "0.018"), ("MC_YAWRATE_I", "0.009"),
            ("MC_YAWRATE_D", "0.0"),
        ):
            self.assertIn("param set-default %s %s" % (name, value), airframe)
        for i, (x, y, km) in enumerate(expected):
            self.assertRegex(airframe, r"CA_ROTOR%d_PX\s+%s" % (i, x))
            self.assertRegex(airframe, r"CA_ROTOR%d_PY\s+%s" % (i, y))
            self.assertRegex(airframe, r"CA_ROTOR%d_KM\s+%s" % (i, km))
            self.assertIn("SIM_GZ_EC_MIN%d 38" % (i+1), airframe)
            self.assertIn("SIM_GZ_EC_MAX%d 1000" % (i+1), airframe)
        self.assertIn("MPC_THR_HOVER 0.34296", airframe)

    def test_prototype_motor_fit_tracks_training_curve_across_output_range(self):
        model = PROTOTYPE_THRUST_MODEL
        a, b, c = (1.8495740127948213e-8, -7.055596047224982e-5,
                   6.546158341970494e-2)
        k = model["motor_thrust_coefficient"]
        worst_relative_error = 0.0
        for output_fraction in (0.0, 0.01, 0.034, 0.1, 0.25, 0.345, 0.5, 0.75, 1.0):
            rpm = 3000.0 + output_fraction * (29400.0 - 3000.0)
            omega = model["motor_omega_min"] + output_fraction * (
                model["motor_omega_max"] - model["motor_omega_min"]
            )
            flightlxx_force = a * rpm**2 + b * rpm + c
            gazebo_force = k * omega**2
            relative_error = abs(gazebo_force / flightlxx_force - 1.0)
            worst_relative_error = max(worst_relative_error, relative_error)
        self.assertLess(worst_relative_error, 0.012)

    def test_px4_allocator_effectiveness_matches_flightlxx_b_matrix(self):
        airframe = (HERE / "px4_airframes/22001_gz_prototype").read_text().splitlines()
        sdf = ET.parse(MODEL_DIR / "prototype_base/model.sdf").getroot()
        values = {}
        for line in airframe:
            words = line.split()
            if len(words) == 4 and words[:2] == ["param", "set-default"]:
                values[words[2]] = float(words[3])
        # SDF is FLU, PX4 allocator is FRD. For thrust +Z in FLU, FlightLxx
        # torque is (y,-x,0) and yaw reaction is kappa*[+,-,+,-].
        px4 = []
        for i in range(4):
            x, y = values["CA_ROTOR%d_PX" % i], values["CA_ROTOR%d_PY" % i]
            km = values["CA_ROTOR%d_KM" % i]
            # PX4 coordinates -> physical FLU. PX4 thrust axis is -Z in FRD.
            x_flu, y_flu = x, -y
            pose = tuple(map(float, sdf.findtext(
                ".//link[@name='rotor_%d']/pose" % i).split()))
            self.assertAlmostEqual(x_flu, pose[0], places=7)
            self.assertAlmostEqual(y_flu, pose[1], places=7)
            px4_yaw_in_flu = -km
            px4.append((y_flu / .08838835, -x_flu / .08838835,
                        px4_yaw_in_flu / .01))
        b_actual = tuple(tuple(row[i] for i in range(4)) for row in zip(*px4))
        b_expected = ((1, -1, -1, 1), (-1, -1, 1, 1), (1, -1, 1, -1))
        for actual_row, expected_row in zip(b_actual, b_expected):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=5)

    def test_px4_installer_is_idempotent_and_refuses_unrelated_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            px4 = root / "PX4"
            models = px4 / "Tools/simulation/gz/models"
            airframes = px4 / "ROMFS/px4fmu_common/init.d-posix/airframes"
            models.mkdir(parents=True)
            airframes.mkdir(parents=True)
            (px4 / "build/px4_sitl_default").mkdir(parents=True)
            cmake = airframes / "CMakeLists.txt"
            cmake.write_text("  4021_gz_x500_flow\n")
            paths = install_prototype_px4.install(px4, HERE.parent)
            self.assertTrue(all(p.exists() for p in paths))
            self.assertTrue((models / "prototype").is_symlink())
            before = cmake.read_text()
            install_prototype_px4.install(px4, HERE.parent)
            self.assertEqual(cmake.read_text(), before)

            with tempfile.TemporaryDirectory() as tmp2:
                other = Path(tmp2)
                other_models = other / "Tools/simulation/gz/models"
                other_airframes = other / "ROMFS/px4fmu_common/init.d-posix/airframes"
                other_models.mkdir(parents=True)
                other_airframes.mkdir(parents=True)
                (other / "build/px4_sitl_default").mkdir(parents=True)
                (other_models / "prototype").mkdir()
                (other_airframes / "CMakeLists.txt").write_text("  4021_gz_x500_flow\n")
                self.assertRaises(FileExistsError,
                                  install_prototype_px4.install, other, HERE.parent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
