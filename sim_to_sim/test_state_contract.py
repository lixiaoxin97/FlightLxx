#!/usr/bin/env python3
"""Offline coordinate regressions; no ROS, inference, or vehicle commands."""

import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from step02_state_contract import (
    px4_frd_to_flu,
    px4_ned_to_flu,
    px4_quaternion_to_flu,
    quaternion_to_training_ypr_rad,
    quaternion_to_ypr_rad,
    rotate_attitude_world_yaw,
    rotate_world_vector_yaw,
    vehicle_odometry_to_observation,
)
from step07_ros2_policy_bridge import message_to_observation
from step08_px4_rates_contract import convert_raw_action_to_px4
from step12_yaw_calibration import raw_flu_yaw_from_px4_q


def matmul(a, b):
    return tuple(tuple(sum(x*y for x, y in zip(row, col)) for col in zip(*b))
                 for row in a)


def matvec(a, v):
    return tuple(sum(x*y for x, y in zip(row, v)) for row in a)


def zyx_matrix(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = ((cy, -sy, 0), (sy, cy, 0), (0, 0, 1))
    ry = ((cp, 0, sp), (0, 1, 0), (-sp, 0, cp))
    rx = ((1, 0, 0), (0, cr, -sr), (0, sr, cr))
    return matmul(matmul(rz, ry), rx)


def quaternion_from_ypr(yaw, pitch, roll):
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    return (cy*cp*cr + sy*sp*sr, cy*cp*sr - sy*sp*cr,
            cy*sp*cr + sy*cp*sr, sy*cp*cr - cy*sp*sr)


def quaternion_matrix(q):
    length = math.sqrt(sum(x*x for x in q))
    w, x, y, z = [v/length for v in q]
    return ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
            (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
            (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))


def flu_input(position, ypr, velocity, body_rate, **kwargs):
    q = quaternion_from_ypr(*ypr)
    return vehicle_odometry_to_observation(
        position_ned=(position[0], -position[1], -position[2]),
        quaternion_ned_frd_wxyz=(q[0], q[1], -q[2], -q[3]),
        velocity_ned=(velocity[0], -velocity[1], -velocity[2]),
        angular_velocity_frd=(body_rate[0], -body_rate[1], -body_rate[2]),
        **kwargs,
    )


class StateContractTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected, tolerance=2e-11):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, delta=tolerance)

    def assertMatrixClose(self, actual, expected, tolerance=2e-11):
        for row_a, row_b in zip(actual, expected):
            self.assertVectorClose(row_a, row_b, tolerance)

    def test_ned_frd_basis_and_arbitrary_attitude(self):
        self.assertEqual(px4_ned_to_flu((1, 2, 3)), (1, -2, -3))
        self.assertEqual(px4_frd_to_flu((4, 5, 6)), (4, -5, -6))
        basis = ((1, 0, 0), (0, -1, 0), (0, 0, -1))
        q = quaternion_from_ypr(-1.3, .7, -2.2)
        self.assertMatrixClose(quaternion_matrix(px4_quaternion_to_flu(q)),
                               matmul(matmul(basis, quaternion_matrix(q)), basis))

    def test_zero_reference_preserves_world_vectors(self):
        obs = vehicle_odometry_to_observation(
            (1, 2, -3), (1, 0, 0, 0), (4, 5, -6), (.1, .2, .3))
        self.assertVectorClose(obs, (1, -2, 3, 0, 0, 0, 4, -5, 6, .1, -.2, -.3))

    def test_quarter_turn_rotates_position_error_and_velocity(self):
        # Translation must happen BEFORE world rotation, and goal AFTER it.
        obs = flu_input((11, 22, 6), (-math.pi/2, 0, 0), (4, 5, 6), (.1, .2, .3),
                        physical_hover=(10, 20, 5), nn_goal=(3, 4, 7),
                        physical_yaw_reference=-math.pi/2)
        self.assertVectorClose(obs, (1, 5, 8, 0, 0, 0, -5, 4, 6, .1, .2, .3))

    def test_hover_origin_maps_to_network_goal(self):
        for ref, yaw_goal in ((-1.584512435, 0), (.8, 1.2), (2.9, -2.8)):
            obs = flu_input((2, -3, 5), (ref, .2, -.4), (0, 0, 0), (0, 0, 0),
                            physical_hover=(2, -3, 5), nn_goal=(1, 4, 6),
                            physical_yaw_reference=ref, nn_yaw_goal=yaw_goal)
            self.assertVectorClose(obs[:3], (1, 4, 6))
            self.assertMatrixClose(zyx_matrix(*obs[3:6]), zyx_matrix(yaw_goal, .2, -.4))

    def test_arbitrary_world_rotation_is_left_multiplication(self):
        rng = random.Random(1701)
        for _ in range(250):
            ypr = tuple(rng.uniform(-math.pi, math.pi) for _ in range(3))
            alpha = rng.uniform(-math.pi, math.pi)
            q = quaternion_from_ypr(*ypr)
            transformed = rotate_attitude_world_yaw(q, alpha)
            expected = matmul(zyx_matrix(alpha, 0, 0), zyx_matrix(*ypr))
            self.assertMatrixClose(quaternion_matrix(transformed), expected)
            self.assertMatrixClose(zyx_matrix(*quaternion_to_ypr_rad(transformed)), expected)

    def test_full_random_states_match_one_world_transform(self):
        rng = random.Random(2311)
        for _ in range(250):
            vectors = [tuple(rng.uniform(-3, 3) for _ in range(3)) for _ in range(5)]
            pos, vel, rate, hover, goal = vectors
            ypr = tuple(rng.uniform(-math.pi, math.pi) for _ in range(3))
            ref, target = rng.uniform(-3, 3), rng.uniform(-3, 3)
            rz = zyx_matrix(target-ref, 0, 0)
            obs = flu_input(pos, ypr, vel, rate, physical_hover=hover, nn_goal=goal,
                            physical_yaw_reference=ref, nn_yaw_goal=target)
            expected_error = matvec(rz, tuple(p-h for p, h in zip(pos, hover)))
            self.assertVectorClose(obs[:3], tuple(e+g for e, g in zip(expected_error, goal)))
            self.assertVectorClose(obs[6:9], matvec(rz, vel))
            self.assertVectorClose(obs[9:12], rate)
            self.assertMatrixClose(zyx_matrix(*obs[3:6]), matmul(rz, zyx_matrix(*ypr)))

    def test_velocity_is_derivative_of_transformed_position(self):
        pos, vel, ypr = (4, -2, 6), (.7, -.4, .2), (-1.58, .2, -.3)
        kwargs = dict(physical_hover=(2, 1, 5), nn_goal=(-1, 3, 5),
                      physical_yaw_reference=-1.58, nn_yaw_goal=.4)
        obs = flu_input(pos, ypr, vel, (.1, .2, .3), **kwargs)
        dt = 1e-4
        obs_next = flu_input(tuple(p+dt*v for p, v in zip(pos, vel)), ypr, vel,
                             (.1, .2, .3), **kwargs)
        self.assertVectorClose(tuple((b-a)/dt for a, b in zip(obs[:3], obs_next[:3])),
                               obs[6:9], tolerance=2e-10)

    def test_thrust_acceleration_and_gravity_share_world_frame(self):
        ypr = (-1.63, .3, -.2)
        alpha, thrust = 1.584512435, 11.0
        obs = flu_input((.1, -.2, 5), ypr, (.2, .3, 0), (.1, -.2, .3),
                        physical_yaw_reference=-alpha)
        raw_force = matvec(zyx_matrix(*ypr), (0, 0, thrust))
        acceleration = tuple(f+g for f, g in zip(raw_force, (0, 0, -9.81)))
        rotated_acceleration = matvec(zyx_matrix(alpha, 0, 0), acceleration)
        predicted_force = matvec(zyx_matrix(*obs[3:6]), (0, 0, thrust))
        self.assertVectorClose(rotated_acceleration,
                               tuple(f+g for f, g in zip(predicted_force, (0, 0, -9.81))))

    def test_body_rate_action_does_not_follow_world_rotation(self):
        obs = flu_input((0, 0, 5), (-1.58, 0, 0), (0, 0, 0), (.1, .2, .3),
                        physical_yaw_reference=-1.58)
        self.assertVectorClose(obs[9:12], (.1, .2, .3))
        rates, thrust, _, _ = convert_raw_action_to_px4((.1, .2, .3, 0))
        # Keep FlightLxx's established 3.1415926 action scaling.
        self.assertVectorClose(rates, (.2*3.1415926, -.4*3.1415926, -.3*3.1415926))
        self.assertLess(thrust[2], 0)

    def test_deployment_euler_is_principal_and_preserves_attitude(self):
        q = quaternion_from_ypr(-.2, .1, -.3)
        self.assertVectorClose(quaternion_to_ypr_rad(q), (-.2, .1, -.3))
        obs = flu_input((0, 0, 5), (-.2, .1, -.3), (0, 0, 0), (0, 0, 0))
        self.assertVectorClose(obs[3:6], (-.2, .1, -.3))
        self.assertMatrixClose(zyx_matrix(*obs[3:6]), quaternion_matrix(q))
        self.assertVectorClose(quaternion_to_training_ypr_rad(q),
                               (math.pi-.2, math.pi-.1, math.pi-.3))
        q_px4 = (q[0], q[1], -q[2], -q[3])
        self.assertAlmostEqual(raw_flu_yaw_from_px4_q(q_px4), -.2)

    def test_deployment_euler_is_continuous_across_zero_yaw(self):
        epsilon = 1.0e-6
        negative = flu_input((0, 0, 5), (-epsilon, 0, 0), (0, 0, 0), (0, 0, 0))
        positive = flu_input((0, 0, 5), (epsilon, 0, 0), (0, 0, 0), (0, 0, 0))
        self.assertVectorClose(negative[3:6], (-epsilon, 0, 0), tolerance=2e-9)
        self.assertVectorClose(positive[3:6], (epsilon, 0, 0), tolerance=2e-9)
        self.assertLess(math.dist(negative[3:6], positive[3:6]), 3e-6)

    def test_physical_yaw_alignment_returns_to_zero_network_yaw(self):
        raw_yaw = -1.583687388
        q = quaternion_from_ypr(raw_yaw, 0.01, -0.02)
        obs = vehicle_odometry_to_observation(
            (0, 0, -5), (q[0], q[1], -q[2], -q[3]),
            (0, 0, 0), (0, 0, 0),
            physical_yaw_reference=raw_yaw,
            nn_yaw_goal=0.0,
        )
        self.assertAlmostEqual(obs[3], 0.0, delta=2e-8)
        self.assertAlmostEqual(obs[4], 0.01, delta=2e-8)
        self.assertAlmostEqual(obs[5], -0.02, delta=2e-8)

    def test_quaternion_sign_scale_and_wraparound(self):
        for ypr in ((1.2, .4, -.8), (-.2, -.3, 2.8), (3.1, 2.1, -2)):
            q = quaternion_from_ypr(*ypr)
            expected = quaternion_to_training_ypr_rad(q)
            for scale in (-1, 4.5):
                self.assertVectorClose(quaternion_to_training_ypr_rad(tuple(scale*x for x in q)), expected)
            for pitch in (-math.pi/2, math.pi/2):
                singular = quaternion_from_ypr(ypr[0], pitch, ypr[2])
                self.assertMatrixClose(zyx_matrix(*quaternion_to_training_ypr_rad(singular)),
                                       quaternion_matrix(singular))
        q = quaternion_from_ypr(math.pi-.01, .2, -.1)
        transformed = rotate_attitude_world_yaw(q, .03)
        self.assertMatrixClose(zyx_matrix(*quaternion_to_training_ypr_rad(transformed)),
                               zyx_matrix(-math.pi+.02, .2, -.1))

    def test_ros_bridge_uses_shared_contract(self):
        q = quaternion_from_ypr(.8, -.2, .1)
        msg = SimpleNamespace(position=(1, 2, -5), q=q, velocity=(.3, -.4, .1),
                              angular_velocity=(.1, .2, .3))
        kwargs = dict(physical_hover=(1, 2, 5), nn_goal=(0, 0, 5),
                      physical_yaw_reference=-.8, nn_yaw_goal=.1)
        expected = vehicle_odometry_to_observation(msg.position, q, msg.velocity,
                                                   msg.angular_velocity, **kwargs)
        self.assertVectorClose(message_to_observation(msg, **kwargs), expected)

    def test_nonfinite_and_invalid_inputs_are_rejected(self):
        arguments = dict(position_ned=(0, 0, -5), quaternion_ned_frd_wxyz=(1, 0, 0, 0),
                         velocity_ned=(0, 0, 0), angular_velocity_frd=(0, 0, 0))
        for key, value in (("position_ned", (math.nan, 0, 0)),
                           ("velocity_ned", (0, math.inf, 0)),
                           ("quaternion_ned_frd_wxyz", (0, 0, 0, 0)),
                           ("physical_yaw_reference", math.inf),
                           ("nn_yaw_goal", math.nan)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                vehicle_odometry_to_observation(**dict(arguments, **{key: value}))

    @unittest.skipUnless(shutil.which("g++") and Path("/usr/include/eigen3/Eigen/Geometry").is_file(),
                         "optional C++ Eigen reference requires g++ and eigen3 headers")
    def test_training_euler_matches_real_eigen_cpp(self):
        # Independent oracle: compile the exact expression in rl_env.cpp.
        source = r'''
#include <Eigen/Geometry>
#include <iomanip>
#include <iostream>
int main() {
    double w, x, y, z;
    std::cout << std::setprecision(17);
    while (std::cin >> w >> x >> y >> z) {
        Eigen::Quaterniond q(w, x, y, z);
        q.normalize();
        auto angles = q.toRotationMatrix().eulerAngles(2, 1, 0).eval();
        std::cout << angles[0] << ' ' << angles[1] << ' ' << angles[2] << '\n';
    }
}
'''
        rng = random.Random(999)
        quats = [quaternion_from_ypr(y, p, r)
                 for y in (-3.1, -.3, -1e-8, 0, 1e-8, .3, 3.1)
                 for p in (-2.5, -.4, .2, 2.5) for r in (-2.7, .3, 2.7)]
        quats += [tuple(rng.uniform(-1, 1) for _ in range(4)) for _ in range(1000)]
        with tempfile.TemporaryDirectory(prefix="flightlxx_frame_test_") as tmp:
            cpp, exe = Path(tmp)/"eigen_reference.cpp", Path(tmp)/"eigen_reference"
            cpp.write_text(source)
            subprocess.run(["g++", "-std=c++17", "-O2", "-I/usr/include/eigen3",
                            str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
            result = subprocess.run([str(exe)], check=True, capture_output=True, text=True,
                                    input="\n".join(" ".join(map(str, q)) for q in quats)+"\n")
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), len(quats))
        for q, line in zip(quats, lines):
            self.assertVectorClose(quaternion_to_training_ypr_rad(q),
                                   tuple(map(float, line.split())), tolerance=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
