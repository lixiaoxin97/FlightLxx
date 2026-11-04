import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

from full_random_initialization import preparation_spawn,validate_preparation_state,preparation_validation
from state_bank import generate
from ramp_initialization import at_offset,spawn_state
from recovery_state import validate_training_state,quaternion_from_zyx
import recovery_state
from sitl_runtime import prepare_slot as prepare_full_random_slot

HERE=Path(__file__).resolve().parent


class FullRandomBankTests(unittest.TestCase):
    def test_seeded_independent_draws_are_kept_exactly_without_strata(self):
        seed=2026103007;bank=generate(seed,1000)
        self.assertEqual(bank,generate(seed,1000))
        self.assertEqual(bank["rejected"],0)
        self.assertEqual(bank["generated_draws"],12000)
        self.assertTrue(all("level" not in s for s in bank["states"]))
        rng=random.Random(seed)
        for s in bank["states"]:
            p=[rng.uniform(-1.,1.) for _ in range(3)]
            angles=[rng.uniform(-math.pi,math.pi),rng.uniform(-math.pi,math.pi),rng.uniform(0.,2*math.pi)]
            v=[rng.uniform(-1.,1.) for _ in range(3)];w=[rng.uniform(-1.,1.) for _ in range(3)]
            self.assertEqual(s["state"],[p[0],p[1],5+p[2],*angles,*v,*w])
            validate_training_state(s["state"])
        for axis in (0,1,6,7,8,9,10,11):
            values=[s["state"][axis] for s in bank["states"]]
            self.assertLess(min(values),-.98);self.assertGreater(max(values),.98)
        self.assertTrue(any(s["state"][4]>math.pi/2 for s in bank["states"]))
        self.assertTrue(any(s["state"][4]<-math.pi/2 for s in bank["states"]))
        self.assertTrue(any(s["state"][5]>math.pi for s in bank["states"]))

    def test_initially_inside_cases_are_not_rejected(self):
        bank=generate(17,12000)
        inside=[s for s in bank["states"] if s["initially_inside_position_speed_band"]]
        self.assertGreater(len(inside),0)
        self.assertEqual(bank["initially_inside_band_kept"],len(inside))
        self.assertEqual(len(bank["states"]),12000)


class FullRandomPreparationTests(unittest.TestCase):
    def test_all_position_velocity_cube_corners_are_supported_without_widening_release_bounds(self):
        for p in itertools.product((-1.,1.),repeat=3):
            for v in itertools.product((-1.,1.),repeat=3):
                target=[p[0],p[1],5+p[2],.2,-.3,6.,*v,.8,-.9,.6]
                back=preparation_spawn(target)
                validate_preparation_state(back)
                for i in range(3):self.assertAlmostEqual(back[i],target[i]-.35*v[i])
                self.assertEqual(back[6:],[0.]*6)
                moving=list(back);moving[6:]=target[6:]
                end=at_offset(moving,.35)
                for i in range(3):self.assertAlmostEqual(end[i],target[i])
                qa=quaternion_from_zyx(*target[3:6]);qb=quaternion_from_zyx(*end[3:6])
                self.assertAlmostEqual(abs(sum(a*b for a,b in zip(qa,qb))),1.,places=12)
        target=[1.,-1.,6.,0.,0.,0.,-1.,1.,-1.,0.,0.,0.]
        with self.assertRaises(ValueError):spawn_state(target,.35)
        back=preparation_spawn(target)
        with self.assertRaises(ValueError):validate_training_state(back)
        before=recovery_state.validate_training_state
        with preparation_validation():
            result=recovery_state.recovery_spawn(back,"principal_zyx")
            self.assertAlmostEqual(result["expected_policy_observation"][0],1.35)
        self.assertIs(recovery_state.validate_training_state,before)

    def test_new_preparation_matches_old_where_old_is_valid(self):
        target=[.2,-.1,5.1,.2,-.3,5.9,.3,-.2,.1,.4,-.3,.2]
        self.assertEqual(preparation_spawn(target),spawn_state(target,.35))
        with self.assertRaises(ValueError):preparation_spawn(target,.36)
        invalid=target.copy();invalid[0]=1.01
        with self.assertRaises(ValueError):preparation_spawn(invalid)

    def test_generated_launcher_extends_only_spawn_pose_probe(self):
        with tempfile.TemporaryDirectory() as d:
            slot=prepare_full_random_slot(Path(d)/"runtime",Path.home()/"PX4-Autopilot",initializer="trajectory")
            program=(slot/"launcher.sh").read_text()
            self.assertIn('"$SCRIPT_DIR/full_random_initialization.py" spawn',program)
            self.assertIn('"$SCRIPT_DIR/full_random_initialization.py" pose',program)
            self.assertIn('"$SCRIPT_DIR/full_random_initialization.py" probe',program)
            self.assertIn('"$SCRIPT_DIR/trajectory_controller.py"',program)
            self.assertIn('--state "${STATE[@]}"',program)
            self.assertIn('--physical-anchor "${SPAWN_STATE[0]}"',program)
            self.assertEqual(program.count('"$SCRIPT_DIR/fixed_pid.py"'),2)
            subprocess.run(["bash","-n",str(slot/"launcher.sh")],check=True)


if __name__=="__main__":unittest.main()
