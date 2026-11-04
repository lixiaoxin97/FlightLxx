import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import math
from pathlib import Path
import tempfile
import unittest
from ramp_initialization import at_offset,spawn_state,initializer_sensor_reference
from recovery_state import quaternion_from_zyx
from sitl_runtime import prepare_slot
from sitl_runtime import command
import subprocess
import sys


class TrajectoryMathTests(unittest.TestCase):
    def test_reference_only_applies_during_constrained_trajectory(self):
        state=dict(holding=True,prepared=True,trajectory_target_sim_time_s=1.,velocity_enu=[.2,.3,.5],
            angular_velocity_world_enu=[.4,.5,.6],prepared_world_velocity_enu=[.1,.2,.3],
            prepared_body_rates_flu=[.2,0.,0.],quaternion_enu_flu_wxyz=[1.,0.,0.,0.])
        ref=initializer_sensor_reference(state)
        self.assertEqual(ref['velocity_enu'],[.1,.2,.3]);self.assertEqual(state['velocity_enu'],[.2,.3,.5])
        state['holding']=False
        self.assertIs(initializer_sensor_reference(state),state)
    def test_leadin_reaches_target_pose_and_twist(self):
        target=[.3,-.4,5.3,.15,-.12,6.2,.4,-.3,.2,-.4,.3,.35]
        back=at_offset(target,-.5);forward=at_offset(back,.5)
        for i in (0,1,2,6,7,8,9,10,11):self.assertAlmostEqual(forward[i],target[i],places=10)
        a=quaternion_from_zyx(*forward[3:6]);b=quaternion_from_zyx(*target[3:6])
        self.assertAlmostEqual(abs(sum(x*y for x,y in zip(a,b))),1.,places=10)

    def test_static_spawn_uses_backward_pose_and_zero_twist(self):
        target=[0,0,5,0,0,0,.4,-.3,.2,.2,.1,-.1]
        state=spawn_state(target,.5)
        self.assertAlmostEqual(state[0],-.2);self.assertEqual(state[6:],[0.]*6)

    def test_generated_launcher_uses_target_for_back_calculation(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);px4=base/'px4';source=px4/'build/px4_sitl_default/rootfs';source.mkdir(parents=True)
            p=prepare_slot(base/'slot',px4,initializer='trajectory')
            text=(p/'launcher.sh').read_text()
            calc=next(line for line in text.splitlines() if line.startswith('read -r -a SPAWN_STATE'))
            self.assertIn('--state "${STATE[@]}"',calc)
            self.assertIn('trajectory_controller.py',text)
            self.assertIn('--physical-anchor "${SPAWN_STATE[0]}"',text)

    def test_tiny_negative_state_arguments_are_decimal_not_options(self):
        target=[0.,-2e-7,5.,0.,0.,0.,-1e-8,0.,0.,0.,0.,0.]
        args=command('/tmp/example','/tmp/px4',target)[-12:]
        self.assertFalse(any('e' in x.lower() for x in args))
        result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'runtime/ramp_initialization.py'),
            '--state',*args,'--shell'],text=True,capture_output=True,check=True)
        self.assertFalse(any('e' in x.lower() for x in result.stdout.split()))
        self.assertAlmostEqual(float(args[6]),-1e-8,places=17)


if __name__=='__main__':unittest.main()
