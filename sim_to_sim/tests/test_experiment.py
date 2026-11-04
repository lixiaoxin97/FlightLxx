"""Regression checks for the consolidated entry point and inference worker."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE/"runtime"))
from policy_worker import PolicyWorker
from runner import schedule
from fixed_pid import PROFILE


class WorkflowTests(unittest.TestCase):
    def test_preserved_banks_keep_identical_model_order_and_requested_states(self):
        sources = {
            "curriculum": HERE/"logs/continued_pid_20261029_v1/rounds/r005/formal_continue_736ee51a5cf0/trials_batch",
            "random": HERE/"logs/full_random_pid_20261030_v1/evaluation",
        }
        for name, folder in sources.items():
            bank = json.loads((HERE/"states"/(name+"_1000.json")).read_text())
            original = json.loads((folder/"manifest.json").read_text())["schedule"]
            actual = schedule(bank)
            self.assertEqual(len(actual), 3000)
            for a, b in zip(actual, original):
                for key in ("trial_id", "state", "model", "profile", "ordinal"):
                    self.assertEqual(a[key], b[key])
                self.assertEqual(a["profile"], PROFILE)

    def test_background_inference_produces_typed_result_with_physical_mapping(self):
        class Client:
            timeout = .05
            def infer(self, observation, sequence):
                return [.1, -.1, .2, 0.]
            def close(self):
                pass
        worker = PolicyWorker(Client())
        try:
            observation = [0,0,5]+[0]*9
            worker.submit(observation, time.monotonic(), {"reset_counter": 3})
            deadline = time.monotonic()+2
            while time.monotonic() < deadline:
                result, error, count = worker.snapshot()
                if result is not None or error:
                    break
                time.sleep(.005)
            self.assertIsNone(error)
            self.assertEqual(count, 1)
            self.assertEqual(result.source_odometry["reset_counter"], 3)
            self.assertAlmostEqual(result.collective_thrust, 9.81)
            self.assertAlmostEqual(result.thrust_body_frd[2], -.3429551081)
        finally:
            self.assertTrue(worker.close())

    def test_preserved_roots_and_ancestors_cannot_be_output_targets(self):
        spec = importlib.util.spec_from_file_location("experiment_entry", HERE/"experiment.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        for root in entry.PROTECTED:
            for path in (root, root/"new_run", root.parent):
                with self.assertRaises(ValueError):
                    entry.writable_output(path)
        self.assertEqual(entry.writable_output(HERE/"logs/new_run"), HERE/"logs/new_run")


if __name__ == "__main__":
    unittest.main()
