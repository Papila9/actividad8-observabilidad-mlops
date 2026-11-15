import tempfile
import unittest
from pathlib import Path

from src.reentrenamiento import evaluate_acceptance
from src.rollback import create_registry, promote_version, register_version, rollback


class ModelLifecycleTests(unittest.TestCase):
    def test_candidate_rejected_without_replacing_stable(self):
        stable = {"version": "stable", "metrics": {"accuracy": 0.80, "f1_macro": 0.75}}
        candidate = {"version": "candidate", "metrics": {"accuracy": 0.78, "f1_macro": 0.69}}
        decision = evaluate_acceptance(candidate, stable)
        registry = create_registry(stable)
        register_version(registry, candidate)
        registry["versions"]["candidate"]["status"] = "rejected_candidate"
        self.assertFalse(decision["accepted"])
        self.assertEqual(registry["active_version"], "stable")

    def test_rollback_returns_previous_stable(self):
        stable = {"version": "stable", "metrics": {"accuracy": 0.80, "f1_macro": 0.75}}
        candidate = {"version": "candidate", "metrics": {"accuracy": 0.81, "f1_macro": 0.76}}
        registry = create_registry(stable)
        register_version(registry, candidate)
        promote_version(registry, "candidate")
        event = rollback(registry, "stable", reason="test", simulation=True)
        self.assertEqual(event["from_version"], "candidate")
        self.assertEqual(registry["active_version"], "stable")


if __name__ == "__main__":
    unittest.main()
