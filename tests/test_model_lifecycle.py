import tempfile
import unittest
from pathlib import Path

from src.reentrenamiento import evaluate_acceptance
from src.rollback import artifact_sha256, create_registry, load_active_model, promote_version, register_version, rollback


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
        import pickle
        with tempfile.TemporaryDirectory() as directory:
            stable_path = Path(directory) / "stable.pkl"
            candidate_path = Path(directory) / "candidate.pkl"
            stable_path.write_bytes(pickle.dumps({"kind": "stable"}))
            candidate_path.write_bytes(pickle.dumps({"kind": "candidate"}))
            stable = {"version": "stable", "artifact_path": str(stable_path), "artifact_sha256": artifact_sha256(stable_path), "metrics": {"accuracy": 0.80, "f1_macro": 0.75}}
            candidate = {"version": "candidate", "artifact_path": str(candidate_path), "artifact_sha256": artifact_sha256(candidate_path), "metrics": {"accuracy": 0.81, "f1_macro": 0.76}}
            registry = create_registry(stable)
            register_version(registry, candidate)
            promote_version(registry, "candidate")
            event = rollback(registry, "stable", reason="test", simulation=True)
            self.assertEqual(event["from_version"], "candidate")
            self.assertTrue(event["model_loaded_successfully"])
            self.assertEqual(registry["active_version"], "stable")

    def test_rejected_candidate_requires_explicit_authorization(self):
        stable = {"version": "stable", "metrics": {"accuracy": 0.80, "f1_macro": 0.75}}
        candidate = {"version": "candidate", "metrics": {"accuracy": 0.78, "f1_macro": 0.69}}
        registry = create_registry(stable)
        register_version(registry, candidate, status="rejected_candidate")
        with self.assertRaises(ValueError):
            promote_version(registry, "candidate")

    def test_load_active_model_verifies_hash_and_pickle(self):
        import pickle
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            with path.open("wb") as stream:
                pickle.dump({"kind": "test"}, stream)
            registry = create_registry({"version": "stable", "artifact_path": str(path), "artifact_sha256": artifact_sha256(path), "metrics": {}})
            self.assertEqual(load_active_model(registry), {"kind": "test"})
            path.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                load_active_model(registry)


if __name__ == "__main__":
    unittest.main()
