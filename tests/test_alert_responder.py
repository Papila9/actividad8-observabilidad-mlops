import json
import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.responder_alertas import respond_to_incident
from src.rollback import artifact_sha256, create_registry, save_registry


class AlertResponderTests(unittest.TestCase):
    def test_latency_response_rolls_back_and_records_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stable_path = root / "stable.pkl"
            stable_path.write_bytes(pickle.dumps({"model": "stable"}))
            registry_path = root / "registry.json"
            save_registry(registry_path, create_registry({"version": "stable", "artifact_path": str(stable_path), "artifact_sha256": artifact_sha256(stable_path), "metrics": {}}))
            result = respond_to_incident({"incident_id": "inc-1", "incident_type": "slo_latency"}, registry_path=registry_path)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["result"]["model_loaded_successfully"])

    def test_drift_response_requires_data_path(self):
        result = respond_to_incident({"incident_id": "inc-2", "incident_type": "data_drift"})
        self.assertEqual(result["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
