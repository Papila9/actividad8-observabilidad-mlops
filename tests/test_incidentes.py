import tempfile
import unittest
from pathlib import Path

from src.incidentes import build_incident, classify_alert


class IncidentTests(unittest.TestCase):
    def test_classifies_drift_and_slo_incidents(self):
        self.assertEqual(classify_alert({"alert_type": "data_drift"}), "data_drift")
        self.assertEqual(classify_alert({"alert_type": "slo_breach", "indicator": "latency_p95"}), "slo_latency")

    def test_runbook_preserves_simulation_boundary(self):
        incident = build_incident({"alert_type": "data_drift", "severity": "high"})
        self.assertTrue(incident["simulation"]["is_simulation"])
        self.assertIn("etiquetas reales", incident["data_drift_boundary"])
        self.assertEqual(incident["status"], "open")
        self.assertEqual([step["step"] for step in incident["lifecycle"]], ["detection", "classification"])


if __name__ == "__main__":
    unittest.main()
