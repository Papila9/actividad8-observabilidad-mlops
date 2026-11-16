import tempfile
import unittest
from pathlib import Path

from src.alertas import AlertManager


class AlertTests(unittest.TestCase):
    def test_deduplication_and_cooldown(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [100.0]
            manager = AlertManager(Path(directory) / "alerts.jsonl", cooldown_seconds=60, clock=lambda: now[0])
            detection = {
                "data_drift": {"severity": "high", "signal": 0.3},
                "class_distribution_shift": {"status": "unavailable_without_labels"},
                "model_drift": {"status": "unavailable_without_real_labels"},
            }
            first = manager.evaluate(detection, execution_id="exec-1", window_id="window-1")
            duplicate_window = manager.evaluate(detection, execution_id="exec-1", window_id="window-1")
            cooldown = manager.evaluate(detection, execution_id="exec-2", window_id="window-2")
            now[0] += 61
            after_cooldown = manager.evaluate(detection, execution_id="exec-3", window_id="window-3")
            self.assertEqual(len(first), 1)
            self.assertEqual(len(duplicate_window), 0)
            self.assertEqual(len(cooldown), 0)
            self.assertEqual(len(after_cooldown), 1)

    def test_state_persists_after_process_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "alerts.jsonl"
            now = [100.0]
            detection = {"data_drift": {"severity": "high", "signal": 0.3}, "class_distribution_shift": {"status": "unavailable_without_labels"}, "model_drift": {"status": "unavailable_without_real_labels"}}
            first = AlertManager(log_path, cooldown_seconds=60, clock=lambda: now[0])
            self.assertEqual(len(first.evaluate(detection, execution_id="exec-1", window_id="window-1")), 1)
            restarted = AlertManager(log_path, cooldown_seconds=60, clock=lambda: now[0])
            self.assertEqual(len(restarted.evaluate(detection, execution_id="exec-2", window_id="window-2")), 0)
            self.assertEqual(restarted.stats()["alerts_suppressed"], 1)

    def test_model_critical_alert_requires_supervised_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = AlertManager(Path(directory) / "alerts.jsonl")
            detection = {
                "data_drift": {"severity": "high"},
                "class_distribution_shift": {"status": "unavailable_without_labels"},
                "model_drift": {"severity": "critical", "requires_real_labels": True},
            }
            alerts = manager.evaluate(detection, execution_id="exec-1", window_id="window-1")
            self.assertEqual([alert["severity"] for alert in alerts], ["high", "critical"])


if __name__ == "__main__":
    unittest.main()
