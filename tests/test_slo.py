import tempfile
import unittest
from pathlib import Path

from src.alertas import AlertManager
from src.metricas import calculate_window_metrics
from src.slo import evaluate_slos


class SLOTests(unittest.TestCase):
    def test_compliant_window_has_remaining_budget(self):
        metrics = calculate_window_metrics([100, 200, 300, 400, 500], error_count=0)
        result = evaluate_slos(metrics, error_count=0)
        self.assertEqual(result["overall_status"], "compliant")
        self.assertTrue(result["slos"]["availability"]["passed"])
        self.assertEqual(result["slos"]["latency_p95"]["passed"], True)
        self.assertIsNone(result["slos"]["accuracy"]["passed"])

    def test_budget_consumption_and_breach(self):
        metrics = calculate_window_metrics([606] * 100, error_count=2)
        result = evaluate_slos(metrics, error_count=2)
        self.assertEqual(result["overall_status"], "violated")
        self.assertAlmostEqual(result["slis"]["availability"], 0.98)
        self.assertGreater(result["slos"]["availability"]["error_budget"]["budget_consumed_fraction"], 1)
        self.assertGreater(result["slos"]["latency_p95"]["error_budget"]["budget_consumed_fraction"], 1)

    def test_labeled_quality_is_evaluated_only_when_labels_exist(self):
        metrics = calculate_window_metrics(
            [100, 100, 100, 100],
            y_true=["positive", "negative", "positive", "negative"],
            y_pred=["positive", "negative", "negative", "negative"],
            labels=["negative", "positive"],
        )
        result = evaluate_slos(metrics)
        self.assertIsNotNone(result["slis"]["f1_macro"])
        self.assertEqual(result["quality_status"], "available")

    def test_slo_alert_uses_deduplication_and_cooldown(self):
        metrics = calculate_window_metrics([606] * 10, error_count=0)
        result = evaluate_slos(metrics)
        with tempfile.TemporaryDirectory() as directory:
            now = [100.0]
            manager = AlertManager(Path(directory) / "slo.jsonl", cooldown_seconds=60, clock=lambda: now[0])
            first = manager.evaluate_slo(result, execution_id="exec-1", window_id="window-1")
            duplicate = manager.evaluate_slo(result, execution_id="exec-1", window_id="window-1")
            cooldown = manager.evaluate_slo(result, execution_id="exec-2", window_id="window-2")
            now[0] += 61
            after = manager.evaluate_slo(result, execution_id="exec-3", window_id="window-3")
        self.assertGreaterEqual(len(first), 1)
        self.assertEqual(len(duplicate), 0)
        self.assertEqual(len(cooldown), 0)
        self.assertGreaterEqual(len(after), 1)


if __name__ == "__main__":
    unittest.main()
