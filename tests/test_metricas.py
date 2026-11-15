import unittest

from src.metricas import calculate_inference_metrics, calculate_supervised_metrics, percentile


class MetricsTests(unittest.TestCase):
    def test_inference_metrics_include_mean_p95_errors_and_volume(self):
        result = calculate_inference_metrics([100, 200, 300, 400], error_count=1)
        self.assertEqual(result["prediction_volume"], 4)
        self.assertEqual(result["latency_mean_ms"], 250.0)
        self.assertEqual(result["latency_p95_ms"], 385.0)
        self.assertEqual(result["error_rate"], 0.25)

    def test_supervised_metrics_require_real_labels(self):
        result = calculate_supervised_metrics(
            ["negative", "positive", "neutral"],
            ["negative", "neutral", "neutral"],
            ["negative", "neutral", "positive"],
        )
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["f1_macro"], (1 + 0 + 2 / 3) / 3)

    def test_percentile_empty_window_is_unknown(self):
        self.assertIsNone(percentile([], 95))


if __name__ == "__main__":
    unittest.main()
