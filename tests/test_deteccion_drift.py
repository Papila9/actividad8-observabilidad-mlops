import unittest

from src.deteccion_drift import build_baseline, detect_drift, jensen_shannon_divergence, ks_statistic


class DriftDetectionTests(unittest.TestCase):
    def setUp(self):
        self.reference = [
            "flight was on time and helpful",
            "service was friendly and helpful",
            "the flight was acceptable",
            "crew provided excellent service",
        ]
        self.baseline = build_baseline(self.reference, source_path="reference.csv", source_sha256="abc")

    def test_same_distribution_does_not_trigger_text_drift(self):
        result = detect_drift(self.baseline, self.reference)
        self.assertEqual(result["data_drift"]["severity"], "none")
        self.assertEqual(result["class_distribution_shift"]["status"], "unavailable_without_labels")
        self.assertEqual(result["model_drift"]["status"], "unavailable_without_real_labels")

    def test_controlled_token_shift_triggers_high_drift(self):
        current = [f"{text} novelword novelword novelword" for text in self.reference]
        result = detect_drift(self.baseline, current)
        self.assertGreater(result["data_drift"]["token_jsd"], 0.1)
        self.assertIn(result["data_drift"]["severity"], {"medium", "high"})

    def test_class_shift_is_separate_from_data_drift(self):
        baseline = build_baseline(
            self.reference,
            labels=["negative", "neutral", "positive", "neutral"],
            source_path="reference.csv",
            source_sha256="abc",
        )
        result = detect_drift(
            baseline,
            self.reference,
            current_labels=["negative", "negative", "negative", "negative"],
        )
        self.assertEqual(result["class_distribution_shift"]["status"], "observed")
        self.assertNotEqual(result["class_distribution_shift"]["jsd"], 0.0)
        self.assertEqual(result["model_drift"]["status"], "unavailable_without_real_labels")

    def test_statistical_helpers(self):
        self.assertEqual(ks_statistic([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertGreater(jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0]), 0.9)


if __name__ == "__main__":
    unittest.main()
