import unittest

from scripts.simular_slo import measure_prediction_window
from scripts.simular_incidentes import build_simulated_window


class SimulationTests(unittest.TestCase):
    def test_latency_p95_comes_from_measured_predictions(self):
        class Model:
            def predict(self, values):
                return ["positive"]

        result = measure_prediction_window(Model(), ["a", "b", "c"], target_latency_ms=1.0)
        self.assertEqual(result["metrics"]["prediction_volume"], 3)
        self.assertGreater(result["metrics"]["latency_p95_ms"], 0)

    def test_long_text_subset_is_a_controlled_reproducible_window(self):
        texts = ["short", "a much longer airline complaint text", "medium text", "the longest text in this fixture"]
        current, description = build_simulated_window(texts, sample_size=2, scenario="long_text_subset")
        self.assertEqual(current, [texts[1], texts[3]])
        self.assertIn("textos más largos", description)


if __name__ == "__main__":
    unittest.main()
