import json
import os
import tempfile
import unittest
from pathlib import Path

import mlflow

from src.monitor import MonitoringSession


class MonitorTests(unittest.TestCase):
    def test_structured_logs_trace_and_mlflow_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            tracking_uri = (root / "mlruns").as_uri()
            session = MonitoringSession(
                model_name="test_model",
                model_version="test-1",
                output_dir=root / "evidencias" / "monitoring",
                baseline_path=Path("evidencias/baseline/metricas_experimentos.json"),
                tracking_uri=tracking_uri,
                experiment_name="Actividad8_Test",
                execution_id="exec-test-001",
            )
            session.record_prediction(predicted_label="negative", true_label="negative", latency_ms=100)
            session.record_prediction(predicted_label="neutral", true_label="positive", latency_ms=300)
            result = session.finalize()

            self.assertEqual(result["execution_id"], "exec-test-001")
            self.assertEqual(result["metrics"]["prediction_volume"], 2)
            self.assertTrue(result["metric_availability"]["supervised"])
            self.assertTrue(Path(result["log_path"]).exists())
            self.assertTrue(Path(result["trace_path"]).exists())
            first_log = json.loads(Path(result["log_path"]).read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first_log["execution_id"], "exec-test-001")

            mlflow.set_tracking_uri(tracking_uri)
            experiment = mlflow.get_experiment_by_name("Actividad8_Test")
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs.iloc[0]["tags.execution_id"], "exec-test-001")
            self.assertAlmostEqual(float(runs.iloc[0]["metrics.f1_macro"]), result["metrics"]["f1_macro"])


if __name__ == "__main__":
    unittest.main()
