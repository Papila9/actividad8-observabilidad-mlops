import json
import tempfile
import unittest
from pathlib import Path

from src.monitor import MonitoringSession


class TraceabilityTests(unittest.TestCase):
    def test_prediction_trace_contains_stage_durations_without_text(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            output_dir = project / "evidencias" / "monitoring"
            (project / "mlruns").mkdir(parents=True)
            session = MonitoringSession("test", "v1", output_dir, tracking_uri=(project / "mlruns").as_uri())
            session.record_prediction(predicted_label="positive", latency_ms=2.0, trace_id="trace-test", span_id="span-test", vectorization_ms=0.7, prediction_ms=1.1, metadata={"sample_index": 1})
            session.finalize()
            events = [json.loads(line) for line in session._trace_path.read_text(encoding="utf-8").splitlines()]
            prediction = next(item for item in events if item["event"] == "prediction_recorded")
            self.assertEqual(prediction["trace_id"], "trace-test")
            self.assertEqual(prediction["stage_durations_ms"]["vectorization_ms"], 0.7)
            self.assertNotIn("text", prediction)


if __name__ == "__main__":
    unittest.main()
