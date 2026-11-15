"""Ejecuta una validación controlada con datos reales y registra observabilidad.

La ejecución queda marcada como smoke_test. No reemplaza el baseline histórico.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.monitor import MonitoringSession  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    texts = [row["text"] for row in rows]
    labels = [row["airline_sentiment"] for row in rows]
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    args = parser.parse_args()
    texts, labels = load_rows(args.data)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    if args.sample_size <= 0:
        raise ValueError("sample-size debe ser positivo")
    test_texts = test_texts[: args.sample_size]
    test_labels = test_labels[: args.sample_size]

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True, min_df=2)),
            ("classifier", LogisticRegression(class_weight="balanced", C=3.0, max_iter=1000, random_state=42)),
        ]
    )
    model.fit(train_texts, train_labels)

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    tracking_uri = (ROOT / "mlruns").as_uri()
    session = MonitoringSession(
        model_name="logistic_regression",
        model_version="smoke_test-c3-tfidf-unigrams-bigrams",
        output_dir=ROOT / "evidencias" / "monitoring",
        baseline_path=ROOT / "evidencias" / "baseline" / "metricas_experimentos.json",
        tracking_uri=tracking_uri,
        experiment_name="Actividad8_Observabilidad",
    )
    for index, (text, true_label) in enumerate(zip(test_texts, test_labels)):
        start = time.perf_counter()
        predicted = str(model.predict([text])[0])
        latency_ms = (time.perf_counter() - start) * 1000
        session.record_prediction(
            predicted_label=predicted,
            true_label=true_label,
            latency_ms=latency_ms,
            metadata={"sample_index": index, "execution_type": "smoke_test"},
        )
    result = session.finalize()
    result["execution_type"] = "smoke_test"
    result["data"] = {
        "path": str(args.data),
        "sha256": sha256_file(args.data),
        "total_rows": len(texts),
        "sample_size": len(test_texts),
        "train_rows": len(train_texts),
        "random_seed": 42,
    }
    output_path = ROOT / "evidencias" / "monitoring" / f"{result['execution_id']}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
