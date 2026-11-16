"""Reentrenamiento controlado y comparación supervisada de candidatos."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.actividad5.datos_prep import EXPECTED_LABELS, load_dataset, sha256_file


def train_and_evaluate(data_path: Path, *, params: dict[str, Any], random_seed: int = 42) -> tuple[Any, dict[str, Any]]:
    rows, columns = load_dataset(data_path)
    texts = [row["text"] for row in rows]
    labels = [row["airline_sentiment"] for row in rows]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=random_seed, stratify=labels
    )
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                sublinear_tf=True,
                max_features=int(params.get("max_features", 30000)),
                min_df=2,
                ngram_range=tuple(params.get("ngram_range", (1, 2))),
            )),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                C=float(params.get("C", 3.0)),
                max_iter=1000,
                random_state=random_seed,
            )),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "f1_macro": float(f1_score(test_labels, predictions, labels=list(EXPECTED_LABELS), average="macro", zero_division=0)),
    }
    return pipeline, {
        "metrics": metrics,
        "data_sha256": sha256_file(data_path),
        "rows": len(rows),
        "train_rows": len(train_texts),
        "test_rows": len(test_texts),
        "columns": columns,
        "random_seed": random_seed,
        "test_size": 0.2,
        "params": params,
    }


def evaluate_acceptance(
    candidate: dict[str, Any],
    stable: dict[str, Any],
    *,
    min_accuracy: float = 0.75,
    min_f1_macro: float = 0.70,
    max_f1_drop: float = 0.0,
) -> dict[str, Any]:
    candidate_metrics = candidate["metrics"]
    stable_metrics = stable["metrics"]
    checks = {
        "accuracy_meets_minimum": candidate_metrics["accuracy"] >= min_accuracy,
        "f1_macro_meets_minimum": candidate_metrics["f1_macro"] >= min_f1_macro,
        "f1_drop_within_tolerance": candidate_metrics["f1_macro"] - stable_metrics["f1_macro"] >= -max_f1_drop,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "thresholds": {"min_accuracy": min_accuracy, "min_f1_macro": min_f1_macro, "max_f1_drop": max_f1_drop},
        "deltas_vs_stable": {
            name: candidate_metrics[name] - stable_metrics[name] for name in ("accuracy", "f1_macro")
        },
    }


def save_model(model: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(model, stream)
    return sha256_file(path)
