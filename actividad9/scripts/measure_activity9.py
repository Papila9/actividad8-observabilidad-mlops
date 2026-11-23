from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path

import psutil
import sklearn
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
EVIDENCE = ROOT / "evidencias"
MODEL_PATH = INPUTS / "logreg-stable-v1.pkl"
ORIGINAL_PATH = INPUTS / "Tweets.csv"
CLEAN_PATH = INPUTS / "Tweets_limpios.csv"
SEED = 42
TEST_SIZE = 0.2
LABELS = ("negative", "neutral", "positive")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def percentiles(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_and_split(model):
    original = read_csv(ORIGINAL_PATH)
    clean = read_csv(CLEAN_PATH)
    if len(original) != len(clean):
        raise RuntimeError("No se puede alinear: los CSV tienen distinto número de filas.")
    required_original = {"text", "airline", "airline_sentiment"}
    required_clean = {"text", "airline_sentiment"}
    if not required_original.issubset(original[0]) or not required_clean.issubset(clean[0]):
        raise RuntimeError("Faltan columnas requeridas para verificar el alineamiento.")
    mismatches = []
    for index, (source, processed) in enumerate(zip(original, clean)):
        source_text = (source["text"] or "").strip()
        source_label = (source["airline_sentiment"] or "").strip().lower()
        if source_text != processed["text"] or source_label != processed["airline_sentiment"]:
            mismatches.append(index)
    if mismatches:
        raise RuntimeError(f"Alineamiento no confiable; discrepancias en filas: {mismatches[:10]}")

    texts = [row["text"] for row in clean]
    labels = [row["airline_sentiment"] for row in clean]
    indexes = list(range(len(clean)))
    train_idx, test_idx, _, _ = train_test_split(
        indexes, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
    )
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    predictions = [str(value) for value in model.predict(test_texts)]

    train_text_set = set(train_texts)
    contaminated_rows = sum(text in train_text_set for text in test_texts)
    shared_texts = train_text_set.intersection(test_texts)
    row_pairs = sum(train_texts.count(text) * test_texts.count(text) for text in shared_texts)

    grouped: dict[str, dict[str, int | float]] = defaultdict(lambda: {"negative_examples": 0, "tp": 0, "fn": 0})
    for row_index, predicted in zip(test_idx, predictions):
        row = original[row_index]
        airline = row["airline"]
        if row["airline_sentiment"].strip().lower() == "negative":
            grouped[airline]["negative_examples"] += 1
            if predicted == "negative":
                grouped[airline]["tp"] += 1
            else:
                grouped[airline]["fn"] += 1
    for values in grouped.values():
        n = int(values["negative_examples"])
        values["recall_negative"] = int(values["tp"]) / n if n else None
    best = max(float(values["recall_negative"]) for values in grouped.values())
    for values in grouped.values():
        values["gap_vs_best"] = best - float(values["recall_negative"])

    fairness_rows = []
    for airline in sorted(grouped):
        fairness_rows.append({"airline": airline, **grouped[airline]})
    with (EVIDENCE / "fairness_por_aerolinea.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fairness_rows[0].keys())
        writer.writeheader()
        writer.writerows(fairness_rows)
    fairness = {
        "model": "logreg-stable-v1",
        "model_sha256": sha256(MODEL_PATH),
        "protocol": {"test_size": TEST_SIZE, "random_state": SEED, "stratified": True},
        "alignment": {"original_rows": len(original), "clean_rows": len(clean), "mismatches": len(mismatches), "verified": True},
        "historical_split": {"train_rows": len(train_idx), "test_rows": len(test_idx), "shared_unique_texts": len(shared_texts), "contaminated_test_rows": contaminated_rows, "train_test_text_overlap_row_pairs": row_pairs},
        "rows": fairness_rows,
        "interpretation": "Evaluación histórica contaminada por textos idénticos entre train y test; no es un holdout independiente.",
    }
    (EVIDENCE / "fairness_metodologia.json").write_text(json.dumps(fairness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def measure_batch(model, batch: list[str], repetitions: int) -> dict[str, float | int]:
    process = psutil.Process(os.getpid())
    for _ in range(10):
        model.predict(batch)
    latencies = []
    cpu_seconds = []
    rss_before_values = []
    rss_after_values = []
    errors = 0
    for _ in range(repetitions):
        before_cpu = process.cpu_times()
        before_rss = process.memory_info().rss
        start = time.perf_counter()
        try:
            model.predict(batch)
        except Exception:
            errors += 1
        elapsed = time.perf_counter() - start
        after_cpu = process.cpu_times()
        after_rss = process.memory_info().rss
        latencies.append(elapsed * 1000)
        cpu_seconds.append((after_cpu.user + after_cpu.system) - (before_cpu.user + before_cpu.system))
        rss_before_values.append(before_rss)
        rss_after_values.append(after_rss)
    total_seconds = sum(latencies) / 1000
    cpu_pct = (sum(cpu_seconds) / total_seconds / (os.cpu_count() or 1)) * 100 if total_seconds else 0.0
    rss_before = statistics.mean(rss_before_values)
    rss_after = statistics.mean(rss_after_values)
    return {"batch_size": len(batch), "repetitions": repetitions, "errors": errors, "latency_mean_ms": statistics.mean(latencies), "latency_median_ms": statistics.median(latencies), "latency_p95_ms": percentiles(latencies, 0.95), "throughput_predictions_per_second": (len(batch) * repetitions) / total_seconds if total_seconds else 0.0, "cpu_process_percent_of_machine": cpu_pct, "rss_before_mean_mb": rss_before / 1024**2, "rss_after_mean_mb": rss_after / 1024**2, "rss_delta_mean_mb": (rss_after - rss_before) / 1024**2}


def performance(model):
    clean = read_csv(CLEAN_PATH)
    texts = [row["text"] for row in clean]
    rows = []
    for batch_size in (1, 8, 32, 128):
        batch = [texts[i % len(texts)] for i in range(batch_size)]
        result = measure_batch(model, batch, 100)
        rows.append(result)
    with (EVIDENCE / "rendimiento_local.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    hardware = {"platform": platform.platform(), "python": platform.python_version(), "machine": platform.machine(), "processor": platform.processor(), "logical_cpus": os.cpu_count(), "ram_total_mb": psutil.virtual_memory().total / 1024**2}
    methodology = {"model": "logreg-stable-v1", "model_sha256": sha256(MODEL_PATH), "hardware": hardware, "repetitions": 100, "warmup_calls_per_batch": 10, "measurement": "time.perf_counter around local model.predict(batch); CPU from process CPU time aggregated across repetitions; RAM from process RSS before/after each call (not a continuous peak sampler); excludes HTTP, network, queues, I/O, serialization and MLflow.", "rows": rows}
    (EVIDENCE / "rendimiento_metodologia.json").write_text(json.dumps(methodology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for path in (MODEL_PATH, ORIGINAL_PATH, CLEAN_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    with MODEL_PATH.open("rb") as stream:
        model = pickle.load(stream)
    if not callable(getattr(model, "predict", None)):
        raise RuntimeError("El artefacto no expone predict().")
    load_and_split(model)
    performance(model)
    print(json.dumps({"status": "completed", "evidence_dir": str(EVIDENCE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
