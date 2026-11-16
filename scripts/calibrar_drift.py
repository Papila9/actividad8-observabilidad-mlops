"""Calibra señales JSD/KS en ventanas aleatorias sin modificar el baseline."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deteccion_drift import detect_drift, load_texts_and_labels


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    texts, _ = load_texts_and_labels(args.data)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    jsd: list[float] = []
    ks: list[float] = []
    records = []
    for index in range(args.windows):
        window = rng.sample(texts, args.window_size)
        detection = detect_drift(baseline, window)
        values = detection["data_drift"]
        jsd.append(float(values["token_jsd"]))
        ks.append(float(values["length_ks"]))
        records.append({"window_id": f"calibration-{index + 1:02d}", "seed": args.seed, "rows": len(window), "token_jsd": jsd[-1], "length_ks": ks[-1]})
    medium = {"token_jsd": percentile(jsd, 95), "length_ks": percentile(ks, 95)}
    high = {"token_jsd": percentile(jsd, 99), "length_ks": percentile(ks, 99)}
    result = {"calibration_type": "same_dataset_random_windows", "is_historical_baseline": False, "data_path": str(args.data.resolve()), "baseline_path": str(args.baseline.resolve()), "parameters": {"windows": args.windows, "window_size": args.window_size, "seed": args.seed}, "statistics": {"token_jsd": {"min": min(jsd), "mean": statistics.fmean(jsd), "max": max(jsd), "p95": medium["token_jsd"], "p99": high["token_jsd"]}, "length_ks": {"min": min(ks), "mean": statistics.fmean(ks), "max": max(ks), "p95": medium["length_ks"], "p99": high["length_ks"]}}, "recommended_thresholds": {"medium": medium, "high": high}, "windows": records, "methodology": "Los umbrales recomendados son p95 y p99 empíricos de ventanas aleatorias; requieren validación con tráfico representativo y no son universales."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
