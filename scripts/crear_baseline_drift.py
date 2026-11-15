"""Construye el baseline de distribución a partir de datos reales."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deteccion_drift import build_baseline, load_texts_and_labels, save_baseline, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "evidencias" / "drift" / "baseline.json")
    args = parser.parse_args()
    texts, labels = load_texts_and_labels(args.data)
    baseline = build_baseline(texts, labels=labels, source_path=str(args.data), source_sha256=sha256_file(args.data))
    save_baseline(args.output, baseline)
    print(f"baseline={args.output} rows={len(texts)} sha256={baseline['source']['sha256']}")


if __name__ == "__main__":
    main()
