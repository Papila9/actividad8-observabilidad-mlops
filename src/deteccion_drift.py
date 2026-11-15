"""Detección de cambios de distribución para textos y etiquetas.

La señal de texto combina KS sobre longitud de caracteres y divergencia de
Jensen-Shannon sobre frecuencias de tokens. Las etiquetas y el model drift son
señales separadas y solo se calculan cuando existen datos observados válidos.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"\b[\w]+\b", re.UNICODE)
DEFAULT_TOP_TOKENS = 500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _distribution(texts: list[str], top_tokens: int) -> tuple[dict[str, int], int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    selected = dict(counts.most_common(top_tokens))
    selected["__other__"] = sum(value for key, value in counts.items() if key not in selected)
    return selected, sum(counts.values())


def build_baseline(
    texts: list[str],
    *,
    source_path: str,
    source_sha256: str,
    labels: list[str] | None = None,
    top_tokens: int = DEFAULT_TOP_TOKENS,
) -> dict[str, Any]:
    """Construye un baseline sin conservar el texto original."""
    token_counts, token_total = _distribution(texts, top_tokens)
    if labels is not None and len(labels) != len(texts):
        raise ValueError("labels debe tener el mismo tamaño que texts")
    baseline: dict[str, Any] = {
        "schema_version": 1,
        "source": {"path": source_path, "sha256": source_sha256, "rows": len(texts)},
        "method": {
            "length_test": "two-sample Kolmogorov-Smirnov statistic",
            "token_test": "Jensen-Shannon divergence over token frequencies",
            "tokenizer": "lowercase Unicode word tokens",
            "top_tokens": top_tokens,
        },
        "text": {
            "lengths_chars": [len(text) for text in texts],
            "token_counts": token_counts,
            "token_total": token_total,
        },
    }
    if labels is not None:
        baseline["class_counts"] = dict(Counter(labels))
    return baseline


def load_texts(path: Path) -> list[str]:
    return load_texts_and_labels(path)[0]


def load_texts_and_labels(path: Path) -> tuple[list[str], list[str]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row["text"] for row in rows], [row["airline_sentiment"] for row in rows]


def _probabilities(counts: dict[str, int], keys: list[str]) -> list[float]:
    total = sum(counts.get(key, 0) for key in keys)
    if total == 0:
        return [0.0 for _ in keys]
    return [counts.get(key, 0) / total for key in keys]


def jensen_shannon_divergence(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Las distribuciones deben tener la misma longitud")
    midpoint = [(a + b) / 2 for a, b in zip(left, right)]

    def kl_divergence(values: list[float], reference: list[float]) -> float:
        return sum(
            value * math.log2(value / base)
            for value, base in zip(values, reference)
            if value > 0 and base > 0
        )

    return (kl_divergence(left, midpoint) + kl_divergence(right, midpoint)) / 2


def ks_statistic(left: list[int], right: list[int]) -> float:
    """Calcula el estadístico KS de dos muestras sin p-value asintótico."""
    if not left or not right:
        return 0.0
    values = sorted(set(left + right))
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    i = j = 0
    maximum = 0.0
    for value in values:
        while i < len(left_sorted) and left_sorted[i] <= value:
            i += 1
        while j < len(right_sorted) and right_sorted[j] <= value:
            j += 1
        maximum = max(maximum, abs(i / len(left_sorted) - j / len(right_sorted)))
    return maximum


def _severity(value: float, medium: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "none"


def detect_drift(
    baseline: dict[str, Any],
    current_texts: list[str],
    *,
    current_labels: list[str] | None = None,
    baseline_labels: list[str] | None = None,
    observed_metrics: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compara una ventana actual contra el baseline sin inferir model drift."""
    limits = {
        "text_medium": 0.10,
        "text_high": 0.20,
        "class_medium": 0.10,
        "class_high": 0.20,
        "model_f1_drop_high": 0.05,
        "model_f1_drop_critical": 0.10,
        **(thresholds or {}),
    }
    baseline_lengths = baseline["text"]["lengths_chars"]
    current_lengths = [len(text) for text in current_texts]
    current_counts, current_total = _distribution(current_texts, baseline["method"]["top_tokens"])
    keys = sorted(set(baseline["text"]["token_counts"]) | set(current_counts))
    baseline_probabilities = _probabilities(baseline["text"]["token_counts"], keys)
    current_probabilities = _probabilities(current_counts, keys)
    token_jsd = jensen_shannon_divergence(baseline_probabilities, current_probabilities)
    length_ks = ks_statistic(baseline_lengths, current_lengths)
    text_signal = max(token_jsd, length_ks)
    result: dict[str, Any] = {
        "method": baseline["method"],
        "sample": {"rows": len(current_texts), "token_total": current_total},
        "data_drift": {
            "token_jsd": token_jsd,
            "length_ks": length_ks,
            "signal": text_signal,
            "severity": _severity(text_signal, limits["text_medium"], limits["text_high"]),
            "thresholds": {"medium": limits["text_medium"], "high": limits["text_high"]},
        },
        "class_distribution_shift": {"status": "unavailable_without_labels"},
        "model_drift": {"status": "unavailable_without_real_labels"},
        "limitations": [
            "KS se reporta como estadístico; no se interpreta como p-value.",
            "Jensen-Shannon depende del tokenizador y de los tokens retenidos.",
            "Data drift no implica por sí mismo pérdida de precisión.",
        ],
    }
    if current_labels is not None and (baseline_labels is not None or baseline.get("class_counts")):
        if len(current_labels) != len(current_texts):
            raise ValueError("current_labels debe tener el mismo tamaño que current_texts")
        if baseline_labels is not None and not baseline_labels:
            raise ValueError("baseline_labels no puede estar vacío")
        base_counts = Counter(baseline_labels or baseline.get("class_counts", {}))
        current_class_counts = Counter(current_labels)
        classes = sorted(set(base_counts) | set(current_labels))
        class_jsd = jensen_shannon_divergence(
            _probabilities(base_counts, classes), _probabilities(current_class_counts, classes)
        )
        result["class_distribution_shift"] = {
            "status": "observed",
            "jsd": class_jsd,
            "severity": _severity(class_jsd, limits["class_medium"], limits["class_high"]),
            "classes": classes,
            "baseline_counts": dict(base_counts),
            "current_counts": dict(current_class_counts),
        }
    if observed_metrics is not None:
        baseline_metrics = baseline.get("model_metrics")
        if baseline_metrics and observed_metrics.get("f1_macro") is not None:
            f1_delta = observed_metrics["f1_macro"] - baseline_metrics["f1_macro"]
            drop = -f1_delta
            severity = "critical" if drop >= limits["model_f1_drop_critical"] else (
                "high" if drop >= limits["model_f1_drop_high"] else "none"
            )
            result["model_drift"] = {
                "status": "observed",
                "f1_macro_baseline": baseline_metrics["f1_macro"],
                "f1_macro_observed": observed_metrics["f1_macro"],
                "f1_macro_delta": f1_delta,
                "severity": severity,
                "requires_real_labels": True,
            }
    return result


def save_baseline(path: Path, baseline: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
