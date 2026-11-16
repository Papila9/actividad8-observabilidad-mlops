"""Ejecuta respuestas controladas a incidentes registrados en JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reentrenamiento import evaluate_acceptance, train_and_evaluate
from src.rollback import load_registry, rollback, save_registry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def respond_to_incident(incident: dict, *, data_path: Path | None = None, registry_path: Path | None = None) -> dict:
    incident_type = incident.get("incident_type") or incident.get("source_alert", {}).get("alert_type")
    action = {"timestamp": _now(), "incident_id": incident.get("incident_id"), "reason": incident_type, "simulation": True, "lifecycle": incident.get("lifecycle", [])[:], "incident_status_before": incident.get("status", "open")}
    if incident_type == "data_drift":
        if data_path is None:
            action.update({"action": "retraining_and_evaluation", "status": "blocked", "result": "data_path_required"})
        else:
            stable_model, stable_eval = train_and_evaluate(data_path, params={"C": 3.0, "ngram_range": [1, 2], "max_features": 30000})
            candidate_model, candidate_eval = train_and_evaluate(data_path, params={"C": 1.0, "ngram_range": [1, 1], "max_features": 30000})
            decision = evaluate_acceptance(candidate_eval, stable_eval)
            action.update({"action": "retraining_and_evaluation", "status": "completed", "result": {"stable": stable_eval["metrics"], "candidate": candidate_eval["metrics"], "acceptance": decision}})
    elif incident_type in {"slo_latency", "error_rate", "slo_breach"}:
        if registry_path is None:
            action.update({"action": "escalation_and_rollback", "status": "blocked", "result": "registry_path_required"})
        else:
            registry = load_registry(registry_path)
            target = registry["stable_version"]
            event = rollback(registry, target, reason=f"Respuesta controlada a {incident_type}", simulation=True)
            save_registry(registry_path, registry)
            action.update({"action": "escalation_and_rollback", "status": "completed", "result": event, "active_version_after": registry["active_version"]})
    else:
        action.update({"action": "manual_review", "status": "not_applicable", "result": "incident_type_not_supported"})
    verification_status = "completed" if action["status"] == "completed" else "pending"
    action["lifecycle"].append({"step": "corrective_action", "timestamp": action["timestamp"], "status": action["status"], "result": action.get("action")})
    action["lifecycle"].append({"step": "verification", "timestamp": _now(), "status": verification_status, "result": action.get("result")})
    action["incident_status_after"] = "closed" if action["status"] == "completed" else "escalated"
    action["lifecycle"].append({"step": "closure" if action["status"] == "completed" else "escalation", "timestamp": _now(), "status": action["incident_status_after"], "result": "controlled_simulation" if action["status"] == "completed" else "manual_intervention_required"})
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incidents", type=Path, default=Path("logs/incidents.jsonl"))
    parser.add_argument("--actions-log", type=Path, default=Path("logs/alert_responses.jsonl"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    actions = []
    for line in args.incidents.read_text(encoding="utf-8").splitlines():
        if line.strip():
            action = respond_to_incident(json.loads(line), data_path=args.data, registry_path=args.registry)
            args.actions_log.parent.mkdir(parents=True, exist_ok=True)
            with args.actions_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(action, ensure_ascii=False, sort_keys=True) + "\n")
            actions.append(action)
    print(json.dumps(actions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
