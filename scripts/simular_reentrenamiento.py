"""Entrena un candidato real sobre el dataset existente y simula rollback."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reentrenamiento import evaluate_acceptance, save_model, train_and_evaluate
from src.rollback import create_registry, promote_version, register_version, rollback, save_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()

    execution_id = f"retraining-simulation-{uuid.uuid4().hex[:12]}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stable_model, stable_eval = train_and_evaluate(args.data, params={"C": 3.0, "ngram_range": [1, 2]})
    candidate_model, candidate_eval = train_and_evaluate(args.data, params={"C": 1.0, "ngram_range": [1, 1]})
    stable_version = "logreg-stable-v1"
    candidate_version = "logreg-candidate-v2"
    stable_path = args.output_dir / f"{stable_version}.pkl"
    candidate_path = args.output_dir / f"{candidate_version}.pkl"
    stable_hash = save_model(stable_model, stable_path)
    candidate_hash = save_model(candidate_model, candidate_path)
    registry = create_registry({
        "version": stable_version,
        "model_name": "logistic_regression",
        "artifact_path": str(stable_path.resolve()),
        "artifact_sha256": stable_hash,
        "metrics": stable_eval["metrics"],
        "data_sha256": stable_eval["data_sha256"],
        "protocol": stable_eval,
    })
    register_version(registry, {
        "version": candidate_version,
        "model_name": "logistic_regression",
        "artifact_path": str(candidate_path.resolve()),
        "artifact_sha256": candidate_hash,
        "metrics": candidate_eval["metrics"],
        "data_sha256": candidate_eval["data_sha256"],
        "protocol": candidate_eval,
    }, status="candidate")
    decision = evaluate_acceptance(candidate_eval, stable_eval)
    registry["versions"][candidate_version]["status"] = "accepted_candidate" if decision["accepted"] else "rejected_candidate"
    registry_path = args.output_dir / "model_registry.json"
    save_registry(registry_path, registry)

    rollback_registry = copy.deepcopy(registry)
    promote_version(rollback_registry, candidate_version)
    rollback_event = rollback(rollback_registry, stable_version, reason="Simulación de recuperación a la versión estable", simulation=True)
    rollback_path = args.output_dir / "rollback_simulation_registry.json"
    save_registry(rollback_path, rollback_registry)
    result = {
        "execution_id": execution_id,
        "execution_type": "controlled_retraining_and_rollback_simulation",
        "simulation": {"is_simulation": True, "description": "El candidato se entrenó con datos existentes; la promoción/rollback se simula en un registro separado."},
        "data_path": str(args.data.resolve()),
        "stable": {"version": stable_version, "evaluation": stable_eval, "artifact_sha256": stable_hash},
        "candidate": {"version": candidate_version, "evaluation": candidate_eval, "artifact_sha256": candidate_hash},
        "acceptance_decision": decision,
        "registry_path": str(registry_path.resolve()),
        "rollback_simulation": {"event": rollback_event, "registry_path": str(rollback_path.resolve()), "active_version_after": rollback_registry["active_version"]},
    }
    result_path = args.output_dir / f"{execution_id}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Actividad8_Observabilidad_ModelLifecycle")
    with mlflow.start_run(run_name=execution_id) as run:
        mlflow.set_tags({"execution_id": execution_id, "execution_type": result["execution_type"], "simulation": "true", "phase": "fase_6", "candidate_decision": "accepted" if decision["accepted"] else "rejected"})
        mlflow.log_params({"stable_version": stable_version, "candidate_version": candidate_version, "data_sha256": stable_eval["data_sha256"], "acceptance_policy": "accuracy>=0.75; f1_macro>=0.70; no_f1_regression"})
        mlflow.log_metrics({"stable_accuracy": stable_eval["metrics"]["accuracy"], "stable_f1_macro": stable_eval["metrics"]["f1_macro"], "candidate_accuracy": candidate_eval["metrics"]["accuracy"], "candidate_f1_macro": candidate_eval["metrics"]["f1_macro"], "candidate_accepted": float(decision["accepted"])})
        mlflow.log_dict(result, "retraining_rollback_result.json")
        mlflow.log_artifact(str(registry_path), artifact_path="registry")
        mlflow.log_artifact(str(stable_path), artifact_path="models")
        mlflow.log_artifact(str(candidate_path), artifact_path="models")
        result["mlflow_run_id"] = run.info.run_id
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
