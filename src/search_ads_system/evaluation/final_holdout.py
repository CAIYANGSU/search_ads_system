"""One-way final Future-B holdout evaluation with frozen artifact guards."""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from search_ads_system.ads.simulator import SimulationConfig, simulate_attribution, simulate_search_conversion
from search_ads_system.common.config import load_yaml_config, resolve_path
from search_ads_system.ranking.attribution_calibration import IsotonicCalibrator, PlattCalibrator, RawCalibrator, calibration_metrics, serving_consistent_probabilities
from search_ads_system.ranking.attribution_calibration_pipeline import _predict_frame, parse_attribution_calibration_config
from search_ads_system.ranking.attribution_esmm_pipeline import NumericNormalization, build_model as build_esmm, resolve_device
from search_ads_system.ranking.fine_rank_inference import write_predictions_for_split
from search_ads_system.ranking.fine_rank_multitask import parse_fine_rank_multitask_config


@dataclass(frozen=True)
class FinalHoldoutConfig:
    output_dir: Path
    moderate_threshold: float
    large_threshold: float


def parse_final_holdout_config(raw: Mapping[str, Any], config_path: Path) -> FinalHoldoutConfig:
    options = raw.get("final_evaluation", {})
    if not isinstance(options, Mapping): raise ValueError("final_evaluation must be a mapping")
    root = config_path.parent.resolve(); moderate = float(options.get("moderate_degradation_threshold", .05)); large = float(options.get("large_degradation_threshold", .15))
    if not 0 <= moderate < large: raise ValueError("final degradation thresholds require 0 <= moderate < large")
    return FinalHoldoutConfig(resolve_path(str(options.get("output_dir", "outputs/final_evaluation")), root), moderate, large)


def marker_path(config: FinalHoldoutConfig) -> Path: return config.output_dir / "FUTURE_B_OPENED.json"


def future_b_opened_warning(config_path: Path) -> str | None:
    try:
        path = marker_path(parse_final_holdout_config(load_yaml_config(config_path), config_path))
    except (OSError, ValueError, TypeError):
        path = config_path.parent / "outputs" / "final_evaluation" / "FUTURE_B_OPENED.json"
    return "Future-B has been opened; subsequent tuning invalidates pristine holdout semantics." if path.is_file() else None


def validate_frozen_artifacts(raw: Mapping[str, Any], config_path: Path) -> dict[str, str]:
    """Read only paths/metadata; does not enumerate or open Future-B data."""
    calibration = parse_attribution_calibration_config(raw, config_path); fine = parse_fine_rank_multitask_config(raw, config_path, stage="all")
    root = config_path.parent; business = raw.get("business_simulation", {})
    if not isinstance(business, Mapping): raise ValueError("business_simulation must be a mapping")
    paths = {
        "esmm_checkpoint": calibration.checkpoint_path,
        "calibration_metrics": calibration.metrics_dir / "calibration_metrics.json",
        "fine_rank_checkpoint": resolve_path(str(raw.get("fine_rank_multitask", {}).get("predictions", {}).get("checkpoint", "outputs/fine_rank/models/dcnv2.pt")), root),
        "future_a_esmm_metrics": calibration.esmm.metrics_dir / "esmm_metrics.json",
        "future_a_fine_rank_metrics": fine.metrics_dir / "fine_rank_metrics.json",
        "future_a_business_metrics": resolve_path(str(business.get("output_dir", "outputs/business_simulation")), root) / "metrics" / "auction_metrics.json",
    }
    calibration_report_path = paths["calibration_metrics"]
    if calibration_report_path.is_file():
        calibration_report = json.loads(calibration_report_path.read_text(encoding="utf-8"))
        selected = calibration_report.get("selected_calibrator", {})
        for target in ("ctr", "ctcvr"):
            kind = selected.get(target)
            if kind == "platt":
                paths[f"selected_{target}_platt_calibrator"] = calibration.output_dir / "calibrators" / f"{target}_platt.json"
            elif kind == "isotonic":
                paths[f"selected_{target}_isotonic_calibrator"] = calibration.output_dir / "calibrators" / f"{target}_isotonic.pkl"
            elif kind != "raw":
                raise ValueError(f"Frozen calibration report has unsupported selected {target} calibrator: {kind!r}")
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing: raise FileNotFoundError("Frozen artifacts required; final evaluation never retrains or refits. Missing: " + "; ".join(missing))
    business_report = json.loads(paths["future_a_business_metrics"].read_text(encoding="utf-8"))
    if not business_report.get("search_conversion", {}).get("available"):
        raise ValueError("Frozen Future-A business report must contain the available standalone Search Conversion value simulation")
    return {name: str(path) for name, path in paths.items()}


def run_final_holdout(raw: Mapping[str, Any], config_path: Path, *, stage: str) -> dict[str, Any]:
    final = parse_final_holdout_config(raw, config_path)
    if stage == "render":
        return render_existing_final_holdout(final)
    frozen = validate_frozen_artifacts(raw, config_path)
    if stage == "sanity":
        return {"stage": "sanity", "frozen_artifacts": frozen, "future_b_read": False, "schema": _schema_stub()}
    if stage != "all": raise ValueError("final holdout stage must be sanity, render, or all")
    final.output_dir.mkdir(parents=True, exist_ok=True)
    if marker_path(final).exists():
        raise RuntimeError(f"Future-B has already been opened ({marker_path(final)}); final holdout evaluation is one-way and cannot be rerun.")
    _write_opened_marker(final, frozen, raw, config_path)
    attribution = _evaluate_attribution_future_b(raw, config_path)
    search = _evaluate_search_future_b(raw, config_path, final.output_dir)
    future_a = _future_a_baselines(frozen)
    drift = _drift(future_a, attribution, search, final)
    report = {
        "frozen_declaration": {"model_and_policy_frozen_before_future_b": True, "future_b_used_for_training": False, "future_b_used_for_model_selection": False, "future_b_used_for_policy_selection": False, "future_b_used_for_final_evaluation": True},
        "future_b_read_for_policy_selection": False,
        "future_b_read_for_search_conversion_simulation": True,
        "future_b_reread": False,
        "reused_existing_final_metrics": False,
        "data_contracts": {"cross_dataset_join": False, "attribution": "impression-level; synthetic offline auction/bidding only", "search_conversion": "clicked-interaction value selection; not impression auction"},
        "frozen_artifacts": frozen, "attribution_final_holdout": attribution, "search_conversion_final_holdout": search, "future_a_vs_future_b": drift,
        "limitations": _limitations(), "final_conclusion": "Future-B is a final time-out holdout. Results are descriptive offline evaluation, not an online A/B test.",
    }
    json_path = final.output_dir / "final_holdout_metrics.json"; markdown_path = final.output_dir / "final_holdout_metrics.md"; csv_path = final.output_dir / "future_a_vs_future_b.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"); markdown_path.write_text(_markdown(report), encoding="utf-8")
    pd.DataFrame(drift["rows"]).to_csv(csv_path, index=False)
    return {"report_path": str(json_path), "markdown_path": str(markdown_path), "comparison_path": str(csv_path), "future_b_read": True}


def render_existing_final_holdout(config: FinalHoldoutConfig) -> dict[str, Any]:
    """Correct report-layer interpretation from the existing final JSON only."""
    json_path = config.output_dir / "final_holdout_metrics.json"
    markdown_path = config.output_dir / "final_holdout_metrics.md"
    csv_path = config.output_dir / "future_a_vs_future_b.csv"
    if not marker_path(config).is_file():
        raise FileNotFoundError(f"Final-holdout marker required for render-only mode: {marker_path(config)}")
    if not json_path.is_file():
        raise FileNotFoundError(f"Existing final metrics required for render-only mode: {json_path}")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    drift = report.get("future_a_vs_future_b")
    if not isinstance(drift, Mapping) or not isinstance(drift.get("rows"), list):
        raise ValueError("Existing final metrics lacks future_a_vs_future_b.rows")
    rows = [_delta(row.get("metric"), row.get("future_a"), row.get("future_b"), config) for row in drift["rows"]]
    report["future_a_vs_future_b"] = {**drift, "rows": rows, "thresholds": {"moderate": config.moderate_threshold, "large": config.large_threshold}}
    report["future_b_reread"] = False
    report["reused_existing_final_metrics"] = True
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"report_path": str(json_path), "markdown_path": str(markdown_path), "comparison_path": str(csv_path), "future_b_read": False, "future_b_reread": False, "reused_existing_final_metrics": True}


def _evaluate_attribution_future_b(raw: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    calibration = parse_attribution_calibration_config(raw, config_path); future_b = calibration.esmm.future_a_path.parent / "future_b"
    if not future_b.is_dir(): raise FileNotFoundError(f"Attribution Future-B missing: {future_b}")
    checkpoint = torch.load(calibration.checkpoint_path, map_location="cpu", weights_only=False); normalization = NumericNormalization(**checkpoint["normalization"]); device = resolve_device(calibration.esmm.device)
    model = build_esmm(calibration.esmm, "esmm").to(device); model.load_state_dict(checkpoint["model_state"]); model.eval()
    frames: list[pd.DataFrame] = []
    for file in sorted(future_b.glob("part-*.csv")):
        for frame in pd.read_csv(file, chunksize=calibration.io_chunk_size, low_memory=False):
            predicted = _predict_frame(model, frame, calibration, normalization, device)
            kept = frame.loc[:, [name for name in ("event_id", "timestamp", "campaign", "cost", "click", "conversion", "click_and_conversion") if name in frame]].copy()
            kept["raw_pctr"], kept["raw_pctcvr"], kept["raw_pcvr"] = predicted["pctr"], predicted["pctcvr"], predicted["pcvr"]
            frames.append(kept)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty: raise ValueError("Attribution Future-B has no rows")
    calibrated = _frozen_calibration(data, calibration)
    data["calibrated_pctr"], data["calibrated_pctcvr"], data["serving_pcvr"] = calibrated["pctr"], calibrated["pctcvr"], calibrated["pcvr"]
    click, ctcvr = data.click.to_numpy(int), data.click_and_conversion.to_numpy(int)
    clicked = click == 1
    metrics = {"esmm": {"ctr": _binary(click, data.raw_pctr), "ctcvr": _binary(ctcvr, data.raw_pctcvr), "cvr_clicked_subset": _binary(data.conversion.to_numpy(int)[clicked], data.raw_pcvr.to_numpy(float)[clicked])}, "calibration": {"ctr": {"raw": calibration_metrics(click, data.raw_pctr.to_numpy(float), calibration.reliability_bins, calibration.epsilon), "calibrated": calibration_metrics(click, data.calibrated_pctr.to_numpy(float), calibration.reliability_bins, calibration.epsilon)}, "ctcvr": {"raw": calibration_metrics(ctcvr, data.raw_pctcvr.to_numpy(float), calibration.reliability_bins, calibration.epsilon), "calibrated": calibration_metrics(ctcvr, data.calibrated_pctcvr.to_numpy(float), calibration.reliability_bins, calibration.epsilon)}, "serving_pcvr_consistency": serving_consistent_probabilities(data.calibrated_pctr.to_numpy(float), data.calibrated_pctcvr.to_numpy(float), calibration.epsilon)[1]}}
    business = raw.get("business_simulation", {}); sim = SimulationConfig(seed=int(business.get("seed", 2026)), candidates_per_auction=int(business.get("candidates_per_auction", 5)), base_bid=float(business.get("base_bid", 1.0)), mechanism=str(business.get("mechanism", "second_price")), total_budget=float(business.get("total_budget", 1000.0)), pacing_min=float(business.get("pacing_min", .5)), pacing_max=float(business.get("pacing_max", 2.0)), budget_levels=tuple(float(value) for value in business.get("budget_levels", (.25, .5, .75, 1.0))))
    simulation, _, _ = simulate_attribution(data, config=sim, calibrated_available=True)
    return {"rows": int(len(data)), "synthetic_offline_simulation": True, "metrics": metrics, "business": simulation}


def _frozen_calibration(frame: pd.DataFrame, config: Any) -> dict[str, np.ndarray]:
    report = json.loads((config.metrics_dir / "calibration_metrics.json").read_text()); selected = report["selected_calibrator"]; directory = config.output_dir / "calibrators"; output: dict[str, np.ndarray] = {}
    for target, column in (("ctr", "raw_pctr"), ("ctcvr", "raw_pctcvr")):
        kind, values = selected[target], frame[column].to_numpy(float)
        if kind == "raw": calibrator = RawCalibrator()
        elif kind == "platt": calibrator = PlattCalibrator(**json.loads((directory / f"{target}_platt.json").read_text()))
        elif kind == "isotonic":
            with (directory / f"{target}_isotonic.pkl").open("rb") as handle: calibrator = pickle.load(handle)
        else: raise ValueError(f"Unsupported frozen calibrator {kind}")
        output["pctr" if target == "ctr" else "pctcvr"] = calibrator.predict(values)
    output["pcvr"] = serving_consistent_probabilities(output["pctr"], output["pctcvr"], config.epsilon)[0]
    return output


def _evaluate_search_future_b(raw: Mapping[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    fine = parse_fine_rank_multitask_config(raw, config_path, stage="all"); source = fine.future_b_path
    if not source.is_dir(): raise FileNotFoundError(f"Search Conversion Future-B missing: {source}")
    prediction_dir = output_dir / "search_conversion_future_b_predictions"; options = raw.get("fine_rank_multitask", {}).get("predictions", {}); checkpoint = resolve_path(str(options.get("checkpoint", "outputs/fine_rank/models/dcnv2.pt")), config_path.parent)
    write_predictions_for_split(fine, source_path=source, checkpoint_path=checkpoint, output_dir=prediction_dir, split_name="future_b")
    frames = [pd.read_csv(part) for part in sorted(prediction_dir.glob("part-*.csv"))]; data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    labels, probabilities = data.conversion_label.to_numpy(int), data.pCVR_clicked.to_numpy(float); valid = data.conversion_label.eq(1) & data.has_conversion_value.eq(1) & data.conversion_value_eur.notna()
    observed, predicted = data.loc[valid, "conversion_value_eur"].to_numpy(float), data.loc[valid, "predicted_conditional_value"].to_numpy(float)
    metrics = {"cvr": _binary(labels, probabilities), "conditional_value": {"rows": int(valid.sum()), "mae": float(mean_absolute_error(observed, predicted)) if len(observed) else None, "rmse": float(np.sqrt(mean_squared_error(observed, predicted))) if len(observed) else None, "rmsle": float(np.sqrt(mean_squared_error(np.log1p(observed), np.log1p(predicted)))) if len(observed) else None, "prediction_mean": float(predicted.mean()) if len(predicted) else None, "label_mean": float(observed.mean()) if len(observed) else None}}
    business = raw.get("business_simulation", {}); fractions = tuple(float(value) for value in business.get("search_selection_fractions", (.1,.25,.5,.75,1.0))); sim = SimulationConfig(seed=int(business.get("seed", 2026)))
    selection, _, deciles = simulate_search_conversion(data, config=sim, selection_fractions=fractions)
    deciles.to_csv(output_dir / "search_conversion_future_b_value_deciles.csv", index=False)
    return {"rows": int(len(data)), "future_b_read_for_search_conversion_simulation": True, "metrics": metrics, "business": selection}


def _binary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities); both = len(labels) and len(np.unique(labels)) == 2
    return {"rows": int(len(labels)), "roc_auc": float(roc_auc_score(labels, probabilities)) if both else None, "pr_auc": float(average_precision_score(labels, probabilities)) if both else None, "logloss": float(log_loss(labels, probabilities, labels=[0,1])) if len(labels) else None, "brier_score": float(np.mean((labels-probabilities)**2)) if len(labels) else None, "prediction_mean": float(probabilities.mean()) if len(probabilities) else None, "label_mean": float(labels.mean()) if len(labels) else None}


def _future_a_baselines(frozen: Mapping[str, str]) -> dict[str, Any]:
    esmm = json.loads(Path(frozen["future_a_esmm_metrics"]).read_text()); calibration = json.loads(Path(frozen["calibration_metrics"]).read_text()); fine = json.loads(Path(frozen["future_a_fine_rank_metrics"]).read_text()); business = json.loads(Path(frozen["future_a_business_metrics"]).read_text())
    return {"attribution": {"ctr": esmm["metrics"]["esmm"]["ctr"], "ctcvr": esmm["metrics"]["esmm"]["ctcvr"], "calibration": calibration, "business": business["attribution"]}, "search": {"cvr": fine["models"]["dcnv2"]["future_a"]["cvr"], "value": fine["models"]["dcnv2"]["future_a"]["value"], "business": business["search_conversion"]}}


def _drift(a: Mapping[str, Any], attribution: Mapping[str, Any], search: Mapping[str, Any], config: FinalHoldoutConfig) -> dict[str, Any]:
    pairs = [
        ("attribution_ctr_pr_auc", a["attribution"]["ctr"].get("pr_auc"), attribution["metrics"]["esmm"]["ctr"].get("pr_auc")),
        ("attribution_ctcvr_pr_auc", a["attribution"]["ctcvr"].get("pr_auc"), attribution["metrics"]["esmm"]["ctcvr"].get("pr_auc")),
        ("attribution_ctr_logloss", a["attribution"]["ctr"].get("logloss"), attribution["metrics"]["esmm"]["ctr"].get("logloss")),
        ("attribution_calibrated_ctr_ece", a["attribution"]["calibration"]["targets"]["ctr"]["methods"][a["attribution"]["calibration"]["selected_calibrator"]["ctr"]]["metrics"].get("ece"), attribution["metrics"]["calibration"]["ctr"]["calibrated"].get("ece")),
        ("search_cvr_pr_auc", a["search"]["cvr"].get("pr_auc"), search["metrics"]["cvr"].get("pr_auc")),
        ("search_cvr_roc_auc", a["search"]["cvr"].get("roc_auc"), search["metrics"]["cvr"].get("roc_auc")),
        ("search_value_mae", a["search"]["value"].get("mae"), search["metrics"]["conditional_value"].get("mae")),
        ("search_value_rmse", a["search"]["value"].get("rmse"), search["metrics"]["conditional_value"].get("rmse")),
        ("attribution_calibrated_bidding_cpa_proxy", _dig(a, "attribution", "business", "policies", "calibrated_ctcvr_scaled", "cpa_proxy"), _dig(attribution, "business", "policies", "calibrated_ctcvr_scaled", "cpa_proxy")),
        ("search_top_10pct_value_capture_rate", _dig(a, "search", "business", "policies", "expected_value_per_click", "10%", "value_capture_rate"), _dig(search, "business", "policies", "expected_value_per_click", "10%", "value_capture_rate")),
        ("search_top_10pct_value_per_click_lift", _dig(a, "search", "business", "policies", "expected_value_per_click", "10%", "value_per_click_lift"), _dig(search, "business", "policies", "expected_value_per_click", "10%", "value_per_click_lift")),
    ]
    rows = [_delta(name, before, after, config) for name, before, after in pairs]
    return {"rows": rows, "thresholds": {"moderate": config.moderate_threshold, "large": config.large_threshold}}


def _dig(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _delta(name: str, before: Any, after: Any, config: FinalHoldoutConfig) -> dict[str, Any]:
    direction = _metric_direction(name)
    if before is None or after is None:
        return {"metric": name, "direction": direction, "future_a": before, "future_b": after, "absolute_delta": None, "relative_delta": None, "interpretation": "unavailable"}
    delta = float(after) - float(before)
    relative = delta / abs(float(before)) if before else None
    if relative is None:
        interpretation = "stable" if delta == 0 else "unavailable"
    else:
        directional_change = relative if direction == "higher_is_better" else -relative
        interpretation = "improved" if directional_change > 0 else "stable" if directional_change >= -config.moderate_threshold else "moderate degradation" if directional_change >= -config.large_threshold else "large degradation"
    return {"metric": name, "direction": direction, "future_a": float(before), "future_b": float(after), "absolute_delta": delta, "relative_delta": relative, "interpretation": interpretation}


def _metric_direction(name: Any) -> str:
    metric = str(name)
    if metric.endswith(("_pr_auc", "_roc_auc", "_value_capture_rate", "_value_per_click_lift")):
        return "higher_is_better"
    if metric.endswith(("_logloss", "_ece", "_brier_score", "_mae", "_rmse", "_rmsle", "_cpa_proxy")):
        return "lower_is_better"
    raise ValueError(f"No direction contract for final-holdout metric: {metric}")


def _write_opened_marker(config: FinalHoldoutConfig, frozen: Mapping[str, str], raw: Mapping[str, Any], config_path: Path) -> None:
    try: commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config_path.parent, capture_output=True, text=True, check=False).stdout.strip() or None
    except OSError: commit = None
    config_hash = hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()
    payload = {
        "opened_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "checkpoints": {name: path for name, path in frozen.items() if "checkpoint" in name},
        "calibration_artifacts": {name: path for name, path in frozen.items() if "calibration" in name or "calibrator" in name},
        "frozen_artifacts": dict(frozen),
        "policy_config": raw.get("business_simulation", {}),
        "config_hash": config_hash,
        "future_b_opened_for": "final evaluation only",
    }
    marker_path(config).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _schema_stub() -> dict[str, Any]: return {"final_holdout_metrics": ["frozen_declaration", "future_b_read_for_policy_selection", "future_b_read_for_search_conversion_simulation", "future_b_reread", "reused_existing_final_metrics", "data_contracts", "frozen_artifacts", "attribution_final_holdout", "search_conversion_final_holdout", "future_a_vs_future_b", "limitations", "final_conclusion"]}
def _limitations() -> list[str]: return ["The Criteo datasets have no reliable ID mapping and were never joined.", "Attribution auction/bidding is synthetic offline simulation, not online ROI.", "Search Conversion contains clicked interactions, not a complete impression auction.", "Fine Rank high scores reflect strong data separability; its formal audit found no direct leakage.", "Unseen-user temporal coverage is limited.", "Future-B is a final holdout, not a real online A/B test."]
def _markdown(report: Mapping[str, Any]) -> str:
    table = _markdown_table(report["future_a_vs_future_b"]["rows"])
    attribution = report["attribution_final_holdout"]
    search = report["search_conversion_final_holdout"]
    return (
        "# Final Future-B Holdout Evaluation\n\n"
        "Future-B was used once for frozen final evaluation only. No training, model selection, calibration fitting, or policy selection occurred after it was opened.\n\n"
        "## Data contracts\n\n"
        "- `cross_dataset_join = false`; the two datasets remain fully separate.\n"
        "- Attribution is impression-level and its bidding result is a synthetic offline simulation.\n"
        "- Search Conversion is clicked-interaction value selection, never an impression auction.\n\n"
        "## Frozen artifacts\n\n"
        "The selected ESMM/DCNv2 checkpoints, selected calibration artifacts, and Future-A policy definitions were validated before Future-B was opened.\n\n"
        "## Attribution final holdout\n\n"
        f"Future-B rows: {attribution['rows']}. ESMM CTR, CTCVR, clicked-subset CVR, frozen calibration, and synthetic business metrics are in the JSON report.\n\n"
        "## Search Conversion final holdout\n\n"
        f"Future-B clicked interactions: {search['rows']}. Frozen DCNv2 predictions are evaluated with standalone capacity-constrained value selection.\n\n"
        "## Future-A versus Future-B\n\n"
        f"{table}\n\n"
        "## Business metrics\n\n"
        "The comparison includes calibrated Attribution bidding CPA proxy and Search expected-value Top-10% value-capture/value-per-click lifts. These are offline proxies, not online ROI.\n\n"
        "## Limitations\n\n"
        + "\n".join(f"- {item}" for item in report["limitations"])
        + "\n\n## Final conclusion\n\n"
        + report["final_conclusion"]
        + "\n"
    )


def _markdown_table(rows: list[Mapping[str, Any]]) -> str:
    """Render the small report table without requiring the optional tabulate package."""
    if not rows:
        return "No comparable metrics were available."
    columns = ["metric", "direction", "future_a", "future_b", "absolute_delta", "relative_delta", "interpretation"]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    def display(value: Any) -> str:
        if value is None: return ""
        if isinstance(value, float): return f"{value:.8g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    body = ["| " + " | ".join(display(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])
