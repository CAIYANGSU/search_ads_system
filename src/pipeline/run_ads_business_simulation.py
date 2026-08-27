"""Offline-only advertising value, bidding and auction simulation CLI."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from search_ads_system.ads.simulator import SimulationConfig, future_b_isolation_contract, simulate_attribution, simulate_search_conversion
from search_ads_system.common.config import load_yaml_config, resolve_path
from search_ads_system.evaluation.final_holdout import future_b_opened_warning
from search_ads_system.ranking.attribution_calibration import IsotonicCalibrator, PlattCalibrator, RawCalibrator, serving_consistent_probabilities
from search_ads_system.ranking.attribution_calibration_pipeline import parse_attribution_calibration_config


def _options(raw: Mapping[str, Any], config_path: Path, stage: str) -> tuple[dict[str, Any], Path]:
    options = raw.get("business_simulation", {})
    if not isinstance(options, Mapping): raise ValueError("business_simulation must be a mapping")
    effective = {**options, **(options.get("sanity", {}) if stage == "sanity" else {})}
    root = config_path.parent.resolve()
    return effective, resolve_path(str(effective.get("output_dir", "outputs/business_simulation")), root)


def _read_parts(path: Path, limit: int | None) -> pd.DataFrame:
    files = sorted(path.glob("part-*.csv")) if path.is_dir() else [path]
    if not files or not all(item.is_file() for item in files): raise FileNotFoundError(f"Simulation input artifacts not found: {path}")
    pieces: list[pd.DataFrame] = []; rows = 0
    for file in files:
        for frame in pd.read_csv(file, chunksize=200_000, low_memory=False):
            if limit is not None: frame = frame.iloc[:max(0, limit-rows)]
            if frame.empty: break
            pieces.append(frame); rows += len(frame)
            if limit is not None and rows >= limit: break
        if limit is not None and rows >= limit: break
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _load_attribution(raw: Mapping[str, Any], path: Path, stage: str, limit: int | None) -> tuple[pd.DataFrame, bool]:
    calibration = parse_attribution_calibration_config(raw, path)
    suffix = "sanity" if stage == "sanity" else ""
    prediction_dir = calibration.output_dir / "predictions" / suffix / "calibration_eval" if suffix else calibration.output_dir / "predictions" / "calibration_eval"
    frame = _read_parts(prediction_dir, limit)
    # This is an Attribution-internal event_id enrichment, never a join to
    # Search Conversion. It recovers accounting cost/campaign when available.
    enrich = _read_attribution_fields(calibration.esmm.future_a_path, set(frame.event_id.astype(str)), limit)
    if not enrich.empty: frame = frame.merge(enrich, on="event_id", how="left", validate="one_to_one")
    calibrated = _apply_selected_calibrators(frame, calibration, suffix)
    frame["calibrated_pctr"], frame["calibrated_pctcvr"] = calibrated["pctr"], calibrated["pctcvr"]
    return frame, True


def _read_attribution_fields(path: Path, wanted: set[str], limit: int | None) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []; found = 0
    for file in sorted(path.glob("part-*.csv")):
        for frame in pd.read_csv(file, usecols=lambda col: col in {"event_id", "campaign", "cost"}, chunksize=200_000, low_memory=False):
            frame = frame.loc[frame.event_id.astype(str).isin(wanted)]
            pieces.append(frame); found += len(frame)
            if limit is not None and found >= limit: break
        if limit is not None and found >= limit: break
    return pd.concat(pieces, ignore_index=True).drop_duplicates("event_id") if pieces else pd.DataFrame(columns=["event_id", "campaign", "cost"])


def _apply_selected_calibrators(frame: pd.DataFrame, calibration: Any, suffix: str) -> dict[str, Any]:
    metrics = calibration.metrics_dir / f"calibration_metrics{'_sanity' if suffix else ''}.json"
    if not metrics.is_file(): raise FileNotFoundError(f"Selected calibration report required for business simulation: {metrics}")
    report = json.loads(metrics.read_text()); selected = report["selected_calibrator"]
    directory = calibration.output_dir / "calibrators" / suffix if suffix else calibration.output_dir / "calibrators"
    outputs = {}
    for target, column in (("ctr", "raw_pctr"), ("ctcvr", "raw_pctcvr")):
        kind = selected[target]; base = frame[column].to_numpy(float)
        if kind == "raw": calibrator = RawCalibrator()
        elif kind == "platt": calibrator = PlattCalibrator(**json.loads((directory / f"{target}_platt.json").read_text()))
        elif kind == "isotonic":
            with (directory / f"{target}_isotonic.pkl").open("rb") as handle: calibrator = pickle.load(handle)
        else: raise ValueError(f"Unsupported selected calibrator {kind}")
        outputs["pctr" if target == "ctr" else "pctcvr"] = calibrator.predict(base)
    serving, _ = serving_consistent_probabilities(outputs["pctr"], outputs["pctcvr"], calibration.epsilon)
    outputs["pcvr"] = serving
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly synthetic offline auction/bidding simulation")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("sanity", "all"), required=True)
    args = parser.parse_args(); config_path = args.config.resolve(); raw = load_yaml_config(config_path); options, output = _options(raw, config_path, args.stage)
    if warning := future_b_opened_warning(config_path): print(f"WARNING: {warning}", file=sys.stderr)
    simulation = SimulationConfig(seed=int(options.get("seed", 2026)), candidates_per_auction=int(options.get("candidates_per_auction", 5)), base_bid=float(options.get("base_bid", 1.0)), mechanism=str(options.get("mechanism", "second_price")), total_budget=float(options.get("total_budget", 1000.0)), pacing_min=float(options.get("pacing_min", .5)), pacing_max=float(options.get("pacing_max", 2.0)), budget_levels=tuple(float(value) for value in options.get("budget_levels", (.25, .5, .75, 1.0))))
    limit = options.get("max_rows"); limit = None if limit is None else int(limit)
    attribution, calibrated = _load_attribution(raw, config_path, args.stage, limit)
    attribution_report, policy, curve = simulate_attribution(attribution, config=simulation, calibrated_available=calibrated)
    search_report: dict[str, Any] = {"available": False, "reason": "No standalone Search Conversion prediction artifact configured.", "future_b_read_for_search_conversion_simulation": False}; search_path = options.get("search_prediction_path")
    if search_path:
        search_frame = _read_parts(resolve_path(str(search_path), config_path.parent), limit)
        fractions = tuple(float(value) for value in options.get("search_selection_fractions", (.10, .25, .50, .75, 1.0)))
        search_report, search_policy, search_deciles = simulate_search_conversion(search_frame, config=simulation, selection_fractions=fractions)
    output.joinpath("metrics").mkdir(parents=True, exist_ok=True); output.joinpath("tables").mkdir(parents=True, exist_ok=True)
    report = {"simulation_semantics": "synthetic_offline_simulation", **future_b_isolation_contract(), "cross_dataset_join": "not performed; Attribution and Search Conversion are simulated independently", "attribution": attribution_report, "search_conversion": search_report}
    (output / "metrics" / "auction_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output / "metrics" / "auction_metrics.md").write_text(_markdown(report), encoding="utf-8")
    policy.to_csv(output / "tables" / "policy_comparison.csv", index=False); curve.to_csv(output / "tables" / "budget_curve.csv", index=False)
    pd.DataFrame([report["attribution"]["calibration_business_impact"]]).to_csv(output / "tables" / "calibration_business_impact.csv", index=False)
    if search_path:
        search_policy.to_csv(output / "tables" / "search_conversion_value_policy_comparison.csv", index=False)
        search_deciles.to_csv(output / "tables" / "search_conversion_value_deciles.csv", index=False)
    print(json.dumps({"metrics": str(output / "metrics" / "auction_metrics.json"), "future_b_read_for_policy_selection": False}, indent=2))


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Advertising value / auction / bidding simulation", "", "All bid, grouping, auction, payment, budget, pacing, and efficiency results are synthetic offline simulation. They are not real advertiser bids, real auction outcomes, or online ROI.", "", "- Future-B read for policy selection: `false`.", "- Attribution and Search Conversion were not joined.", "", "## Attribution policies", ""]
    for name, value in report["attribution"]["policies"].items():
        if not value.get("available", True):
            lines.append(f"- {name}: unavailable")
        else:
            lines.append(f"- {name}: spend={value['simulated_spend']}, win_rate={value['win_rate']}, CPA proxy={value['cpa_proxy']}")
    search = report["search_conversion"]
    lines.extend(("", "## Search Conversion standalone selection", ""))
    if not search.get("available"):
        lines.append(f"- unavailable: {search['reason']}")
    else:
        lines.append("- `clicked_interaction_value_selection_simulation`: capacity-constrained selection only; not an impression auction, spend, or ROI simulation.")
        for policy, by_fraction in search["policies"].items():
            full = by_fraction.get("100%")
            if full: lines.append(f"- {policy} @100%: selected={full['selected_rows']}, value/click={full['actual_value_per_selected_click']}, value capture={full['value_capture_rate']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__": main()
