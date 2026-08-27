"""Independent Attribution auction and Search Conversion selection simulations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .auction import group_candidates, run_auction
from .bidding import synthetic_bid, target_cpa_bid, value_based_bid
from .value_scoring import attribution_scores, search_value_scores


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 2026
    candidates_per_auction: int = 5
    base_bid: float = 1.0
    mechanism: str = "second_price"
    total_budget: float = 1000.0
    pacing_min: float = 0.5
    pacing_max: float = 2.0
    budget_levels: tuple[float, ...] = (.25, .50, .75, 1.0)
    target_cpa_enabled: bool = True
    target_cpa_values: tuple[float, ...] = (.5, 1.0, 2.0)
    default_target_cpa: float = 1.0
    target_cpa_bid_scale: float = 1.0
    value_based_enabled: bool = True
    target_roas_values: tuple[float, ...] = (1.0, 2.0, 4.0)
    default_target_roas: float = 2.0
    value_based_bid_scale: float = 1.0
    value_prediction_column: str | None = None
    pacing_enabled: bool = True
    pacing_alpha: float = 0.2
    pacing_epsilon: float = 1e-12
    pacing_comparison_policies: tuple[str, ...] = ("calibrated_ctcvr_scaled", "target_cpa_bidding")
    trajectory_buckets: int = 10


def future_b_isolation_contract() -> dict[str, object]:
    return {
        "future_b_read_for_policy_selection": False,
        "future_b_read_for_bidding_experiment": False,
        "enforcement": "The runner rejects any input path containing Future-B; policies use approved Attribution Future-A predictions only.",
    }


def simulate_attribution(frame: pd.DataFrame, *, config: SimulationConfig, calibrated_available: bool) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run label-blind grouped auctions; labels are only post-selection metrics."""
    _validate_config(config)
    grouped = group_candidates(frame, candidates_per_auction=config.candidates_per_auction, seed=config.seed)
    specs: list[dict[str, Any]] = [
        {"name": "fixed_bid", "calibrated": False, "quality_score": "score_ctr", "bid_policy": "fixed_bid", "bid_semantics": "synthetic_fixed_bid"},
        {"name": "ctr_scaled", "calibrated": False, "quality_score": "score_ctr", "bid_policy": "ctr_scaled", "bid_semantics": "synthetic_pctr_scaled_bid"},
        {"name": "ctcvr_scaled", "calibrated": False, "quality_score": "score_ctcvr", "bid_policy": "ctcvr_scaled", "bid_semantics": "synthetic_pctcvr_scaled_bid"},
        {"name": "calibrated_ctcvr_scaled", "calibrated": True, "quality_score": "score_ctcvr", "bid_policy": "calibrated_ctcvr_scaled", "bid_semantics": "synthetic_calibrated_pctcvr_scaled_bid"},
    ]
    if config.target_cpa_enabled:
        specs.append({"name": "target_cpa_bidding", "calibrated": True, "quality_score": "score_ctcvr", "bid_policy": "target_cpa_bidding", "bid_semantics": "synthetic_target_cpa_bidding", "target_cpa": config.default_target_cpa})

    results: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    winner_ids: dict[str, set[int]] = {}
    for spec in specs:
        if spec["calibrated"] and not calibrated_available:
            results[spec["name"]] = {"available": False, "reason": "No selected calibrated pCTR/pCTCVR artifact is available."}
            continue
        scored, quality_key = _scored_bids(grouped, spec, config)
        result, rows, policy_curves = _evaluate_policy(scored, quality_key, spec, config)
        results[spec["name"]] = result
        comparison.extend(rows); curves.extend(policy_curves)
        winner_ids[spec["name"]] = set(result["budget_levels"].get("100%", {}).get("winner_row_indices", []))

    value = _value_based_result(grouped, config, calibrated_available)
    if value["available"]:
        results["value_based_bidding"] = value["result"]
        comparison.extend(value["comparison_rows"]); curves.extend(value["curves"])
    else:
        results["value_based_bidding"] = {"available": False, "reason": value["reason"], "real_online_roas_available": False}
    sensitivity = _target_cpa_sensitivity(grouped, config, calibrated_available)
    raw_ids, calibrated_ids = winner_ids.get("ctcvr_scaled", set()), winner_ids.get("calibrated_ctcvr_scaled", set())
    report = {
        "bid_semantics": "synthetic_offline_simulation",
        "candidate_grouping": "synthetic deterministic random, label blind",
        "mechanism": config.mechanism,
        "cross_dataset_join": False,
        "data_contract": "Attribution impression-level modeling plus synthetic auction simulation only.",
        "policies": results,
        "policy_comparison": comparison,
        "target_cpa_sensitivity": sensitivity,
        "value_based_bidding": results["value_based_bidding"],
        "pacing_comparison": _pacing_comparison(results, config),
        "calibration_business_impact": {
            "available": bool(calibrated_available),
            "comparison": "raw CTCVR synthetic bid vs selected calibrated CTCVR synthetic bid",
            "winner_change_count": len(raw_ids.symmetric_difference(calibrated_ids)),
            "winner_change_rate": len(raw_ids.symmetric_difference(calibrated_ids)) / max(len(raw_ids | calibrated_ids), 1),
            "raw": results.get("ctcvr_scaled"), "calibrated": results.get("calibrated_ctcvr_scaled"),
        },
    }
    clean = [{key: value for key, value in row.items() if key != "winner_row_indices"} for row in comparison]
    return report, pd.DataFrame(clean), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def _validate_config(config: SimulationConfig) -> None:
    if not config.budget_levels or any(not 0 < value <= 1 for value in config.budget_levels): raise ValueError("budget_levels must be fractions in (0, 1]")
    if config.trajectory_buckets <= 0: raise ValueError("trajectory_buckets must be positive")
    if not 0 < config.pacing_alpha <= 1 or config.pacing_epsilon <= 0: raise ValueError("pacing alpha must be in (0, 1] and epsilon positive")


def _scored_bids(grouped: pd.DataFrame, spec: Mapping[str, Any], config: SimulationConfig, *, predicted_value: np.ndarray | None = None) -> tuple[pd.DataFrame, str]:
    scored = attribution_scores(grouped, calibrated=bool(spec["calibrated"]))
    quality_key = str(spec["quality_score"]); quality = scored[quality_key].to_numpy(float); policy = str(spec["bid_policy"])
    if policy == "target_cpa_bidding":
        bids = target_cpa_bid(calibrated_pctcvr=quality, target_cpa=float(spec["target_cpa"]), bid_scale=config.target_cpa_bid_scale)
    elif policy == "value_based_bidding":
        if predicted_value is None: raise ValueError("value_based_bidding requires prediction values")
        bids = value_based_bid(calibrated_pctcvr=quality, predicted_conditional_conversion_value=predicted_value, target_roas=float(spec["target_roas"]), bid_scale=config.value_based_bid_scale)
    else:
        bids = synthetic_bid(policy, base_bid=config.base_bid, pctr=quality, pctcvr=quality)
    scored["synthetic_bid"] = bids
    return scored, quality_key


def _evaluate_policy(scored: pd.DataFrame, quality_key: str, spec: Mapping[str, Any], config: SimulationConfig) -> tuple[dict[str, Any], list[dict[str, Any]], list[pd.DataFrame]]:
    levels: dict[str, Any] = {}; rows: list[dict[str, Any]] = []; curves: list[pd.DataFrame] = []
    for level in config.budget_levels:
        budget = config.total_budget * level
        modes = ("no_pacing", "budget_aware_pacing") if config.pacing_enabled and str(spec["name"]) in config.pacing_comparison_policies else ("budget_aware_pacing",)
        by_mode: dict[str, Any] = {}
        for mode in modes:
            feedback = None if mode == "no_pacing" else {"initial_multiplier": 1.0, "minimum": config.pacing_min, "maximum": config.pacing_max, "alpha": config.pacing_alpha, "epsilon": config.pacing_epsilon}
            winners, curve = run_auction(scored, quality_column=quality_key, bid_column="synthetic_bid", mechanism=config.mechanism, budget=budget, feedback_pacing=feedback)
            metrics = _attribution_metrics(scored, winners, budget)
            metrics.update(_pacing_metrics(winners, curve, budget, config.trajectory_buckets)); metrics["pacing_mode"] = mode
            by_mode[mode] = metrics
            row = {"policy_name": spec["name"], "bid_semantics": spec["bid_semantics"], "quality_score": quality_key, "budget": budget, **metrics}
            if "target_cpa" in spec: row["target_cpa"] = spec["target_cpa"]
            if "target_roas" in spec:
                row["target_roas"] = spec["target_roas"]
                row["expected_conversion_value"] = spec["expected_conversion_value"]
                row["real_online_roas_available"] = False
            rows.append(row)
            curve = curve.copy(); curve["policy_name"] = spec["name"]; curve["bid_semantics"] = spec["bid_semantics"]; curve["budget_level"] = level; curve["budget"] = budget; curve["pacing_mode"] = mode
            curves.append(curve)
        selected = by_mode["budget_aware_pacing"]
        levels[f"{int(level * 100)}%"] = {**selected, "pacing_comparison": by_mode}
    primary = levels["100%"]
    result = {"available": True, "quality_score": quality_key, "bid_policy": spec["bid_policy"], "bid_semantics": spec["bid_semantics"], "budget_levels": levels, **{key: value for key, value in primary.items() if key not in {"pacing_comparison", "winner_row_indices"}}}
    if "target_cpa" in spec: result.update({"target_cpa": spec["target_cpa"], "bid_scale": config.target_cpa_bid_scale, "normalization": "none; direct deterministic pCTCVR * target_CPA * bid_scale in synthetic units"})
    if "target_roas" in spec: result.update({"target_roas": spec["target_roas"], "bid_scale": config.value_based_bid_scale, "real_online_roas_available": False})
    return result, rows, curves


def _target_cpa_sensitivity(grouped: pd.DataFrame, config: SimulationConfig, calibrated_available: bool) -> dict[str, Any]:
    if not config.target_cpa_enabled: return {"available": False, "reason": "target_cpa.enabled=false"}
    if not calibrated_available: return {"available": False, "reason": "selected calibrated pCTCVR artifact unavailable"}
    values: dict[str, Any] = {}
    for target in config.target_cpa_values:
        spec = {"name": "target_cpa_bidding", "calibrated": True, "quality_score": "score_ctcvr", "bid_policy": "target_cpa_bidding", "bid_semantics": "synthetic_target_cpa_bidding", "target_cpa": float(target)}
        scored, quality = _scored_bids(grouped, spec, config); result, _, _ = _evaluate_policy(scored, quality, spec, config)
        values[str(target)] = result["budget_levels"]
    return {"available": True, "formula": "bid = calibrated_pCTCVR * target_CPA * bid_scale", "synthetic_units": True, "bid_scale": config.target_cpa_bid_scale, "default_target_cpa": config.default_target_cpa, "values": values}


def _value_based_result(grouped: pd.DataFrame, config: SimulationConfig, calibrated_available: bool) -> dict[str, Any]:
    column = config.value_prediction_column
    if not config.value_based_enabled: return {"available": False, "reason": "value_based_bidding.enabled=false"}
    if not calibrated_available: return {"available": False, "reason": "selected calibrated pCTCVR artifact unavailable"}
    if not column: return {"available": False, "reason": "No declared Attribution predicted conditional conversion-value artifact/column is configured; real labels and Search Conversion values are forbidden."}
    forbidden = {"conversion", "conversion_label", "conversion_value", "conversion_value_eur", "click"}
    if column in forbidden or "conversion_value" in column.lower() or not column.startswith("predicted_"): return {"available": False, "reason": f"Configured value column {column!r} is not an explicitly named prediction-only field and is forbidden for bidding."}
    if column not in grouped: return {"available": False, "reason": f"Configured leakage-safe value prediction column {column!r} is absent from Attribution Future-A prediction artifact."}
    values = pd.to_numeric(grouped[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all() or (values < 0).any(): return {"available": False, "reason": f"Configured value prediction column {column!r} contains invalid predictions."}
    spec = {"name": "value_based_bidding", "calibrated": True, "quality_score": "score_ctcvr", "bid_policy": "value_based_bidding", "bid_semantics": "synthetic_expected_value_over_target_roas_bid", "target_roas": config.default_target_roas, "expected_conversion_value": "calibrated_pCTCVR * predicted_conditional_conversion_value"}
    scored, quality = _scored_bids(grouped, spec, config, predicted_value=values.to_numpy(float)); result, rows, curves = _evaluate_policy(scored, quality, spec, config)
    sensitivity: dict[str, Any] = {}
    for target_roas in config.target_roas_values:
        sensitivity_spec = {**spec, "target_roas": float(target_roas)}
        sensitivity_scored, sensitivity_quality = _scored_bids(grouped, sensitivity_spec, config, predicted_value=values.to_numpy(float))
        sensitivity_result, _, _ = _evaluate_policy(sensitivity_scored, sensitivity_quality, sensitivity_spec, config)
        sensitivity[str(target_roas)] = sensitivity_result["budget_levels"]
    result["expected_conversion_value"] = "calibrated_pCTCVR * predicted_conditional_conversion_value; prediction-only input"; result["target_roas_sensitivity"] = sensitivity
    return {"available": True, "result": result, "comparison_rows": rows, "curves": curves}


def _pacing_comparison(results: Mapping[str, Any], config: SimulationConfig) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for name in config.pacing_comparison_policies:
        result = results.get(name)
        if isinstance(result, Mapping) and result.get("available"):
            policies[name] = {level: value["pacing_comparison"] for level, value in result["budget_levels"].items()}
    return {"available": bool(policies), "definition": "no_pacing and deterministic feedback budget_aware_pacing share grouped auction order, prediction, budget, and seed", "feedback_formula": "next_multiplier = clip(current_multiplier * (elapsed_horizon_fraction / max(cumulative_spend / total_budget, epsilon)) ** alpha, pacing_min, pacing_max)", "trajectory_buckets": config.trajectory_buckets, "early_budget_exhaustion_definition": "first cumulative_spend_fraction >= 0.99 before 90% of auction order", "spend_smoothness_definition": "mean absolute deviation between cumulative spend fraction and cumulative horizon fraction; lower is smoother", "trajectory_contract": {"cumulative_budget_fraction": "planned linear budget fraction, equal to cumulative horizon fraction", "cumulative_spend_fraction": "actual cumulative spend divided by configured total budget", "cumulative_impression_fraction": "won impressions divided by final won impressions", "remaining_budget_fraction": "unspent configured budget divided by configured budget"}, "policies": policies}


def _pacing_metrics(winners: pd.DataFrame, curve: pd.DataFrame, budget: float, buckets: int) -> dict[str, Any]:
    trajectory = _spend_trajectory(winners, curve, budget, buckets)
    spend_fraction = curve.cumulative_spend.to_numpy(float) / budget if budget else np.zeros(len(curve), dtype=float)
    exhausted = np.flatnonzero(spend_fraction >= .99)
    exhaustion_fraction = float((int(exhausted[0]) + 1) / len(curve)) if len(exhausted) else None
    return {"budget_exhaustion_horizon_fraction": exhaustion_fraction, "early_budget_exhaustion": bool(exhaustion_fraction is not None and exhaustion_fraction < .9), "spend_smoothness": float(trajectory["spend_smoothness"].iat[0]) if len(trajectory) else None, "spend_trajectory": trajectory.to_dict(orient="records"), "winner_row_indices": winners.loc[winners.won, "winner_row_index"].astype(int).tolist()}


def _spend_trajectory(winners: pd.DataFrame, curve: pd.DataFrame, budget: float, buckets: int) -> pd.DataFrame:
    if curve.empty: return pd.DataFrame(columns=["bucket", "cumulative_horizon_fraction", "cumulative_budget_fraction", "cumulative_spend_fraction", "cumulative_impression_fraction", "remaining_budget_fraction", "spend_smoothness"])
    total, wins_total = len(curve), max(int(winners.won.sum()), 1); rows: list[dict[str, Any]] = []
    for bucket in range(1, buckets + 1):
        end = min(total, math.ceil(total * bucket / buckets)); at_end = curve.iloc[end - 1]; horizon = end / total; spend = float(at_end.cumulative_spend)
        rows.append({"bucket": f"bucket_{bucket}", "cumulative_horizon_fraction": horizon, "cumulative_budget_fraction": horizon, "cumulative_spend_fraction": spend / budget if budget else None, "cumulative_impression_fraction": int(winners.iloc[:end].won.sum()) / wins_total, "remaining_budget_fraction": max(budget - spend, 0.0) / budget if budget else None})
    smoothness = float(np.mean([abs(row["cumulative_spend_fraction"] - row["cumulative_horizon_fraction"]) for row in rows]))
    for row in rows: row["spend_smoothness"] = smoothness
    return pd.DataFrame(rows)


def simulate_search_conversion(frame: pd.DataFrame, *, config: SimulationConfig, selection_fractions: tuple[float, ...] = (.10, .25, .50, .75, 1.0)) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Standalone clicked-interaction capacity selection, never an auction."""
    scored = search_value_scores(frame); required = {"conversion_label", "conversion_value_eur"}
    if missing := required - set(scored): raise ValueError(f"Search selection input missing {sorted(missing)}")
    if not selection_fractions or any(not 0 < value <= 1 for value in selection_fractions): raise ValueError("selection fractions must be in (0, 1]")
    scores = {"random_baseline": None, "pCVR_clicked": "pCVR_clicked", "predicted_conditional_value": "predicted_conditional_value", "expected_value_per_click": "score_value_per_click"}; rng = np.random.default_rng(config.seed); random_order = rng.permutation(len(scored))
    actual_value = pd.to_numeric(scored.conversion_value_eur, errors="coerce").fillna(0.0).clip(lower=0.0); conversions = pd.to_numeric(scored.conversion_label, errors="coerce").fillna(0).astype(int); total_value, total_conversions = float(actual_value.sum()), int(conversions.sum()); rows: list[dict[str, Any]] = []; nested: dict[str, Any] = {}
    for fraction in selection_fractions:
        capacity = min(len(scored), max(1, int(math.ceil(len(scored) * fraction)))); baseline: dict[str, Any] | None = None
        for policy, column in scores.items():
            positions = random_order[:capacity] if column is None else np.argsort(-pd.to_numeric(scored[column], errors="coerce").fillna(0.0).to_numpy(float), kind="stable")[:capacity]; selected_value, selected_conversion = actual_value.iloc[positions], conversions.iloc[positions]
            metrics = {"policy": policy, "selection_fraction": fraction, "selected_rows": int(capacity), "observed_conversions": int(selected_conversion.sum()), "observed_conversion_rate": float(selected_conversion.mean()), "actual_conversion_value_sum": float(selected_value.sum()), "actual_value_per_selected_click": float(selected_value.mean()), "value_capture_rate": float(selected_value.sum() / total_value) if total_value else None, "conversion_capture_rate": float(selected_conversion.sum() / total_conversions) if total_conversions else None}
            if baseline is None: baseline = metrics
            metrics["value_per_click_lift"] = float(metrics["actual_value_per_selected_click"] / baseline["actual_value_per_selected_click"]) if baseline["actual_value_per_selected_click"] else None; metrics["value_capture_lift"] = float(metrics["value_capture_rate"] / baseline["value_capture_rate"]) if baseline["value_capture_rate"] else None; rows.append(metrics); nested.setdefault(policy, {})[f"{int(fraction * 100)}%"] = metrics
    deciles = _search_deciles(scored, conversions, actual_value, total_value)
    return {"available": True, "definition": "clicked_interaction_value_selection_simulation; capacity-constrained Top-K selection, not impression-level bidding, auction, spend, or ROI", "selection_capacity_semantics": "fractions are allowed clicked-interaction selection capacity, not advertiser monetary budgets", "future_b_read_for_search_conversion_simulation": False, "overall_rows": int(len(scored)), "overall_actual_conversion_value": total_value, "overall_observed_conversions": total_conversions, "policies": nested}, pd.DataFrame(rows), deciles


def _search_deciles(scored: pd.DataFrame, conversions: pd.Series, actual_value: pd.Series, total_value: float) -> pd.DataFrame:
    order = np.argsort(-scored.score_value_per_click.to_numpy(float), kind="stable"); groups = np.array_split(order, 10); cumulative = 0.0; rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups, 1):
        value = float(actual_value.iloc[group].sum()); cumulative += value
        rows.append({"decile": index, "decile_order": "1=highest predicted expected_value_per_click", "rows": int(len(group)), "predicted_score_mean": float(scored.score_value_per_click.iloc[group].mean()) if len(group) else None, "actual_CVR": float(conversions.iloc[group].mean()) if len(group) else None, "actual_value_per_click": float(actual_value.iloc[group].mean()) if len(group) else None, "cumulative_value_capture": cumulative / total_value if total_value else None})
    return pd.DataFrame(rows)


def _attribution_metrics(candidates: pd.DataFrame, winners: pd.DataFrame, budget: float) -> dict[str, Any]:
    chosen = candidates.loc[winners.loc[winners.won, "winner_row_index"].astype(int)] if len(winners) else candidates.iloc[:0]; spend = float(winners.payment.sum()); clicks = pd.to_numeric(chosen.get("click", pd.Series(dtype=float)), errors="coerce").fillna(0.0); conversions = pd.to_numeric(chosen.get("conversion", pd.Series(dtype=float)), errors="coerce").fillna(0.0); observed_cost = pd.to_numeric(chosen.get("cost", pd.Series(dtype=float)), errors="coerce").dropna()
    observed_ctr = float(clicks.mean()) if len(chosen) else None; observed_conversion_rate = float(conversions.mean()) if len(chosen) else None
    return {"auctions": int(len(winners)), "impressions_won": int(winners.won.sum()), "win_rate": float(winners.won.mean()) if len(winners) else None, "simulated_spend": spend, "budget_utilization": spend / budget if budget else None, "end_of_horizon_budget_delta": budget - spend, "observed_ctr_among_won": observed_ctr, "observed_conversion_rate_among_won": observed_conversion_rate, "observed_ctr": observed_ctr, "observed_conversion_rate": observed_conversion_rate, "cpa_proxy": spend / float(conversions.sum()) if conversions.sum() else None, "observed_cost_among_won": float(observed_cost.sum()) if len(observed_cost) else None, "accounting_note": "simulated spend/payment and observed source cost are separate definitions; neither is real online ROI."}
