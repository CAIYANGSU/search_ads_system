"""Independent Attribution and Search Conversion offline simulations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .auction import group_candidates, run_auction
from .bidding import synthetic_bid
from .pacing import pacing_multipliers
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


def future_b_isolation_contract() -> dict[str, object]:
    """Document the structural policy: simulations have no Future-B reader."""
    return {"future_b_read_for_policy_selection": False, "enforcement": "The simulator accepts only already-approved Attribution Future-A and standalone Search Conversion inputs."}


def simulate_attribution(frame: pd.DataFrame, *, config: SimulationConfig, calibrated_available: bool) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compare synthetic bids on an impression-level Attribution frame only."""
    grouped = group_candidates(frame, candidates_per_auction=config.candidates_per_auction, seed=config.seed)
    policies: list[tuple[str, bool, str, str]] = [
        ("fixed_bid", False, "score_ctr", "fixed_bid"),
        ("ctr_scaled", False, "score_ctr", "ctr_scaled"),
        ("ctcvr_scaled", False, "score_ctcvr", "ctcvr_scaled"),
    ]
    if calibrated_available:
        policies.append(("calibrated_ctcvr_scaled", True, "score_ctcvr", "calibrated_ctcvr_scaled"))
    else:
        policies.append(("calibrated_ctcvr_scaled", True, "score_ctcvr", "unavailable"))
    results: dict[str, Any] = {}; comparison: list[dict[str, Any]] = []; curves: list[pd.DataFrame] = []; winner_ids: dict[str, set[int]] = {}
    for name, calibrated, quality_key, bid_policy in policies:
        if bid_policy == "unavailable":
            results[name] = {"available": False, "reason": "No selected calibrated pCTR/pCTCVR artifact is available."}; continue
        scored = attribution_scores(grouped, calibrated=calibrated)
        quality = scored[quality_key].to_numpy(float)
        bids = synthetic_bid(bid_policy, base_bid=config.base_bid, pctr=quality, pctcvr=quality)
        scored["synthetic_bid"] = bids
        _, estimated = run_auction(scored, quality_column=quality_key, bid_column="synthetic_bid", mechanism=config.mechanism)
        by_budget: dict[str, Any] = {}; full_winners = None
        for level in config.budget_levels:
            budget = config.total_budget * level
            multipliers = pacing_multipliers(estimated.payment.to_numpy(float), budget=budget, minimum=config.pacing_min, maximum=config.pacing_max)
            winners, curve = run_auction(scored, quality_column=quality_key, bid_column="synthetic_bid", mechanism=config.mechanism, budget=budget, pacing_multipliers=multipliers)
            metrics = _attribution_metrics(scored, winners, budget)
            by_budget[f"{int(level * 100)}%"] = metrics
            curve["policy"] = name; curve["budget_level"] = level; curves.append(curve)
            if level == 1.0: full_winners = winners
        metrics = by_budget["100%"]
        results[name] = {"available": True, "quality_score": quality_key, "bid_policy": bid_policy, "budget_levels": by_budget, **metrics}
        comparison.append({"policy": name, **metrics}); winner_ids[name] = set(full_winners.loc[full_winners.won, "winner_row_index"].astype(int)) if full_winners is not None else set()
    raw_winners = winner_ids.get("ctcvr_scaled", set()); calibrated_winners = winner_ids.get("calibrated_ctcvr_scaled", set())
    calibration_impact = {"available": bool(calibrated_available), "comparison": "raw CTCVR synthetic bid vs selected calibrated CTCVR synthetic bid", "winner_change_count": len(raw_winners.symmetric_difference(calibrated_winners)), "winner_change_rate": len(raw_winners.symmetric_difference(calibrated_winners)) / max(len(raw_winners | calibrated_winners), 1), "raw": results.get("ctcvr_scaled"), "calibrated": results.get("calibrated_ctcvr_scaled")}
    return {"bid_semantics": "synthetic_offline_simulation", "candidate_grouping": "synthetic deterministic random, label blind", "mechanism": config.mechanism, "policies": results, "calibration_business_impact": calibration_impact}, pd.DataFrame(comparison), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def simulate_search_conversion(frame: pd.DataFrame, *, config: SimulationConfig, selection_fractions: tuple[float, ...] = (.10, .25, .50, .75, 1.0)) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Standalone clicked-interaction capacity selection, never an auction."""
    scored = search_value_scores(frame)
    required = {"conversion_label", "conversion_value_eur"}
    if missing := required - set(scored): raise ValueError(f"Search selection input missing {sorted(missing)}")
    if not selection_fractions or any(not 0 < value <= 1 for value in selection_fractions): raise ValueError("selection fractions must be in (0, 1]")
    scores = {"random_baseline": None, "pCVR_clicked": "pCVR_clicked", "predicted_conditional_value": "predicted_conditional_value", "expected_value_per_click": "score_value_per_click"}
    rng = np.random.default_rng(config.seed); random_order = rng.permutation(len(scored))
    actual_value = pd.to_numeric(scored.conversion_value_eur, errors="coerce").fillna(0.0).clip(lower=0.0)
    conversions = pd.to_numeric(scored.conversion_label, errors="coerce").fillna(0).astype(int)
    total_value, total_conversions = float(actual_value.sum()), int(conversions.sum())
    rows: list[dict[str, Any]] = []; nested: dict[str, Any] = {}
    for fraction in selection_fractions:
        capacity = min(len(scored), max(1, int(math.ceil(len(scored) * fraction))))
        baseline: dict[str, Any] | None = None
        for policy, column in scores.items():
            positions = random_order[:capacity] if column is None else np.argsort(-pd.to_numeric(scored[column], errors="coerce").fillna(0.0).to_numpy(float), kind="stable")[:capacity]
            selected_value, selected_conversion = actual_value.iloc[positions], conversions.iloc[positions]
            metrics = {"policy": policy, "selection_fraction": fraction, "selected_rows": int(capacity), "observed_conversions": int(selected_conversion.sum()), "observed_conversion_rate": float(selected_conversion.mean()), "actual_conversion_value_sum": float(selected_value.sum()), "actual_value_per_selected_click": float(selected_value.mean()), "value_capture_rate": float(selected_value.sum() / total_value) if total_value else None, "conversion_capture_rate": float(selected_conversion.sum() / total_conversions) if total_conversions else None}
            if baseline is None:
                baseline = metrics
            metrics["value_per_click_lift"] = float(metrics["actual_value_per_selected_click"] / baseline["actual_value_per_selected_click"]) if baseline["actual_value_per_selected_click"] else None
            metrics["value_capture_lift"] = float(metrics["value_capture_rate"] / baseline["value_capture_rate"]) if baseline["value_capture_rate"] else None
            rows.append(metrics); nested.setdefault(policy, {})[f"{int(fraction * 100)}%"] = metrics
    deciles = _search_deciles(scored, conversions, actual_value, total_value)
    report = {"available": True, "definition": "clicked_interaction_value_selection_simulation; capacity-constrained Top-K selection, not impression-level bidding, auction, spend, or ROI", "selection_capacity_semantics": "fractions are allowed clicked-interaction selection capacity, not advertiser monetary budgets", "future_b_read_for_search_conversion_simulation": False, "overall_rows": int(len(scored)), "overall_actual_conversion_value": total_value, "overall_observed_conversions": total_conversions, "policies": nested}
    return report, pd.DataFrame(rows), deciles


def _search_deciles(scored: pd.DataFrame, conversions: pd.Series, actual_value: pd.Series, total_value: float) -> pd.DataFrame:
    order = np.argsort(-scored.score_value_per_click.to_numpy(float), kind="stable")
    groups = np.array_split(order, 10); cumulative = 0.0; rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups, 1):
        value = float(actual_value.iloc[group].sum()); cumulative += value
        rows.append({"decile": index, "decile_order": "1=highest predicted expected_value_per_click", "rows": int(len(group)), "predicted_score_mean": float(scored.score_value_per_click.iloc[group].mean()) if len(group) else None, "actual_CVR": float(conversions.iloc[group].mean()) if len(group) else None, "actual_value_per_click": float(actual_value.iloc[group].mean()) if len(group) else None, "cumulative_value_capture": cumulative / total_value if total_value else None})
    return pd.DataFrame(rows)


def _attribution_metrics(candidates: pd.DataFrame, winners: pd.DataFrame, budget: float) -> dict[str, Any]:
    chosen = candidates.loc[winners.loc[winners.won, "winner_row_index"].astype(int)] if len(winners) else candidates.iloc[:0]
    spend = float(winners.payment.sum())
    clicks = pd.to_numeric(chosen.get("click", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    conversions = pd.to_numeric(chosen.get("conversion", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    observed_cost = pd.to_numeric(chosen.get("cost", pd.Series(dtype=float)), errors="coerce").dropna()
    return {"auctions": int(len(winners)), "impressions_won": int(winners.won.sum()), "win_rate": float(winners.won.mean()) if len(winners) else None, "simulated_spend": spend, "budget_utilization": spend / budget if budget else None, "end_of_horizon_budget_delta": budget - spend, "observed_ctr_among_won": float(clicks.mean()) if len(chosen) else None, "observed_conversion_rate_among_won": float(conversions.mean()) if len(chosen) else None, "cpa_proxy": spend / float(conversions.sum()) if conversions.sum() else None, "observed_cost_among_won": float(observed_cost.sum()) if len(observed_cost) else None, "accounting_note": "simulated spend/payment and observed source cost are separate definitions; neither is real online ROI."}
