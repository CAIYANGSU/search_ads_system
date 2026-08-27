"""Independent Attribution and Search Conversion offline simulations."""

from __future__ import annotations

from dataclasses import dataclass
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


def simulate_search_conversion(frame: pd.DataFrame, *, config: SimulationConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    """Standalone clicked-interaction value selection; never an impression auction."""
    scored = search_value_scores(frame); grouped = group_candidates(scored, candidates_per_auction=config.candidates_per_auction, seed=config.seed)
    bids = synthetic_bid("value_scaled", base_bid=config.base_bid, value_per_click=grouped.score_value_per_click.to_numpy(float)); grouped["synthetic_bid"] = bids
    winners, curve = run_auction(grouped, quality_column="score_value_per_click", bid_column="synthetic_bid", mechanism=config.mechanism, budget=config.total_budget)
    selected = grouped.loc[winners.loc[winners.won, "winner_row_index"].astype(int)] if len(winners) else grouped.iloc[:0]
    actual = pd.to_numeric(selected.get("conversion_value_eur", pd.Series(0.0, index=selected.index)), errors="coerce").fillna(0.0)
    total = pd.to_numeric(grouped.get("conversion_value_eur", pd.Series(0.0, index=grouped.index)), errors="coerce").fillna(0.0)
    metrics = {"available": True, "definition": "clicked-interaction expected value selection; not impression-level auction ROI", "selected_clicks": int(len(selected)), "simulated_spend": float(winners.payment.sum()), "actual_value_captured": float(actual.sum()), "actual_value_per_click": float(actual.mean()) if len(actual) else None, "overall_actual_value_per_click": float(total.mean()) if len(total) else None, "value_per_click_lift": float(actual.mean() / total.mean()) if len(actual) and total.mean() else None, "budget_efficiency_proxy": float(actual.sum() / winners.payment.sum()) if winners.payment.sum() else None}
    return metrics, curve


def _attribution_metrics(candidates: pd.DataFrame, winners: pd.DataFrame, budget: float) -> dict[str, Any]:
    chosen = candidates.loc[winners.loc[winners.won, "winner_row_index"].astype(int)] if len(winners) else candidates.iloc[:0]
    spend = float(winners.payment.sum())
    clicks = pd.to_numeric(chosen.get("click", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    conversions = pd.to_numeric(chosen.get("conversion", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    observed_cost = pd.to_numeric(chosen.get("cost", pd.Series(dtype=float)), errors="coerce").dropna()
    return {"auctions": int(len(winners)), "impressions_won": int(winners.won.sum()), "win_rate": float(winners.won.mean()) if len(winners) else None, "simulated_spend": spend, "budget_utilization": spend / budget if budget else None, "end_of_horizon_budget_delta": budget - spend, "observed_ctr_among_won": float(clicks.mean()) if len(chosen) else None, "observed_conversion_rate_among_won": float(conversions.mean()) if len(chosen) else None, "cpa_proxy": spend / float(conversions.sum()) if conversions.sum() else None, "observed_cost_among_won": float(observed_cost.sum()) if len(observed_cost) else None, "accounting_note": "simulated spend/payment and observed source cost are separate definitions; neither is real online ROI."}
