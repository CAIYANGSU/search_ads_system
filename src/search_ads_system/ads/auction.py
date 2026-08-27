"""Transparent deterministic synthetic auction grouping and clearing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pacing import feedback_pacing_multiplier


def group_candidates(frame: pd.DataFrame, *, candidates_per_auction: int, seed: int) -> pd.DataFrame:
    """Assign rows to synthetic auctions via a seed-fixed label-blind shuffle."""
    if candidates_per_auction < 2:
        raise ValueError("candidates_per_auction must be at least two")
    output = frame.reset_index(drop=True).copy()
    order = np.random.default_rng(seed).permutation(len(output))
    group = np.empty(len(output), dtype=np.int64)
    group[order] = np.arange(len(output), dtype=np.int64) // candidates_per_auction
    output["synthetic_auction_id"] = group
    output["candidate_grouping"] = "deterministic_random_label_blind"
    return output


def run_auction(candidates: pd.DataFrame, *, quality_column: str, bid_column: str, mechanism: str, budget: float | None = None, pacing_multipliers: np.ndarray | None = None, feedback_pacing: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank by quality*bid and apply first/second-price clearing per group."""
    if mechanism not in {"first_price", "second_price"}:
        raise ValueError("mechanism must be first_price or second_price")
    required = {"synthetic_auction_id", quality_column, bid_column}
    if missing := required - set(candidates): raise ValueError(f"auction candidates missing {sorted(missing)}")
    if pacing_multipliers is not None and feedback_pacing is not None:
        raise ValueError("choose either static pacing_multipliers or feedback_pacing")
    if feedback_pacing is not None and budget is None:
        raise ValueError("feedback_pacing requires a total budget")
    rows: list[dict[str, object]] = []; curve: list[dict[str, float | int]] = []; spend = 0.0
    total_auctions = int(candidates["synthetic_auction_id"].nunique())
    feedback_multiplier = float(feedback_pacing.get("initial_multiplier", 1.0)) if feedback_pacing else 1.0
    for order, (auction_id, group) in enumerate(candidates.groupby("synthetic_auction_id", sort=True)):
        quality = np.maximum(pd.to_numeric(group[quality_column], errors="raise").to_numpy(float), 0.0)
        multiplier = float(pacing_multipliers[order]) if pacing_multipliers is not None else feedback_multiplier if feedback_pacing is not None else 1.0
        bids = np.maximum(pd.to_numeric(group[bid_column], errors="raise").to_numpy(float), 0.0) * multiplier
        score = quality * bids
        ranking = np.lexsort((np.arange(len(group)), -score)); winner = int(ranking[0]); second = int(ranking[1]) if len(ranking) > 1 else winner
        winning_bid, winning_score = float(bids[winner]), float(score[winner])
        market_price = winning_bid if mechanism == "first_price" else min(winning_bid, float(score[second] / max(quality[winner], 1e-12)))
        can_win = budget is None or spend + market_price <= budget + 1e-12
        spend += market_price if can_win else 0.0
        if feedback_pacing is not None:
            feedback_multiplier = feedback_pacing_multiplier(current_multiplier=multiplier, elapsed_horizon_fraction=(order + 1) / max(total_auctions, 1), cumulative_spend=spend, total_budget=float(budget), minimum=float(feedback_pacing["minimum"]), maximum=float(feedback_pacing["maximum"]), alpha=float(feedback_pacing["alpha"]), epsilon=float(feedback_pacing.get("epsilon", 1e-12)))
        winner_index = int(group.index[winner])
        rows.append({"synthetic_auction_id": int(auction_id), "winner_row_index": winner_index, "winning_bid": winning_bid, "rank_score": winning_score, "market_price": market_price, "payment": market_price if can_win else 0.0, "won": bool(can_win), "pacing_multiplier": multiplier})
        curve.append({"auction_order": order, "synthetic_auction_id": int(auction_id), "cumulative_spend": spend, "cumulative_spend_fraction": spend / float(budget) if budget else None, "payment": market_price if can_win else 0.0, "pacing_multiplier": multiplier, "next_pacing_multiplier": feedback_multiplier if feedback_pacing is not None else multiplier})
    return pd.DataFrame(rows), pd.DataFrame(curve)
