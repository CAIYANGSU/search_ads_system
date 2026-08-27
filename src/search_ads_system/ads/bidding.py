"""Clearly labelled synthetic bid policies for offline simulations."""

from __future__ import annotations

import numpy as np


def synthetic_bid(policy: str, *, base_bid: float, pctr: np.ndarray | None = None, pctcvr: np.ndarray | None = None, value_per_click: np.ndarray | None = None) -> np.ndarray:
    """Return synthetic bids; these are never observed advertiser bids."""
    if base_bid < 0:
        raise ValueError("base_bid must be non-negative")
    if policy == "fixed_bid":
        size = len(pctr) if pctr is not None else len(pctcvr) if pctcvr is not None else len(value_per_click) if value_per_click is not None else 0
        return np.full(size, base_bid, dtype=float)
    if policy == "ctr_scaled":
        if pctr is None: raise ValueError("ctr_scaled requires pCTR")
        return base_bid * np.clip(np.asarray(pctr, dtype=float), 0.0, 1.0)
    if policy in {"ctcvr_scaled", "calibrated_ctcvr_scaled"}:
        if pctcvr is None: raise ValueError(f"{policy} requires pCTCVR")
        return base_bid * np.clip(np.asarray(pctcvr, dtype=float), 0.0, 1.0)
    if policy == "value_scaled":
        if value_per_click is None: raise ValueError("value_scaled requires expected conversion value per click")
        values = np.maximum(np.asarray(value_per_click, dtype=float), 0.0)
        scale = float(np.quantile(values, .95)) if len(values) else 0.0
        return base_bid * np.clip(values / max(scale, 1e-12), 0.0, 1.0)
    raise ValueError(f"Unknown synthetic bid policy: {policy}")


def target_cpa_bid(*, calibrated_pctcvr: np.ndarray, target_cpa: float, bid_scale: float) -> np.ndarray:
    """Synthetic Target-CPA bid: calibrated pCTCVR * target CPA * scale.

    `target_cpa` and the returned bid are deliberately synthetic units.  The
    function accepts predictions only; it has no access to labels or actual
    conversion value.
    """
    if target_cpa < 0 or bid_scale < 0:
        raise ValueError("target_cpa and bid_scale must be non-negative")
    probability = np.clip(np.asarray(calibrated_pctcvr, dtype=float), 0.0, 1.0)
    return probability * float(target_cpa) * float(bid_scale)


def value_based_bid(*, calibrated_pctcvr: np.ndarray, predicted_conditional_conversion_value: np.ndarray, target_roas: float, bid_scale: float) -> np.ndarray:
    """Synthetic expected-value bid using model predictions only.

    expected value per impression = calibrated pCTCVR * predicted conditional
    conversion value; bid = expected value / target ROAS * scale.
    """
    if target_roas <= 0 or bid_scale < 0:
        raise ValueError("target_roas must be positive and bid_scale non-negative")
    probability = np.clip(np.asarray(calibrated_pctcvr, dtype=float), 0.0, 1.0)
    value = np.maximum(np.asarray(predicted_conditional_conversion_value, dtype=float), 0.0)
    return probability * value / float(target_roas) * float(bid_scale)
