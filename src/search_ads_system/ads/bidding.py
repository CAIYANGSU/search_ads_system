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
