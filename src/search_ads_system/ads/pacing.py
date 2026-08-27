"""A deterministic offline pacing baseline, not an online controller."""

from __future__ import annotations

import numpy as np


def pacing_multipliers(estimated_payments: np.ndarray, *, budget: float, minimum: float, maximum: float) -> np.ndarray:
    """Target linear cumulative spend with clipped target/observed adjustment."""
    if budget < 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("invalid pacing budget or multiplier bounds")
    payments = np.maximum(np.asarray(estimated_payments, dtype=float), 0.0)
    result = np.ones(len(payments), dtype=float); spent = 0.0
    for index in range(len(payments)):
        target = budget * (index + 1) / max(len(payments), 1)
        observed_rate = max(spent, 1e-12)
        result[index] = float(np.clip(target / observed_rate, minimum, maximum))
        spent += payments[index] * result[index]
    return result


def feedback_pacing_multiplier(*, current_multiplier: float, elapsed_horizon_fraction: float, cumulative_spend: float, total_budget: float, minimum: float, maximum: float, alpha: float, epsilon: float = 1e-12) -> float:
    """Deterministically update the next bid multiplier from realized spend only."""
    if total_budget <= 0 or minimum <= 0 or maximum < minimum or not 0 < alpha <= 1 or epsilon <= 0:
        raise ValueError("invalid feedback pacing settings")
    expected = float(np.clip(elapsed_horizon_fraction, 0.0, 1.0))
    actual = max(float(cumulative_spend) / total_budget, 0.0)
    pace_ratio = expected / max(actual, epsilon)
    return float(np.clip(float(current_multiplier) * pace_ratio ** alpha, minimum, maximum))
