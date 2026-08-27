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
