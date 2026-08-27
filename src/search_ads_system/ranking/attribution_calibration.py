"""Probability calibration primitives for Attribution ESMM predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


CalibratorKind = Literal["raw", "platt", "isotonic"]


@dataclass(frozen=True)
class RawCalibrator:
    kind: str = "raw"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return clip_probabilities(probabilities)


@dataclass(frozen=True)
class PlattCalibrator:
    coefficient: float
    intercept: float
    epsilon: float
    kind: str = "platt"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        logits = probability_logit(probabilities, self.epsilon)
        return sigmoid(self.coefficient * logits + self.intercept)

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IsotonicCalibrator:
    model: IsotonicRegression
    kind: str = "isotonic"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return clip_probabilities(np.asarray(self.model.predict(clip_probabilities(probabilities)), dtype=np.float64))


@dataclass(frozen=True)
class UnavailableCalibrator:
    kind: str
    reason: str


def clip_probabilities(probabilities: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Convert to finite float64 probabilities and apply the required guard."""

    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Calibration probabilities contain non-finite values")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    return np.clip(values, epsilon, 1.0 - epsilon)


def probability_logit(probabilities: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    values = clip_probabilities(probabilities, epsilon)
    return np.log(values) - np.log1p(-values)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_platt(labels: np.ndarray, probabilities: np.ndarray, epsilon: float = 1e-7) -> PlattCalibrator | UnavailableCalibrator:
    """Fit logistic calibration to logit(raw_probability), never to labels alone."""

    y, p = _validate_fit_inputs(labels, probabilities, epsilon)
    if len(np.unique(y)) < 2:
        return UnavailableCalibrator("platt", "degenerate labels: both classes are required")
    try:
        estimator = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=1000, random_state=2026)
        estimator.fit(probability_logit(p, epsilon).reshape(-1, 1), y)
    except (ValueError, FloatingPointError) as error:
        return UnavailableCalibrator("platt", f"fit failed: {error}")
    return PlattCalibrator(float(estimator.coef_[0, 0]), float(estimator.intercept_[0]), epsilon)


def fit_isotonic(labels: np.ndarray, probabilities: np.ndarray, epsilon: float = 1e-7) -> IsotonicCalibrator | UnavailableCalibrator:
    """Fit monotonic calibration only when the fit window is non-degenerate."""

    y, p = _validate_fit_inputs(labels, probabilities, epsilon)
    if len(np.unique(y)) < 2:
        return UnavailableCalibrator("isotonic", "degenerate labels: both classes are required")
    if len(np.unique(p)) < 2:
        return UnavailableCalibrator("isotonic", "degenerate predictions: at least two values are required")
    try:
        estimator = IsotonicRegression(y_min=epsilon, y_max=1.0 - epsilon, out_of_bounds="clip")
        estimator.fit(p, y)
    except (ValueError, FloatingPointError) as error:
        return UnavailableCalibrator("isotonic", f"fit failed: {error}")
    return IsotonicCalibrator(estimator)


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int, epsilon: float = 1e-7) -> dict[str, Any]:
    """Ranking and probability-quality metrics plus fixed-width reliability bins."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    y, p = _validate_fit_inputs(labels, probabilities, epsilon)
    positives = int(y.sum())
    metrics: dict[str, Any] = {
        "rows": int(len(y)),
        "positive_count": positives,
        "negative_count": int(len(y) - positives),
        "label_mean": float(y.mean()),
        "prediction_mean": float(p.mean()),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "reliability_bins": reliability_bins(y, p, bins),
    }
    metrics["ece"] = float(sum(bin_["count"] * bin_["absolute_gap"] for bin_ in metrics["reliability_bins"]) / len(y))
    if positives in {0, len(y)}:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
        metrics["pr_auc"] = float(average_precision_score(y, p))
    metrics.update(calibration_intercept_slope(y, p, epsilon))
    return metrics


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> list[dict[str, Any]]:
    """Fixed-width [0, 1] bins whose counts sum exactly to the input rows."""

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    indices = np.minimum((p * bins).astype(np.int64), bins - 1)
    result: list[dict[str, Any]] = []
    for index in range(bins):
        mask = indices == index
        count = int(mask.sum())
        result.append({
            "bin_index": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": count,
            "prediction_mean": float(p[mask].mean()) if count else None,
            "label_mean": float(y[mask].mean()) if count else None,
            "absolute_gap": float(abs(p[mask].mean() - y[mask].mean())) if count else 0.0,
        })
    return result


def calibration_intercept_slope(labels: np.ndarray, probabilities: np.ndarray, epsilon: float = 1e-7) -> dict[str, float | None]:
    """Diagnostic logistic calibration intercept/slope estimated on an audit set."""

    y, p = _validate_fit_inputs(labels, probabilities, epsilon)
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return {"calibration_intercept": None, "calibration_slope": None}
    try:
        diagnostic = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=1000, random_state=2026)
        diagnostic.fit(probability_logit(p, epsilon).reshape(-1, 1), y)
    except ValueError:
        return {"calibration_intercept": None, "calibration_slope": None}
    return {"calibration_intercept": float(diagnostic.intercept_[0]), "calibration_slope": float(diagnostic.coef_[0, 0])}


def select_calibrator(metrics_by_kind: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Select by ECE, then LogLoss, then Brier; unavailable methods cannot win."""

    candidates = [(kind, metrics) for kind, metrics in metrics_by_kind.items() if metrics.get("available", True)]
    if not candidates:
        return "raw", {"reason": "all fitted calibrators unavailable; raw retained"}
    selected, metrics = min(candidates, key=lambda item: (item[1]["ece"], item[1]["logloss"], item[1]["brier_score"], item[0]))
    return selected, {"reason": "lowest ECE; ties resolved by LogLoss then Brier Score", "selection_metrics": {key: metrics[key] for key in ("ece", "logloss", "brier_score")}}


def serving_consistent_probabilities(calibrated_pctr: np.ndarray, calibrated_pctcvr: np.ndarray, epsilon: float = 1e-7) -> tuple[np.ndarray, dict[str, float]]:
    """Derive serving pCVR from selected CTR/CTCVR values to retain ESMM identity."""

    pctr = clip_probabilities(calibrated_pctr, epsilon)
    pctcvr = clip_probabilities(calibrated_pctcvr, epsilon)
    raw_ratio = pctcvr / np.maximum(pctr, epsilon)
    serving_pcvr = np.clip(raw_ratio, 0.0, 1.0)
    reconstructed = pctr * serving_pcvr
    error = np.abs(pctcvr - reconstructed)
    return serving_pcvr, {
        "max_abs_error": float(error.max()) if len(error) else 0.0,
        "mean_abs_error": float(error.mean()) if len(error) else 0.0,
        "fraction_requiring_clipping": float(((raw_ratio < 0.0) | (raw_ratio > 1.0)).mean()) if len(raw_ratio) else 0.0,
    }


def _validate_fit_inputs(labels: np.ndarray, probabilities: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    p = clip_probabilities(probabilities, epsilon)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Calibration labels and probabilities must be non-empty and aligned")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Calibration labels must be binary")
    return y, p
