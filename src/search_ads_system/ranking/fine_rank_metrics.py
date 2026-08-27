"""Metrics for click-conditioned CVR, conversion value and value ranking."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score


def evaluate_fine_rank_predictions(rows: Iterable[dict[str, Any]], *, cutoffs: tuple[int, ...] = (10, 20)) -> dict[str, Any]:
    """Calculate prediction and per-user ranking metrics from streamed batches.

    Each element is expected to contain 1-D arrays plus user/candidate lists.
    Only compact numeric arrays are retained for global pCVR/value metrics;
    ranking is completed whenever a user boundary is encountered.
    """
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    current_user: str | None = None
    group: list[dict[str, float | str]] = []
    ndcg = defaultdict(float); recall = defaultdict(float); hit = defaultdict(float); users_with_conversion = 0
    comparison = {name: _new_topk_summary(cutoffs) for name in ("coarse_score", "pcvr", "expected_value")}

    def flush_group() -> None:
        nonlocal group, users_with_conversion
        if not group:
            return
        converted = [row for row in group if int(row["label"]) == 1]
        if converted:
            users_with_conversion += 1
            for cutoff in cutoffs:
                for score_name, score_key in (("coarse_score", "coarse_score"), ("pcvr", "pcvr"), ("expected_value", "expected_value")):
                    ordered = sorted(group, key=lambda row: (-float(row[score_key]), str(row["candidate_id"])))[:cutoff]
                    observed = np.asarray([float(row["label"]) for row in ordered])
                    ideal = np.ones(min(cutoff, len(converted)), dtype=float)
                    dcg = float(np.sum(observed / np.log2(np.arange(2, len(observed) + 2))))
                    ideal_dcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
                    if score_name == "expected_value":
                        ndcg[cutoff] += dcg / ideal_dcg if ideal_dcg else 0.0
                        recovered = float(observed.sum())
                        recall[cutoff] += recovered / len(converted)
                        hit[cutoff] += float(recovered > 0)
                    _accumulate_topk(comparison[score_name][cutoff], ordered)
        group = []

    for batch in rows:
        batch_labels = np.asarray(batch["label"], dtype=np.float32)
        batch_probability = np.asarray(batch["pcvr"], dtype=np.float32)
        batch_values = np.asarray(batch["predicted_value"], dtype=np.float32)
        batch_masks = np.asarray(batch["value_mask"], dtype=np.float32)
        labels.append(batch_labels); probabilities.append(batch_probability); predicted_values.append(batch_values); values.append(np.asarray(batch["observed_value"], dtype=np.float32)); masks.append(batch_masks)
        for index, user in enumerate(batch["user_id"]):
            user = str(user)
            if current_user is not None and user != current_user:
                flush_group()
            current_user = user
            probability = float(batch_probability[index]); predicted_value = float(batch_values[index])
            group.append({"candidate_id": str(batch["candidate_ad_id"][index]), "label": float(batch_labels[index]), "observed_value": float(batch["observed_value"][index]) if batch_masks[index] else 0.0, "coarse_score": float(batch["coarse_score"][index]), "pcvr": probability, "expected_value": probability * predicted_value})
    flush_group()
    y = np.concatenate(labels) if labels else np.empty(0, dtype=np.float32)
    p = np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)
    predicted = np.concatenate(predicted_values) if predicted_values else np.empty(0, dtype=np.float32)
    observed = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    value_mask = np.concatenate(masks).astype(bool) if masks else np.empty(0, dtype=bool)
    cvr = {
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "brier_score": float(np.mean(np.square(p.astype(np.float64) - y.astype(np.float64)))) if len(y) else None,
        "positive_rate": float(np.mean(y)) if len(y) else None,
    }
    value_metrics = _value_metrics(observed[value_mask], predicted[value_mask])
    result = {"pcvr": cvr, "pcvr_distribution": {"conversion_label_1": _distribution(p[y == 1]), "conversion_label_0": _distribution(p[y == 0])}, "calibration": _calibration(y, p), "value": value_metrics, "ranking": {f"ndcg@{k}": _divide(ndcg[k], users_with_conversion) for k in cutoffs} | {f"recall@{k}": _divide(recall[k], users_with_conversion) for k in cutoffs} | {f"hit_rate@{k}": _divide(hit[k], users_with_conversion) for k in cutoffs}, "expected_value_comparison": _finalize_comparison(comparison), "users_with_conversion": users_with_conversion, "rows": int(len(y))}
    return result


def value_regression_metrics(observed_values: np.ndarray, predicted_values: np.ndarray, masks: np.ndarray) -> dict[str, float | None]:
    mask = np.asarray(masks, dtype=bool)
    return _value_metrics(np.asarray(observed_values)[mask], np.asarray(predicted_values)[mask])


def _value_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if not len(observed) or not len(predicted) or np.isnan(predicted).all():
        return {"mae": None, "rmse": None, "log_value_mae": None, "positive_value_rows": int(len(observed))}
    return {"mae": float(mean_absolute_error(observed, predicted)), "rmse": float(math.sqrt(mean_squared_error(observed, predicted))), "log_value_mae": float(mean_absolute_error(np.log1p(np.maximum(observed, 0)), np.log1p(np.maximum(predicted, 0)))), "positive_value_rows": int(len(observed))}


def _new_topk_summary(cutoffs: tuple[int, ...]) -> dict[int, dict[str, float]]:
    return {cutoff: {"observed_conversions": 0.0, "observed_conversion_value": 0.0, "selected_rows": 0.0} for cutoff in cutoffs}


def _accumulate_topk(summary: dict[str, float], rows: list[dict[str, float | str]]) -> None:
    labels = np.asarray([float(row["label"]) for row in rows])
    observed_values = np.asarray([float(row["observed_value"]) for row in rows])
    summary["observed_conversions"] += float(labels.sum())
    summary["observed_conversion_value"] += float(observed_values.sum())
    summary["selected_rows"] += len(rows)


def _finalize_comparison(comparison: dict[str, dict[int, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for strategy, by_cutoff in comparison.items():
        result[strategy] = {}
        for cutoff, values in by_cutoff.items():
            count = values["observed_conversions"]
            result[strategy][f"topk@{cutoff}"] = {**values, "average_conversion_value": values["observed_conversion_value"] / count if count else 0.0, "cumulative_conversion_value": values["observed_conversion_value"]}
    return result


def _divide(value: float, divisor: int) -> float:
    return float(value / divisor) if divisor else 0.0


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"rows": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {"rows": int(len(values)), "min": float(values.min()), "p50": float(np.quantile(values, .50)), "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)), "max": float(values.max())}


def _calibration(labels: np.ndarray, probabilities: np.ndarray, bins: int = 20) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.float32); probabilities = np.asarray(probabilities, dtype=np.float32)
    output: list[dict[str, float | int | None]] = []; ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (probabilities >= lower) & ((probabilities < upper) if index < bins - 1 else (probabilities <= upper))
        rows = int(mask.sum())
        if not rows:
            output.append({"bin": index, "lower": lower, "upper": upper, "rows": 0, "mean_predicted_pcvr": None, "actual_conversion_rate": None, "calibration_error": None})
            continue
        predicted = float(probabilities[mask].mean()); actual = float(labels[mask].mean()); error = abs(predicted - actual)
        ece += rows / len(labels) * error
        output.append({"bin": index, "lower": lower, "upper": upper, "rows": rows, "mean_predicted_pcvr": predicted, "actual_conversion_rate": actual, "calibration_error": error})
    return {"bin_count": bins, "ece": float(ece), "bins": output}
