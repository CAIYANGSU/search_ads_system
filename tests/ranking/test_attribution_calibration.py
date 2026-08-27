"""Regression tests for strict-time Attribution ESMM calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from search_ads_system.ranking.attribution_calibration import (
    UnavailableCalibrator,
    calibration_metrics,
    fit_isotonic,
    fit_platt,
    reliability_bins,
    select_calibrator,
    serving_consistent_probabilities,
)
from search_ads_system.ranking.attribution_calibration_pipeline import _future_a_boundary


def test_reliability_ece_and_brier_are_exact() -> None:
    labels = np.asarray([0, 1])
    predictions = np.asarray([0.25, 0.75])
    bins = reliability_bins(labels, predictions, 2)
    metrics = calibration_metrics(labels, predictions, 2)
    assert sum(bin_["count"] for bin_ in bins) == 2
    assert metrics["brier_score"] == 0.0625
    assert metrics["ece"] == 0.25


def test_platt_and_isotonic_fit_only_use_fit_window() -> None:
    fit_labels = np.asarray([0, 0, 1, 1])
    fit_predictions = np.asarray([0.1, 0.2, 0.8, 0.9])
    eval_predictions = np.asarray([0.15, 0.85])
    platt = fit_platt(fit_labels, fit_predictions)
    isotonic = fit_isotonic(fit_labels, fit_predictions)
    assert not isinstance(platt, UnavailableCalibrator)
    assert not isinstance(isotonic, UnavailableCalibrator)
    assert np.all((platt.predict(eval_predictions) >= 0) & (platt.predict(eval_predictions) <= 1))
    assert np.all((isotonic.predict(eval_predictions) >= 0) & (isotonic.predict(eval_predictions) <= 1))


def test_degenerate_calibrators_and_raw_selection_are_safe() -> None:
    assert isinstance(fit_platt(np.asarray([1, 1]), np.asarray([0.3, 0.7])), UnavailableCalibrator)
    assert isinstance(fit_isotonic(np.asarray([0, 1]), np.asarray([0.5, 0.5])), UnavailableCalibrator)
    selected, reason = select_calibrator({
        "raw": {"ece": 0.01, "logloss": 0.2, "brier_score": 0.04},
        "platt": {"ece": 0.02, "logloss": 0.19, "brier_score": 0.03},
        "isotonic": {"available": False},
    })
    assert selected == "raw"
    assert "lowest ECE" in reason["reason"]


def test_serving_consistency_uses_ratio_guard_and_reports_clipping() -> None:
    serving_pcvr, diagnostics = serving_consistent_probabilities(
        np.asarray([0.2, 0.5]), np.asarray([0.1, 0.8])
    )
    assert np.allclose(serving_pcvr, np.asarray([0.5, 1.0]))
    assert np.isclose(diagnostics["fraction_requiring_clipping"], 0.5)
    assert np.isclose(diagnostics["max_abs_error"], 0.3)


def test_timestamp_calibration_boundary_keeps_ties_together(tmp_path: Path) -> None:
    source = tmp_path / "future_a"
    source.mkdir()
    (source / "part-00000.csv").write_text("timestamp\n1\n1\n2\n2\n3\n3\n", encoding="utf-8")
    boundary = _future_a_boundary(source, chunk_size=2, fit_ratio=0.5, max_rows=None)
    assert boundary == 3
    timestamps = np.asarray([1, 1, 2, 2, 3, 3])
    assert timestamps[timestamps < boundary].max() < timestamps[timestamps >= boundary].min()
