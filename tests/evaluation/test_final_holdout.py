"""Final holdout guards: sanity is path/schema-only and never opens Future-B."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from search_ads_system.evaluation import final_holdout
from search_ads_system.evaluation.final_holdout import (
    FinalHoldoutConfig, _delta, marker_path, run_final_holdout,
)


def _raw(tmp_path: Path) -> tuple[dict, Path]:
    attribution = tmp_path / "attribution"; calibration = tmp_path / "calibration"; metrics = tmp_path / "metrics"; fine = tmp_path / "fine"; business = tmp_path / "business"
    for path in (attribution / "split" / "future_a", attribution / "split" / "future_b", calibration, metrics, fine / "models", fine / "metrics", business / "metrics"):
        path.mkdir(parents=True, exist_ok=True)
    # An invalid Future-B sentinel proves sanity never parses it.
    (attribution / "split" / "future_b" / "DO_NOT_READ.txt").write_text("holdout")
    (calibration / "esmm.pt").write_text("checkpoint")
    (fine / "models" / "dcnv2.pt").write_text("checkpoint")
    (metrics / "calibration_metrics.json").write_text(json.dumps({"selected_calibrator": {"ctr": "raw", "ctcvr": "raw"}}))
    (metrics / "esmm_metrics.json").write_text(json.dumps({"metrics": {"esmm": {"ctr": {"pr_auc": .2, "logloss": .5}, "ctcvr": {"pr_auc": .1, "logloss": .2}}}}))
    (fine / "metrics" / "fine_rank_metrics.json").write_text(json.dumps({"models": {"dcnv2": {"future_a": {"cvr": {"pr_auc": .3, "roc_auc": .4}, "value": {"mae": 1., "rmse": 2.}}}}}))
    (business / "metrics" / "auction_metrics.json").write_text(json.dumps({"attribution": {}, "search_conversion": {"available": True}}))
    raw = {"attribution_preprocessing": {"temporal_output_dir": str(attribution)}, "attribution_esmm": {"enabled": True, "metrics_dir": str(metrics), "hash_buckets": {"user_id": 17, "campaign_id": 17, "categories": 17}}, "attribution_calibration": {"checkpoint_path": str(calibration / "esmm.pt"), "output_dir": str(calibration), "metrics_dir": str(metrics)}, "temporal": {"output_dir": str(tmp_path / "search")}, "fine_rank_multitask": {"model_dir": str(fine / "models"), "metrics_dir": str(fine / "metrics"), "predictions": {"checkpoint": str(fine / "models" / "dcnv2.pt")}}, "business_simulation": {"output_dir": str(business)}, "final_evaluation": {"output_dir": str(tmp_path / "final"), "moderate_degradation_threshold": .05, "large_degradation_threshold": .15}}
    config_path = tmp_path / "config.yaml"; config_path.write_text("{}")
    return raw, config_path


def test_sanity_checks_frozen_artifacts_without_reading_future_b(tmp_path: Path) -> None:
    raw, config_path = _raw(tmp_path)
    result = run_final_holdout(raw, config_path, stage="sanity")
    assert result["future_b_read"] is False
    assert not marker_path(FinalHoldoutConfig(tmp_path / "final", .05, .15)).exists()


def test_frozen_checkpoint_is_required_without_retraining(tmp_path: Path) -> None:
    raw, config_path = _raw(tmp_path)
    (tmp_path / "calibration" / "esmm.pt").unlink()
    with pytest.raises(FileNotFoundError, match="never retrains or refits"):
        run_final_holdout(raw, config_path, stage="sanity")


def test_delta_and_marker_contract(tmp_path: Path) -> None:
    config = FinalHoldoutConfig(tmp_path, .05, .15)
    assert _delta("search_value_mae", 93.5175, 89.1515, config)["interpretation"] == "improved"
    assert _delta("search_value_rmse", 234.9550, 226.5022, config)["interpretation"] == "improved"
    assert _delta("search_top_10pct_value_per_click_lift", 8.4726, 9.4294, config)["interpretation"] == "improved"
    assert _delta("attribution_ctr_pr_auc", .5, .45, config)["interpretation"] == "moderate degradation"
    assert _delta("attribution_ctr_logloss", .5, .55, config)["interpretation"] == "moderate degradation"


def test_render_reuses_only_existing_final_metrics_without_future_b(tmp_path: Path) -> None:
    raw, config_path = _raw(tmp_path)
    config = FinalHoldoutConfig(tmp_path / "final", .05, .15)
    config.output_dir.mkdir()
    marker_path(config).write_text("{}")
    (config.output_dir / "final_holdout_metrics.json").write_text(json.dumps({
        "future_a_vs_future_b": {"rows": [
            {"metric": "search_value_mae", "future_a": 93.5175, "future_b": 89.1515},
            {"metric": "search_top_10pct_value_per_click_lift", "future_a": 8.4726, "future_b": 9.4294},
        ]},
        "attribution_final_holdout": {"rows": 1},
        "search_conversion_final_holdout": {"rows": 1},
        "limitations": [],
        "final_conclusion": "final",
    }))
    result = run_final_holdout(raw, config_path, stage="render")
    rendered = json.loads(Path(result["report_path"]).read_text())
    assert result["future_b_read"] is False
    assert rendered["future_b_reread"] is False and rendered["reused_existing_final_metrics"] is True
    assert [row["interpretation"] for row in rendered["future_a_vs_future_b"]["rows"]] == ["improved", "improved"]
    assert "| metric | direction |" in Path(result["markdown_path"]).read_text()


def test_all_is_the_only_stage_that_opens_future_b_and_writes_final_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw, config_path = _raw(tmp_path)
    calls: list[str] = []
    attribution = {"rows": 2, "metrics": {"esmm": {"ctr": {}, "ctcvr": {}}, "calibration": {"ctr": {"calibrated": {}}}}, "business": {"policies": {}}}
    search = {"rows": 2, "metrics": {"cvr": {}, "conditional_value": {}}, "business": {"policies": {}}}
    monkeypatch.setattr(final_holdout, "_evaluate_attribution_future_b", lambda *_: calls.append("attribution") or attribution)
    monkeypatch.setattr(final_holdout, "_evaluate_search_future_b", lambda *_: calls.append("search") or search)
    monkeypatch.setattr(final_holdout, "_future_a_baselines", lambda *_: {})
    monkeypatch.setattr(final_holdout, "_drift", lambda *_: {"rows": [], "thresholds": {}})
    result = run_final_holdout(raw, config_path, stage="all")
    marker = marker_path(FinalHoldoutConfig(tmp_path / "final", .05, .15))
    assert calls == ["attribution", "search"]
    marker_data = json.loads(marker.read_text())
    assert marker.is_file() and marker_data["future_b_opened_for"] == "final evaluation only"
    assert marker_data["checkpoints"] and marker_data["calibration_artifacts"]
    report = json.loads(Path(result["report_path"]).read_text())
    assert set(report) >= {"frozen_declaration", "data_contracts", "attribution_final_holdout", "search_conversion_final_holdout", "future_a_vs_future_b", "limitations", "final_conclusion"}
    with pytest.raises(RuntimeError, match="already been opened"):
        run_final_holdout(raw, config_path, stage="all")
