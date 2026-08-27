"""Regression coverage for the Attribution-only impression data contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from search_ads_system.data.attribution import (
    ATTRIBUTION_COLUMNS,
    build_attribution_audit,
    build_attribution_impressions,
    split_attribution_temporally,
)
from search_ads_system.data.interfaces import AttributionPreprocessConfig, DelimitedDatasetConfig
from search_ads_system.data.storage import iter_csv_parts


RAW_COLUMNS = (
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp", "conversion_id", "attribution",
    "click", "click_pos", "click_nb", "cost", "cpo", "time_since_last_click", "cat1", "cat2", "cat3",
    "cat4", "cat5", "cat6", "cat7", "cat8", "cat9",
)


def _write_raw(path: Path, timestamps: list[int], *, out_of_order: bool = False) -> None:
    rows: list[list[str]] = []
    outcomes = [("0", "0"), ("1", "0"), ("1", "1"), ("0", "1")]
    for index, timestamp in enumerate(timestamps):
        click, conversion = outcomes[index % len(outcomes)]
        has_conversion = conversion == "1"
        rows.append([
            str(timestamp), f"user-{index % 5}", f"campaign-{index % 3}", conversion,
            str(timestamp + 10) if has_conversion else "-1", f"conversion-{index}" if has_conversion else "-1",
            "1" if has_conversion else "0", click, "0" if click == "1" else "-1", "1" if click == "1" else "-1",
            "0.001", "0.05" if has_conversion else "-1", str(index),
            *[f"cat-{category}-{index % 2}" for category in range(1, 10)],
        ])
    if out_of_order:
        rows[4], rows[5] = rows[5], rows[4]
    path.write_text("\t".join(RAW_COLUMNS) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")


def _config(tmp_path: Path, raw_path: Path) -> AttributionPreprocessConfig:
    return AttributionPreprocessConfig(
        dataset=DelimitedDatasetConfig(
            path=raw_path,
            delimiter="\t",
            has_header=True,
            encoding="utf-8",
            chunk_size=3,
            column_names=RAW_COLUMNS,
            label_columns=("click", "conversion"),
            missing_value_tokens=("", "-1"),
            missing_value_tokens_by_column={},
        ),
        processed_data=tmp_path / "outputs" / "processed" / "criteo_attribution_impression",
        temporal_output_dir=tmp_path / "outputs" / "attribution_temporal",
        audit_path=tmp_path / "outputs" / "metrics" / "attribution_impression_data_audit.json",
        build_metadata_path=tmp_path / "outputs" / "metrics" / "attribution_impression_build_metadata.json",
        past_ratio=0.5,
        future_a_ratio=0.5,
    )


def test_attribution_streaming_labels_temporal_contract_and_audit(tmp_path: Path) -> None:
    raw = tmp_path / "attribution.tsv"
    _write_raw(raw, list(range(12)))
    config = _config(tmp_path, raw)

    built = build_attribution_impressions(config)
    assert built.rows_written == 12
    assert built.parts_written == 4
    frame = pd.concat(iter_csv_parts(config.processed_data, chunk_size=20), ignore_index=True)
    assert tuple(frame.columns) == ATTRIBUTION_COLUMNS
    assert (frame["click_and_conversion"] == ((frame["click"] == 1) & (frame["conversion"] == 1)).astype(int)).all()
    post_view = frame.loc[(frame["click"] == 0) & (frame["conversion"] == 1)]
    assert len(post_view) == 3
    assert post_view["click_and_conversion"].eq(0).all()
    assert pd.isna(frame.loc[0, "conversion_timestamp"])

    split = split_attribution_temporally(config)
    assert split.boundaries == {"future_a_start": 6, "future_b_start": 9}
    split_frames = {
        name: pd.concat(iter_csv_parts(path, chunk_size=20), ignore_index=True)
        for name, path in split.split_directories.items()
    }
    assert max(split_frames["past"]["timestamp"]) < min(split_frames["future_a"]["timestamp"])
    assert min(split_frames["future_a"]["timestamp"]) < min(split_frames["future_b"]["timestamp"])
    identifiers = [set(value["event_id"]) for value in split_frames.values()]
    assert not identifiers[0] & identifiers[1]
    assert not identifiers[0] & identifiers[2]
    assert not identifiers[1] & identifiers[2]
    assert sum(len(value) for value in split_frames.values()) == len(frame)

    audit = build_attribution_audit(config)
    assert audit["raw_rows"] == 12
    assert audit["processed_rows"] == 12
    assert audit["label_diagnostics"]["label_cross_table"]["click_0_conversion_1"] == 3
    assert audit["search_conversion_join"] is False
    assert "conversion_id" not in audit["feature_eligibility"]["A_safe_impression_time_features"]
    assert "attribution" not in audit["feature_eligibility"]["A_safe_impression_time_features"]
    assert "cpo" in audit["feature_eligibility"]["D_label_derived_or_leakage_risk"]
    persisted = json.loads(config.audit_path.read_text(encoding="utf-8"))
    assert persisted["temporal_split"]["splits"]["future_b"]["rows"] == len(split_frames["future_b"])
    assert config.audit_path.with_suffix(".md").is_file()


def test_attribution_rerun_does_not_append_dirty_parts(tmp_path: Path) -> None:
    raw = tmp_path / "attribution.tsv"
    _write_raw(raw, list(range(12)))
    config = _config(tmp_path, raw)
    build_attribution_impressions(config)
    with pytest.raises(FileExistsError):
        build_attribution_impressions(config)
    rebuilt = build_attribution_impressions(config, overwrite=True)
    assert rebuilt.rows_written == 12
    parts = sorted(config.processed_data.glob("part-*.csv"))
    assert len(parts) == 4
    assert sum(len(chunk) for chunk in iter_csv_parts(config.processed_data, chunk_size=2)) == 12


def test_attribution_temporal_order_guard_fails_on_unsorted_input(tmp_path: Path) -> None:
    raw = tmp_path / "attribution.tsv"
    _write_raw(raw, list(range(12)), out_of_order=True)
    config = _config(tmp_path, raw)
    build_attribution_impressions(config)
    with pytest.raises(ValueError, match="timestamp order"):
        split_attribution_temporally(config)
