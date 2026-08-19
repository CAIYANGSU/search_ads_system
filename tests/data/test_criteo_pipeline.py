"""Integration coverage for the streaming Criteo data pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from search_ads_system.data.conversion import convert_criteo_to_unified
from search_ads_system.data.eda import run_eda
from search_ads_system.data.features import build_features
from search_ads_system.data.interfaces import DelimitedDatasetConfig, EdaConfig, FeatureConfig
from search_ads_system.data.storage import iter_csv_parts


SOURCE_COLUMNS = (
    "Sale", "SalesAmountInEuro", "time_delay_for_conversion", "click_timestamp", "nb_clicks_1week",
    "product_price", "product_age_group", "device_type", "audience_id", "product_gender", "product_brand",
    "product_category_1", "product_category_2", "product_category_3", "product_category_4", "product_category_5",
    "product_category_6", "product_category_7", "product_country", "product_id", "product_title", "partner_id", "user_id",
)


def test_conversion_eda_and_features_streaming(tmp_path: Path) -> None:
    raw_file = tmp_path / "CriteoSearchData.tsv"
    row_one = [
        "0", "-1", "-1", "1598891820", "-1", "0.0", "-1", "mobile", "-1", "-1", "-1",
        "-1", "-1", "-1", "-1", "-1", "-1", "-1", "FR", "product-a", "title-a", "partner-a", "user-a",
    ]
    row_two = [
        "1", "12.5", "60", "1598895420", "3", "20.0", "adult", "desktop", "audience-a", "female", "brand-a",
        "cat-1", "cat-2", "-1", "-1", "-1", "-1", "-1", "DE", "product-b", "title-b", "partner-b", "user-b",
    ]
    raw_file.write_text("\t".join(row_one) + "\n" + "\t".join(row_two) + "\n", encoding="utf-8")
    dataset = DelimitedDatasetConfig(
        path=raw_file,
        delimiter="\t",
        has_header=False,
        encoding="utf-8",
        chunk_size=1,
        column_names=SOURCE_COLUMNS,
        label_columns=("Sale",),
        missing_value_tokens=("", "-1"),
        missing_value_tokens_by_column={"click_timestamp": ("0",)},
    )
    unified_path = tmp_path / "outputs" / "processed" / "unified"
    converted = convert_criteo_to_unified(dataset, unified_path)
    assert converted.rows_written == 2
    assert converted.parts_written == 2

    unified = pd.concat(iter_csv_parts(unified_path, chunk_size=10), ignore_index=True)
    assert unified["event_id"].tolist() == ["criteo-000000000000", "criteo-000000000001"]
    assert pd.isna(unified.loc[0, "conversion_value_eur"])
    assert unified.loc[1, "conversion_label"] == 1

    summary_path = tmp_path / "outputs" / "eda" / "summary.json"
    category_path = tmp_path / "outputs" / "eda" / "top_categories.csv"
    summary = run_eda(
        unified_path,
        summary_path,
        category_path,
        EdaConfig(categorical_columns=("device_type", "product_country"), top_k=5),
        chunk_size=1,
    )
    assert summary["conversion_rate"] == 0.5
    assert json.loads(summary_path.read_text(encoding="utf-8"))["row_count"] == 2
    assert category_path.is_file()

    feature_path = tmp_path / "outputs" / "features" / "data"
    metadata_path = tmp_path / "outputs" / "features" / "metadata.json"
    result = build_features(
        unified_path,
        feature_path,
        metadata_path,
        FeatureConfig(categorical_columns=("device_type", "product_country"), missing_category_token="__MISSING__"),
        chunk_size=1,
    )
    assert result.rows_written == 2
    features = pd.concat(iter_csv_parts(feature_path, chunk_size=10), ignore_index=True)
    assert {"click_hour_utc", "log1p_product_price", "cat_device_type"}.issubset(features.columns)
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["feature_version"] == "1.0"
