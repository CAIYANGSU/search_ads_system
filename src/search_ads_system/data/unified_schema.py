"""Canonical click-conversion schema used by downstream pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from search_ads_system.data.interfaces import DelimitedDatasetConfig

UNIFIED_COLUMNS: tuple[str, ...] = (
    "event_id",
    "source_row_number",
    "click_timestamp",
    "conversion_label",
    "conversion_value_eur",
    "conversion_delay_seconds",
    "clicks_last_7d",
    "product_price",
    "product_age_group",
    "device_type",
    "audience_id",
    "product_gender",
    "product_brand",
    "product_category_1",
    "product_category_2",
    "product_category_3",
    "product_category_4",
    "product_category_5",
    "product_category_6",
    "product_category_7",
    "product_country",
    "product_id",
    "product_title",
    "partner_id",
    "user_id",
)

_SOURCE_TO_UNIFIED: Mapping[str, str] = {
    "Sale": "conversion_label",
    "SalesAmountInEuro": "conversion_value_eur",
    "time_delay_for_conversion": "conversion_delay_seconds",
    "click_timestamp": "click_timestamp",
    "nb_clicks_1week": "clicks_last_7d",
    "product_price": "product_price",
    "product_age_group": "product_age_group",
    "device_type": "device_type",
    "audience_id": "audience_id",
    "product_gender": "product_gender",
    "product_brand": "product_brand",
    "product_category_1": "product_category_1",
    "product_category_2": "product_category_2",
    "product_category_3": "product_category_3",
    "product_category_4": "product_category_4",
    "product_category_5": "product_category_5",
    "product_category_6": "product_category_6",
    "product_category_7": "product_category_7",
    "product_country": "product_country",
    "product_id": "product_id",
    "product_title": "product_title",
    "partner_id": "partner_id",
    "user_id": "user_id",
}
_NUMERIC_COLUMNS = {
    "click_timestamp",
    "conversion_label",
    "conversion_value_eur",
    "conversion_delay_seconds",
    "clicks_last_7d",
    "product_price",
}


def normalize_criteo_chunk(
    raw_chunk: pd.DataFrame,
    source_row_offset: int,
    config: DelimitedDatasetConfig,
) -> pd.DataFrame:
    """Convert one Criteo Search Conversion chunk into the canonical schema.

    Raw missing-value indicators are converted to nullable values. The Criteo file
    does not expose a click identifier, so ``event_id`` is a stable identifier based
    on the source row number and is reproducible when the source file is unchanged.
    """

    unknown_columns = set(_SOURCE_TO_UNIFIED) - set(raw_chunk.columns)
    if unknown_columns:
        raise ValueError(f"Criteo source is missing required columns: {sorted(unknown_columns)}")

    normalized = pd.DataFrame(index=raw_chunk.index)
    source_rows = pd.Series(
        range(source_row_offset, source_row_offset + len(raw_chunk)), index=raw_chunk.index, dtype="Int64"
    )
    normalized["source_row_number"] = source_rows
    normalized["event_id"] = "criteo-" + source_rows.astype("string").str.zfill(12)

    for source_name, unified_name in _SOURCE_TO_UNIFIED.items():
        values = _replace_missing_values(raw_chunk[source_name], source_name, config)
        if unified_name in _NUMERIC_COLUMNS:
            normalized[unified_name] = pd.to_numeric(values, errors="coerce")
        else:
            normalized[unified_name] = values.astype("string")

    _validate_labels(normalized["conversion_label"])
    normalized["conversion_label"] = normalized["conversion_label"].astype("Int8")
    normalized["click_timestamp"] = normalized["click_timestamp"].astype("Int64")
    normalized["clicks_last_7d"] = normalized["clicks_last_7d"].astype("Float64")
    for column in ("conversion_value_eur", "conversion_delay_seconds", "product_price"):
        normalized[column] = normalized[column].astype("Float64")
    return normalized.loc[:, UNIFIED_COLUMNS]


def _replace_missing_values(
    values: pd.Series,
    source_name: str,
    config: DelimitedDatasetConfig,
) -> pd.Series:
    tokens = set(config.missing_value_tokens)
    tokens.update(config.missing_value_tokens_by_column.get(source_name, ()))
    string_values = values.astype("string")
    return string_values.mask(string_values.isin(tokens), pd.NA)


def _validate_labels(labels: pd.Series) -> None:
    numeric_labels = pd.to_numeric(labels, errors="coerce")
    if numeric_labels.isna().any():
        raise ValueError("Criteo conversion label contains missing or non-numeric values")
    observed_labels = set(numeric_labels.astype(int).unique())
    if not observed_labels.issubset({0, 1}):
        raise ValueError(f"Criteo conversion label must be binary; observed {sorted(observed_labels)}")
