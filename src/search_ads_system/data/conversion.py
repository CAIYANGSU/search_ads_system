"""Streaming conversion from raw Criteo Search Conversion rows to canonical data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from search_ads_system.data.dataset import iter_delimited_chunks
from search_ads_system.data.interfaces import DelimitedDatasetConfig
from search_ads_system.data.storage import prepare_output_directory, write_csv_part
from search_ads_system.data.unified_schema import normalize_criteo_chunk

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversionResult:
    """Summary of a completed conversion run."""

    rows_written: int
    parts_written: int
    output_directory: Path


def convert_criteo_to_unified(
    dataset_config: DelimitedDatasetConfig,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> ConversionResult:
    """Convert raw Criteo data in bounded chunks and write canonical CSV parts."""

    prepare_output_directory(output_directory, overwrite=overwrite)
    row_offset = 0
    parts_written = 0
    for part_number, raw_chunk in enumerate(iter_delimited_chunks(dataset_config)):
        unified_chunk = normalize_criteo_chunk(raw_chunk, row_offset, dataset_config)
        write_csv_part(unified_chunk, output_directory, part_number)
        row_offset += len(unified_chunk)
        parts_written += 1
        LOGGER.info("Converted %s rows so far", row_offset)
    return ConversionResult(row_offset, parts_written, output_directory)
