"""Chunked, schema-driven loading for delimited datasets."""

from __future__ import annotations

from collections.abc import Iterator
import logging

import pandas as pd

from search_ads_system.data.interfaces import DelimitedDatasetConfig

LOGGER = logging.getLogger(__name__)


def iter_delimited_chunks(config: DelimitedDatasetConfig) -> Iterator[pd.DataFrame]:
    """Yield raw rows in bounded chunks, applying only the declared column contract.

    Values remain strings deliberately: inference and missing-value definitions are
    reported separately by schema inspection instead of being silently coerced.
    """

    reader = pd.read_csv(
        config.path,
        sep=config.delimiter,
        header=0 if config.has_header else None,
        names=list(config.column_names) if not config.has_header and config.column_names else None,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        encoding=config.encoding,
        chunksize=config.chunk_size,
        on_bad_lines="error",
    )
    for chunk in reader:
        if not config.has_header and not config.column_names:
            chunk.columns = [f"column_{position}" for position in range(chunk.shape[1])]
        LOGGER.info("Read raw chunk with %s rows from %s", len(chunk), config.path)
        yield chunk
