"""Chunk-file storage helpers for large pipeline outputs."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def prepare_output_directory(directory: Path, overwrite: bool = False) -> None:
    """Create an empty output directory or reject accidental artifact mixing."""

    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {directory}. Use --overwrite to replace generated part files."
            )
        for part in directory.glob("part-*.csv"):
            part.unlink()
    directory.mkdir(parents=True, exist_ok=True)


def write_csv_part(frame: pd.DataFrame, directory: Path, part_number: int) -> Path:
    """Write one data chunk atomically as a headered CSV part."""

    target = directory / f"part-{part_number:05d}.csv"
    temporary = target.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)
    LOGGER.info("Wrote %s rows to %s", len(frame), target)
    return target


def iter_csv_parts(directory: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Stream all CSV part files in deterministic order."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Expected output data directory does not exist: {directory}")
    parts = sorted(directory.glob("part-*.csv"))
    if not parts:
        raise FileNotFoundError(f"No CSV parts found in: {directory}")
    for part in parts:
        LOGGER.info("Reading %s", part)
        yield from pd.read_csv(part, chunksize=chunk_size, low_memory=False)
