"""Value-score contracts; Attribution and Search Conversion stay separate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attribution_scores(frame: pd.DataFrame, *, calibrated: bool) -> pd.DataFrame:
    """Attach Attribution-only CTR, CTCVR and derived serving-CVR scores."""
    pctr_key = "calibrated_pctr" if calibrated else "raw_pctr"
    pctcvr_key = "calibrated_pctcvr" if calibrated else "raw_pctcvr"
    required = {pctr_key, pctcvr_key, "click", "conversion"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Attribution score input is missing {sorted(missing)}")
    output = frame.copy()
    pctr = np.clip(pd.to_numeric(output[pctr_key], errors="raise").to_numpy(float), 1e-7, 1.0 - 1e-7)
    pctcvr = np.clip(pd.to_numeric(output[pctcvr_key], errors="raise").to_numpy(float), 1e-7, 1.0 - 1e-7)
    output["score_ctr"] = pctr
    output["score_ctcvr"] = pctcvr
    output["score_pcvr"] = np.clip(pctcvr / pctr, 0.0, 1.0)
    output["probability_variant"] = "calibrated" if calibrated else "raw"
    return output


def search_value_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach Search Conversion's clicked-interaction value score only."""
    required = {"pCVR_clicked", "predicted_conditional_value"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Search Conversion score input is missing {sorted(missing)}")
    output = frame.copy()
    pcvr = np.clip(pd.to_numeric(output["pCVR_clicked"], errors="raise").to_numpy(float), 0.0, 1.0)
    value = np.maximum(pd.to_numeric(output["predicted_conditional_value"], errors="raise").to_numpy(float), 0.0)
    output["score_value_per_click"] = pcvr * value
    return output
