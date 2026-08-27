"""Strictly time-separated offline experiment helpers.

Temporal artifacts are deliberately isolated below ``outputs/temporal``.  The
split writer is streaming and only writes rows for users eligible in both time
windows; all recall training inputs therefore contain Past rows only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import sqlite3
import pickle
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_ads_system.common.config import resolve_path
from search_ads_system.data.storage import iter_csv_parts
from search_ads_system.ranking.coarse_rank import CoarseRankConfig, build_interaction_feature_index, preprocess_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalConfig:
    interaction_path: Path
    output_dir: Path
    timestamp_column: str = "click_timestamp"
    past_ratio: float = 0.8
    max_users: int = 100_000
    seed: int = 2026
    chunk_size: int = 200_000


def parse_temporal_config(raw: Mapping[str, Any], config_path: Path) -> TemporalConfig:
    paths, options = raw.get("paths"), raw.get("temporal", {})
    if not isinstance(paths, Mapping) or not isinstance(options, Mapping):
        raise ValueError("Configuration must define paths and temporal mappings")
    root = config_path.parent.resolve()
    split, users = options.get("split", {}), options.get("users", {})
    if not isinstance(split, Mapping) or not isinstance(users, Mapping):
        raise ValueError("temporal.split and temporal.users must be mappings")
    config = TemporalConfig(
        interaction_path=resolve_path(str(paths.get("unified_data")), root),
        output_dir=resolve_path(str(options.get("output_dir", "outputs/temporal")), root),
        timestamp_column=str(split.get("timestamp_column", "click_timestamp")),
        past_ratio=float(split.get("past_ratio", 0.8)),
        max_users=int(users.get("max_users", 100_000)),
        seed=int(users.get("seed", raw.get("project", {}).get("seed", 2026))),
        chunk_size=int(options.get("chunk_size", raw.get("coarse_rank", {}).get("chunk_size", 200_000))),
    )
    if not 0 < config.past_ratio < 1 or min(config.max_users, config.chunk_size) <= 0:
        raise ValueError("temporal past_ratio must be in (0,1); user/chunk limits must be positive")
    return config


def build_temporal_split(config: TemporalConfig) -> dict[str, Any]:
    """Create selected-user Past/Future parts and immutable split metadata.

    The timestamp quantile is exact over a compact int64 array, never a full
    interaction dataframe.  User eligibility is tracked in a temporary SQLite
    table, avoiding a multi-million-user Python set.
    """
    metadata_path = config.output_dir / "split" / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _split_matches(existing, config):
            LOGGER.info("Reusing temporal split metadata: %s", metadata_path)
            return existing
    timestamps = _collect_timestamps(config)
    if not len(timestamps):
        raise ValueError("No valid timestamps found for temporal split")
    threshold = int(np.quantile(timestamps, config.past_ratio, method="higher"))
    state_path = config.output_dir / "split" / ".user_state.sqlite"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists(): state_path.unlink()
    connection = sqlite3.connect(state_path)
    try:
        connection.execute("CREATE TABLE users (user_id TEXT PRIMARY KEY, past INTEGER NOT NULL DEFAULT 0, future INTEGER NOT NULL DEFAULT 0)")
        connection.execute("CREATE TABLE products (product_id TEXT PRIMARY KEY, past INTEGER NOT NULL DEFAULT 0, future INTEGER NOT NULL DEFAULT 0)")
        summary = _index_user_windows(config, threshold, connection)
        selected = _select_eligible_users(connection, config)
    finally:
        connection.close()
    _write_selected_parts(config, threshold, selected)
    state_path.unlink(missing_ok=True)
    metadata = {**summary, "full_time_min": int(timestamps.min()), "full_time_max": int(timestamps.max()), "split_timestamp": threshold,
                "past_ratio": config.past_ratio, "selected_users": len(selected), "eligible_users": summary.pop("eligible_users"),
                "selected_users_hash": _hash_ids(selected), "seed": config.seed, "timestamp_column": config.timestamp_column}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Temporal leakage checks passed: selected Past/Future artifacts written below %s", config.output_dir)
    return metadata


def evaluate_recall_file(candidate_path: Path, future_path: Path, *, cutoffs: Iterable[int] = (10,20,50,100), chunk_size: int = 200_000) -> dict[str, Any]:
    """Evaluate multi-positive Recall@K from a future-only label directory."""
    positives = _future_positives(future_path, chunk_size)
    header = set(pd.read_csv(candidate_path, nrows=0).columns)
    if "candidate_ad_id" not in header:
        raise ValueError(f"Recall candidates at {candidate_path} must include candidate_ad_id")
    source = _recall_source_name(candidate_path)
    if "user_id" not in header:
        if "rank" not in header:
            raise ValueError(f"Global recall candidates at {candidate_path} must include rank")
        LOGGER.info("Evaluating %s as global recall", source)
        global_candidates = [(str(ad), _recall_rank(rank)) for ad, rank in pd.read_csv(candidate_path, usecols=["candidate_ad_id", "rank"]).itertuples(index=False, name=None)]
        return _evaluate_candidate_mapping({user: global_candidates for user in positives}, positives, cutoffs)
    has_explicit_rank = "rank" in header
    LOGGER.info(
        "Evaluating %s with %s", source,
        "explicit rank" if has_explicit_rank else "derived per-user rank",
    )
    cutoffs=tuple(int(k) for k in cutoffs); recalls={k:0. for k in cutoffs}; users_hit={k:0 for k in cutoffs}
    unique_ads: set[str] = set(); rows = 0
    for user, candidates in _candidate_groups(candidate_path, chunk_size, has_explicit_rank=has_explicit_rank):
        target = positives.get(user)
        if not target: continue
        rows += len(candidates); unique_ads.update(ad for ad, _ in candidates)
        ranks = {ad: rank for ad, rank in candidates}
        for cutoff in cutoffs:
            matched = sum(ad in ranks and ranks[ad] <= cutoff for ad in target)
            recalls[cutoff] += matched / len(target)
            users_hit[cutoff] += int(matched > 0)
    users = len(positives)
    return {"users_evaluated": users, "average_future_positives_per_user": _divide(sum(map(len, positives.values())), users), "candidate_coverage": _divide(rows, users), "unique_recalled_ads": len(unique_ads), "users_with_hit": users_hit,
            "metrics": {f"recall@{k}": _divide(recalls[k], users) for k in cutoffs} | {f"hit_rate@{k}": _divide(users_hit[k], users) for k in cutoffs}}


def diagnose_temporal_recall_sources(paths: Mapping[str, Path], future_path: Path, *, chunk_size: int, cutoff: int = 100) -> dict[str, Any]:
    """Compare recall routes by Future hit coverage, overlap and incrementality."""
    positives = _future_positives(future_path, chunk_size); target_pairs = {(user, product) for user, products in positives.items() for product in products}
    source_hits: dict[str, set[tuple[str, str]]] = {}; source_rows: dict[str, int] = {}; source_users: dict[str, set[str]] = {}; source_items: dict[str, set[str]] = {}
    for name, path in paths.items():
        header = set(pd.read_csv(path, nrows=0).columns); rows = 0; users: set[str] = set(); items: set[str] = set(); hits: set[tuple[str, str]] = set()
        if "user_id" not in header:
            candidates: list[str] = []
            for chunk in pd.read_csv(path, usecols=["candidate_ad_id", "rank"], chunksize=chunk_size, low_memory=False):
                for product, rank in chunk.itertuples(index=False, name=None):
                    if _recall_rank(rank) <= cutoff and (candidate := _id(product)) is not None:
                        candidates.append(candidate); items.add(candidate); rows += 1
            candidate_set = set(candidates)
            hits = {(user, product) for user, product in target_pairs if product in candidate_set}
            users = set(positives)
        else:
            current_user: str | None = None; position = 0
            columns = ["user_id", "candidate_ad_id"] + (["rank"] if "rank" in header else [])
            for chunk in pd.read_csv(path, usecols=columns, chunksize=chunk_size, low_memory=False):
                for row in chunk.itertuples(index=False, name=None):
                    user, product = _id(row[0]), _id(row[1])
                    if user is None or product is None:
                        continue
                    if user != current_user:
                        current_user = user; position = 0
                    position += 1; rank = _recall_rank(row[2]) if "rank" in header else position
                    if rank <= cutoff:
                        rows += 1; users.add(user); items.add(product)
                        if (user, product) in target_pairs: hits.add((user, product))
        source_hits[name] = hits; source_rows[name] = rows; source_users[name] = users; source_items[name] = items
    sources: dict[str, Any] = {}
    for name, hits in source_hits.items():
        sources[name] = {"candidate_rows_at_cutoff": source_rows[name], "candidate_users_at_cutoff": len(source_users[name]), "candidate_user_coverage": _divide(len(source_users[name] & set(positives)), len(positives)), "unique_items": len(source_items[name]), "hit_positive_pairs": len(hits), "positive_pair_coverage": _divide(len(hits), len(target_pairs)), "hit_users": len({user for user, _ in hits}), "hit_rate": _divide(len({user for user, _ in hits}), len(positives))}
    overlap = {f"{left}__{right}": {"shared_hit_positive_pairs": len(source_hits[left] & source_hits[right]), "jaccard": _divide(len(source_hits[left] & source_hits[right]), len(source_hits[left] | source_hits[right]))} for index, left in enumerate(source_hits) for right in list(source_hits)[index + 1:]}
    union: set[tuple[str, str]] = set(); incremental: dict[str, Any] = {}
    for name in ("itemcf", "two_tower", "popularity", "fused"):
        if name not in source_hits: continue
        added = source_hits[name] - union; union |= source_hits[name]
        incremental[name] = {"incremental_hit_positive_pairs": len(added), "incremental_positive_pair_coverage": _divide(len(added), len(target_pairs)), "union_positive_pair_coverage": _divide(len(union), len(target_pairs))}
    popularity = source_hits.get("popularity", set()); fused = source_hits.get("fused", set())
    return {"cutoff": cutoff, "future_positive_users": len(positives), "future_positive_pairs": len(target_pairs), "sources": sources, "hit_overlap": overlap, "incremental_recall": incremental, "rrf_popularity_hit_retention": _divide(len(popularity & fused), len(popularity)), "interpretation": "A low RRF popularity-hit retention means RRF/top-K fusion displaced popularity hits; tune source weights/top_k only after this diagnostic, not from aggregate recall alone."}


def diagnose_two_tower_cold_start(past_path: Path, future_path: Path, *, chunk_size: int) -> dict[str, Any]:
    """ID-only Two Tower cannot represent unseen IDs except through fallback behavior."""
    past_users: set[str] = set(); past_products: set[str] = set()
    for chunk in iter_csv_parts(past_path, chunk_size):
        past_users.update(value for value in (_id(item) for item in chunk["user_id"]) if value is not None); past_products.update(value for value in (_id(item) for item in chunk["product_id"]) if value is not None)
    rows = 0; seen_user = 0; seen_product = 0; seen_both = 0
    for chunk in iter_csv_parts(future_path, chunk_size):
        for user, product in chunk[["user_id", "product_id"]].itertuples(index=False, name=None):
            user_id, product_id = _id(user), _id(product)
            if user_id is None or product_id is None: continue
            rows += 1; user_seen = user_id in past_users; product_seen = product_id in past_products
            seen_user += int(user_seen); seen_product += int(product_seen); seen_both += int(user_seen and product_seen)
    return {"rows": rows, "seen_user_rate": _divide(seen_user, rows), "seen_product_rate": _divide(seen_product, rows), "seen_user_and_product_rate": _divide(seen_both, rows), "unseen_user_rate": _divide(rows - seen_user, rows), "unseen_product_rate": _divide(rows - seen_product, rows), "model_capability_warning": "Current TwoTowerRecall uses only user/ad ID embeddings. Unseen IDs cannot gain content-based representations; add product category/brand/price and user-context towers before expecting strict temporal cold-start gains."}


def temporal_pipeline_diagnostics(config: TemporalConfig) -> dict[str, Any]:
    """Rows/users/observed positives for the revised temporal contract."""
    past_ab = build_past_ab_split(config); future_ab = build_future_ab_split(config); root = config.output_dir
    def window(path: Path) -> dict[str, int]:
        rows = 0; users: set[str] = set(); products: set[str] = set(); conversions = 0
        for chunk in iter_csv_parts(path, config.chunk_size):
            rows += len(chunk); users.update(value for value in (_id(item) for item in chunk["user_id"]) if value is not None); products.update(value for value in (_id(item) for item in chunk["product_id"]) if value is not None)
            conversions += int(pd.to_numeric(chunk.get("conversion_label", pd.Series(0, index=chunk.index)), errors="coerce").eq(1).sum())
        observed_pairs = _future_positives(path, config.chunk_size)
        return {"rows": rows, "users": len(users), "observed_click_positive_pairs": sum(len(value) for value in observed_pairs.values()), "unique_products": len(products), "conversion_positive_rows": conversions}
    split_metadata = json.loads((root / "split" / "metadata.json").read_text(encoding="utf-8")) if (root / "split" / "metadata.json").is_file() else {}
    selected = int(split_metadata.get("selected_users", 0)); eligible = int(split_metadata.get("eligible_users", 0))
    return {"contract": {"past_a": "history, product features and sampled-negative pool", "past_b": "direct observed-click coarse supervision", "future_a": "coarse candidate-inventory evaluation", "future_b": "held out for downstream final evaluation"}, "windows": {name: window(root / "split" / name) for name in ("past_a", "past_b", "future_a", "future_b")}, "past_ab": past_ab, "future_ab": future_ab, "selected_users": selected, "eligible_users": eligible, "eligible_user_scale_factor": _divide(eligible, selected), "max_users_interpretation": "Increasing max_users can only scale observed Past-B positives approximately linearly. It cannot repair the prior recall-hit supervision bottleneck; the revised contract removes that bottleneck first."}


def _evaluate_candidate_mapping(mapping: Mapping[str,list[tuple[str,int]]], positives: Mapping[str,set[str]], cutoffs: Iterable[int]) -> dict[str,Any]:
    cutoffs=tuple(int(k) for k in cutoffs); recalls={k:0. for k in cutoffs}; hits={k:0 for k in cutoffs}; ads=set()
    for user,target in positives.items():
        ranks={ad:rank for ad,rank in mapping.get(user,[])}; ads.update(ranks)
        for k in cutoffs:
            matched=sum(ad in ranks and ranks[ad]<=k for ad in target); recalls[k]+=matched/len(target); hits[k]+=int(matched>0)
    users=len(positives); rows=sum(len(v) for v in mapping.values())
    return {"users_evaluated":users,"average_future_positives_per_user":_divide(sum(map(len,positives.values())),users),"candidate_coverage":_divide(rows,users),"unique_recalled_ads":len(ads),"users_with_hit":hits,"metrics":{f"recall@{k}":_divide(recalls[k],users) for k in cutoffs}|{f"hit_rate@{k}":_divide(hits[k],users) for k in cutoffs}}


def future_candidate_labels(candidate_path: Path, future_path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Yield candidate chunks with labels from Future only; never full history."""
    positives = _future_positives(future_path, chunk_size)
    for chunk in pd.read_csv(candidate_path, chunksize=chunk_size, low_memory=False):
        users = chunk["user_id"].astype("string").str.strip(); ads = chunk["candidate_ad_id"].astype("string").str.strip()
        labels = [int(str(ad) in positives.get(str(user), set())) for user, ad in zip(users, ads)]
        labeled = chunk.copy(); labeled["future_label"] = labels
        yield labeled


def build_future_ab_split(config: TemporalConfig) -> dict[str, Any]:
    """Strictly split the already-Future selected-user events into A then B."""
    future=config.output_dir/'split'/'future'; metadata=config.output_dir/'split'/'future_ab_metadata.json'
    if metadata.exists(): return json.loads(metadata.read_text())
    values=[]
    for chunk in iter_csv_parts(future,config.chunk_size): values.append(pd.to_numeric(chunk[config.timestamp_column],errors='coerce').dropna().to_numpy(dtype=np.int64))
    timestamps=np.concatenate(values) if values else np.empty(0,dtype=np.int64)
    if not len(timestamps): raise ValueError('Future split has no timestamps')
    threshold=int(np.quantile(timestamps,.5,method='higher'))
    _write_window_parts(future,config.output_dir/'split'/'future_a',config,lambda ts:ts<=threshold)
    _write_window_parts(future,config.output_dir/'split'/'future_b',config,lambda ts:ts>threshold)
    def stats(name):
        rows=users=0; ids=set(); lo=hi=None
        for chunk in iter_csv_parts(config.output_dir/'split'/name,config.chunk_size):
            rows+=len(chunk); ids.update(chunk.user_id.astype(str)); t=pd.to_numeric(chunk[config.timestamp_column],errors='coerce').dropna()
            if len(t): lo=int(t.min()) if lo is None else min(lo,int(t.min())); hi=int(t.max()) if hi is None else max(hi,int(t.max()))
        return {'rows':rows,'users':len(ids),'time_min':lo,'time_max':hi}
    result={'future_ab_split_timestamp':threshold,'future_a':stats('future_a'),'future_b':stats('future_b')}
    if result['future_a']['time_max'] is not None and result['future_b']['time_min'] is not None and result['future_a']['time_max']>=result['future_b']['time_min']: raise ValueError('Temporal leakage: Future-A/B overlap')
    metadata.write_text(json.dumps(result,indent=2,sort_keys=True)); return result


def build_past_ab_split(config: TemporalConfig) -> dict[str, Any]:
    """Split Past into Past-A (history) and Past-B (coarse supervision).

    This keeps coarse training independent from whether a later recall system
    happens to retrieve an observed click.  Both windows precede Future-A/B.
    """
    source = config.output_dir / "split" / "past"; metadata = config.output_dir / "split" / "past_ab_metadata.json"
    if metadata.exists():
        return json.loads(metadata.read_text())
    values = [pd.to_numeric(chunk[config.timestamp_column], errors="coerce").dropna().to_numpy(dtype=np.int64) for chunk in iter_csv_parts(source, config.chunk_size)]
    timestamps = np.concatenate(values) if values else np.empty(0, dtype=np.int64)
    if not len(timestamps):
        raise ValueError("Past split has no timestamps for Past-A/Past-B")
    threshold = int(np.quantile(timestamps, .5, method="higher"))
    past_a, past_b = config.output_dir / "split" / "past_a", config.output_dir / "split" / "past_b"
    _write_window_parts(source, past_a, config, lambda timestamp: timestamp <= threshold)
    _write_window_parts(source, past_b, config, lambda timestamp: timestamp > threshold)
    def stats(path: Path) -> dict[str, int | None]:
        rows = 0; users: set[str] = set(); lower: int | None = None; upper: int | None = None
        for chunk in iter_csv_parts(path, config.chunk_size):
            rows += len(chunk); users.update(str(value) for value in chunk["user_id"].dropna())
            current = pd.to_numeric(chunk[config.timestamp_column], errors="coerce").dropna()
            if len(current):
                lower = int(current.min()) if lower is None else min(lower, int(current.min()))
                upper = int(current.max()) if upper is None else max(upper, int(current.max()))
        return {"rows": rows, "users": len(users), "time_min": lower, "time_max": upper}
    result = {"past_ab_split_timestamp": threshold, "past_a": stats(past_a), "past_b": stats(past_b), "contract": "Past-A builds history/statistics and sampled-negative pools; Past-B observed clicks supply coarse train positives."}
    if result["past_a"]["time_max"] is None or result["past_b"]["time_min"] is None or result["past_a"]["time_max"] >= result["past_b"]["time_min"]:
        raise ValueError("Temporal leakage: Past-A/Past-B overlap")
    metadata.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_temporal_coarse(config: TemporalConfig, *, max_train_rows:int=2_000_000, top_k:int=50, negatives_per_positive:int=5) -> dict[str,Any]:
    """Train coarse ranker from Past-B observed clicks, not recall hits.

    A sampled non-positive is a training surrogate drawn from a Past-A
    popularity pool.  It is never claimed to be an observed impression
    negative.  Future-A is candidate-inventory evaluation only; Future-B is
    untouched by this coarse training contract.
    """
    future_ab = build_future_ab_split(config); past_ab = build_past_ab_split(config); root=config.output_dir; fused=root/'recall_candidates'/'formal_top1000'/'fused_top1000.csv'; past_a=root/'split'/'past_a'; past_b=root/'split'/'past_b'; future_a=root/'split'/'future_a'
    if not fused.exists(): raise FileNotFoundError(f'Build temporal RRF first: {fused}')
    train_positives = _future_positives(past_b, config.chunk_size); train_conversions = _future_conversion_positives(past_b, config.chunk_size); evaluation_positives = _future_positives(future_a, config.chunk_size)
    popularity_pool = _past_popularity_pool(past_a, config.chunk_size)
    cc=CoarseRankConfig(fused,past_a,root/'ranking'/'coarse_rank_topk.csv',root/'models'/'coarse_rank_model.pkl',max_train_rows=max_train_rows,max_users=config.max_users,top_k=top_k,chunk_size=config.chunk_size,random_seed=config.seed,negatives_per_positive=negatives_per_positive)
    store=build_interaction_feature_index(cc,root/'models'/'.temporal_past_a_features.sqlite')
    try:
        xs=[]; ys=[]; ws=[]; train_users=set(); train_pos=0; sampled_negatives=0; dropped_for_limit=0
        for user, positives in train_positives.items():
            remaining = max_train_rows - sum(len(value) for value in ys)
            if remaining <= 1:
                dropped_for_limit += len(positives); continue
            max_positive = max(1, remaining // (1 + negatives_per_positive))
            selected_positives = sorted(positives)[:max_positive]
            dropped_for_limit += len(positives) - len(selected_positives)
            negatives = _sample_nonpositive_candidates(user, set(selected_positives), popularity_pool, min(len(selected_positives) * negatives_per_positive, remaining - len(selected_positives)), config.seed)
            pairs = [(ad, 1) for ad in selected_positives] + [(ad, 0) for ad in negatives]
            if not pairs:
                continue
            frame = pd.DataFrame({"user_id": user, "candidate_ad_id": [ad for ad, _ in pairs], "rrf_score": 0.0, "source_count": 0.0})
            picked = store.enrich(frame); labels = np.asarray([label for _, label in pairs], dtype=np.int8)
            xs.append(preprocess_features(picked)); ys.append(labels); ws.append(_future_a_sample_weights(labels, picked.candidate_ad_id, train_conversions.get(user, set())))
            train_users.add(user); train_pos += int(labels.sum()); sampled_negatives += int((labels == 0).sum())
        if not xs or len(np.unique(np.concatenate(ys)))<2: raise ValueError('Temporal coarse needs Past-B positives and sampled non-positive candidates')
        X=np.concatenate(xs); y=np.concatenate(ys); w=np.concatenate(ws)
        model=HistGradientBoostingClassifier(max_iter=100,random_state=config.seed).fit(X,y,sample_weight=w)
        cc.model_path.parent.mkdir(parents=True,exist_ok=True); pickle.dump(model,cc.model_path.open('wb'))
        scores=[]; labels=[]; before=0; retained={20:0,30:0,50:0,70:0}; users_before=set(); users_after={k:set() for k in retained}; eval_users=set(); candidate_rows=0; candidate_users=set(); output_rows=[]
        for user,candidates in _candidate_frame_groups(fused,config.chunk_size):
            target=evaluation_positives.get(user,set())
            if not target: continue
            eval_users.add(user); candidate_users.add(user); candidate_rows += len(candidates); frame=store.enrich(candidates); s=model.predict_proba(preprocess_features(frame))[:,1]; l=frame.candidate_ad_id.astype(str).isin(target).to_numpy(dtype=np.int8); scores.extend(s); labels.extend(l)
            positive=np.flatnonzero(l); before+=len(positive); users_before.update([user] if len(positive) else [])
            order=np.lexsort((frame.candidate_ad_id.astype(str).to_numpy(),-pd.to_numeric(frame.rrf_score).to_numpy(),-s))
            for k in retained:
                kept=set(order[:k]); n=sum(i in kept for i in positive); retained[k]+=n
                if n: users_after[k].add(user)
            for rank, index in enumerate(order[:top_k], start=1): output_rows.append((user, str(frame.candidate_ad_id.iloc[index]), float(s[index]), rank))
        labels=np.asarray(labels); scores=np.asarray(scores); all_train_positive_pairs=sum(len(value) for value in train_positives.values()); all_eval_positive_pairs=sum(len(value) for value in evaluation_positives.values())
        metrics={'roc_auc':float(roc_auc_score(labels,scores)) if len(np.unique(labels))==2 else None,'pr_auc':float(average_precision_score(labels,scores)) if len(np.unique(labels))==2 else None,'logloss':float(log_loss(labels,scores,labels=[0,1])) if len(labels) else None,
                 'training_contract':'Past-B observed clicks are direct positives; negatives are sampled non-positive candidates from a Past-A popularity pool, not exposure negatives. Recall hit is never required for a train positive.', 'past_b_observed_positive_candidates':all_train_positive_pairs,'past_b_observed_positive_users':len(train_positives),'train_positive_candidates_used':train_pos,'users_used_for_coarse_train':len(train_users),'sampled_nonpositive_candidates':sampled_negatives,'dropped_positive_candidates_by_max_train_rows':dropped_for_limit,'future_a_positive_candidates_before_coarse':all_eval_positive_pairs,'future_a_positive_candidates_in_recall_inventory':before,'future_a_recall_positive_coverage':_divide(before,all_eval_positive_pairs),'users_with_future_a_positive':len(evaluation_positives),'users_with_future_a_positive_in_inventory':len(users_before),'users_used_for_coarse_eval':len(eval_users),'candidate_rows_evaluated':candidate_rows,'candidate_coverage_per_future_a_user':_divide(candidate_rows,len(evaluation_positives)),'retention':{f'retention@{k}':_divide(retained[k],before) for k in retained},'positive_after_topk':retained,'users_with_positive_after_topk':{k:len(v) for k,v in users_after.items()},'evaluation_caveat':'Future-A labels mark observed clicks inside recall inventory; non-matches are candidate-inventory non-positives, not verified exposure negatives.','leakage_passed':True,'past_ab':past_ab,'future_ab':future_ab}
        cc.output_path.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(output_rows,columns=['user_id','candidate_ad_id','coarse_score','rank']).to_csv(cc.output_path,index=False)
        target=root/'metrics'; target.mkdir(parents=True,exist_ok=True); (target/'coarse_metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)); pd.DataFrame([{'metric':k,'value':v} for k,v in metrics.items() if isinstance(v,(float,int))]+[{'metric':k,'value':v} for k,v in metrics['retention'].items()]).to_csv(target/'coarse_metrics.csv',index=False)
        return metrics
    finally:
        store.close(); (root/'models'/'.temporal_past_a_features.sqlite').unlink(missing_ok=True)


def _past_popularity_pool(path: Path, chunk_size: int) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for chunk in iter_csv_parts(path, chunk_size):
        for product in chunk["product_id"]:
            if (candidate := _id(product)) is not None:
                counts[candidate] += 1
    if not counts:
        raise ValueError("Past-A has no products for sampled-negative pool")
    return [candidate for candidate, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _sample_nonpositive_candidates(user: str, positive_ads: set[str], pool: Sequence[str], count: int, seed: int) -> list[str]:
    """Deterministically sample training surrogates, never asserted negatives."""
    if count <= 0:
        return []
    start = _stable(user, seed) % len(pool); sampled: list[str] = []
    for offset in range(len(pool)):
        candidate = pool[(start + offset) % len(pool)]
        if candidate in positive_ads:
            continue
        sampled.append(candidate)
        if len(sampled) >= count:
            break
    return sampled


def assert_temporal_leakage_safe(past_path: Path, future_path: Path) -> None:
    if past_path.resolve() == future_path.resolve():
        raise ValueError("Temporal leakage: Past and Future paths are identical")
    if "temporal" not in past_path.parts or "temporal" not in future_path.parts:
        raise ValueError("Temporal leakage: experiment artifacts must be isolated under outputs/temporal")


def _collect_timestamps(config: TemporalConfig) -> np.ndarray:
    values: list[np.ndarray] = []
    for chunk in iter_csv_parts(config.interaction_path, config.chunk_size):
        parsed = pd.to_numeric(chunk[config.timestamp_column], errors="coerce").dropna().to_numpy(dtype=np.int64)
        values.append(parsed)
    return np.concatenate(values) if values else np.empty(0, dtype=np.int64)


def _index_user_windows(config: TemporalConfig, threshold: int, connection: sqlite3.Connection) -> dict[str, int]:
    counts = defaultdict(int)
    for chunk in iter_csv_parts(config.interaction_path, config.chunk_size):
        times=pd.to_numeric(chunk[config.timestamp_column],errors="coerce"); users=chunk["user_id"].map(_id); products=chunk["product_id"].map(_id)
        rows=[]; product_rows=[]
        for user, product, timestamp in zip(users,products,times):
            if user is None or product is None or pd.isna(timestamp): continue
            future=int(int(timestamp)>threshold); rows.append((user,1-future,future)); product_rows.append((product,1-future,future)); counts["future_rows" if future else "past_rows"]+=1
        connection.executemany("INSERT INTO users VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET past=MAX(past,excluded.past), future=MAX(future,excluded.future)",rows); connection.commit()
        connection.executemany("INSERT INTO products VALUES (?, ?, ?) ON CONFLICT(product_id) DO UPDATE SET past=MAX(past,excluded.past), future=MAX(future,excluded.future)",product_rows); connection.commit()
    eligible=connection.execute("SELECT COUNT(*) FROM users WHERE past=1 AND future=1").fetchone()[0]
    return {**counts,"past_users":connection.execute("SELECT COUNT(*) FROM users WHERE past=1").fetchone()[0],"future_users":connection.execute("SELECT COUNT(*) FROM users WHERE future=1").fetchone()[0],"past_products":connection.execute("SELECT COUNT(*) FROM products WHERE past=1").fetchone()[0],"future_products":connection.execute("SELECT COUNT(*) FROM products WHERE future=1").fetchone()[0],"eligible_users":eligible}


def _select_eligible_users(connection: sqlite3.Connection, config: TemporalConfig) -> set[str]:
    users=[str(row[0]) for row in connection.execute("SELECT user_id FROM users WHERE past=1 AND future=1")]
    return set(sorted(users,key=lambda user: _stable(user,config.seed))[:config.max_users])


def _write_selected_parts(config: TemporalConfig, threshold: int, selected: set[str]) -> None:
    for window in ("past","future"):
        target=config.output_dir/"split"/window; target.mkdir(parents=True,exist_ok=True)
        for file in target.glob("part-*.csv"): file.unlink()
    buffers={"past":[],"future":[]}; part={"past":0,"future":0}
    for chunk in iter_csv_parts(config.interaction_path,config.chunk_size):
        users=chunk["user_id"].map(_id); times=pd.to_numeric(chunk[config.timestamp_column],errors="coerce")
        for window,mask in (("past",times<=threshold),("future",times>threshold)):
            chosen=chunk.loc[mask & users.isin(selected)]
            if not chosen.empty: buffers[window].append(chosen)
            if sum(len(x) for x in buffers[window])>=config.chunk_size:
                pd.concat(buffers[window],ignore_index=True).to_csv(config.output_dir/"split"/window/f"part-{part[window]:05d}.csv",index=False); part[window]+=1; buffers[window]=[]
    for window in buffers:
        if buffers[window]: pd.concat(buffers[window],ignore_index=True).to_csv(config.output_dir/"split"/window/f"part-{part[window]:05d}.csv",index=False)

def _write_window_parts(source:Path,target:Path,config:TemporalConfig,predicate)->None:
    target.mkdir(parents=True,exist_ok=True)
    for file in target.glob('part-*.csv'): file.unlink()
    part=0
    for chunk in iter_csv_parts(source,config.chunk_size):
        times=pd.to_numeric(chunk[config.timestamp_column],errors='coerce'); chosen=chunk.loc[predicate(times)]
        if not chosen.empty: chosen.to_csv(target/f'part-{part:05d}.csv',index=False); part+=1

def _candidate_frame_groups(path:Path,chunk_size:int)->Iterator[tuple[str,pd.DataFrame]]:
    current=None; rows=[]
    for chunk in pd.read_csv(path,usecols=['user_id','candidate_ad_id','rrf_score','source_count'],chunksize=chunk_size,low_memory=False):
        for row in chunk.itertuples(index=False,name=None):
            user=_id(row[0]); ad=_id(row[1])
            if user is None or ad is None: continue
            if current is not None and user!=current:
                yield current,pd.DataFrame(rows,columns=['user_id','candidate_ad_id','rrf_score','source_count']); rows=[]
            current=user; rows.append((user,ad,row[2],row[3]))
    if current is not None: yield current,pd.DataFrame(rows,columns=['user_id','candidate_ad_id','rrf_score','source_count'])


def _future_positives(path: Path, chunk_size: int) -> dict[str,set[str]]:
    result:dict[str,set[str]]=defaultdict(set)
    for chunk in iter_csv_parts(path,chunk_size):
        for user,product in chunk[["user_id","product_id"]].itertuples(index=False,name=None):
            if (u:=_id(user)) is not None and (p:=_id(product)) is not None: result[u].add(p)
    return result


def _future_conversion_positives(path: Path, chunk_size: int) -> dict[str,set[str]]:
    """Return Future click pairs that have at least one conversion.

    This is deliberately separate from ``_future_positives``: every Future
    click remains a training positive, while conversion is used only to set
    its sample weight.
    """
    result:dict[str,set[str]]=defaultdict(set)
    for chunk in iter_csv_parts(path,chunk_size):
        if "conversion_label" not in chunk:
            raise ValueError("Temporal coarse requires conversion_label for Future-A sample weights")
        converted=pd.to_numeric(chunk["conversion_label"],errors="coerce").eq(1)
        for user,product in chunk.loc[converted,["user_id","product_id"]].itertuples(index=False,name=None):
            if (u:=_id(user)) is not None and (p:=_id(product)) is not None: result[u].add(p)
    return result


def _future_a_sample_weights(labels: np.ndarray, candidate_ad_ids: Iterable[object], conversion_ads: set[str]) -> np.ndarray:
    """Weight Future-A conversions at 3x; keep clicks and negatives at 1x."""
    weights=np.ones(len(labels),dtype=np.float32)
    converted=np.fromiter(((_id(ad) in conversion_ads) for ad in candidate_ad_ids),dtype=bool,count=len(labels))
    weights[(labels==1) & converted]=3.0
    return weights


def _candidate_groups(path:Path,chunk_size:int,*,has_explicit_rank:bool)->Iterator[tuple[str,list[tuple[str,int]]]]:
    """Yield complete user groups, deriving ranks from ordered RRF rows if needed."""
    columns=["user_id","candidate_ad_id"] + (["rank"] if has_explicit_rank else [])
    current=None; rows=[]; completed_users:set[str]=set()
    for chunk in pd.read_csv(path,usecols=columns,chunksize=chunk_size,low_memory=False):
        for row in chunk.itertuples(index=False,name=None):
            user,ad=_id(row[0]),_id(row[1])
            if user is None or ad is None: continue
            if current is not None and user!=current:
                if current in completed_users:
                    raise ValueError("Recall candidates must keep each user's rows contiguous for streaming evaluation")
                completed_users.add(current)
                yield current,rows; rows=[]
            current=user
            rank=_recall_rank(row[2]) if has_explicit_rank else len(rows)+1
            rows.append((ad,rank))
    if current is not None:
        if current in completed_users:
            raise ValueError("Recall candidates must keep each user's rows contiguous for streaming evaluation")
        yield current,rows


def _id(value:object)->str|None:
    if pd.isna(value): return None
    value=str(value).strip(); return value or None
def _recall_source_name(path:Path)->str:
    return "rrf" if path.stem=="fused_candidates" else path.stem.removesuffix("_topk")
def _recall_rank(value:object)->int:
    try: rank=float(value)
    except (TypeError,ValueError) as error: raise ValueError(f"Recall rank must be a positive integer: {value!r}") from error
    if not rank.is_integer() or rank<=0: raise ValueError(f"Recall rank must be a positive integer: {value!r}")
    return int(rank)
def _divide(a:float,b:int)->float: return float(a/b) if b else 0.0
def _stable(value:str,seed:int)->int: return int.from_bytes(hashlib.blake2b(f"{seed}:{value}".encode(),digest_size=8).digest(),"big")
def _hash_ids(ids:set[str])->str: return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
def _split_matches(m:Mapping[str,Any],c:TemporalConfig)->bool: return m.get("past_ratio")==c.past_ratio and m.get("seed")==c.seed and m.get("timestamp_column")==c.timestamp_column
