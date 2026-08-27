"""Leakage-safe development sweep for strict-temporal recall fusion.

The module scores only candidate lists produced from Past.  Future-A is read
solely after ranking to measure variants and select a recommendation; Future-B
is intentionally not an input to this module.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from search_ads_system.evaluation.temporal import _future_positives
from search_ads_system.recall.rrf_fusion import _SOURCE_BITS, rank_rrf_candidates

SOURCES = ("itemcf", "two_tower", "popularity")
CUTOFFS = (10, 20, 50, 100)


@dataclass(frozen=True)
class FusionVariant:
    """One deterministic, label-free candidate-ranking variant."""

    name: str
    k: int
    weights: Mapping[str, float]
    min_quotas: Mapping[str, int]
    order: int

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "rrf_k": self.k, "weights": dict(self.weights), "min_quotas": dict(self.min_quotas), "variant_order": self.order}


def default_variants(*, top_k: int, popularity_quota: int, balanced_quota: int) -> list[FusionVariant]:
    """Small fixed grid requested for interpretable temporal development."""
    weights = (
        (1.0, 1.0, 1.0), (1.0, 1.0, 2.0), (1.0, 1.0, 3.0),
        (1.0, 2.0, 2.0), (2.0, 1.0, 2.0),
    )
    variants: list[FusionVariant] = []
    order = 0
    for k in (10, 30, 60, 100):
        for itemcf, two_tower, popularity in weights:
            variants.append(FusionVariant(
                name=f"rrf_k{k}_i{itemcf:g}_t{two_tower:g}_p{popularity:g}", k=k,
                weights={"itemcf": itemcf, "two_tower": two_tower, "popularity": popularity},
                min_quotas={}, order=order,
            ))
            order += 1
    # Baselines deliberately use fixed, declared quotas. They are not tuned per
    # user or from labels; only the choice among these variants uses Future-A.
    variants.extend([
        FusionVariant("rrf_baseline", 60, {"itemcf": 1.0, "two_tower": 1.0, "popularity": 1.0}, {}, order),
        FusionVariant("rrf_popularity_protected", 60, {"itemcf": 1.0, "two_tower": 1.0, "popularity": 1.0}, {"popularity": min(popularity_quota, top_k)}, order + 1),
        FusionVariant("rrf_balanced_protected", 60, {"itemcf": 1.0, "two_tower": 1.0, "popularity": 1.0}, {source: min(balanced_quota, top_k) for source in SOURCES}, order + 2),
    ])
    return variants


def run_temporal_fusion_sweep(
    *,
    itemcf_path: Path,
    two_tower_path: Path,
    popularity_path: Path,
    future_a_path: Path,
    output_dir: Path,
    chunk_size: int,
    top_k: int = 100,
    popularity_quota: int = 25,
    balanced_quota: int = 20,
) -> dict[str, Any]:
    """Evaluate fixed fusion variants on Future-A only and write reports.

    No fused candidate CSV is generated or changed.  This streams personalised
    candidates one user at a time, keeping only one user's lists in memory.
    """
    if top_k != 100:
        # Metrics are specified through @100 and the oracle capacity is Top100.
        raise ValueError("Temporal fusion sweep currently requires top_k=100")
    positive_by_user = _future_positives(future_a_path, chunk_size)
    if not positive_by_user:
        raise ValueError(f"Future-A contains no observed clicked pairs: {future_a_path}")
    paths = {"itemcf": itemcf_path, "two_tower": two_tower_path, "popularity": popularity_path}
    _validate_sources(paths)
    popularity = _load_global_candidates(popularity_path, chunk_size)
    variants = default_variants(top_k=top_k, popularity_quota=popularity_quota, balanced_quota=balanced_quota)
    states = {variant.name: _new_state() for variant in variants}
    source_stats = _new_source_stats()
    selected_users = set(positive_by_user)
    sources = {
        "itemcf": _iter_personalized_groups(itemcf_path, "itemcf", selected_users, chunk_size),
        "two_tower": _iter_personalized_groups(two_tower_path, "two_tower", selected_users, chunk_size),
    }
    pending: dict[str, tuple[str, list[tuple[str, int]]] | None] = {}
    for source, iterator in sources.items():
        pending[source] = next(iterator, None)

    source_hit_sets: dict[str, set[tuple[str, str]]] = {source: set() for source in SOURCES}
    union_hits: set[tuple[str, str]] = set()
    for user_id in sorted(positive_by_user):
        lists: dict[str, list[tuple[str, int]]] = {"popularity": popularity}
        for source, iterator in sources.items():
            current = pending[source]
            if current is not None and current[0] == user_id:
                lists[source] = current[1]
                pending[source] = next(iterator, None)
            else:
                lists[source] = []
        target = positive_by_user[user_id]
        source_hits = _source_hits(user_id, target, lists, top_k)
        for source, hits in source_hits.items():
            source_stats[source]["candidate_rows_at_100"] += len({candidate for candidate, rank in lists[source] if rank <= top_k})
            source_stats[source]["candidate_users_at_100"] += int(bool(lists[source]))
            source_stats[source]["unique_items"].update(candidate for candidate, rank in lists[source] if rank <= top_k)
            source_stats[source]["hit_pairs"] += len(hits)
            source_stats[source]["hit_users"] += int(bool(hits))
            source_hit_sets[source].update(hits)
        # This union is an offline candidate-generation ceiling, not a ranking
        # score and never feeds any production candidate list.
        for hits in source_hits.values():
            union_hits.update(hits)
        for variant in variants:
            ranked = rank_rrf_candidates(lists, k=variant.k, weights=variant.weights, top_k=top_k, min_quotas=variant.min_quotas)
            _accumulate_state(states[variant.name], target, ranked, source_hits, user_id)

    total_pairs = sum(len(products) for products in positive_by_user.values())
    source_report = _finish_source_stats(source_stats, positive_by_user, total_pairs)
    strongest = min(SOURCES, key=lambda source: (-source_report[source]["positive_pair_coverage_at_100"], SOURCES.index(source)))
    oracle_hits = sum(min(top_k, len({product for user, product in union_hits if user == user_id})) for user_id in positive_by_user)
    oracle = {
        "label": "DIAGNOSTIC ORACLE — USES FUTURE-A LABELS — NOT A VALID MODEL/FUSION STRATEGY",
        "diagnostic_only": True,
        "strongest_single_source_at_100": strongest,
        "strongest_single_source_positive_pair_coverage_at_100": source_report[strongest]["positive_pair_coverage_at_100"],
        "union_positive_pair_coverage_at_100": _divide(len(union_hits), total_pairs),
        "oracle_union_top100_positive_pair_coverage": _divide(oracle_hits, total_pairs),
        "oracle_hit_positive_pairs_at_100": oracle_hits,
    }
    reports = [_finish_variant(state, variant, source_hit_sets, positive_by_user, total_pairs, source_report) for variant, state in ((variant, states[variant.name]) for variant in variants)]
    best = sorted(reports, key=lambda row: (-row["positive_pair_coverage@100"], -row["hit_rate@100"], -row["source_retention@100"][strongest], row["variant_order"]))[0]
    audit = _implementation_audit(paths, popularity, top_k)
    result = {
        "development_window": "Future-A only; Future-B is intentionally untouched and cannot affect scores, variants, or selection.",
        "temporal_leakage_guard": {"passed": True, "future_b_read": False, "candidate_inputs": "Past-built recall artifacts only", "labels_used_only_for": "offline Future-A metrics, oracle diagnostic, and deterministic best-config recommendation"},
        "implementation_audit": audit,
        "future_a": {"positive_users": len(positive_by_user), "positive_pairs": total_pairs},
        "source_baselines": source_report,
        "oracle": oracle,
        "variants": reports,
        "best_recommendation": {"selection_rule": "primary positive_pair_coverage@100; secondary hit_rate@100; tie-break strongest-source retention then declared variant order", "strongest_source": strongest, "selected": best, "does_not_modify_production_config": True},
    }
    _write_reports(result, output_dir)
    return result


def _new_state() -> dict[str, Any]:
    return {"recall_sum": {cutoff: 0.0 for cutoff in CUTOFFS}, "hit_users": {cutoff: 0 for cutoff in CUTOFFS}, "hit_pairs": 0, "output_users": 0, "items": set(), "source_count_sum": 0, "retained": {source: 0 for source in SOURCES}, "composition": defaultdict(int), "hit_composition": defaultdict(int)}


def _accumulate_state(state: dict[str, Any], target: set[str], ranked: list[tuple[str, float, int]], source_hits: Mapping[str, set[tuple[str, str]]], user_id: str) -> None:
    state["output_users"] += 1
    ranks = {candidate: index for index, (candidate, _, _) in enumerate(ranked, start=1)}
    for cutoff in CUTOFFS:
        matched = {product for product in target if ranks.get(product, 10**9) <= cutoff}
        state["recall_sum"][cutoff] += _divide(len(matched), len(target))
        state["hit_users"][cutoff] += int(bool(matched))
    for candidate, _, mask in ranked:
        state["items"].add(candidate)
        state["source_count_sum"] += int(mask).bit_count()
        key = _composition(mask)
        state["composition"][key] += 1
        if candidate in target:
            state["hit_pairs"] += 1
            state["hit_composition"][key] += 1
            for source in SOURCES:
                if (user_id, candidate) in source_hits[source]:
                    state["retained"][source] += 1


def _finish_variant(state: Mapping[str, Any], variant: FusionVariant, source_hits: Mapping[str, set[tuple[str, str]]], positives: Mapping[str, set[str]], total_pairs: int, source_report: Mapping[str, Any]) -> dict[str, Any]:
    users = len(positives)
    result = variant.as_dict()
    for cutoff in CUTOFFS:
        result[f"recall@{cutoff}"] = _divide(state["recall_sum"][cutoff], users)
        result[f"hit_rate@{cutoff}"] = _divide(state["hit_users"][cutoff], users)
    result.update({
        "positive_pair_coverage@100": _divide(state["hit_pairs"], total_pairs), "hit_positive_pairs@100": state["hit_pairs"],
        "hit_users@100": state["hit_users"][100], "unique_items@100": len(state["items"]),
        "candidate_user_coverage": _divide(state["output_users"], users),
        "average_source_count_per_selected_candidate": _divide(state["source_count_sum"], sum(state["composition"].values())),
        "popularity_hit_retention@100": _divide(state["retained"]["popularity"], len(source_hits["popularity"])),
        "itemcf_hit_retention@100": _divide(state["retained"]["itemcf"], len(source_hits["itemcf"])),
        "two_tower_hit_retention@100": _divide(state["retained"]["two_tower"], len(source_hits["two_tower"])),
        "source_retention@100": {source: _divide(state["retained"][source], len(source_hits[source])) for source in SOURCES},
        "source_composition_top100": {key: int(state["composition"].get(key, 0)) for key in ("only_itemcf", "only_two_tower", "only_popularity", "multi_source")},
        "hit_source_composition@100": {key: int(state["hit_composition"].get(key, 0)) for key in ("only_itemcf", "only_two_tower", "only_popularity", "multi_source")},
        "incremental_hit_contribution": _incremental_contribution(source_hits),
    })
    return result


def _new_source_stats() -> dict[str, dict[str, Any]]:
    return {source: {"candidate_rows_at_100": 0, "candidate_users_at_100": 0, "unique_items": set(), "hit_pairs": 0, "hit_users": 0} for source in SOURCES}


def _finish_source_stats(stats: Mapping[str, Mapping[str, Any]], positives: Mapping[str, set[str]], total_pairs: int) -> dict[str, Any]:
    users = len(positives)
    return {source: {"candidate_rows_at_100": int(value["candidate_rows_at_100"]), "candidate_users_at_100": int(value["candidate_users_at_100"]), "candidate_user_coverage": _divide(value["candidate_users_at_100"], users), "unique_items@100": len(value["unique_items"]), "hit_positive_pairs@100": int(value["hit_pairs"]), "positive_pair_coverage_at_100": _divide(value["hit_pairs"], total_pairs), "hit_users@100": int(value["hit_users"]), "hit_rate@100": _divide(value["hit_users"], users)} for source, value in stats.items()}


def _source_hits(user_id: str, target: set[str], lists: Mapping[str, list[tuple[str, int]]], top_k: int) -> dict[str, set[tuple[str, str]]]:
    return {source: {(user_id, candidate) for candidate, rank in candidates if rank <= top_k and candidate in target} for source, candidates in lists.items()}


def _incremental_contribution(source_hits: Mapping[str, set[tuple[str, str]]]) -> dict[str, int]:
    seen: set[tuple[str, str]] = set(); result: dict[str, int] = {}
    for source in SOURCES:
        result[source] = len(source_hits[source] - seen)
        seen.update(source_hits[source])
    return result


def _composition(mask: int) -> str:
    names = [source for source in SOURCES if mask & _SOURCE_BITS[source]]
    return f"only_{names[0]}" if len(names) == 1 else "multi_source"


def _iter_personalized_groups(path: Path, source: str, selected_users: set[str], chunk_size: int) -> Iterator[tuple[str, list[tuple[str, int]]]]:
    current: str | None = None; rows: list[tuple[str, int]] = []; previous: str | None = None
    for chunk in pd.read_csv(path, usecols=["user_id", "candidate_ad_id", "rank"], chunksize=chunk_size, low_memory=False):
        for raw_user, raw_candidate, raw_rank in chunk.itertuples(index=False, name=None):
            user, candidate = _id(raw_user), _id(raw_candidate)
            if user is None or candidate is None:
                continue
            if current is not None and user != current:
                if previous is not None and current <= previous:
                    raise ValueError(f"{source} candidates must be grouped by ascending user_id")
                if current in selected_users:
                    yield current, rows
                previous, current, rows = current, user, []
            elif current is None:
                current = user
            rows.append((candidate, _rank(raw_rank, source)))
    if current is not None:
        if previous is not None and current <= previous:
            raise ValueError(f"{source} candidates must be grouped by ascending user_id")
        if current in selected_users:
            yield current, rows


def _load_global_candidates(path: Path, chunk_size: int) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    for chunk in pd.read_csv(path, usecols=["candidate_ad_id", "rank"], chunksize=chunk_size, low_memory=False):
        for raw_candidate, raw_rank in chunk.itertuples(index=False, name=None):
            if (candidate := _id(raw_candidate)) is not None:
                candidates.append((candidate, _rank(raw_rank, "popularity")))
    return candidates


def _implementation_audit(paths: Mapping[str, Path], popularity: list[tuple[str, int]], top_k: int) -> dict[str, Any]:
    return {
        "rrf_formula": "score(item) = sum_source(weight_source / (rrf_k + rank_source)); each source contributes at most once per item",
        "current_formal_temporal_rrf": {"rrf_k": 60, "weights": {"itemcf": 1.0, "two_tower": 1.0, "popularity": 0.5}, "final_top_k": top_k},
        "candidate_input_paths": {source: str(path) for source, path in paths.items()},
        "input_top_k": {"itemcf": _max_rank(paths["itemcf"]), "two_tower": _max_rank(paths["two_tower"]), "popularity": max((rank for _, rank in popularity), default=0)},
        "duplicate_handling": "Same item from multiple sources accumulates one weighted RRF contribution per source; duplicate rows inside a source do not inflate score.",
        "tie_break": "Descending RRF score, then ascending candidate_ad_id string.",
        "popularity_expansion": "Popularity is a global list and is applied to every personalised user in formal RRF. The diagnostic sweep also expands it for every Future-A evaluation user, including users without personalised candidates.",
        "source_truncation": "No additional source truncation occurs inside fusion; it consumes ranks emitted by source artifacts and only truncates final output to Top100.",
    }


def _max_rank(path: Path) -> int:
    maximum = 0
    for chunk in pd.read_csv(path, usecols=["rank"], chunksize=200_000, low_memory=False):
        if not chunk.empty:
            maximum = max(maximum, int(pd.to_numeric(chunk["rank"], errors="raise").max()))
    return maximum


def _validate_sources(paths: Mapping[str, Path]) -> None:
    for source, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Temporal fusion sweep requires existing {source} artifact: {path}")
        expected = {"candidate_ad_id", "rank"} | ({"user_id"} if source != "popularity" else set())
        columns = set(pd.read_csv(path, nrows=0).columns)
        if not expected <= columns:
            raise ValueError(f"{source} candidate artifact is missing columns {sorted(expected - columns)}: {path}")


def _write_reports(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recall_fusion_sweep.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "recall_fusion_best.json").write_text(json.dumps(result["best_recommendation"], ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    flat = []
    for variant in result["variants"]:
        row = {key: value for key, value in variant.items() if not isinstance(value, (dict, list))}
        row.update({f"weight_{source}": variant["weights"][source] for source in SOURCES})
        row.update({f"quota_{source}": variant["min_quotas"].get(source, 0) for source in SOURCES})
        flat.append(row)
    pd.DataFrame(flat).to_csv(output_dir / "recall_fusion_sweep.csv", index=False)
    best = result["best_recommendation"]["selected"]
    lines = ["# Strict Temporal Recall Fusion Sweep", "", "Future-A only for offline development; Future-B was not read.", "", "## Best recommendation", "", f"- Variant: `{best['name']}`", f"- positive_pair_coverage@100: {best['positive_pair_coverage@100']:.6f}", f"- hit_rate@100: {best['hit_rate@100']:.6f}", "", "## Oracle warning", "", result["oracle"]["label"], "", "## Variants", "", "| variant | pair coverage@100 | hit rate@100 | popularity retention@100 |", "| --- | ---: | ---: | ---: |"]
    for row in result["variants"]:
        lines.append(f"| {row['name']} | {row['positive_pair_coverage@100']:.6f} | {row['hit_rate@100']:.6f} | {row['popularity_hit_retention@100']:.6f} |")
    (output_dir / "recall_fusion_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _id(value: object) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _rank(value: object, source: str) -> int:
    try:
        rank = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} rank must be a positive integer: {value!r}") from error
    if not rank.is_integer() or rank <= 0:
        raise ValueError(f"{source} rank must be a positive integer: {value!r}")
    return int(rank)


def _divide(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
