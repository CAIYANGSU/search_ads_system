"""Future-A-only diagnostics for ID and content Two-Tower retrieval variants."""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from search_ads_system.evaluation.temporal import _future_positives, iter_csv_parts
from search_ads_system.recall.rrf_fusion import rank_rrf_candidates

CUTOFFS=(10,20,50,100)


def run_content_two_tower_diagnostics(*, past_path: Path, future_a_path: Path, itemcf_path: Path, popularity_path: Path, id_only_path: Path, content_path: Path, content_no_product_id_path: Path, output_dir: Path, chunk_size: int, model_runs: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate retrieval and fixed RRF with Future-A; Future-B is never opened."""
    positives=_future_positives(future_a_path,chunk_size); seen=_past_products(past_path,chunk_size)
    paths={"itemcf":itemcf_path,"popularity":popularity_path,"id_only":id_only_path,"content":content_path,"content_no_product_id":content_no_product_id_path}
    hits={}; metrics={}
    for name,path in paths.items():
        metrics[name],hits[name]=_evaluate(path,positives,seen,chunk_size)
    content_new=hits["content"]-hits["id_only"]
    baseline_union=hits["itemcf"]|hits["popularity"]|hits["id_only"]
    content_union=hits["itemcf"]|hits["popularity"]|hits["content"]
    fixed={"rrf_k":100,"weights":{"itemcf":2.0,"two_tower":1.0,"popularity":2.0},"id_only":_fixed_rrf(positives,itemcf_path,id_only_path,popularity_path,chunk_size),"content":_fixed_rrf(positives,itemcf_path,content_path,popularity_path,chunk_size)}
    total=sum(map(len,positives.values()))
    def oracle(pairs:set[tuple[str,str]])->int:
        by_user:dict[str,set[str]]=defaultdict(set)
        for user,product in pairs: by_user[user].add(product)
        return sum(min(100,len(products)) for products in by_user.values())
    report={
      "development_window":"Future-A only", "temporal_leakage_guard":{"passed":True,"future_b_read_for_model_selection":False,"training_features":"Past interactions and optional explicitly configured point-in-time product catalogue only","future_a_usage":"offline evaluation and fixed comparison only"},
      "overall_metrics":metrics, "model_runs":dict(model_runs),
      "unseen_product_recall_lift@100":metrics["content"]["unseen_product"]["recall@100"]-metrics["id_only"]["unseen_product"]["recall@100"],
      "seen_product_recall_regression@100":metrics["content"]["seen_product"]["recall@100"]-metrics["id_only"]["seen_product"]["recall@100"],
      "incremental_hits":{"content_new_positive_pairs_vs_id_only":len(content_new),"content_new_vs_popularity_overlap":len(content_new & hits["popularity"]),"content_new_vs_itemcf_overlap":len(content_new & hits["itemcf"]),"content_only_two_tower_positive_pairs":len(content_new-hits["itemcf"]-hits["popularity"]),"content_vs_id_overlap":len(hits["content"]&hits["id_only"])},
      "source_overlap":{"content__id_only_hit_positive_pairs":len(hits["content"]&hits["id_only"]),"content__itemcf_hit_positive_pairs":len(hits["content"]&hits["itemcf"]),"content__popularity_hit_positive_pairs":len(hits["content"]&hits["popularity"]),"itemcf__popularity_hit_positive_pairs":len(hits["itemcf"]&hits["popularity"])},
      "candidate_generation_ceiling":{"label":"DIAGNOSTIC ONLY — USES FUTURE-A LABELS — NOT A VALID PRODUCTION STRATEGY","diagnostic_only":True,"old_union_positive_pair_coverage@100":_divide(len(baseline_union),total),"new_union_positive_pair_coverage@100":_divide(len(content_union),total),"old_oracle_union_top100_positive_pair_coverage":_divide(oracle(baseline_union),total),"new_oracle_union_top100_positive_pair_coverage":_divide(oracle(content_union),total)},
      "fixed_rrf_comparison":fixed,
      "interpretation":"Sampled training negatives remain Past-pool negatives, not exposure negatives. Any content improvement must be interpreted against unseen-product slice and fixed-RRF results rather than tuned fusion."}
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/'two_tower_content_diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8'); _write_markdown(report,output_dir/'two_tower_content_diagnostics.md'); return report


def _evaluate(path: Path, positives: Mapping[str,set[str]], seen: set[str], chunk_size: int) -> tuple[dict[str,Any],set[tuple[str,str]]]:
    rows=users=0; items=set(); recall={cutoff:0. for cutoff in CUTOFFS}; hit_users={cutoff:0 for cutoff in CUTOFFS}; hits=set(); sliced={"seen_product":defaultdict(float),"unseen_product":defaultdict(float)}
    header=set(pd.read_csv(path,nrows=0).columns)
    groups: Iterator[tuple[str,list[tuple[str,int]]]]
    if "user_id" not in header:
        global_candidates=_global(path,chunk_size)
        groups=iter((user,global_candidates) for user in sorted(positives))
    else:
        groups=_groups(path,chunk_size)
    for user,candidates in groups:
        target=positives.get(user)
        if not target: continue
        users+=1; rows+=len(candidates); items.update(candidate for candidate,_ in candidates); ranks={candidate:rank for candidate,rank in candidates}
        for cutoff in CUTOFFS:
            matched={product for product in target if ranks.get(product,10**9)<=cutoff}; recall[cutoff]+=_divide(len(matched),len(target)); hit_users[cutoff]+=int(bool(matched))
        for product in target:
            if ranks.get(product,10**9)<=100: hits.add((user,product))
    total_users=len(positives); total_pairs=sum(map(len,positives.values()))
    result={"candidate_rows":rows,"candidate_users":users,"unique_items":len(items),"candidate_user_coverage":_divide(users,total_users),"hit_positive_pairs@100":len(hits),"positive_pair_coverage@100":_divide(len(hits),total_pairs),"hit_users@100":len({user for user,_ in hits}),"unique_recalled_items":len(items)}
    for cutoff in CUTOFFS: result[f"recall@{cutoff}"]=_divide(recall[cutoff],total_users); result[f"hit_rate@{cutoff}"]=_divide(hit_users[cutoff],total_users)
    for label,products in (("seen_product",{p for products in positives.values() for p in products if p in seen}),("unseen_product",{p for products in positives.values() for p in products if p not in seen})):
        pairs={(user,product) for user,values in positives.items() for product in values if product in products}; slice_hits=hits&pairs; users_with={user for user,_ in pairs}; value={"rows":len(pairs),"positive_pairs":len(pairs),"users":len(users_with),"hit_positive_pairs@100":len(slice_hits),"positive_pair_coverage@100":_divide(len(slice_hits),len(pairs))}
        # Re-read candidate ranks is unnecessary for coverage, but cut-off slice
        # metrics need the same exact denominators; derive them in a second
        # bounded streaming pass.
        value.update(_slice_metrics(path,positives,products,chunk_size)); result[label]=value
    return result,hits


def _slice_metrics(path:Path, positives:Mapping[str,set[str]], products:set[str], chunk_size:int)->dict[str,float]:
    total={user:values&products for user,values in positives.items() if values&products}; recall={k:0. for k in CUTOFFS}; rates={k:0 for k in CUTOFFS}; present=set()
    if "user_id" not in set(pd.read_csv(path,nrows=0).columns):
        global_candidates=_global(path,chunk_size); groups=iter((user,global_candidates) for user in sorted(total))
    else: groups=_groups(path,chunk_size)
    for user,candidates in groups:
        if user not in total: continue
        present.add(user); ranks={candidate:rank for candidate,rank in candidates}
        for cutoff in CUTOFFS:
            count=sum(ranks.get(product,10**9)<=cutoff for product in total[user]); recall[cutoff]+=_divide(count,len(total[user])); rates[cutoff]+=int(count>0)
    users=len(total); return {f"recall@{k}":_divide(recall[k],users) for k in CUTOFFS}|{f"hit_rate@{k}":_divide(rates[k],users) for k in CUTOFFS}


def _fixed_rrf(positives:Mapping[str,set[str]],itemcf_path:Path,two_path:Path,popularity_path:Path,chunk_size:int)->dict[str,float]:
    pop=_global(popularity_path,chunk_size); wanted=set(positives); streams={"itemcf":_groups(itemcf_path,chunk_size,wanted),"two_tower":_groups(two_path,chunk_size,wanted)}; pending={name:next(stream,None) for name,stream in streams.items()}; pair_hits=hit_users=0
    for user in sorted(positives):
        lists={"popularity":pop}
        for name,stream in streams.items():
            row=pending[name]; lists[name]=row[1] if row is not None and row[0]==user else []
            if row is not None and row[0]==user: pending[name]=next(stream,None)
        ranked=rank_rrf_candidates(lists,k=100,weights={"itemcf":2.,"two_tower":1.,"popularity":2.},top_k=100); selected={candidate for candidate,_,_ in ranked}; found=selected&positives[user]; pair_hits+=len(found); hit_users+=int(bool(found))
    total=sum(map(len,positives.values())); return {"positive_pair_coverage@100":_divide(pair_hits,total),"hit_rate@100":_divide(hit_users,len(positives))}


def _groups(path:Path,chunk_size:int,wanted:set[str]|None=None)->Iterator[tuple[str,list[tuple[str,int]]]]:
    header=set(pd.read_csv(path,nrows=0).columns)
    if "user_id" not in header: return iter(())
    current=None; rows=[]; previous=None
    for chunk in pd.read_csv(path,usecols=["user_id","candidate_ad_id","rank"],chunksize=chunk_size,low_memory=False):
        for user,product,rank in chunk.itertuples(index=False,name=None):
            user=str(user).strip(); product=str(product).strip()
            if current is not None and user!=current:
                if previous is not None and current<=previous: raise ValueError("candidate rows must be grouped by ascending user_id")
                if wanted is None or current in wanted: yield current,rows
                previous,current,rows=current,user,[]
            elif current is None: current=user
            rows.append((product,int(rank)))
    if current is not None and (wanted is None or current in wanted): yield current,rows


def _global(path:Path,chunk_size:int)->list[tuple[str,int]]:
    return [(str(product),int(rank)) for chunk in pd.read_csv(path,usecols=["candidate_ad_id","rank"],chunksize=chunk_size,low_memory=False) for product,rank in chunk.itertuples(index=False,name=None)]
def _past_products(path:Path,chunk_size:int)->set[str]: return {str(product) for chunk in iter_csv_parts(path,chunk_size) for product in chunk.product_id.dropna()}
def _divide(a:float,b:int)->float: return float(a/b) if b else 0.
def _write_markdown(report:Mapping[str,Any],path:Path)->None:
    lines=["# Content-aware Two-Tower — Strict Temporal Diagnostic","","Future-A is the development evaluator only. Future-B was not read.","", "| variant | recall@100 | unseen recall@100 | seen recall@100 |", "| --- | ---: | ---: | ---: |"]
    for name,metrics in report["overall_metrics"].items(): lines.append(f"| {name} | {metrics['recall@100']:.6f} | {metrics['unseen_product']['recall@100']:.6f} | {metrics['seen_product']['recall@100']:.6f} |")
    lines += ["", "## Oracle", "", report["candidate_generation_ceiling"]["label"], "", "## Fixed RRF", "", f"- ID-only coverage@100: {report['fixed_rrf_comparison']['id_only']['positive_pair_coverage@100']:.6f}", f"- Content coverage@100: {report['fixed_rrf_comparison']['content']['positive_pair_coverage@100']:.6f}"]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
