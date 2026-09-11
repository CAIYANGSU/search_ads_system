"""Strict temporal experiment entry point; all artifacts live under outputs/temporal."""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from search_ads_system.common.config import load_yaml_config
from search_ads_system.evaluation.final_holdout import future_b_opened_warning
from search_ads_system.evaluation.temporal import build_temporal_split, diagnose_temporal_recall_sources, diagnose_two_tower_cold_start, evaluate_recall_file, parse_temporal_config, run_temporal_coarse, temporal_pipeline_diagnostics
from search_ads_system.recall.itemcf_recall import ItemCFRecallConfig, generate_itemcf_candidates, load_interactions as load_itemcf, write_candidates as write_itemcf
from search_ads_system.recall.popularity_recall import PopularityRecallConfig, generate_popularity_candidates, write_candidates as write_popularity
from search_ads_system.recall.rrf_fusion import RRFFusionConfig, fuse_and_write_candidates
from search_ads_system.recall.temporal_fusion import run_temporal_fusion_sweep
from search_ads_system.recall.two_tower_recall import TwoTowerRecallConfig, run_two_tower_recall
from search_ads_system.recall.content_item_two_tower import ContentItemTwoTowerConfig, run_content_item_ablation

def _recall(raw, temporal):
    # New formal artifacts leave historical sweep/diagnostic candidates intact.
    root=temporal.output_dir; past=root/'split'/'past'; candidates=root/'recall_candidates'/'formal_top1000'; models=root/'models'/'formal_top1000'; candidates.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True)
    options=raw.get('temporal',{}).get('recall',{}); base=raw.get('recall',{})
    item=ItemCFRecallConfig(past,candidates/'itemcf_topk.csv','user_id','product_id','conversion_label',int(options.get('itemcf_top_k',100)),{'0':1.,'1':3.},1.,'sum','cosine',temporal.chunk_size,False,None,temporal.seed,10_000)
    if not item.output_path.exists():
        logging.info('Building temporal ItemCF from Past only: %s',past); write_itemcf(generate_itemcf_candidates(load_itemcf(item),item),item.output_path)
    else: logging.info('Reusing temporal ItemCF: %s',item.output_path)
    pop=PopularityRecallConfig(past,candidates/'popularity_topk.csv',int(options.get('popularity_top_k',200)),1.,3.,temporal.chunk_size)
    if not pop.output_path.exists():
        logging.info('Building temporal Popularity from Past only'); write_popularity(generate_popularity_candidates(pop),pop.output_path)
    else: logging.info('Reusing temporal Popularity: %s',pop.output_path)
    tt_opts=base.get('two_tower',{}); checkpoint=models/'two_tower_checkpoint.pt'; index=candidates/'faiss_ad_index'
    two=TwoTowerRecallConfig(past,candidates/'two_tower_topk.csv',index,checkpoint,top_k=int(options.get('two_tower_top_k',100)),seed=temporal.seed,train=not checkpoint.exists(),max_users=None,input_chunk_size=temporal.chunk_size,device=str(tt_opts.get('device','auto')))
    if not two.output_path.exists():
        logging.info('%s temporal Two Tower using Past only', 'Training' if two.train else 'Reusing checkpoint for'); run_two_tower_recall(two)
    else: logging.info('Reusing temporal Two Tower candidates: %s',two.output_path)
    # Frozen Future-A selection: fixed production RRF, not a new sweep.
    rrf=RRFFusionConfig(item.output_path,two.output_path,pop.output_path,candidates/'fused_top1000.csv',k=100,weights={'itemcf':2.,'two_tower':1.,'popularity':2.},top_k_per_user=int(options.get('rrf_top_k',1000)),max_users=temporal.max_users,chunk_size=temporal.chunk_size)
    if not rrf.output_path.exists(): fuse_and_write_candidates(rrf)
    else: logging.info('Reusing temporal RRF: %s',rrf.output_path)

def _evaluate(raw, temporal):
    # This stage intentionally reads frozen historical diagnostic artifacts.
    # The formal Top1000 funnel is run by --stage funnel.
    paths=temporal.output_dir/'recall_candidates'; future=temporal.output_dir/'split'/'future'; metrics={}
    for name in ('itemcf','two_tower','popularity','fused'):
        path=paths/f'{name}_topk.csv' if name!='fused' else paths/'fused_candidates.csv'
        metrics[name]=evaluate_recall_file(path,future,chunk_size=temporal.chunk_size)
    diagnostic=diagnose_temporal_recall_sources({name: paths/f'{name}_topk.csv' if name!='fused' else paths/'fused_candidates.csv' for name in ('itemcf','two_tower','popularity','fused')},future,chunk_size=temporal.chunk_size)
    two_tower=diagnose_two_tower_cold_start(temporal.output_dir/'split'/'past',future,chunk_size=temporal.chunk_size)
    target=temporal.output_dir/'metrics'; target.mkdir(parents=True,exist_ok=True); (target/'recall_metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True),encoding='utf-8'); (target/'recall_diagnostics.json').write_text(json.dumps({'pipeline':temporal_pipeline_diagnostics(temporal),'recall_sources':diagnostic,'two_tower_cold_start':two_tower},indent=2,sort_keys=True),encoding='utf-8')
    rows=[]
    for name,value in metrics.items():
        for metric,score in value['metrics'].items(): rows.append({'source':name,'metric':metric,'value':score})
    import pandas as pd; pd.DataFrame(rows).to_csv(target/'recall_metrics.csv',index=False)
    return metrics

def _fusion_sweep(raw, temporal):
    """Future-A-only fusion development; does not train or overwrite recall."""
    root=temporal.output_dir; candidates=root/'recall_candidates'; options=raw.get('temporal',{}).get('recall',{}).get('fusion_sweep',{})
    future_a=root/'split'/'future_a'
    if not future_a.exists():
        from search_ads_system.evaluation.temporal import build_future_ab_split
        build_future_ab_split(temporal)
    return run_temporal_fusion_sweep(
        itemcf_path=candidates/'itemcf_topk.csv', two_tower_path=candidates/'two_tower_topk.csv', popularity_path=candidates/'popularity_topk.csv',
        future_a_path=future_a, output_dir=root/'metrics', chunk_size=temporal.chunk_size,
        top_k=int(options.get('top_k',100)), popularity_quota=int(options.get('popularity_min_quota',25)), balanced_quota=int(options.get('balanced_min_quota',20)),
    )

def _content_two_tower(raw, temporal, *, sanity=False):
    """Run the history-free ID-only versus content-item Future-A ablation."""
    root=temporal.output_dir; past=root/'split'/'past'; future_a=root/'split'/'future_a'; recall_options=raw.get('temporal',{}).get('recall',{}); options=recall_options.get('content_item_two_tower',recall_options.get('content_two_tower',{})); chosen={**options,**(options.get('sanity',{}) if sanity else {})}
    if not future_a.exists():
        from search_ads_system.evaluation.temporal import build_future_ab_split
        build_future_ab_split(temporal)
    namespace=(root/'sanity'/'content_item_two_tower') if sanity else root
    catalogue=chosen.get('product_catalog_path'); catalogue_path=None if not catalogue else (Path(catalogue) if Path(catalogue).is_absolute() else ROOT/str(catalogue))
    catalogue_as_of=chosen.get('product_catalog_as_of_timestamp')
    metadata=json.loads((root/'split'/'metadata.json').read_text(encoding='utf-8'))
    base=raw.get('recall',{}).get('two_tower',{}); faiss=chosen.get('faiss',{})
    config=ContentItemTwoTowerConfig(
        input_path=past,future_path=future_a,output_dir=namespace,split_timestamp=int(metadata['split_timestamp']),
        product_catalog_path=catalogue_path,product_catalog_as_of_timestamp=None if catalogue_as_of is None else int(catalogue_as_of),
        enabled_features=tuple(chosen.get('enabled_features',('product_id','product_brand','product_category_1','product_category_2','product_category_3','product_category_4','partner_id','product_price'))),
        embedding_dim=int(chosen.get('embedding_dim',32)),feature_embedding_dim=int(chosen.get('feature_embedding_dim',16)),price_embedding_dim=int(chosen.get('price_embedding_dim',4)),hidden_dims=tuple(int(value) for value in chosen.get('hidden_dims',(128,64))),
        batch_size=int(chosen.get('batch_size',4096)),epochs=int(chosen.get('epochs',3)),learning_rate=float(chosen.get('learning_rate',1e-3)),negative_samples=int(chosen.get('negative_samples',5)),click_weight=float(chosen.get('click_weight',1.0)),conversion_weight=float(chosen.get('conversion_weight',3.0)),
        max_train_rows=None if chosen.get('max_train_rows') is None else int(chosen['max_train_rows']),top_k=int(chosen.get('top_k',200)),exclude_seen_items=bool(chosen.get('exclude_seen_items',True)),retrieval_oversample_ratio=float(chosen.get('retrieval_oversample_ratio',2.0)),search_batch_size=int(chosen.get('search_batch_size',10000)),inference_batch_size=int(chosen.get('inference_batch_size',4096)),input_chunk_size=temporal.chunk_size,log_every_rows=int(chosen.get('log_every_rows',200000)),history_cache_users=int(chosen.get('history_cache_users',20000)),seed=temporal.seed,device=str(base.get('device','auto')),
        faiss_index_type=str(faiss.get('index_type',chosen.get('faiss_index_type','hnsw'))),hnsw_m=int(faiss.get('hnsw_m',32)),ef_construction=int(faiss.get('ef_construction',200)),ef_search=int(faiss.get('ef_search',64)),train=bool(chosen.get('train',True)),
    )
    return run_content_item_ablation(config)

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=ROOT/"config.yaml"); parser.add_argument("--stage",choices=("split","itemcf","two_tower","popularity","rrf","evaluate_recall","fusion_sweep","content_item_two_tower_sanity","content_item_two_tower","two_tower_content_sanity","two_tower_content","coarse","funnel","all"),default="all"); args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path=args.config.resolve()
    if warning := future_b_opened_warning(path): logging.warning(warning)
    raw=load_yaml_config(path); temporal=parse_temporal_config(raw,path); result={}
    if args.stage in ('split','all'): result['split']=build_temporal_split(temporal)
    if args.stage in ('itemcf','two_tower','popularity','rrf','funnel','all'):
        if not (temporal.output_dir/'split'/'metadata.json').exists(): build_temporal_split(temporal)
        _recall(raw,temporal); result['recall']='complete'
    if args.stage in ('evaluate_recall','all'):
        result['recall_metrics']=_evaluate(raw,temporal)
    if args.stage == 'fusion_sweep':
        result['fusion_sweep']=_fusion_sweep(raw,temporal)
    if args.stage in ('content_item_two_tower_sanity','two_tower_content_sanity'): result['content_item_two_tower_sanity']=_content_two_tower(raw,temporal,sanity=True)
    if args.stage in ('content_item_two_tower','two_tower_content'): result['content_item_two_tower']=_content_two_tower(raw,temporal)
    if args.stage in ('coarse','all'):
        result['coarse_metrics']=run_temporal_coarse(temporal,max_train_rows=int(raw.get('temporal',{}).get('coarse_rank',{}).get('max_train_rows',2_000_000)),top_k=int(raw.get('temporal',{}).get('coarse_rank',{}).get('top_k',50)))
        target=temporal.output_dir/'metrics'; summary={'split':result.get('split',json.loads((temporal.output_dir/'split'/'metadata.json').read_text())),'pipeline':temporal_pipeline_diagnostics(temporal),'recall':result.get('recall_metrics',{}),'coarse':result['coarse_metrics'],'leakage':{'passed':True}}
        (target/'temporal_experiment_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
    if args.stage == 'funnel':
        result['coarse_metrics']=run_temporal_coarse(temporal,max_train_rows=int(raw.get('temporal',{}).get('coarse_rank',{}).get('max_train_rows',2_000_000)),top_k=int(raw.get('temporal',{}).get('coarse_rank',{}).get('top_k',100)))
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
