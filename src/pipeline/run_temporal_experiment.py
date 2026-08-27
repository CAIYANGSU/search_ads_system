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
from search_ads_system.recall.two_tower_recall import TwoTowerRecallConfig, checkpoint_parameter_counts, run_two_tower_recall
from search_ads_system.recall.content_two_tower import ContentTwoTowerConfig, content_checkpoint_parameter_counts, run_content_two_tower_recall
from search_ads_system.recall.two_tower_content_audit import run_content_two_tower_diagnostics

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
    """Train isolated content ablations from Past and evaluate Future-A only."""
    root=temporal.output_dir; past=root/'split'/'past'; future_a=root/'split'/'future_a'; options=raw.get('temporal',{}).get('recall',{}).get('content_two_tower',{}); chosen={**options,**(options.get('sanity',{}) if sanity else {})}
    if not future_a.exists():
        from search_ads_system.evaluation.temporal import build_future_ab_split
        build_future_ab_split(temporal)
    namespace=(root/'sanity'/'two_tower_content') if sanity else root
    candidates=namespace/'recall_candidates'; models=namespace/'models'; candidates.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True)
    dim=int(chosen.get('embedding_dim',16)); epochs=int(chosen.get('epochs',3)); max_rows=chosen.get('max_train_rows'); base=raw.get('recall',{}).get('two_tower',{})
    id_config=TwoTowerRecallConfig(past,candidates/'two_tower_id_only_topk.csv',candidates/'faiss_id_only_product_index',models/'two_tower_id_only.pt',embedding_dim=dim,batch_size=int(chosen.get('batch_size',4096)),epochs=epochs,learning_rate=float(chosen.get('learning_rate',1e-3)),top_k=100,negative_samples=int(chosen.get('negative_samples',5)),seed=temporal.seed,device=str(base.get('device','auto')),train=not (models/'two_tower_id_only.pt').exists(),input_chunk_size=temporal.chunk_size,max_train_rows=None if max_rows is None else int(max_rows))
    if not id_config.output_path.exists() or not id_config.checkpoint_path.exists():
        run_two_tower_recall(id_config)
    id_result={"checkpoint":str(id_config.checkpoint_path),"output":str(id_config.output_path),"variant":"id_only","parameter_counts":checkpoint_parameter_counts(id_config.checkpoint_path)}
    catalogue=chosen.get('product_catalog_path'); catalogue_path=None if not catalogue else (Path(catalogue) if Path(catalogue).is_absolute() else Path.cwd()/str(catalogue))
    catalogue_as_of=chosen.get('product_catalog_as_of_timestamp')
    if catalogue_path is not None:
        metadata=json.loads((root/'split'/'metadata.json').read_text(encoding='utf-8'))
        if catalogue_as_of is None or int(catalogue_as_of)>int(metadata['split_timestamp']):
            raise ValueError('content product catalogue must declare an as-of timestamp no later than the Past/Future split boundary')
    runs={"id_only":id_result}
    for variant,filename in (("content","two_tower_content"),("content_no_product_id","two_tower_content_no_product_id")):
        config=ContentTwoTowerConfig(input_path=past,output_path=candidates/f'{filename}_topk.csv',index_path=candidates/("faiss_content_product_index" if variant=="content" else "faiss_content_no_product_id_index"),checkpoint_path=models/f'{filename}.pt',product_catalog_path=catalogue_path,catalog_as_of_timestamp=catalogue_as_of,variant=variant,embedding_dim=dim,hidden_dim=int(chosen.get('hidden_dim',64)),categorical_buckets=int(chosen.get('categorical_buckets',4096)),product_id_buckets=int(chosen.get('product_id_buckets',65537)),batch_size=int(chosen.get('batch_size',4096)),epochs=epochs,learning_rate=float(chosen.get('learning_rate',1e-3)),top_k=100,negative_samples=int(chosen.get('negative_samples',5)),max_history_items=int(chosen.get('max_history_items',100)),max_train_rows=None if max_rows is None else int(max_rows),seed=temporal.seed,device=str(base.get('device','auto')),train=not (models/f'{filename}.pt').exists(),input_chunk_size=temporal.chunk_size)
        if not config.output_path.exists() or not config.checkpoint_path.exists(): run_content_two_tower_recall(config)
        runs[variant]={"checkpoint":str(config.checkpoint_path),"output":str(config.output_path),"variant":variant,"parameter_counts":content_checkpoint_parameter_counts(config.checkpoint_path)}
    if sanity: return {"runs":runs,"namespace":str(namespace),"future_b_read_for_model_selection":False}
    return run_content_two_tower_diagnostics(past_path=past,future_a_path=future_a,itemcf_path=root/'recall_candidates'/'itemcf_topk.csv',popularity_path=root/'recall_candidates'/'popularity_topk.csv',id_only_path=id_config.output_path,content_path=candidates/'two_tower_content_topk.csv',content_no_product_id_path=candidates/'two_tower_content_no_product_id_topk.csv',output_dir=root/'metrics',chunk_size=temporal.chunk_size,model_runs=runs)

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=ROOT/"config.yaml"); parser.add_argument("--stage",choices=("split","itemcf","two_tower","popularity","rrf","evaluate_recall","fusion_sweep","two_tower_content_sanity","two_tower_content","coarse","funnel","all"),default="all"); args=parser.parse_args()
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
    if args.stage == 'two_tower_content_sanity': result['two_tower_content_sanity']=_content_two_tower(raw,temporal,sanity=True)
    if args.stage == 'two_tower_content': result['two_tower_content']=_content_two_tower(raw,temporal)
    if args.stage in ('coarse','all'):
        result['coarse_metrics']=run_temporal_coarse(temporal,max_train_rows=int(raw.get('temporal',{}).get('coarse_rank',{}).get('max_train_rows',2_000_000)),top_k=int(raw.get('temporal',{}).get('coarse_rank',{}).get('top_k',50)))
        target=temporal.output_dir/'metrics'; summary={'split':result.get('split',json.loads((temporal.output_dir/'split'/'metadata.json').read_text())),'pipeline':temporal_pipeline_diagnostics(temporal),'recall':result.get('recall_metrics',{}),'coarse':result['coarse_metrics'],'leakage':{'passed':True}}
        (target/'temporal_experiment_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8')
    if args.stage == 'funnel':
        result['coarse_metrics']=run_temporal_coarse(temporal,max_train_rows=int(raw.get('temporal',{}).get('coarse_rank',{}).get('max_train_rows',2_000_000)),top_k=int(raw.get('temporal',{}).get('coarse_rank',{}).get('top_k',100)))
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
