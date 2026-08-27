from pathlib import Path
import pandas as pd
import pytest

import numpy as np

from search_ads_system.evaluation.temporal import (
    TemporalConfig,
    _future_a_sample_weights,
    _future_conversion_positives,
    build_past_ab_split,
    build_temporal_split,
    diagnose_temporal_recall_sources,
    diagnose_two_tower_cold_start,
    evaluate_recall_file,
    future_candidate_labels,
    run_temporal_coarse,
    temporal_pipeline_diagnostics,
)
from search_ads_system.ranking.coarse_rank import FEATURE_COLUMNS, assert_no_leakage_features


def test_temporal_split_and_multi_positive_recall(tmp_path: Path) -> None:
    source=tmp_path/'source'; source.mkdir()
    pd.DataFrame([
        {'user_id':'u1','product_id':'a','click_timestamp':1,'conversion_label':0},
        {'user_id':'u1','product_id':'b','click_timestamp':2,'conversion_label':0},
        {'user_id':'u1','product_id':'c','click_timestamp':5,'conversion_label':0},
        {'user_id':'u1','product_id':'d','click_timestamp':6,'conversion_label':1},
        {'user_id':'u2','product_id':'e','click_timestamp':1,'conversion_label':0},
    ]).to_csv(source/'part-00000.csv',index=False)
    config=TemporalConfig(source,tmp_path/'outputs'/'temporal',past_ratio=.5,max_users=10,chunk_size=2)
    meta=build_temporal_split(config)
    assert meta['selected_users']==1
    past=pd.concat([pd.read_csv(p) for p in (config.output_dir/'split'/'past').glob('part-*.csv')])
    future=pd.concat([pd.read_csv(p) for p in (config.output_dir/'split'/'future').glob('part-*.csv')])
    assert set(past.user_id)=={'u1'} and set(future.user_id)=={'u1'}
    assert past.click_timestamp.max() <= meta['split_timestamp'] < future.click_timestamp.min()
    candidates=tmp_path/'candidates.csv'
    pd.DataFrame([('u1','c',.9,1),('u1','x',.8,2),('u1','d',.7,3)],columns=['user_id','candidate_ad_id','score','rank']).to_csv(candidates,index=False)
    metrics=evaluate_recall_file(candidates,config.output_dir/'split'/'future',cutoffs=(1,2,3),chunk_size=2)
    assert metrics['metrics']['recall@1']==.5
    assert metrics['metrics']['recall@3']==1.0
    labels=pd.concat(list(future_candidate_labels(candidates,config.output_dir/'split'/'future',2)))
    assert labels.future_label.tolist()==[1,0,1]


def test_temporal_future_a_conversion_weights_are_not_features(tmp_path: Path) -> None:
    future_a=tmp_path/'future_a'; future_a.mkdir()
    pd.DataFrame([
        {'user_id':'u1','product_id':'ordinary-click','conversion_label':0},
        {'user_id':'u1','product_id':'conversion-click','conversion_label':1},
    ]).to_csv(future_a/'part-00000.csv',index=False)

    conversions=_future_conversion_positives(future_a,chunk_size=10)
    weights=_future_a_sample_weights(
        np.array([1,1,0],dtype=np.int8),
        ['ordinary-click','conversion-click','negative'],
        conversions['u1'],
    )

    assert weights.tolist()==[1.0,3.0,1.0]
    assert 'conversion_label' not in FEATURE_COLUMNS
    assert_no_leakage_features(FEATURE_COLUMNS)


def _write_future_positives(path: Path, rows: list[tuple[str, str]]) -> None:
    path.mkdir()
    pd.DataFrame(rows,columns=['user_id','product_id']).to_csv(path/'part-00000.csv',index=False)


def test_recall_evaluation_uses_explicit_rank(tmp_path: Path, caplog) -> None:
    future=tmp_path/'future'
    _write_future_positives(future,[('u1','a'),('u1','c')])
    candidates=tmp_path/'itemcf_topk.csv'
    pd.DataFrame([('u1','a',1),('u1','x',2),('u1','c',11)],columns=['user_id','candidate_ad_id','rank']).to_csv(candidates,index=False)

    with caplog.at_level('INFO'):
        metrics=evaluate_recall_file(candidates,future,chunk_size=2)

    assert metrics['metrics']['recall@10']==.5
    assert metrics['metrics']['recall@20']==1.0
    assert 'Evaluating itemcf with explicit rank' in caplog.text


def test_rrf_evaluation_derives_continuous_per_user_rank_across_chunks(tmp_path: Path, caplog) -> None:
    future=tmp_path/'future'
    _write_future_positives(future,[('u1','p10'),('u1','p11'),('u2','first')])
    candidates=tmp_path/'fused_candidates.csv'
    rows=[('u1',f'p{index}',1.0/index,1) for index in range(1,13)]
    rows += [('u2','first',.9,1),('u2','other',.8,1)]
    pd.DataFrame(rows,columns=['user_id','candidate_ad_id','rrf_score','source_count']).to_csv(candidates,index=False)

    with caplog.at_level('INFO'):
        metrics=evaluate_recall_file(candidates,future,chunk_size=2)

    assert metrics['metrics']['recall@10']==.75
    assert metrics['metrics']['recall@20']==1.0
    assert metrics['metrics']['recall@50']==1.0
    assert metrics['metrics']['recall@100']==1.0
    assert 'Evaluating rrf with derived per-user rank' in caplog.text


def test_popularity_evaluation_remains_global_recall(tmp_path: Path, caplog) -> None:
    future=tmp_path/'future'
    _write_future_positives(future,[('u1','popular'),('u2','not-recalled')])
    candidates=tmp_path/'popularity_topk.csv'
    pd.DataFrame([('popular',.9,1),('other',.8,2)],columns=['candidate_ad_id','popularity_score','rank']).to_csv(candidates,index=False)

    with caplog.at_level('INFO'):
        metrics=evaluate_recall_file(candidates,future,chunk_size=1)

    assert metrics['metrics']['recall@10']==.5
    assert metrics['metrics']['recall@20']==.5
    assert 'Evaluating popularity as global recall' in caplog.text


def test_recall_evaluation_validates_required_schema(tmp_path: Path) -> None:
    future=tmp_path/'future'
    _write_future_positives(future,[('u1','a')])
    candidates=tmp_path/'invalid.csv'
    pd.DataFrame([('u1',1)],columns=['user_id','rank']).to_csv(candidates,index=False)

    with pytest.raises(ValueError,match='candidate_ad_id'):
        evaluate_recall_file(candidates,future)


def test_temporal_recall_diagnostics_detects_popularity_retention_and_cold_start(tmp_path: Path) -> None:
    past=tmp_path/'past'; future=tmp_path/'future'; past.mkdir(); future.mkdir()
    pd.DataFrame([('u1','seen'),('u2','known')],columns=['user_id','product_id']).to_csv(past/'part-00000.csv',index=False)
    pd.DataFrame([('u1','popular'),('u3','known')],columns=['user_id','product_id']).to_csv(future/'part-00000.csv',index=False)
    paths={}
    for name, rows in {
        'itemcf':[('u1','other',1)],
        'two_tower':[('u3','known',1)],
        'popularity':[('popular',1)],
        'fused':[('u1','other',1),('u3','known',1)],
    }.items():
        path=tmp_path/f'{name}.csv'; paths[name]=path
        columns=['candidate_ad_id','rank'] if name=='popularity' else ['user_id','candidate_ad_id','rank']
        pd.DataFrame(rows,columns=columns).to_csv(path,index=False)
    diagnostic=diagnose_temporal_recall_sources(paths,future,chunk_size=10)
    assert diagnostic['sources']['popularity']['hit_positive_pairs']==1
    assert diagnostic['rrf_popularity_hit_retention']==0.0
    cold=diagnose_two_tower_cold_start(past,future,chunk_size=10)
    assert cold['unseen_user_rate']==.5 and cold['seen_product_rate']==.5


def test_revised_temporal_coarse_uses_past_b_observed_clicks_without_recall_hit(tmp_path: Path) -> None:
    source=tmp_path/'source'; source.mkdir()
    pd.DataFrame([
        {'user_id':'u1','product_id':'a','click_timestamp':1,'conversion_label':0},
        {'user_id':'u1','product_id':'b','click_timestamp':2,'conversion_label':0},
        {'user_id':'u1','product_id':'c','click_timestamp':3,'conversion_label':0},
        {'user_id':'u1','product_id':'d','click_timestamp':4,'conversion_label':1},
        {'user_id':'u1','product_id':'e','click_timestamp':5,'conversion_label':0},
        {'user_id':'u1','product_id':'f','click_timestamp':6,'conversion_label':1},
        {'user_id':'u1','product_id':'g','click_timestamp':7,'conversion_label':0},
        {'user_id':'u1','product_id':'h','click_timestamp':8,'conversion_label':1},
    ]).to_csv(source/'part-00000.csv',index=False)
    config=TemporalConfig(source,tmp_path/'outputs'/'temporal',past_ratio=.5,max_users=10,chunk_size=10)
    build_temporal_split(config); past_ab=build_past_ab_split(config)
    assert past_ab['past_a']['time_max'] < past_ab['past_b']['time_min']
    candidates=config.output_dir/'recall_candidates'/'formal_top1000'/'fused_top1000.csv'; candidates.parent.mkdir(parents=True)
    # Deliberately omit Past-B products c/d: revised training must still use them.
    pd.DataFrame([('u1','g',.9,1),('u1','a',.8,1)],columns=['user_id','candidate_ad_id','rrf_score','source_count']).to_csv(candidates,index=False)
    metrics=run_temporal_coarse(config,max_train_rows=100,top_k=10,negatives_per_positive=1)
    assert metrics['past_b_observed_positive_candidates'] >= 1
    assert metrics['train_positive_candidates_used']==metrics['past_b_observed_positive_candidates']
    assert metrics['training_contract'].startswith('Past-B observed clicks')
    pipeline=temporal_pipeline_diagnostics(config)
    assert pipeline['contract']['past_b'].startswith('direct observed-click')
