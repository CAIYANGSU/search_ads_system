from pathlib import Path
import pandas as pd

import numpy as np

from search_ads_system.evaluation.temporal import (
    TemporalConfig,
    _future_a_sample_weights,
    _future_conversion_positives,
    build_temporal_split,
    evaluate_recall_file,
    future_candidate_labels,
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
