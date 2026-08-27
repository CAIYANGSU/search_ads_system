"""Regression coverage for strict-temporal content Two-Tower components."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from search_ads_system.recall.content_two_tower import (
    CONTENT_COLUMNS, ContentTwoTowerConfig, ContentTwoTowerModel, ContentNegativeSamplingDataset,
    prepare_content_training_data, save_content_checkpoint,
)
from search_ads_system.recall.two_tower_content_audit import run_content_two_tower_diagnostics


def _past(tmp_path:Path)->Path:
    path=tmp_path/'past'; path.mkdir(exist_ok=True)
    pd.DataFrame([
        {'user_id':'u1','product_id':'seen','conversion_label':0,'product_brand':'b','product_category_1':'c','product_price':10},
        {'user_id':'u2','product_id':'seen2','conversion_label':1,'product_brand':None,'product_category_1':None,'product_price':20},
    ]).to_csv(path/'part-00000.csv',index=False)
    return path

def _catalogue(tmp_path:Path)->Path:
    path=tmp_path/'catalogue.csv'
    pd.DataFrame([
        {'product_id':'seen','product_brand':'b','product_category_1':'c','product_price':10},
        {'product_id':'seen2','product_brand':None,'product_category_1':None,'product_price':20},
        {'product_id':'cold','product_brand':'b','product_category_1':'c','product_price':12},
    ]).to_csv(path,index=False)
    return path

def _config(tmp_path:Path,variant='content')->ContentTwoTowerConfig:
    return ContentTwoTowerConfig(input_path=_past(tmp_path),output_path=tmp_path/'out.csv',index_path=tmp_path/'index',checkpoint_path=tmp_path/'model.pt',product_catalog_path=_catalogue(tmp_path),catalog_as_of_timestamp=1,variant=variant,embedding_dim=4,hidden_dim=8,categorical_buckets=17,product_id_buckets=19,batch_size=2,epochs=1,negative_samples=1,max_history_items=2)

def test_content_tower_encodes_unseen_product_from_catalogue_content(tmp_path:Path)->None:
    data=prepare_content_training_data(_config(tmp_path)); model=ContentTwoTowerModel(data,_config(tmp_path))
    cold=list(data.product_ids).index('cold'); encoded=model.encode_products(torch.tensor([cold]))
    assert encoded.shape == (1,4) and torch.isfinite(encoded).all()
    assert data.metadata['content_representable_cold_products'] == 1

def test_missing_content_uses_unknown_bucket(tmp_path:Path)->None:
    data=prepare_content_training_data(_config(tmp_path))
    seen2=list(data.product_ids).index('seen2')
    assert (data.product_category_indices[seen2] == 0).all()

def test_no_product_id_ablation_forwards_normally(tmp_path:Path)->None:
    config=_config(tmp_path,'content_no_product_id'); data=prepare_content_training_data(config); model=ContentTwoTowerModel(data,config)
    user,product=model(torch.tensor([0]),torch.tensor([0]))
    assert model.product_id_embedding is None and user.shape == product.shape == (1,4)

def test_history_pooling_uses_only_past_input(tmp_path:Path)->None:
    data=prepare_content_training_data(_config(tmp_path))
    assert data.metadata['user_history'].startswith('Past-only')
    assert data.metadata['training_contract'].startswith('Past observed')
    assert max(len(history) for history in data.histories) == 1

def test_future_catalogue_paths_are_rejected(tmp_path:Path)->None:
    config=_config(tmp_path); future=tmp_path/'future_a'; future.mkdir(); config=ContentTwoTowerConfig(**{**config.__dict__,'product_catalog_path':future})
    with pytest.raises(ValueError,match='Future'):
        prepare_content_training_data(config)

def test_content_checkpoint_schema_isolated_from_other_ablation(tmp_path:Path)->None:
    config=_config(tmp_path); data=prepare_content_training_data(config); model=ContentTwoTowerModel(data,config); save_content_checkpoint(model,data,config,[])
    wrong=ContentTwoTowerConfig(**{**config.__dict__,'variant':'content_no_product_id'})
    from search_ads_system.recall.content_two_tower import load_content_checkpoint
    with pytest.raises(ValueError,match='schema/variant'):
        load_content_checkpoint(wrong,torch.device('cpu'))

def _write_candidates(path:Path,rows:list[tuple[str,str,int]])->None: pd.DataFrame(rows,columns=['user_id','candidate_ad_id','rank']).to_csv(path,index=False)

def test_content_audit_seen_unseen_incremental_and_fixed_rrf(tmp_path:Path)->None:
    past=_past(tmp_path); future=tmp_path/'future_a'; future.mkdir(); pd.DataFrame([('u1','seen'),('u1','cold')],columns=['user_id','product_id']).to_csv(future/'part-00000.csv',index=False)
    item=tmp_path/'item.csv'; id_only=tmp_path/'id.csv'; content=tmp_path/'content.csv'; no_id=tmp_path/'noid.csv'; pop=tmp_path/'pop.csv'
    _write_candidates(item,[('u1','seen',1)]); _write_candidates(id_only,[('u1','seen',1)]); _write_candidates(content,[('u1','cold',1),('u1','seen',2)]); _write_candidates(no_id,[('u1','cold',1)])
    pd.DataFrame([('popular',1)],columns=['candidate_ad_id','rank']).to_csv(pop,index=False)
    report=run_content_two_tower_diagnostics(past_path=past,future_a_path=future,itemcf_path=item,popularity_path=pop,id_only_path=id_only,content_path=content,content_no_product_id_path=no_id,output_dir=tmp_path/'metrics',chunk_size=1,model_runs={})
    assert report['overall_metrics']['content']['unseen_product']['recall@100'] == 1.0
    assert report['incremental_hits']['content_new_positive_pairs_vs_id_only'] == 1
    assert report['fixed_rrf_comparison']['rrf_k'] == 100 and report['fixed_rrf_comparison']['weights'] == {'itemcf':2.0,'two_tower':1.0,'popularity':2.0}
    assert report['temporal_leakage_guard']['future_b_read_for_model_selection'] is False
    assert (tmp_path/'metrics'/'two_tower_content_diagnostics.md').is_file()

def test_content_dataset_keeps_past_negative_semantics(tmp_path:Path)->None:
    config=_config(tmp_path); data=prepare_content_training_data(config); dataset=ContentNegativeSamplingDataset(data,config)
    assert len(dataset) and set(dataset.pool.tolist()) <= set(data.training_product_indices.tolist())

def test_ann_can_index_content_representable_cold_product(tmp_path:Path)->None:
    pytest.importorskip('faiss')
    from search_ads_system.recall.content_two_tower import extract_content_product_embeddings
    from search_ads_system.recall.faiss_index import build_faiss_index
    config=_config(tmp_path); data=prepare_content_training_data(config); model=ContentTwoTowerModel(data,config)
    index=build_faiss_index(extract_content_product_embeddings(model,len(data.product_ids),2,torch.device('cpu')),'flat')
    assert index.ntotal == len(data.product_ids) and 'cold' in data.product_ids.tolist()
