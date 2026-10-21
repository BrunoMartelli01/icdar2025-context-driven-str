from scripts.run_guarded_v3_graph_cut import as_rows, metrics, choose, encode_labels
import numpy as np


def fixture():
    proposals=[{'bbox':[1,2,20,10],'crop_array_sha256':'abc'}]
    shards=[{'sample_id':'total_text:1','selected':{'baseline':{
        'proposals':proposals,'chosen_groups':[],'changed':False}}}]
    gt={'total_text:1':[{'bbox':[1,2,20,10],'text':'TEST'}]}
    ocr={'abc':{'raw_text':'test'}}
    policy={'id':'baseline','method':'baseline'}
    return shards,gt,ocr,policy


def test_rank_is_preserved_for_existing_metric_adapter():
    shards,gt,ocr,policy=fixture()
    rows=as_rows(shards,gt,ocr,policy)
    assert rows[0]['predictions'][0]['rank']==1
    assert metrics(shards,gt,ocr,policy)['correct_words']==1


def test_no_regression_promotion_and_active_diagnostic_are_distinct():
    base={'correct_words':10,'f1':.5,'changed_images':0,'split_operations':0}
    worse={'correct_words':11,'f1':.49,'changed_images':3,'split_operations':4}
    params=[{'id':'baseline','method':'baseline'},{'id':'graph','method':'graphcut'}]
    selection=choose({'baseline':base,'graph':worse},params,{})
    assert selection['graphcut']['promoted_policy']['id']=='baseline'
    assert selection['graphcut']['diagnostic_policy']['id']=='graph'


def test_rle_roundtrip_keeps_instance_labels_and_background():
    a=np.array([[0,0,1],[1,2,2]],dtype=np.int32);r=encode_labels(a)
    b=np.repeat(r['values'],r['lengths']).reshape(r['shape'])
    assert np.array_equal(a,b)
