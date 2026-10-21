"""Scientific mode, artifact integrity and GT-blind crop selection boundaries."""
from copy import deepcopy
from hashlib import md5,sha256
import json
import math
from types import SimpleNamespace

import pytest

from scripts.evaluate_totaltext_text_detectors import (DEFAULT_CONFIG,care_ground_truth,
    installed_package_inventory,select_arms,validate_config)
from scripts.evaluate_totaltext_yolo_direct_parseq import metrics
from scripts.prepare_totaltext_text_detectors import verify_weight
from scripts.run_crop_group_word_selection import evaluate_predictions


def config(): return json.loads(DEFAULT_CONFIG.read_text())


def reading(index,view,text,prob):
    return {'candidate_id':f'{index}:{view}','source_index':index,'view':view,
            'raw_text':text,'bbox':[index*20,0,index*20+10,10],'detector_score':.9,
            'token_probabilities':[prob]*len(text),'token_log_probabilities':[math.log(prob)]*len(text)}


def test_best_crop_is_selected_by_ocr_confidence_without_gt_and_outputs_one_per_region():
    cfg=config()
    rows=[reading(0,'axis_aligned','wrong',.9),reading(0,'rectified','correct',.8),
          reading(1,'axis_aligned','Same!',.8),reading(1,'rectified','same',.8)]
    arms=select_arms(rows,cfg,cfg['models']['craft'])
    assert [p['raw_text'] for p in arms['best_ocr/raw_0.7']] == ['wrong','Same!']
    assert len(arms['axis_aligned/raw_0.7']) == len(arms['rectified/raw_0.7']) == 2


def test_empty_predictions_are_counted_in_raw_arm_and_removed_only_explicitly():
    cfg=config()
    rows=[reading(0,view,'',.8) for view in cfg['crop_views']]
    arms=select_arms(rows,cfg,cfg['models']['craft'])
    assert len(arms['best_ocr/raw_0.7']) == 1
    assert arms['best_ocr/nonempty'] == arms['best_ocr/joint_text_nms_0.6'] == []


def test_word_metric_does_not_inflate_correct_count_with_two_crops():
    rows=[reading(0,view,'Hello!',.9) for view in ['axis_aligned','rectified']]
    gt=[{'bbox':[0,0,10,10],'text':'HELLO'}]
    score=evaluate_predictions(rows,gt,.5)
    assert score['correct'] == 1 and score['predictions'] == 2
    assert metrics(1,2,1)['hmean'] == pytest.approx(2/3)


@pytest.mark.parametrize('patch',[{'reproduction_mode':'strict'},{'split':'TRAIN'},
                                 {'table_5_claim':True},{'crop_views':['axis_aligned']}])
def test_config_is_approximate_test_only(patch):
    cfg=config(); validate_config(cfg)
    with pytest.raises(ValueError): validate_config({**cfg,**patch})


def test_no_score_control_can_request_missing_below_floor_candidates():
    cfg=deepcopy(config()); cfg['models']['craft']['score_controls']=[.1,.7]
    with pytest.raises(ValueError): validate_config(cfg)


def test_artifact_corruption_cannot_be_accepted(tmp_path):
    path=tmp_path/'weight.pt'; path.write_bytes(b'qualified')
    evidence={'official_md5':md5(b'qualified').hexdigest()}
    assert verify_weight(path,evidence) == sha256(b'qualified').hexdigest()
    assert verify_weight(path,{'official_sha256_prefix':sha256(b'qualified').hexdigest()[:8]})
    path.write_bytes(b'corrupted')
    with pytest.raises(RuntimeError): verify_weight(path,evidence)
    with pytest.raises(RuntimeError): verify_weight(path,{'official_sha256_prefix':'ffffffff'})
    with pytest.raises(ValueError): verify_weight(path,{})


def test_official_legacy_edge_coordinates_are_clipped_and_ignored_words_counted(tmp_path):
    path=tmp_path/'annotation.txt'
    path.write_text("x: [[1 11 11 1]], y: [[1 1 5 5]], ornt: [u'h'], transcriptions: [u'Hello!']\n"
                    "x: [[1 5 5 1]], y: [[1 1 5 5]], ornt: [u'#'], transcriptions: [u'###']\n")
    care,counts=care_ground_truth(path,(10,10))
    assert care == [{'bbox':[1,1,10,6],'text':'Hello!'}]
    assert counts == {'ignored_words':1,'clipped_polygons':1}


def test_incomplete_egg_info_is_recorded_without_none_string_sort_failure():
    distributions=[
        SimpleNamespace(metadata={'Name':'torch'},version='2.4.1',_path='/env/torch.dist-info'),
        SimpleNamespace(metadata={},version=None,_path='UNKNOWN.egg-info'),
        SimpleNamespace(metadata={'Name':'numpy'},version='1.26.4',_path='/env/numpy.dist-info'),
        SimpleNamespace(metadata={'Name':'torch'},version='2.4.1',_path='/env/torch.dist-info'),
        SimpleNamespace(metadata={'Name':'broken'},version='',_path='/env/broken.egg-info')]
    result=installed_package_inventory(distributions)
    assert result['packages'] == [['numpy','1.26.4'],['torch','2.4.1']]
    assert len(result['metadata_anomalies']) == 2
    assert any(row['metadata_path']=='UNKNOWN.egg-info' and row['name'] is None
               and row['version'] is None for row in result['metadata_anomalies'])
    assert installed_package_inventory(reversed(distributions)) == result
    assert json.loads(json.dumps(result)) == result
