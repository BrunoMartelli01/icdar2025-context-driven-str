from pathlib import Path

import pytest

from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
from scripts.evaluate_word_multitask_crops import (CONFIG, load_config, ratios, score_predictions,
                                                  validate_ocr, verify_geometry_exports, write_json)
from scripts.prune_segmentation_checkpoints import sha256_file


def test_crop_scoring_requires_spatial_matching_and_one_to_one_transcriptions():
    gt=[{'bbox':[0,0,10,10],'text':'Hello'},{'bbox':[20,0,30,10],'text':'World!'}]
    predictions=[{'bbox':[0,0,10,10],'text':'HELLO'},{'bbox':[0,0,10,10],'text':'Hello'},
                 {'bbox':[20,0,30,10],'text':'world'}]
    scored=score_predictions(predictions,gt,text_available=True)
    assert scored['geometry_iou_ge_50']==2
    assert scored['exact_casefold_iou_ge_50']==1
    assert scored['exact_no_symbols_iou_ge_50']==2
    value=ratios(scored)
    assert value['exact_casefold_recall_all_gt']==.5
    assert value['exact_casefold_precision']==pytest.approx(1/3)
    assert value['exact_casefold_f1']==.4
    predictions[0]['bbox']=[100,100,110,110]
    predictions[1]['bbox']=[100,100,110,110]
    assert score_predictions(predictions,gt,text_available=True)['exact_casefold_iou_ge_50']==0


def test_empty_predictions_keep_every_gt_word_in_denominator():
    score=score_predictions([], [{'bbox':[0,0,10,10],'text':'a'}],text_available=True)
    assert ratios(score)['exact_casefold_recall_all_gt']==0
    assert score['gt_words']==1
    assert ratios(score)['geometry_precision']==0


def test_config_isolation_and_pinned_decode():
    cfg=load_config(CONFIG,'approximate')
    assert cfg['recognition']['decode_ar'] is True
    with pytest.raises(ValueError,match='APPROXIMATE'): load_config(CONFIG,'strict')
    with pytest.raises(ValueError,match='APPROXIMATE'): load_config(Path('/tmp/config.json'),'approximate')


def test_writes_reuse_identical_json_and_reject_overwrites(tmp_path):
    path=tmp_path/'result.json'
    write_json(path,{'value':1});original=path.read_bytes()
    write_json(path,{'value':1})
    assert path.read_bytes()==original
    with pytest.raises(ValueError,match='NEW output'): write_json(path,{'value':2})
    assert path.read_bytes()==original


@pytest.mark.parametrize('changed',[{'crop_rgb_sha256':'wrong'}, {'decode_ar':False},
                                  {'parseq_checkpoint_sha256':'wrong'}, {'refine_iters':0}, {'predicted_length':3}])
def test_ocr_cache_rejects_changed_crop_weights_or_decoding(changed):
    record={'crop_rgb_sha256':'crop','parseq_checkpoint_sha256':PARSEQ_CHECKPOINT_SHA256,
            'parseq_source_archive_sha256':PARSEQ_RUNTIME_ARCHIVE_SHA256,'raw_text':'word',
            'predicted_length':4,'decode_ar':True,'refine_iters':1}
    assert validate_ocr(record,'crop')=='word'
    with pytest.raises(ValueError): validate_ocr({**record,**changed},'crop')


def test_completed_geometry_cache_verifies_original_images_and_annotations(tmp_path):
    image=tmp_path/'rgb';image.write_bytes(b'RGB')
    annotation=tmp_path/'annotation';annotation.write_bytes(b'GT')
    output=tmp_path/'result.json';output.write_text('{}')
    row={'image_path':str(image),'image_sha256':sha256_file(image),'annotation_path':str(annotation),
         'annotation_sha256':sha256_file(annotation),'exports':{'result.json':sha256_file(output)}}
    result={'branches':{'text':{'findcontours':{'per_image':[row]}}}}
    verify_geometry_exports(result,tmp_path)
    annotation.write_bytes(b'changed')
    with pytest.raises(ValueError,match='annotation changed'): verify_geometry_exports(result,tmp_path)
