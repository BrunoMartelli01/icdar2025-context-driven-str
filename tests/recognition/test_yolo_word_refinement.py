import numpy as np
import pytest
from recognition.yolo_word_refinement import crop_box, joint_score, nms, tile_windows, translated_slots


def reading(cid,box,text,score=.4,p=.9):
    return {'candidate_id':cid,'bbox':box,'raw_text':text,'detector_score':score,
            'token_probabilities':[p]*len(text),'token_log_probabilities':[float(np.log(p))]*len(text)}


def test_crop_variants_use_each_axis_and_clip():
    assert crop_box([10,20,110,60],(200,100),.1)==[0,16,120,64]
    assert crop_box([10,20,110,60],(200,100),-.1)==[20,24,100,56]
    with pytest.raises(ValueError):crop_box([0,0,10,10],(20,20),-.5)


def test_tiled_boxes_map_back_to_original_without_exif_rotation():
    windows=tile_windows((100,80),.2)
    assert windows==[[0,0,56,45],[44,0,100,45],[0,35,56,80],[44,35,100,80]]
    slots=[{'bbox':[2,3,12,13],'source_index':0,'score':.8}]
    assert translated_slots(slots,windows[-1],(100,80),'tile')[0]['bbox']==[46,38,56,48]


def test_joint_score_uses_ocr_and_empty_has_no_support():
    assert joint_score(reading('a',[0,0,10,10],'word',.4,.9))==pytest.approx(.6)
    assert joint_score(reading('a',[0,0,10,10],'',.9,.9))==0


def test_text_nms_keeps_different_readings_and_spatially_separate_same_words():
    rows=[reading('a',[0,0,10,10],'Hello',.8),reading('b',[0,0,10,10],'HELLO',.7),
          reading('c',[0,0,10,10],'World',.6),reading('d',[20,0,30,10],'Hello',.5)]
    assert [r['candidate_id'] for r in nms(rows,.5,same_text=True)]==['a','c','d']
    assert [r['candidate_id'] for r in nms(rows,.5)]==['a','d']
