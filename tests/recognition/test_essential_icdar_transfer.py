import numpy as np

from scripts.run_essential_icdar_transfer import accept,cut,ignored_rows,normalize_yolo
from recognition.unet_corrected_token_interfaces import lexical_counts


def test_unlabeled_calibration_cut_and_eligibility_ties():
    scores={'a':2.,'b':1.,'c':1.,'d':None}
    policy=cut(scores,.5)
    assert policy['status']=='SUCCESS' and accept(scores,policy)==['a','b']
    assert accept({'x':0.,'y':None,'a':1.},policy)==['a']
    unavailable=cut({'a':None,'b':1.,'c':None,'d':None},.5)
    assert unavailable['status']=='UNATTAINABLE_CALIBRATION_BUDGET'
    assert accept(scores,unavailable)==[]


def test_native_ignore_uses_predictions_only_and_retains_tiny_boxes():
    rows=[dict(bbox=[0,0,2,2]),dict(bbox=[5,0,10,10])]
    valid=np.ones((256,256),bool);valid[:,:64]=False
    assert ignored_rows(rows,[10,10],valid)==[True,False]
    assert ignored_rows(rows,[10,10],np.ones((256,256),bool))==[False,False]


def test_independent_audit_sampler_preserves_odd_rectangular_letterbox():
    from PIL import Image,ImageDraw
    from recognition.clip_router_evaluation import native_mask
    from scripts.audit_essential_thresholds import native_binary
    image=Image.new('L',(23,10),0);ImageDraw.Draw(image).rectangle([0,0,9,9],fill=255)
    actual=native_binary(image)
    np.testing.assert_array_equal(actual,native_mask(image,256))
    assert not actual[0].any() and not actual[-1].any()


def test_symbol_free_lexical_counts_repeats_empty_and_ignored():
    rows=[dict(raw_text='A!'),dict(raw_text='a'),dict(raw_text='***'),dict(raw_text='ignored')]
    gt=[dict(text='A'),dict(text='A'),dict(text='?')]
    result=lexical_counts(rows,gt,[False,False,False,True])
    assert (result['tp'],result['fp'],result['fn'],result['gt_words'],result['all_outputs'])==(2,1,1,3,4)


def test_yolo_balanced_confidence_and_frozen_joint_selection():
    def reading(cid,text,box,detector,p):
        return dict(candidate_id=cid,raw_text=text,bbox=box,detector_score=detector,
            token_log_probabilities=[np.log(p)]*len(text),token_probabilities=[p]*len(text))
    rows=[reading('a','A',[0,0,4,4],.9,.9),reading('b','A',[0,0,4,4],.8,.8),reading('c','B',[5,0,10,4],.1,.9)]
    cfg=dict(score_controls=[.4],reference_score=.4,joint_controls=[.6],merge_iou=.5,primary_method='joint_text_nms_0.6')
    selected=normalize_yolo(rows,cfg)
    assert len(selected)==1 and selected[0]['candidate_id']=='a'
    assert abs(selected[0]['confidence']-.9)<1e-12
