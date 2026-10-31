"""Predeclared TRAIN-only resolution and epoch tie breaks."""

from scripts.screen_icdar2015_unet_resolutions import select_larger
from scripts.select_icdar2015_improvement import choose
from segmentation.icdar2015_improvement_runtime import selection_key


def metric(f1, recall=.5, fused=10, iou=.5):
    return {"proposal_f1_iou_50":f1,"word_recall_iou_50":recall,
        "gt_words_touched_by_fused_components":fused,"foreground_iou":iou}


def test_resolution_excludes_512_and_prefers_f1_then_recall_then_cost():
    config={"resolutions":{"square512":[512,512],"square768":[768,768],
        "square1024":[1024,1024],"wide1024":[576,1024]}}
    rows={name:{"aggregate":metric(.5)} for name in config["resolutions"]}
    rows["square512"]["aggregate"]=metric(1.)
    assert select_larger(rows,config)=="square768" # same pixels as wide1024: name tie
    rows["wide1024"]["aggregate"]=metric(.5,.6)
    assert select_larger(rows,config)=="wide1024"
    rows["square1024"]["aggregate"]=metric(.51,.1)
    assert select_larger(rows,config)=="square1024"


def test_epoch_prefers_geometry_then_less_fusion_and_retains_epoch_zero_on_ties():
    assert selection_key(metric(.5),0)>selection_key(metric(.5),1)
    assert selection_key(metric(.5,fused=9),5)>selection_key(metric(.5,fused=10),0)
    assert selection_key(metric(.51,recall=.1,fused=50),10)>selection_key(metric(.5),0)


def test_final_selection_uses_correct_words_before_geometry_and_cost():
    rows={"a":{"word_f1":.3,"geometry_f1":.8,"rgb_pixels":100},
          "b":{"word_f1":.31,"geometry_f1":.4,"rgb_pixels":200}}
    assert choose(rows)=="b"
    rows["a"]["word_f1"]=.31
    assert choose(rows)=="a"
    rows["b"]["geometry_f1"]=.8
    assert choose(rows)=="a"
