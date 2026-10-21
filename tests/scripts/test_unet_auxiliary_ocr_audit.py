from scripts.audit_unet_auxiliary_ocr import independent_counts


def test_hungarian_audit_enforces_spatial_exact_and_one_to_one():
    gt=[dict(bbox=[0,0,10,10],text='One!'),dict(bbox=[20,0,30,10],text='two')]
    predictions=[dict(bbox=[0,0,10,10],text='ONE!'),
                 dict(bbox=[0,0,10,10],text='ONE!'),
                 dict(bbox=[20,0,30,10],text='wrong'),
                 dict(bbox=[40,0,50,10],text='two')]
    actual=independent_counts(predictions,gt)
    assert actual==dict(gt_words=2,proposals=4,geometry_iou_ge_50=2,exact_casefold_iou_ge_50=1)


def test_empty_outputs_keep_ground_truth_misses():
    assert independent_counts([], [dict(bbox=[0,0,1,1],text='x')])==dict(
        gt_words=1,proposals=0,geometry_iou_ge_50=0,exact_casefold_iou_ge_50=0)
