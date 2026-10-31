from scripts.router_localization_control import nested_coverage


def test_one_merged_proposal_cannot_cover_two_gt_words_in_one_to_one_matching():
    row=dict(bbox=[0,0,20,10],text='a',sequence_confidence_native=.5)
    gt=[dict(bbox=[0,0,10,10],text='a'),dict(bbox=[10,0,20,10],text='b')]
    c=nested_coverage([row],gt,.5,.7)
    assert c['geometry_covered']==1 and c['no_geometric_coverage']==1
    assert c['text_correct_before_filter']==1 and c['correct_reading_removed_by_filter']==1
    assert c['retained_spatial_correct']==0


def test_nested_error_counts_partition_every_gt_without_changing_denominator():
    rows=[dict(bbox=[0,0,10,10],text='wrong',sequence_confidence_native=.99),
          dict(bbox=[20,0,30,10],text='B!',sequence_confidence_native=.7)]
    gt=[dict(bbox=[0,0,10,10],text='a'),dict(bbox=[20,0,30,10],text='b'),dict(bbox=[40,0,50,10],text='c')]
    c=nested_coverage(rows,gt,.5,.7)
    assert c['no_geometric_coverage']==1 and c['covered_but_no_correct_reading']==1
    assert c['retained_spatial_correct']==1
    assert sum(c[k] for k in ['no_geometric_coverage','covered_but_no_correct_reading','correct_reading_removed_by_filter','retained_spatial_correct'])==3
