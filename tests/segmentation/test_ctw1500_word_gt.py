import pytest
from segmentation.ctw1500_word_gt import parse_vimts_word_gt
from segmentation.ctw1500_word_gt import restore_split_single_token_lines
from segmentation.ctw1500_word_gt import remove_redundant_word_annotations
from recognition.scene_text_datasets import SceneTextInstance


def test_variable_vertex_polygons_preserve_word_identity_and_ignored_flags():
    rows = ('1,1,8,1,8,9,1,9,####DOUGL\n'
            '11,1,20,1,20,9,11,9,####ASTON\n'
            '25,1,26,1,27,2,27,9,26,10,25,9,#######\n')
    words = parse_vimts_word_gt(rows)
    assert [x.transcription for x in words] == ["DOUGL", "ASTON", "###"]
    assert all(x.granularity == "word" for x in words)
    assert [len(x.polygon_xy) for x in words] == [4,4,6]
    assert [x.ignore for x in words] == [False,False,True]


@pytest.mark.parametrize('rows', ['1,2,3,####word', '1,1,8,1,8,9,1,9,####two words',
                                  '1,1,8,1,8,9,1,9,word', ''])
def test_malformed_or_line_level_annotations_fail(rows):
    with pytest.raises(ValueError):
        parse_vimts_word_gt(rows)


def test_split_single_word_recovers_original_geometry_and_text():
    words=parse_vimts_word_gt('1,1,8,1,8,8,1,8,####DOUGL\n10,1,19,1,19,8,10,8,####ASTON\n')
    polygon=((0,0),(3,0),(6,0),(9,0),(12,0),(16,0),(20,0),
             (20,9),(16,9),(12,9),(9,9),(6,9),(3,9),(0,9))
    source=SceneTextInstance(polygon,'DOUGLASTON',False,'text_line')
    recovered,audit=restore_split_single_token_lines(words,(source,))
    assert len(recovered)==1 and recovered[0].transcription=='DOUGLASTON'
    assert recovered[0].polygon_xy==polygon and recovered[0].granularity=='word'
    assert audit['merged_original_single_token_lines']==1


def test_different_words_or_conflicting_text_cannot_be_merged():
    words=parse_vimts_word_gt('1,1,8,1,8,8,1,8,####ONE\n10,1,19,1,19,8,10,8,####TWO\n')
    polygon=((0,0),(3,0),(6,0),(9,0),(12,0),(16,0),(20,0),
             (20,9),(16,9),(12,9),(9,9),(6,9),(3,9),(0,9))
    for text in ['ONE TWO','DIFFERENT']:
        out,audit=restore_split_single_token_lines(words,(SceneTextInstance(polygon,text,False,'text_line'),))
        assert out==words and audit['merged_original_single_token_lines']==0


def test_only_duplicate_geometry_and_fully_contained_text_fragments_are_removed():
    rows=('1,1,20,1,20,8,1,8,####1862-64.\n'
          '12,2,19,2,19,7,12,7,####64\n'
          '12,2,19,2,19,7,12,7,####64\n'
          '25,2,31,2,31,7,25,7,####64\n')
    out,audit=remove_redundant_word_annotations(parse_vimts_word_gt(rows),(40,10))
    assert [x.transcription for x in out]==['1862-64.','64']
    assert audit['exact_duplicates_removed']==1
    assert audit['contained_lexical_fragments_removed']==1
    unrelated=parse_vimts_word_gt('1,1,20,1,20,8,1,8,####foo\n3,2,7,2,7,7,3,7,####bar\n')
    assert remove_redundant_word_annotations(unrelated,(40,10))[0]==unrelated


def test_nested_alternative_boxes_for_same_word_keep_tighter_region():
    words=parse_vimts_word_gt('1,1,20,1,20,8,1,8,####A\n4,2,10,2,10,7,4,7,####A\n')
    out,audit=remove_redundant_word_annotations(words,(30,10))
    assert len(out)==1 and out[0].bbox_xyxy==(4,2,10,7)
    assert audit['equal_text_nested_duplicates_removed']==1
