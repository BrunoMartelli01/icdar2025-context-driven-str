import pytest

from scripts.prepare_selected_deepsolo_export import selected_records


def test_selected_export_has_only_frozen_images_and_no_annotation_paths():
    records = [dict(sample_id='a', image_path='/a', image_sha256='1', annotation_path='/secret-a'),
               dict(sample_id='b', image_path='/b', image_sha256='2', annotation_path='/secret-b')]
    decision = dict(fallback_ids=['b'], accepted_ids=['a'], ground_truth_used=False,
                    fallback_outputs_loaded=False)
    assert selected_records(records, decision, 2, 1) == [
        dict(sample_id='b', image_path='/b', image_sha256='2')]
    for changed in (dict(ground_truth_used=True), dict(fallback_outputs_loaded=True),
                    dict(fallback_ids=['a'])):
        with pytest.raises(ValueError, match='GT-free'):
            selected_records(records, dict(decision, **changed), 2, 1)
