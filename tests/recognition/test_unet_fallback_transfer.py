"""Regression for array/sample alignment during frozen-router transfer."""
import unittest
from scripts.analyze_unet_fallback_transfer import aligned_sample_ids

class TransferAlignmentTests(unittest.TestCase):
    def test_array_order_follows_population_not_sorted_json_keys(self):
        samples=[{'sample_id':'total_text:2'},{'sample_id':'total_text:10'}]
        rows={'total_text:10':{},'total_text:2':{}}
        self.assertEqual(aligned_sample_ids(samples,rows,2),['total_text:2','total_text:10'])

    def test_missing_duplicated_or_wrong_size_population_fails(self):
        for samples,rows,n in [([{'sample_id':'a'}],{'b':{}},1),
            ([{'sample_id':'a'},{'sample_id':'a'}],{'a':{}},2),([{'sample_id':'a'}],{'a':{}},2)]:
            with self.assertRaises(ValueError):aligned_sample_ids(samples,rows,n)

if __name__=='__main__':unittest.main()
