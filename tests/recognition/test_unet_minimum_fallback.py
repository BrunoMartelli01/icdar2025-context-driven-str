"""Scientific checks for minimum-cost spatial word fallback."""
import itertools
import unittest

import numpy as np

from recognition.unet_minimum_fallback import (exact_counts,independent_counts,
    aggregate_counts,routing_curve,minimum_oracle,fallback_ids,observable_features)


class MatchingTests(unittest.TestCase):
    def test_punctuation_geometry_null_and_duplicates(self):
        gt=[dict(text='A',bbox=[0,0,10,10]),dict(text="B's",bbox=[20,0,30,10])]
        pred=[dict(text='a',bbox=[0,0,10,10]),dict(text='A',bbox=[0,0,10,10]),
              dict(text='Bs',bbox=[20,0,30,10]),dict(text="B's",bbox=None)]
        a=exact_counts(pred,gt,.5);b=independent_counts(pred,gt,.5)
        self.assertEqual((a['tp'],a['predictions'],a['gt']),(1,4,2))
        self.assertEqual({k:a[k] for k in b},b)

    def test_empty_outputs_and_empty_texts_remain_in_denominator(self):
        self.assertEqual(independent_counts([],[],.5),dict(tp=0,predictions=0,gt=0))
        m=exact_counts([dict(text='',bbox=[0,0,1,1])],[dict(text='',bbox=[0,0,1,1])],.5)
        self.assertEqual(m['tp'],0);self.assertEqual(m['predictions'],1)


class BudgetTests(unittest.TestCase):
    def test_fixed_rank_ties_and_boundary_budgets(self):
        self.assertEqual(fallback_ids({'b':.5,'a':.5,'c':.1},0),[])
        self.assertEqual(fallback_ids({'b':.5,'a':.5,'c':.1},1/3),['a'])
        self.assertEqual(fallback_ids({'b':.5,'a':.5,'c':.1},1),['a','b','c'])

    def test_curve_matches_direct_micro_counts(self):
        light={'a':dict(tp=1,predictions=3,gt=3),'b':dict(tp=1,predictions=2,gt=3)}
        heavy={'a':dict(tp=3,predictions=4,gt=3),'b':dict(tp=2,predictions=2,gt=3)}
        order,curve=routing_curve(light,heavy,{'a':.2,'b':.1},['b','a'])
        for k,p in enumerate(curve):
            rows=[heavy[s] if s in order[:k] else light[s] for s in ['b','a']]
            actual=aggregate_counts(rows,k,2)
            self.assertEqual(p['f1'],actual['f1']);self.assertEqual(p['deepsolo_images'],k)

    def test_oracle_minimum_matches_exhaustive_subsets(self):
        rng=np.random.default_rng(8)
        for _ in range(15):
            ids=list('abcdef')
            def branch():
                pred=int(rng.integers(1,8));tp=int(rng.integers(0,min(pred,4)+1))
                return dict(tp=tp,predictions=pred,gt=4)
            light={s:branch() for s in ids};heavy={s:branch() for s in ids}
            target=.6;possible=[]
            for k in range(len(ids)+1):
                for subset in itertools.combinations(ids,k):
                    m=aggregate_counts([heavy[s] if s in subset else light[s] for s in ids],k,len(ids))
                    if m['f1']>target:possible.append(k)
            result=minimum_oracle(light,heavy,ids,target)
            self.assertEqual(result['feasible'],bool(possible))
            if possible:self.assertEqual(result['deepsolo_images'],min(possible))

    def test_strict_target_equality_does_not_count_as_success(self):
        a={'a':dict(tp=1,predictions=2,gt=2)}
        self.assertFalse(minimum_oracle(a,a,['a'],.5)['feasible'])

    def test_oracle_can_keep_light_when_heavy_is_worse(self):
        a={'a':dict(tp=2,predictions=2,gt=2)};b={'a':dict(tp=0,predictions=2,gt=2)}
        self.assertEqual(minimum_oracle(a,b,['a'],.9)['deepsolo_images'],0)

    def test_missing_pairs_and_nonfinite_scores_fail(self):
        with self.assertRaises(ValueError):routing_curve({'a':{}},{},{'a':0.},['a'])
        with self.assertRaises(ValueError):routing_curve({'a':{}},{'a':{}},{'a':float('nan')},['a'])


class FeatureTests(unittest.TestCase):
    def test_features_use_only_RGB_CLIP_arrays_and_available_light_readings(self):
        p=dict(id='p',text='A',bbox=[0,0,10,10]);clip=dict(patches=np.ones((4,3)),patch_valid=np.ones(4,bool),word_text=np.zeros((1,5)))
        def get():return observable_features([20,20],[p],[p],{'p':.9},np.ones((1,2,3)),clip,[0.,.5,1.])
        before=get();clip['word_text'][:]=999
        after=get()
        np.testing.assert_array_equal(before['vision'],after['vision'])
        self.assertTrue(np.isfinite(after['tabular']).all())

    def test_empty_light_output_has_finite_missing_features(self):
        p=dict(id='p',text='A',bbox=[0,0,10,10]);clip=dict(patches=np.ones((4,3)),patch_valid=np.ones(4,bool))
        value=observable_features([20,20],[p],[],{'p':.9},np.ones((1,2,3)),clip,[0.,.5,1.])
        self.assertTrue(np.isfinite(value['vision']).all())

    def test_unknown_corrected_source_is_rejected(self):
        p=dict(id='p',text='A',bbox=[0,0,10,10]);q=dict(p,id='edit:x',source_proposal_id='unknown')
        clip=dict(patches=np.ones((4,3)),patch_valid=np.ones(4,bool))
        with self.assertRaises(ValueError):observable_features([20,20],[p],[q],{'p':.9},np.ones((1,2,3)),clip,[.5])


if __name__=='__main__':unittest.main()
