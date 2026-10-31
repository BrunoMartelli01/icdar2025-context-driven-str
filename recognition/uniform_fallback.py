"""Shared annotation-free DeepSolo word filtering and optional benchmark lexicon."""

import math

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.deepsolo_word_f1 import apply_policy
from scripts.infer_deepsolo_word_f1 import raw_predictions
from segmentation.uniform_unet_fusion import file_hash


class UniformFallbackPostprocessor:
    def __init__(self, profile):
        if profile['mode'] != 'approximate' or profile['claim_label'] != 'approximate reproduction':
            raise ValueError('Explicit approximate fallback profile required')
        self.profile = profile
        policy = profile['policy']
        if (policy['decoder'] != 'greedy' or policy['same_text_nms_iou'] is not None or
                policy['resolution'] != 'native' or
                not 0 <= policy['score_threshold'] <= 1 or
                not 0 <= policy['recognition_threshold'] <= 1):
            raise ValueError('Qualified single-view greedy fallback policy required')
        self.lexicon = None
        spec = profile['lexicon']
        if spec is not None:
            for name in ('vocabulary', 'pairs'):
                if file_hash(spec[name]) != spec[name+'_sha256']:
                    raise ValueError('Changed fallback lexicon: '+name)
            self.lexicon = DeepSoloLexicon.from_files(spec['vocabulary'], spec['pairs'],
                distance_exclusive=spec['distance_exclusive'])

    def apply(self, response, image_wh, orientation, checkpoint_sha256):
        if not response['success']:
            raise RuntimeError('DeepSolo failure must propagate before fusion')
        if checkpoint_sha256 != self.profile['checkpoint_sha256']:
            raise ValueError('Wrong qualified DeepSolo checkpoint')
        indices = [r['source_index'] for r in response['detections']]
        if len(indices) != len(set(indices)) or any(not math.isfinite(r['score']) or
                not 0 <= r['score'] <= 1 for r in response['detections']):
            raise ValueError('Invalid or duplicated DeepSolo detection')
        rows = raw_predictions(response['detections'], image_wh, orientation)
        if any(not math.isfinite(r['recognition_confidence']) for r in rows):
            raise ValueError('Invalid DeepSolo recognition confidence')
        rows = apply_policy(rows, self.profile['policy'])
        trace = []
        if self.lexicon is not None:
            rows, trace = self.lexicon.correct(rows, unmatched_policy=self.profile['lexicon']['unmatched_policy'])
        rows = [dict(r, raw_text='' if r.get('invalid_envelope') is not None else r['text'],
            word_source='deepsolo') for r in rows]
        return dict(rows=rows, lexical_trace=trace, ground_truth_used=False,
            checkpoint_sha256=checkpoint_sha256)
