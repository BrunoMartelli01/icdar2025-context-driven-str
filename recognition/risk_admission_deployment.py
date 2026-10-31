"""GT-free risk admission and image routing using one trained word model."""

from pathlib import Path

import joblib
import numpy as np

from recognition.common_budget_pipeline import choose_light
from recognition.fixed_expert_budget import (
    fixed_trace, observed_features, fixed_words_with_risk,
    fixed_heuristic, conditional_complement,
)
from scripts.icdar_budget_data import read, sha


class RiskAdmissionPipeline:
    def __init__(self, run_root, *, verify_qualification=True):
        root = Path(run_root)
        if verify_qualification:
            q = read(root/'deployment_qualification.json')
            if q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256']:
                raise ValueError('Changed independent risk-admission qualification')
            for path, digest in q['artifact_sha256'].items():
                if sha(path) != digest:
                    raise ValueError('Changed qualified risk-admission artifact: '+path)
        self.plan = read(root/'resolved_configuration.json')
        frozen = read(root/'selection_frozen.json')
        if (self.plan['mode'] != 'approximate' or not self.plan['risk_admission'] or
                frozen['status'] != 'FROZEN' or frozen['test_used_for_selection']):
            raise ValueError('A TRAIN-only frozen approximate risk-admission gate is required')
        for path, digest in frozen['artifact_sha256'].items():
            if sha(root/path) != digest:
                raise ValueError('Changed frozen risk-admission artifact: '+path)
        self.choice = frozen['winner']; self.winner = self.choice['winner']
        if self.winner['heuristic'] not in ('expected_errors', 'weakest_word') or self.winner['models']:
            raise ValueError('Risk admission requires an explicitly trained word-risk gate')
        self.complement = self.choice['complement']
        self.word_model = joblib.load(root/'models'/self.choice['view']/'word_full.joblib')

    def observe(self, raw, emitted, image_wh, *, trace=None, map_summary=None):
        # Word probabilities are computed once on the original emission trace.
        # Applying a threshold does not feed the filtered trace back into the model.
        trace = fixed_trace(raw, emitted, trace)
        x, _ = observed_features(raw, emitted, trace, image_wh, None)
        p = self.word_model.predict_proba(x)[:, 1] if len(x) else np.empty(0)
        all_words = fixed_words_with_risk(emitted, p)
        rows = [r for r in all_words if r['word_probability'] >= self.winner['admission_threshold']]
        score = fixed_heuristic(raw, rows, p, self.winner['heuristic'])
        if not np.isfinite(score):
            raise ValueError('Nonfinite risk-admission route score')
        return dict(score=float(score), rows=rows, probabilities=p.tolist())

    def route(self, observations, rgb_hashes, *, fallback_images):
        eligible = ([s for s, o in observations.items() if o['rows']]
                    if self.winner['guard'] else None)
        light = choose_light({s: o['score'] for s, o in observations.items()},
                             rgb_hashes, fallback_images, eligible)
        fallback = sorted(set(observations)-set(light))
        return dict(accepted_ids=sorted(light), fallback_ids=fallback,
            deepsolo_images=len(fallback), total_images=len(observations),
            deepsolo_percentage=100*len(fallback)/len(observations), mode='approximate',
            scope='logical routing; online latency unmeasured')

    def combine_after_fallback(self, observation, fallback_rows):
        return conditional_complement(observation['rows'], fallback_rows, self.complement)
