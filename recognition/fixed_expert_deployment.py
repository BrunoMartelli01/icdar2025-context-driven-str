"""Annotation-free inference for a qualified fixed-emission budget gate."""

from pathlib import Path

import joblib
import numpy as np

from recognition.common_budget_pipeline import choose_light
from recognition.fixed_expert_budget import (
    fixed_trace, observed_features, with_word_risk, fixed_words_with_risk,
    fixed_heuristic, count_scores, conditional_complement,
)
from scripts.icdar_budget_data import read, sha


class FixedExpertPipeline:
    def __init__(self, run_root, *, verify_qualification=True):
        root = Path(run_root)
        if verify_qualification:
            qualification = read(root/'deployment_qualification.json')
            if (qualification['status'] != 'PASS' or
                    sha(qualification['audit_path']) != qualification['audit_sha256']):
                raise ValueError('Changed fixed-expert independent qualification')
            for path, digest in qualification['artifact_sha256'].items():
                if sha(path) != digest:
                    raise ValueError('Changed qualified fixed-expert artifact: '+path)
        self.plan = read(root/'resolved_configuration.json')
        if self.plan['mode'] != 'approximate':
            raise ValueError('APPROXIMATE only')
        frozen = read(root/'selection_frozen.json')
        if frozen['status'] != 'FROZEN' or frozen['test_used_for_selection']:
            raise ValueError('A TRAIN-only frozen fixed-expert selection is required')
        for path, digest in frozen['artifact_sha256'].items():
            if sha(root/path) != digest:
                raise ValueError('Changed frozen fixed-expert artifact: '+path)
        self.choice = frozen['winner']; self.winner = self.choice['winner']
        self.complement = self.choice['complement']
        folder = root/'models'/self.choice['view']
        target = folder/self.choice['target_name']
        self.heads = [joblib.load(target/name) for name in self.winner['models']]
        self.needs_risk = (any('risk' in h['groups'] for h in self.heads) or
            self.winner['heuristic'] in ('expected_errors', 'weakest_word') or
            self.complement['confidence'] is not None)
        self.word_model = joblib.load(folder/'word_full.joblib') if self.needs_risk else None

    def observe(self, raw, emitted, image_wh, *, trace=None, map_summary=None):
        trace = fixed_trace(raw, emitted, trace)
        xword, groups = observed_features(raw, emitted, trace, image_wh, map_summary)
        probabilities = None
        if self.needs_risk:
            probabilities = (self.word_model.predict_proba(xword)[:, 1]
                             if len(xword) else np.empty(0))
            groups = with_word_risk(groups, probabilities, trace)
        if self.heads:
            matrices = {tuple(h['groups']): np.concatenate([groups[g] for g in h['groups']])[None]
                        for h in self.heads}
            score = count_scores(self.heads, matrices, [len(raw)+1], self.winner['reference'])[0]
        else:
            score = fixed_heuristic(raw, emitted, probabilities, self.winner['heuristic'])
        if not np.isfinite(score):
            raise ValueError('Nonfinite fixed-expert routing score')
        rows = (fixed_words_with_risk(emitted, probabilities)
                if probabilities is not None else [dict(r) for r in emitted])
        return dict(score=float(score), rows=rows,
                    probabilities=None if probabilities is None else probabilities.tolist())

    def route(self, observations, rgb_hashes, *, fallback_images):
        eligible = ([s for s, o in observations.items() if o['rows']]
                    if self.winner['guard'] else None)
        light = choose_light({s: o['score'] for s, o in observations.items()},
                             rgb_hashes, fallback_images, eligible)
        fallback = sorted(set(observations)-set(light))
        return dict(accepted_ids=sorted(light), fallback_ids=fallback,
                    deepsolo_images=len(fallback), total_images=len(observations),
                    deepsolo_percentage=100*len(fallback)/len(observations),
                    mode='approximate', scope='logical routing; online latency unmeasured')

    def combine_after_fallback(self, observation, fallback_rows):
        return conditional_complement(observation['rows'], fallback_rows, self.complement)
