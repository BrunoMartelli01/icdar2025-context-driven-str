"""Annotation-free deployment of the trained admission and fixed-budget router."""

from pathlib import Path

import joblib
import numpy as np

from recognition.group_word_selection import reading_confidence
from recognition.lightweight_word_selection import proposals, feature_matrix, admit, reconcile
from recognition.lightweight_word_router import image_groups, image_score, exact_plan
from scripts.icdar_budget_data import read, sha


class LightweightWordPipeline:
    def __init__(self, router_root, *, arm=None, verify_qualification=True):
        root = Path(router_root)
        if verify_qualification:
            q = read(root/'deployment_qualification.json')
            if q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256']:
                raise ValueError('Changed independent lightweight qualification')
            for path, digest in q['artifact_sha256'].items():
                if sha(path) != digest:
                    raise ValueError('Changed qualified lightweight artifact: '+path)
        plan = read(root/'resolved_configuration.json')
        if plan['mode'] != 'approximate':
            raise ValueError('APPROXIMATE only')
        self.plan = plan; self.admission_plan = plan['admission_plan']
        frozen = read(root/'pipeline_frozen.json')
        self.arm = arm or frozen['choice']['arm']
        self.sources = self.admission_plan['config']['arms'][self.arm]
        admission = Path(plan['config']['admission_root'])/'models'/self.arm
        word = read(admission/'admission_frozen.json')
        if sha(admission/word['winner']['model']) != word['model_sha256']:
            raise ValueError('Changed word admission weights')
        self.word_policy = word['winner']
        self.word_model = joblib.load(admission/self.word_policy['model'])
        router = read(root/'models'/self.arm/'router_frozen.json')
        if sha(admission/'admission_frozen.json') != router['admission_frozen_sha256']:
            raise ValueError('Changed frozen admission policy')
        self.router = router['winner']; self.head = None
        if self.router['model']:
            path = root/'models'/self.arm/self.router['model']
            if sha(path) != router['model_sha256']:
                raise ValueError('Changed image routing weights')
            self.head = joblib.load(path)
        path = root/'models'/self.arm/'reconciliation_frozen.json'
        if sha(path) != router['reconciliation_sha256']:
            raise ValueError('Changed conditional fallback policy')
        self.complement = read(path)['winner']['policy']

    def observe(self, sources, image_wh, lexicon):
        if set(sources) != set(self.sources):
            raise ValueError('Exactly the selected source experts are required')
        raw = {}
        for name in self.sources:
            if self.admission_plan['config']['sources'][name]['kind'] == 'word_unet':
                raw[name] = [dict(r, confidence=reading_confidence(r)['confidence_balanced'])
                             for r in sources[name]]
            else:
                raw[name] = sources[name]
        rows = proposals(raw, lexicon)
        c = self.admission_plan['config']
        x = feature_matrix(rows, image_wh, c['source_names'], c['token_feature_cutoffs'])
        p = self.word_model.predict_proba(x)[:, 1] if len(rows) else np.empty(0)
        selected = admit(rows, p, threshold=self.word_policy['threshold'],
                         duplicate_iou=self.word_policy['duplicate_iou'])
        groups = image_groups(rows, selected, p, image_wh, c['token_feature_cutoffs'])
        scale = 1+sum(not r['lexical_alternative'] for r in rows)
        if self.head is not None:
            score = image_score(groups, self.head, self.router['reference'], scale)
        else:
            index = self.router['heuristic_index']
            score = float(groups['risk'][index])*(-1 if index in (1, 3) else 1)
        if not np.isfinite(score):
            raise ValueError('Nonfinite deployed lightweight score')
        return dict(score=score, rows=selected, proposals=rows,
                    probabilities=p.tolist(), groups=groups)

    def route(self, observations, rgb_hashes, *, fallback_images):
        return exact_plan({s: v['score'] for s, v in observations.items()},
                          rgb_hashes, fallback_images)

    def combine_after_fallback(self, observation, fallback_rows):
        return reconcile(observation['rows'], fallback_rows, **self.complement)
