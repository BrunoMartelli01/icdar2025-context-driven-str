"""Held-out validation identities and cuts for an unchanged deployed router."""
from __future__ import annotations

import hashlib

from recognition.unet_token_routing import fit_thresholds


def expert_holdout_ids(ordered_ids, seed, count, expected_sha256):
    if len(set(ordered_ids)) != len(ordered_ids) or not 0 < count < len(ordered_ids):
        raise ValueError('Unique expert population and nonempty holdout required')
    selected = set(sorted(ordered_ids, key=lambda s:(hashlib.sha256(f'{seed}:{s}'.encode()).digest(),s))[:count])
    values = [s for s in ordered_ids if s in selected]
    digest = hashlib.sha256(''.join(s+'\n' for s in values).encode()).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Expert holdout does not reproduce its qualified identity')
    return values


def independent_partitions(samples, validation_ids, test_samples):
    by_id = {s['sample_id']:s for s in samples}
    if len(by_id)!=len(samples) or not set(validation_ids)<=set(by_id):
        raise ValueError('Validation identities are missing or duplicated')
    validation = sorted(validation_ids)
    fit = sorted(set(by_id)-set(validation))
    test = sorted(s['sample_id'] for s in test_samples)
    if not fit or not validation or not test:
        raise ValueError('All three populations must be nonempty')
    sets = [{by_id[s]['rgb_sha256'] for s in members} for members in (fit,validation)]
    sets.append({s['rgb_sha256'] for s in test_samples})
    if any(sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError('RGB leakage across fit, validation or TEST')
    return dict(fit=fit,validation=validation,test=test)


def validation_cuts(scores, fractions, model_sha256):
    if not model_sha256:
        raise ValueError('Bind validation cuts to deployed weights')
    return [dict(action=p['action'], threshold=p['threshold'],
                 validation_target_fraction=p['train_target_fraction'],
                 model_sha256=model_sha256, selected_on='VALIDATION')
            for p in fit_thresholds(scores,fractions)]


def require_same_model(policy, actual_sha256):
    if policy['selected_on']!='VALIDATION' or policy['model_sha256']!=actual_sha256:
        raise ValueError('Validation threshold cannot transfer to refitted or different weights')
