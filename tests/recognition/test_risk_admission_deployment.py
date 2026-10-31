import json

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from recognition.fixed_expert_budget import fixed_trace, observed_features
from recognition.risk_admission_deployment import RiskAdmissionPipeline
from tests.recognition.test_fixed_expert_budget import raw_word


def test_admission_uses_original_trace_once_and_routes_exact_budget(tmp_path):
    raw = [dict(raw_word('cat'), confidence=.99, detector_score=.99),
           dict(raw_word('dog', (10, 0, 20, 5)), confidence=.05, detector_score=.05)]
    emitted = [dict(row, source_row=i) for i, row in enumerate(raw)]
    x, _ = observed_features(raw, emitted, fixed_trace(raw, emitted), [30, 20], None)
    train_x = np.repeat(x, 30, axis=0); target = np.repeat([1, 0], 30)
    model = HistGradientBoostingClassifier(max_iter=30, min_samples_leaf=2, random_state=1).fit(train_x, target)
    folder = tmp_path/'models/oof'; folder.mkdir(parents=True)
    joblib.dump(model, folder/'word_full.joblib')
    (tmp_path/'resolved_configuration.json').write_text(json.dumps(dict(mode='approximate', risk_admission=True)))
    winner = dict(models=[], heuristic='expected_errors', guard=True, admission_threshold=.5)
    choice = dict(view='oof', winner=winner, complement=dict(confidence=None, maximum_overlap=.1))
    (tmp_path/'selection_frozen.json').write_text(json.dumps(dict(status='FROZEN',
        test_used_for_selection=False, artifact_sha256={}, winner=choice)))
    gate = RiskAdmissionPipeline(tmp_path, verify_qualification=False)
    observed = gate.observe(raw, emitted, [30, 20]); empty = gate.observe([], [], [30, 20])
    assert [row['raw_text'] for row in observed['rows']] == ['cat']
    np.testing.assert_allclose(observed['probabilities'], model.predict_proba(x)[:, 1])
    decision = gate.route(dict(a=observed, b=empty), dict(a='hash1', b='hash2'), fallback_images=1)
    assert decision['accepted_ids'] == ['a'] and decision['fallback_ids'] == ['b']
    assert decision['deepsolo_images'] == 1 and decision['total_images'] == 2
