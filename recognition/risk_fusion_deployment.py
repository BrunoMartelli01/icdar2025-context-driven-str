"""Add a frozen post-fallback fusion policy to a trained risk-admission gate."""

from pathlib import Path

from recognition.fallback_word_fusion import replace_fallback_transcripts
from recognition.risk_admission_deployment import RiskAdmissionPipeline
from scripts.icdar_budget_data import read, sha


class RiskFusionPipeline:
    def __init__(self, run_root, *, verify_qualification=True):
        root = Path(run_root)
        if verify_qualification:
            q = read(root/'deployment_qualification.json')
            if q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256']:
                raise ValueError('Changed independent fusion qualification')
            for path, digest in q['artifact_sha256'].items():
                if sha(path) != digest:
                    raise ValueError('Changed qualified fusion artifact: '+path)
        self.plan = read(root/'resolved_configuration.json'); frozen = read(root/'selection_frozen.json')
        if (self.plan['mode'] != 'approximate' or not self.plan['risk_fusion'] or
                frozen['status'] != 'FROZEN' or frozen['test_used_for_selection']):
            raise ValueError('A TRAIN-only frozen approximate fusion policy is required')
        for path, digest in frozen['artifact_sha256'].items():
            if sha(root/path) != digest:
                raise ValueError('Changed frozen fusion artifact: '+path)
        self.parent = RiskAdmissionPipeline(self.plan['source_root'], verify_qualification=verify_qualification)
        if self.parent.choice != frozen['winner']:
            raise ValueError('The trained image gate changed after fusion selection')
        self.choice = self.parent.choice; self.winner = self.parent.winner
        self.complement = self.parent.complement; self.policy = frozen['fusion_policy']

    def observe(self, raw, emitted, image_wh, *, trace=None, map_summary=None):
        return self.parent.observe(raw, emitted, image_wh, trace=trace, map_summary=map_summary)

    def route(self, observations, rgb_hashes, *, fallback_images):
        return self.parent.route(observations, rgb_hashes, fallback_images=fallback_images)

    def combine_after_fallback(self, observation, fallback_rows):
        replaced = replace_fallback_transcripts(observation['rows'], fallback_rows, self.policy)
        return self.parent.combine_after_fallback(observation, replaced)
