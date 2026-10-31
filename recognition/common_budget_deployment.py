"""One GT-free deployment interface for every shared-pipeline dataset/expert."""
from pathlib import Path
import json
import hashlib
import joblib
import numpy as np

from recognition.common_budget_pipeline import (map_features,word_features,
    risk_features,image_features,predict_counts,choose_light)
from recognition.lexical_budget_router import correct_rows
from recognition.deepsolo_lexicon import DeepSoloLexicon


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CommonBudgetPipeline:
    """Consumes qualified light OCR/maps; requires no annotation or fallback output."""
    def __init__(self,root,case_name,*,vocabulary_split='test',verify_qualification=True):
        root=Path(root)
        if verify_qualification:
            qualification=read(root/'deployment_qualification.json')
            if qualification['status']!='PASS' or sha(root/qualification['audit_path'])!=qualification['audit_sha256']:
                raise ValueError('Missing or changed independent deployment qualification')
            for name,digest in qualification['artifact_sha256'].items():
                if sha(root/name)!=digest:raise ValueError('Changed qualified deployment artifact: '+name)
        self.plan=read(root/'resolved_configuration.json');self.case=read(root/'cases.json')[case_name]
        if self.plan['mode']!='approximate':raise ValueError('APPROXIMATE pipeline required')
        folder=root/'models'/case_name;frozen=read(folder/'selection_frozen.json');self.selection=frozen['winner']
        if sha(folder/'word_full.joblib')!=frozen['word_model_sha256']:raise ValueError('Changed word-risk weights')
        self.word_model=joblib.load(folder/'word_full.joblib');self.models=[]
        for name in self.selection['models']:
            if sha(folder/name)!=frozen['model_sha256'][name]:raise ValueError('Changed image-router weights')
            self.models.append(joblib.load(folder/name))
        self.policy=read(folder/'correction_frozen.json')['winner']['policy']
        profile=self.plan['config']['lexicons'].get(self.case['dataset']);self.lexicon=None
        if profile is not None:
            vocabulary=profile[vocabulary_split+'_vocabulary'];pairs=profile[vocabulary_split+'_pairs']
            for path in (vocabulary,pairs):
                if sha(path)!=self.plan['config']['pinned_inputs'][path]:raise ValueError('Changed vocabulary')
            if profile.get(vocabulary_split+'_pair_format')=='json':
                self.lexicon=DeepSoloLexicon(Path(vocabulary).read_text().splitlines(),read(pairs),distance_exclusive=profile['distance_exclusive'])
            else:self.lexicon=DeepSoloLexicon.from_files(vocabulary,pairs,distance_exclusive=profile['distance_exclusive'])

    def score(self,rows,image_wh,probabilities):
        if self.lexicon is None:
            corrected=[dict(r,source_row=i) for i,r in enumerate(rows)]
            trace=[dict(source_row=i,original_text=r['raw_text'],active=False,accepted=True,
                match=None,potential_match=None,changed=False,confidence=r['confidence']) for i,r in enumerate(rows)]
        else:corrected,trace=correct_rows(rows,self.lexicon,self.policy)
        maps=map_features(probabilities,image_wh,rows,self.plan['config']['map_features'])
        groups=image_features(rows,corrected,trace,image_wh,maps);x=word_features(rows,trace,image_wh)
        risk=self.word_model.predict_proba(x)[:,1] if len(x) else np.empty(0)
        groups['risk']=risk_features(risk,trace);predictions=[]
        for model in self.models:
            x=np.concatenate([groups[g] for g in model['groups']])[None]
            predictions.append(predict_counts(model,x)[0])
        pair=np.mean(predictions,axis=0);score=float(2*pair[0]-self.selection['reference']*pair[1])*(len(rows)+1)
        if not np.isfinite(score):raise ValueError('Nonfinite deployed utility')
        return dict(score=score,eligible=bool(rows) and all(r['raw_text'].strip() for r in rows),
            rows=corrected,trace=trace,word_risk=risk.tolist(),groups=groups)

    def route_batch(self,observations,tie_keys,*,fallback_images):
        scores={s:v['score'] for s,v in observations.items()}
        eligible=[s for s,v in observations.items() if v['eligible']] if self.selection['guard'] else list(scores)
        accepted=choose_light(scores,tie_keys,fallback_images,eligible);fallback=sorted(set(scores)-set(accepted))
        return dict(accepted_ids=accepted,fallback_ids=fallback,deepsolo_images=len(fallback),
            total_images=len(scores),deepsolo_percentage=100*len(fallback)/len(scores),
            light_rows={s:observations[s]['rows'] for s in accepted},
            scope='GT-free exact batch plan; caller invokes its authenticated dataset-specific DeepSolo backend')
