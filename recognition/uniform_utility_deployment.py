"""Qualified annotation-free common utility/fusion deployment boundary."""

from pathlib import Path

import joblib

from recognition.uniform_lexical_risk import lexicon
from recognition.uniform_utility_fusion import score
from recognition.uniform_risk_fusion import route, combine
from scripts.run_uniform_unet_training import read
from segmentation.uniform_unet_fusion import file_hash
from recognition.uniform_encoder_context import projection, project
from recognition.uniform_word_union import add_union_rows, combine as combine_union, FEATURE_SCHEMA


class UniformUtilityPipeline:
    def __init__(self, config_path, dataset, fallback_images, *, split='test'):
        self.config=read(config_path)
        if self.config['mode']!='approximate' or split not in ('train','test'):
            raise ValueError('Explicit approximate inference resource split required')
        self.root=Path(self.config['output_root'])/dataset
        frozen=read(self.root/'selection_frozen.json'); audit=read(self.root/'independent_audit.json')
        if (frozen['status']!='FROZEN' or frozen['test_used_for_selection'] or audit['status']!='PASS' or
                audit['frozen_sha256']!=file_hash(self.root/'selection_frozen.json') or
                frozen['config_sha256']!=file_hash(config_path)):
            raise ValueError('Qualified frozen common utility pipeline required')
        self.policy=frozen['policies'][str(fallback_images)]
        self.fallback_images=fallback_images
        for name in ['word_model.joblib',self.policy['model']]:
            if file_hash(self.root/name)!=frozen['model_sha256'][name]:
                raise ValueError('Changed qualified deployed model')
        profile=self.config['lexicons'][dataset]
        if profile is not None:
            for field in ('vocabulary','pairs'):
                path=profile[split][field]
                if file_hash(path)!=self.config['pinned_inputs'][path]:
                    raise ValueError('Changed qualified inference vocabulary')
        self.dictionary=lexicon(self.config['lexicons'][dataset],split)
        self.lexical_policy=frozen['lexical_policy']
        self.word_model=joblib.load(self.root/'word_model.joblib')
        self.count_head=joblib.load(self.root/self.policy['model'])
        self.context_projection=projection(self.config)
        self.union_model=None
        if 'learned_word_union' in self.config:
            if self.config['learned_word_union'].get('feature_schema')!=FEATURE_SCHEMA:
                raise ValueError('Use the frozen source matching this word-union feature schema')
            name='union_model.joblib'
            if file_hash(self.root/name)!=frozen['model_sha256'][name]:raise ValueError('Changed learned union model')
            self.union_model=joblib.load(self.root/name)

    def observe(self, raw_rows, image_wh, predicted_probabilities, encoder_context=None):
        if self.context_projection is not None and encoder_context is None:
            raise ValueError('Actual shared U-Net encoder context required by this frozen gate')
        context=project(self.context_projection,encoder_context) if self.context_projection is not None else None
        observation=score(raw_rows,image_wh,predicted_probabilities,self.dictionary,self.lexical_policy,
            self.word_model,self.count_head,self.policy['admission_threshold'],self.policy['reference'],
            self.config['map_settings'],context)
        if self.union_model is not None:
            observation=add_union_rows(observation,raw_rows,self.dictionary,self.lexical_policy)
            observation['image_wh']=list(image_wh)
        return observation

    def route_batch(self, observations, rgb_hashes):
        return route(observations,rgb_hashes,self.fallback_images,self.policy['guard'])

    def combine_after_fallback(self, observation, fallback_rows):
        if self.union_model is not None and self.policy['learned_union_threshold'] is not None:
            return combine_union(observation,fallback_rows,observation['image_wh'],self.union_model,
                self.policy,self.config['learned_word_union'])
        return combine(observation,fallback_rows,self.policy['fusion'],self.policy['complement'])
