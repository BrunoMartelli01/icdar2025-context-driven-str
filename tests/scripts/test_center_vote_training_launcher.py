from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.train_word_center_votes as runner
from scripts.train_word_center_offsets import evaluate_offsets as old_evaluator
from segmentation.training import build_optimizer, seed_everything
from segmentation.word_center_offsets import make_word_center_offsets
from segmentation.word_center_votes import params_from_config


def batch():
    owners = np.zeros((8,12),np.int32)
    owners[1:7,1:5] = 1;owners[1:7,7:11] = 2
    targets,_ = make_word_center_offsets(owners,np.ones_like(owners,bool))
    generator = torch.Generator().manual_seed(42)
    return {'image':torch.randn(2,3,8,12,generator=generator),'sample_id':['word:1','word:2'],
        'center_offsets':torch.from_numpy(targets)[None].repeat(2,1,1,1),
        'owners_native':torch.from_numpy(owners)[None].repeat(2,1,1),
        'validity':torch.ones(2,1,8,12),'expected_word_count':torch.tensor([2,2])}


def config(variant='coherent'):
    cfg,_ = runner.read_config(runner.DEFAULT_CONFIG,'approximate',variant)
    cfg['training'].update(epochs=3,batch_size=2)
    cfg['dataset'].update(images=4,validation_images=2,validation_expected_words=4)
    return cfg


class Tiny(torch.nn.Conv2d):
    architecture_identity = 'synthetic-center-vote-launcher-test'
    def __init__(self):super().__init__(3,2,1)


class EpochState:
    def set_epoch(self,value):self.epoch = value


def execute(path,cfg,stop_after_epoch=None):
    seed_everything(37,deterministic_algorithms=True)
    model = Tiny();optimizer = build_optimizer(model,runner.training_config(cfg));data = batch()
    path.mkdir(exist_ok=True)
    result = runner.train(path,model,optimizer,[data],[data],EpochState(),EpochState(),cfg,
        {'variant':'synthetic-center-votes','config':cfg},torch.device('cpu'),stop_after_epoch=stop_after_epoch)
    return result,model


def test_modes_namespace_and_matched_variants(tmp_path):
    control,_ = runner.read_config(runner.DEFAULT_CONFIG,'approximate','control')
    coherent,_ = runner.read_config(runner.DEFAULT_CONFIG,'approximate','coherent')
    assert control['training'] == coherent['training']
    assert control['dataset'] == coherent['dataset'] and control['voting'] == coherent['voting']
    assert control['architecture_identity'] == coherent['architecture_identity']
    tc = runner.training_config(coherent)
    assert tc.neighbor_weight == .5 and tc.separation_weight == 1.  # Distinct existing segmentation field.
    assert tc.loss == 'word_balanced_coherent_center_votes' and tc.offset_beta == .01
    assert runner.training_config(control).consensus_weight == runner.training_config(control).neighbor_weight == 0
    assert tc.checkpoint_selection == 'validation_offset_loss'
    with pytest.raises(ValueError,match='approximate'):runner.read_config(runner.DEFAULT_CONFIG,'strict','coherent')
    with pytest.raises(ValueError,match='fallback'):runner.read_config(tmp_path/'cfg.json','approximate','coherent')
    with pytest.raises(ValueError,match='variant'):runner.read_config(runner.DEFAULT_CONFIG,'approximate','missing')


def test_validation_zero_targets_empty_samples_and_export_compatibility(tmp_path):
    data = batch();data['owners_native'][1] = 0
    class Ideal(torch.nn.Module):
        def forward(self,image):return torch.atanh(data['center_offsets'])
    cfg = config()
    result = runner.evaluate_offsets(Ideal(),[data],torch.device('cpu'),params=params_from_config(cfg),export_dir=tmp_path)
    old = old_evaluator(Ideal(),[data],torch.device('cpu'),beta=.01)
    a = result['aggregate'];b = old['aggregate']
    assert a['loss'] < 1e-12 and a['endpoint_scatter_native_pixels'] < 1e-5 and a['wrong_neighbor_vote_rate'] == 0
    for name in ('offset_loss','endpoint_error_native_pixels','samples','supervised_images','empty_images','expected_words','surviving_words','lost_words'):
        assert a[name] == pytest.approx(b[name])
    assert a['expected_words'] == 4 and a['surviving_words'] == 2 and a['lost_words'] == 2
    assert result['per_image'][1]['offset_loss'] is None
    with np.load(tmp_path/'word__1/offsets.npz') as saved:
        assert set(saved.files) == {'normalized_offsets','target_offsets','owners','validity'}
        assert saved['normalized_offsets'].shape == (2,8,12)
    runner.evaluate_offsets(Ideal(),[data],torch.device('cpu'),params=params_from_config(cfg),export_dir=tmp_path)
    path = tmp_path/'word__1/offsets.npz';path.write_bytes(b'changed')
    with pytest.raises((ValueError,OSError)):
        runner.evaluate_offsets(Ideal(),[data],torch.device('cpu'),params=params_from_config(cfg),export_dir=tmp_path)


@pytest.mark.parametrize('variant',['control','coherent'])
def test_partial_resume_matches_full_and_saves_loss_provenance(tmp_path,variant):
    cfg = config(variant)
    full,fm = execute(tmp_path/'full',cfg)
    part,_ = execute(tmp_path/'resume',cfg,1)
    assert part['status'] == 'IN_PROGRESS' and not (tmp_path/'resume/run_manifest.json').exists()
    resumed,rm = execute(tmp_path/'resume',cfg)
    assert full['validation'] == resumed['validation'] and resumed['completed_epochs'] == 3
    assert full['checkpoint']['epoch'] == resumed['checkpoint']['epoch']
    assert all(torch.equal(v,rm.state_dict()[k]) for k,v in fm.state_dict().items())
    state = torch.load(tmp_path/'resume/last.pt',weights_only=False)
    assert state['training_config']['loss'] == 'word_balanced_coherent_center_votes'
    assert state['training_config']['neighbor_weight'] == cfg['loss']['neighbor_weight']


def test_checkpoint_before_log_recovers_without_repeating_epoch(tmp_path,monkeypatch):
    cfg = config();full,fm = execute(tmp_path/'full',cfg)
    append = runner._append_epoch
    def stop(*args):raise RuntimeError('synthetic atomic checkpoint interruption')
    monkeypatch.setattr(runner,'_append_epoch',stop)
    with pytest.raises(RuntimeError,match='interruption'):execute(tmp_path/'resume',cfg)
    monkeypatch.setattr(runner,'_append_epoch',append)
    resumed,rm = execute(tmp_path/'resume',cfg)
    assert full['validation'] == resumed['validation']
    assert all(torch.equal(v,rm.state_dict()[k]) for k,v in fm.state_dict().items())
    assert runner._epoch_log(tmp_path/'resume/epochs.jsonl')[0]['recovered_from_checkpoint']


def test_resume_rejects_different_neighbor_objective(tmp_path):
    cfg = config();execute(tmp_path/'run',cfg,1)
    changed = deepcopy(cfg);changed['loss']['neighbor_weight'] = .75
    with pytest.raises(ValueError,match='contract'):execute(tmp_path/'run',changed)


def test_selection_remains_original_regression_loss_with_earliest_ties(tmp_path,monkeypatch):
    cfg = config();evaluate = runner.evaluate_offsets
    def tied(*args,**kwargs):
        result = evaluate(*args,**kwargs);result['aggregate']['offset_loss'] = .5
        return result
    monkeypatch.setattr(runner,'evaluate_offsets',tied)
    result,_ = execute(tmp_path/'run',cfg)
    assert result['checkpoint']['epoch'] == 1


def test_cli_oom_propagates_with_loss_and_memory_metadata(tmp_path,monkeypatch):
    monkeypatch.setattr('sys.argv',['train','--mode','approximate','--variant','coherent',
        '--target-root',str(tmp_path),'--output-root',str(tmp_path),'--device','cpu','--action','smoke'])
    def fail(args):
        args._owns_output = True;args._resolved_cfg = config()
        raise torch.cuda.OutOfMemoryError('synthetic vote-loss OOM')
    monkeypatch.setattr(runner,'run',fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):runner.main()
    failure = json.loads((tmp_path/'failure.json').read_text())
    assert failure['oom'] and failure['variant'] == 'coherent' and failure['loss_parameters']['neighbor_weight'] == .5
    assert failure['model_set'].startswith('auxiliary') and failure['precision'] == 'float32'
    assert failure['batch_size'] == 2 and failure['input_shape'] == [2,3,512,512]
