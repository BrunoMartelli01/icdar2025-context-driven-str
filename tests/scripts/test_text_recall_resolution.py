import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import scripts.run_text_recall_resolution as runner
from segmentation.training import build_optimizer, seed_everything, load_checkpoint


def test_explicit_approximate_namespace_and_contract_no_overwrite(tmp_path):
    with pytest.raises(ValueError,match="approximate"):runner.read_config(runner.DEFAULT_CONFIG,'strict')
    outside=tmp_path/'profile.json';outside.write_text(runner.DEFAULT_CONFIG.read_text())
    with pytest.raises(ValueError,match="approximate"):runner.read_config(outside,'approximate')
    path=tmp_path/'run_contract.json';runner.fixed_json(path,{'betas':(.9,.999)})
    runner.fixed_json(path,{'betas':[.9,.999]})
    before=path.read_bytes()
    with pytest.raises(ValueError,match="NEW output"):runner.fixed_json(path,{'betas':[.8,.999]})
    assert path.read_bytes()==before


@pytest.mark.parametrize('field,value',[('loss',{'fn_weight':float('nan')}),('training',{'precision':'float16'}),
    ('evaluation',{'canonical_size':384}),('evaluation',{'missing_word':'omit'}),('training',{'epochs':0})])
def test_profile_tampering_fails_before_execution(tmp_path,monkeypatch,field,value):
    cfg=json.loads(runner.DEFAULT_CONFIG.read_text());cfg[field].update(value)
    source=tmp_path/cfg['source_config'];source.parent.mkdir(parents=True);source.write_bytes(runner.SOURCE_PATH.read_bytes())
    monkeypatch.setattr(runner,'REPO',tmp_path);monkeypatch.setattr(runner,'SOURCE_PATH',source)
    path=source.parent/'test.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):runner.read_config(path,'approximate')


def test_output_cannot_overlap_old_artifacts(tmp_path):
    target=tmp_path/'data';base=tmp_path/'old'/'text'
    for path in (target,target/'run',base,base.parent/'other',tmp_path):
        with pytest.raises(ValueError):runner.guard_output(path,target,base)
    runner.guard_output(tmp_path/'new',target,base)


class TinyModel(torch.nn.Module):
    architecture_identity='unit-test-recall'
    def __init__(self):
        super().__init__();self.head=torch.nn.Conv2d(3,1,1)
    def forward(self,x):return self.head(F.avg_pool2d(x,2))


def setup_training():
    cfg,_=runner.read_config(runner.DEFAULT_CONFIG,'approximate');cfg=copy.deepcopy(cfg)
    cfg['training'].update(epochs=2,batch_size=1);cfg['evaluation']['canonical_size']=8
    cfg['dataset'].update(validation_images=1,validation_words=1)
    mask=torch.zeros(1,8,8);mask[:,2:6,2:6]=1
    sample={'sample_id':'test:1','image':torch.ones(3,16,16),'mask':mask,
            'validity':torch.ones_like(mask),'separation':torch.zeros_like(mask)}
    loader=DataLoader([sample],batch_size=1)
    ref={'test:1':dict(target=mask[0].numpy().astype(bool),validity=np.ones((8,8),bool),
        canvas=np.ones((8,8),bool),owners=mask[0].numpy().astype(np.int32),expected_owner_ids=[1])}
    noop=type('Epoch',(),{'set_epoch':lambda self,epoch:None})()
    return cfg,loader,ref,noop


def test_resume_matches_uninterrupted_model_and_recovers_unlogged_checkpoint(tmp_path):
    cfg,loader,refs,noop=setup_training();device=torch.device('cpu');branch='tversky512';contract={'test':'resume'}
    full=tmp_path/'full';resumed=tmp_path/'resumed';full.mkdir();resumed.mkdir()
    def objects():
        seed_everything(17,deterministic_algorithms=True)
        model=TinyModel();optimizer=build_optimizer(model,runner.training_config(cfg,branch))
        return model,optimizer
    model,opt=objects()
    result=runner.train(full,model,opt,[loader,loader],refs,noop,noop,cfg,branch,contract,device)
    assert result['status']=='SUCCESS'
    model,opt=objects()
    partial=runner.train(resumed,model,opt,[loader,loader],refs,noop,noop,cfg,branch,contract,device,1)
    assert partial['status']=='IN_PROGRESS'
    # Simulate interruption after the atomic last.pt commit, before log append.
    (resumed/'epochs.jsonl').write_text('')
    model,opt=objects()
    result=runner.train(resumed,model,opt,[loader,loader],refs,noop,noop,cfg,branch,contract,device)
    a=load_checkpoint(full/'last.pt',model=TinyModel(),expected_architecture=TinyModel.architecture_identity)
    b=load_checkpoint(resumed/'last.pt',model=TinyModel(),expected_architecture=TinyModel.architecture_identity)
    assert result['status']=='SUCCESS' and runner._epoch_log(resumed/'epochs.jsonl')[0]['recovered_from_checkpoint']
    assert all(torch.equal(a['model_state_dict'][k],v) for k,v in b['model_state_dict'].items())
    assert a['validation_metrics']==b['validation_metrics']


def test_validation_predictions_do_not_use_gt_and_exports_reuse(tmp_path):
    cfg,loader,refs,noop=setup_training();model=TinyModel()
    with torch.no_grad():model.head.weight.zero_();model.head.bias.fill_(3)
    image=tmp_path/'image.jpg'
    from PIL import Image
    Image.new('RGB',(40,20),'white').save(image)
    kwargs=dict(export_dir=tmp_path/'exports',image_paths={'test:1':image},export_threshold=.5)
    a=runner.evaluate(model,loader,refs,torch.device('cpu'),cfg,**kwargs)
    b=runner.evaluate(model,loader,refs,torch.device('cpu'),cfg,**kwargs)
    assert a==b
    recipe=json.loads((tmp_path/'exports/test__1/proposals.json').read_text())
    assert recipe['proposals'][0]['bbox']==[0,0,40,20]
    refs['test:1']['owners'][:]=0;refs['test:1']['target'][:]=False
    runner.evaluate(model,loader,refs,torch.device('cpu'),cfg,export_dir=tmp_path/'changed',
        image_paths={'test:1':image},export_threshold=.5)
    other=json.loads((tmp_path/'changed/test__1/proposals.json').read_text())
    assert recipe['proposals']==other['proposals']


def test_oom_is_recorded_and_propagated_without_batch_fallback(tmp_path,monkeypatch):
    import sys
    output=tmp_path/'run';output.mkdir()
    def failing_run(args):
        args._owns_output=True;args._active_branch='control768'
        raise torch.cuda.OutOfMemoryError('synthetic OOM: no CUDA qualification')
    monkeypatch.setattr(runner,'run',failing_run)
    monkeypatch.setattr(sys,'argv',['run_text_recall_resolution','--mode','approximate',
        '--target-root',str(tmp_path/'data'),'--base-run',str(tmp_path/'base'),
        '--output-root',str(output),'--action','smoke','--device','cpu'])
    with pytest.raises(torch.cuda.OutOfMemoryError):runner.main()
    failure=json.loads(next(output.glob('failure-*.json')).read_text())
    assert failure['oom'] and failure['input_shape']==[4,3,768,768]
    assert failure['peak_allocated_bytes'] is None and failure['batch_size']==4


def test_cuda_tracking_allocates_before_reset_and_uses_integer_index(monkeypatch):
    events=[];initialized=False
    def allocate(*args,**kwargs):
        nonlocal initialized
        initialized=True;events.append(('allocate',kwargs['device']))
    def reset(index):
        assert initialized and index==0 and type(index) is int
        events.append(('reset',index))
    monkeypatch.setattr(torch.cuda,'set_device',lambda device:events.append(('select',device)))
    monkeypatch.setattr(torch,'empty',allocate)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:events.append(('synchronize',device)))
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:events.append(('empty_cache',None)))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',reset)
    assert runner._prepare_cuda_memory_tracking(torch.device('cuda:0'))==0
    assert [e[0] for e in events]==['select','allocate','synchronize','empty_cache','reset']


def test_cuda_initialization_failure_is_propagated(monkeypatch):
    def fail(device):raise RuntimeError('CUDA setup failed')
    monkeypatch.setattr(torch.cuda,'set_device',fail)
    with pytest.raises(RuntimeError,match='CUDA setup failed'):
        runner._prepare_cuda_memory_tracking(torch.device('cuda:0'))
