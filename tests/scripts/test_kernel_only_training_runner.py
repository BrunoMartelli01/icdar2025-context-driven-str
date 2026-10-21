from pathlib import Path
import json
import torch
import pytest
from scripts import train_kernel_only as m


def test_mode_isolation_and_actual_loss_metadata():
    cfg,_,source,_=m.read_config(m.DEFAULT_CONFIG,'approximate')
    tc=m.training_config(cfg)
    assert cfg['supervision']=='kernel_only' and tc.loss=='masked_BCE_Dice_kernel_only_v1'
    assert tc.learning_rate==1e-5 and cfg['epochs']==50
    with pytest.raises(ValueError,match='approximate'):m.read_config(m.DEFAULT_CONFIG,'strict')


def test_checkpoint_selection_uses_kernel_not_word_or_padding_metrics():
    a={'kernel_iou_macro':.6,'kernel_f1_macro':.7,'word_correct':1}
    b={**a,'word_correct':999}
    assert m.selection(a,2)==m.selection(b,2)
    assert m.selection(a,2)>m.selection(b,3)


def test_exact_optimizer_rng_resume_and_missing_epoch_log_recovery(tmp_path):
    torch.set_num_threads(1)
    cfg,_,source,_=m.read_config(m.DEFAULT_CONFIG,'approximate');cfg={**cfg,'epochs':3}
    class Epoch:
        def set_epoch(self,e):pass
    class Batches(list):
        dataset=[0,1]
    batch={'image':torch.ones(2,1,4,4),'kernel':torch.ones(2,1,4,4),'validity':torch.ones(2,1,4,4),'sample_id':['fixture0','fixture1']}
    loader=Batches([batch]);contract={'fixture':True,'supervision':'kernel_only'};device=torch.device('cpu')
    def model_and_optimizer():
        model=torch.nn.Sequential(torch.nn.Conv2d(1,1,1),torch.nn.Dropout2d(.3));model.architecture_identity='toy-kernel-fixture'
        return model,torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    for root in [tmp_path/'full',tmp_path/'resume']:root.mkdir()
    m.seed_everything(41,deterministic_algorithms=True);model,opt=model_and_optimizer()
    full=m.train_loop(tmp_path/'full',model,opt,loader,loader,Epoch(),Epoch(),cfg,source,contract,device)
    m.seed_everything(41,deterministic_algorithms=True);model,opt=model_and_optimizer()
    partial=m.train_loop(tmp_path/'resume',model,opt,loader,loader,Epoch(),Epoch(),cfg,source,contract,device,stop_after=2)
    assert not partial['training_complete'] and partial['completed_epochs']==2
    log=tmp_path/'resume/epochs.jsonl';log.write_text(log.read_text().splitlines()[0]+'\n')
    torch.rand(100);model,opt=model_and_optimizer()
    resumed=m.train_loop(tmp_path/'resume',model,opt,loader,loader,Epoch(),Epoch(),cfg,source,contract,device)
    assert resumed['training_complete'] and resumed['global_step']==full['global_step']==3
    a=torch.load(tmp_path/'full/last.pt',weights_only=False);b=torch.load(tmp_path/'resume/last.pt',weights_only=False)
    assert all(torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items())
    assert torch.equal(a['rng_state']['torch_cpu'],b['rng_state']['torch_cpu'])
    assert json.loads(log.read_text().splitlines()[1])['recovered_from_checkpoint']


def test_cpu_training_is_rejected_before_source_loading(tmp_path):
    args=type('Args',(),dict(config=m.DEFAULT_CONFIG,mode='approximate',output_root=tmp_path/'out',
        reference_run=tmp_path/'reference',word_root=tmp_path/'data',parseq_artifacts=tmp_path/'artifacts',device='cpu'))()
    with pytest.raises(RuntimeError,match='CUDA'):m.run(args)


def test_final_padding_boundary_and_qualified_cache_without_loading_ocr(tmp_path,monkeypatch):
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    out=tmp_path/'out';ref=tmp_path/'reference';data=tmp_path/'data'
    for p in [out,ref/'ocr_cache',data]:p.mkdir(parents=True)
    rgb=np.full((16,16,3),180,np.uint8);Image.fromarray(rgb).save(data/'sample.png')
    labels=np.zeros((16,16),np.int32);labels[6:10,6:10]=1
    folder=out/'validation_exports/fixture__one';folder.mkdir(parents=True)
    np.savez_compressed(folder/'maps.npz',labels=labels,validity=np.ones((16,16),bool))
    sample=SimpleNamespace(sample_id='fixture:one',image_path=Path('sample.png'),image_sha256=m.sha256_file(data/'sample.png'))
    manifest=SimpleNamespace(samples=[sample])
    old={}
    for pad in [0,2]:
        proposals=m.P.parseq_word_proposals(labels,(16,16),padding=pad,min_area=0,max_instances=None)
        x1,y1,x2,y2=proposals[0]['bbox'];digest=m.crop_array_hash(rgb[y1:y2,x1:x2].copy())
        record={'crop_rgb_sha256':digest,'parseq_checkpoint_sha256':m.PARSEQ_CHECKPOINT_SHA256,
            'parseq_source_archive_sha256':m.PARSEQ_RUNTIME_ARCHIVE_SHA256,'provenance':{'decode_ar':True,'refine_iters':1},'raw_text':'WORD'}
        (ref/'ocr_cache'/(digest+'.json')).write_text(json.dumps(record))
        for kind in ['kernel_only','text_kernel']:old[kind+'/pad'+str(pad)]={'aggregate':{'geometry_iou_ge_50':0,'exact_casefold_iou_ge_50':0}}
    (ref/'results.json').write_text(json.dumps({'arms':old}))
    monkeypatch.setattr(m,'_train_gt',lambda *a:([{'bbox':[4,4,12,12],'text':'word'}],'fixture-annotation'))
    monkeypatch.setattr(m,'_prepare_cuda_memory_tracking',lambda *a:None)
    monkeypatch.setattr(m,'memory',lambda *a:{'fixture':True})
    def forbidden(*a,**k):raise AssertionError('All fixture crops have qualified cache hits')
    monkeypatch.setattr(m,'ParseqApproxRecognizer',forbidden)
    result=m.padding_ocr(out,ref,manifest,[0],data,{'padding_native_pixels':[0,2],'ocr_batch_size':8},
        {'expected_validation_words':1},tmp_path/'unused',torch.device('cpu'))
    assert result['arms']['pad0']['aggregate']['geometry_iou_ge_50']==0
    assert result['arms']['pad2']['aggregate']['geometry_iou_ge_50']==1
    assert result['arms']['pad2']['aggregate']['exact_casefold_iou_ge_50']==1
    assert result['arms']['pad2']['aggregate']['unpadded_geometry_iou_ge_50']==0
