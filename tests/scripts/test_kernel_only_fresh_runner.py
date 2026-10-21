import json
from pathlib import Path
import torch
import pytest
from scripts import train_kernel_only_fresh as m


def test_fresh_config_and_mode_isolation():
    cfg,_,source,_=m.read_config(m.DEFAULT_CONFIG,'approximate')
    assert cfg['epochs']==100 and cfg['learning_rate']==1e-4
    assert not cfg['segmentation_checkpoint_initialization'] and 'source_checkpoint_sha256' not in cfg
    assert m.T.training_config(cfg).loss=='masked_BCE_Dice_kernel_only_v1'
    with pytest.raises(ValueError,match='approximate'):m.read_config(m.DEFAULT_CONFIG,'strict')


def test_warm_start_configuration_is_rejected(tmp_path,monkeypatch):
    cfg=json.loads(m.DEFAULT_CONFIG.read_text());cfg['source_checkpoint_sha256']='forbidden-segmentation-checkpoint'
    root=tmp_path/'configs/approximate';root.mkdir(parents=True);path=root/'wrong.json';path.write_text(json.dumps(cfg))
    monkeypatch.setattr(m,'REPO',tmp_path)
    with pytest.raises(ValueError,match='Fresh initialization'):m.read_config(path,'approximate')


def test_only_local_encoder_is_loaded_and_no_segmentation_checkpoint(tmp_path,monkeypatch):
    checkpoint=tmp_path/'imagenet-fixture.pt';torch.save({'weight':torch.full((1,1),.75)},checkpoint)
    cfg={'encoder_checkpoint_sha256':m.T.sha256_file(checkpoint)}
    class Toy(torch.nn.Module):
        def __init__(self,**kwargs):
            super().__init__();assert kwargs['encoder_pretrained'] is False
            self.encoder=torch.nn.Linear(1,1,bias=False);self.head=torch.nn.Linear(1,1)
        def initialize_encoder(self,state):
            self.encoder.load_state_dict(state,strict=True)
            return {'segmentation_checkpoint_loaded':False}
    monkeypatch.setattr(m,'FreshKernelAGUNet',Toy)
    original=torch.load;seen=[]
    def recorded(path,**kwargs):
        seen.append(Path(path));assert kwargs['weights_only'] is True
        return original(path,**kwargs)
    monkeypatch.setattr(m.torch,'load',recorded)
    model,info=m.build_fresh_model(cfg,{'skip_floor':.25},checkpoint)
    assert seen==[checkpoint] and torch.all(model.encoder.weight==.75)
    assert info['encoder_checkpoint_sha256']==cfg['encoder_checkpoint_sha256']
    assert len(info['decoder_head_sha256'])==64 and not info['segmentation_checkpoint_loaded']
    with pytest.raises(ValueError,match='checksum'):m.build_fresh_model({'encoder_checkpoint_sha256':'wrong'},{'skip_floor':.25},checkpoint)


def test_cpu_training_is_rejected(tmp_path):
    args=type('Args',(),dict(config=m.DEFAULT_CONFIG,mode='approximate',output_root=tmp_path/'out',reference_run=tmp_path/'ref',
        word_root=tmp_path/'data',parseq_artifacts=tmp_path/'artifacts',encoder_checkpoint=tmp_path/'encoder',device='cpu'))()
    with pytest.raises(RuntimeError,match='CUDA'):m.run(args)
