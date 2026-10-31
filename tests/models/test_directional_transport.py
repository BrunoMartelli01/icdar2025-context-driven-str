import pytest
import torch
from torch.nn import functional as F

from models.adaspot_gap import AdaSpotGapAGUNet
from models.directional_transport_unet import DirectionalTransport, DirectionalTransportUNet, shifted
from segmentation.directional_transport import balanced_edge_loss, transform
from segmentation.transforms import SegmentationTransformConfig
from recognition.directional_transport_metrics import aggregate, counts, normalize


@pytest.mark.parametrize('dy,dx', [(0,1),(0,-1),(1,0),(-1,0),(2,-2)])
def test_shift_has_no_wrap_and_correct_neighbor(dy, dx):
    x = torch.arange(20.).reshape(1,1,4,5)
    y, valid = shifted(x,dy,dx)
    for row in range(4):
        for col in range(5):
            inside = 0 <= row+dy < 4 and 0 <= col+dx < 5
            assert bool(valid[0,0,row,col]) == inside
            assert y[0,0,row,col] == (x[0,0,row+dy,col+dx] if inside else 0)


def test_transport_identity_and_supervision_gradients():
    x = torch.randn(1,8,8,8,requires_grad=True)
    block = DirectionalTransport(8,4,(1,2),spatial=True)
    y, edges = block(x,return_edges=True)
    assert torch.equal(x,y)
    owner = torch.zeros(1,8,8,dtype=torch.long)
    owner[:,:4,:4] = 1
    owner[:,:4,4:] = 2
    loss = sum(balanced_edge_loss(z,owner,torch.ones(1,1,8,8),dy,dx,v) for z,v,dy,dx in edges)
    (y.square().mean()+loss).backward()
    assert block.edge[0].weight.grad.abs().sum() > 0
    assert block.gain.grad is not None and torch.isfinite(block.gain.grad).all()
    for z,v,dy,dx in edges:
        assert torch.isfinite(balanced_edge_loss(z,owner,torch.zeros(1,1,8,8),dy,dx,v))


def test_edge_targets_separate_distinct_owner_and_exclude_background():
    owner = torch.tensor([[[1,1,2,2],[0,0,0,0]]])
    correct = torch.tensor([[[[8.,-8.,8.,0.],[8.,8.,8.,8.]]]])
    _, border = shifted(correct,0,1)
    loss = balanced_edge_loss(correct,owner,torch.ones_like(correct),0,1,border)
    wrong = balanced_edge_loss(-correct,owner,torch.ones_like(correct),0,1,border)
    assert loss < .001 and wrong > 7.


def test_warm_start_full_resolution_identity_and_parameter_budget():
    torch.set_num_threads(2)
    source = AdaSpotGapAGUNet('control',skip_floor=.25,initial_gap_bias=-2.).eval()
    model = DirectionalTransportUNet(skip_floor=.25,initial_gap_bias=-2.,widths=(32,24,16,12),
        radii=(1,4),detail_width=16,spatial=True).eval()
    model.initialize_source(source.state_dict())
    n = sum(p.numel() for p in source.parameters())
    assert n == 58155146
    assert n < sum(p.numel() for p in model.parameters()) <= n*1.1
    x = torch.randn(1,3,128,128)
    with torch.no_grad():
        assert torch.equal(model(x),F.interpolate(source(x),size=(128,128),mode='nearest'))
    model.train()
    assert not model.encoder.training
    assert not any(p.requires_grad for p in model.encoder.parameters())
    with pytest.raises(ValueError,match='Every original'):
        model.initialize_source({})


def test_matched_global_control_uses_same_parameter_tensors():
    a = DirectionalTransport(16,8,(1,4),spatial=True)
    b = DirectionalTransport(16,8,(1,4),spatial=False)
    assert [(k,v.shape) for k,v in a.state_dict().items()] == [(k,v.shape) for k,v in b.state_dict().items()]


def test_full_resolution_transform_does_not_relax_original_namespace():
    c = dict(seed=2,augmentation=dict(horizontal_flip_probability=0.,rotation_probability=0.,
        rotation_degrees=[0.,0.],canvas_fill_rgb=[124,116,104]))
    paired = transform(c,native_size=512,train=False)
    from PIL import Image
    image, mask = paired(Image.new('RGB',(40,20)),Image.new('L',(40,20)),0)
    assert image.shape == (3,512,512) and mask.shape == (1,512,512)
    from dataclasses import asdict
    values = asdict(paired.config)
    values['classification'] = 'PRACTICAL_APPROXIMATION_HIGH_RESOLUTION'
    with pytest.raises(ValueError,match='sizes'):
        SegmentationTransformConfig(**values)


def test_symbol_free_one_to_one_and_unmatched_denominators():
    assert normalize(' Straße-$23! ') == 'strasse23'
    word = dict(text='A-B',bbox=[0,0,10,10])
    pred = [dict(text='a$b',bbox=[0,0,10,10])]*2 + [dict(text='!!!',bbox=[0,0,10,10])]
    result = counts(pred,[word],.5)
    assert (result['tp'],result['predictions'],result['gt']) == (1,3,1)
    assert aggregate([result])['f1'] == .5
    wrong_location = counts([dict(text='ab',bbox=[100,100,110,110])],[word],.5)
    assert wrong_location['tp'] == 0 and wrong_location['lexical_tp'] == 1
    assert counts([],[],.5)['tp'] == 0


def test_only_documented_restored_input_roots_are_accepted():
    from scripts.run_directional_transport import qualified_path
    assert qualified_path('/old/a.jpg','/old','/restored') == '/restored/a.jpg'
    assert qualified_path('/restored/a.jpg','/old','/restored') == '/restored/a.jpg'
    with pytest.raises(ValueError):
        qualified_path('/unknown/a.jpg','/old','/restored')


def test_frozen_source_keeps_all_tensors_and_batchnorm_statistics_after_update():
    torch.set_num_threads(2)
    source = AdaSpotGapAGUNet('control',skip_floor=.25,initial_gap_bias=-2.).eval()
    model = DirectionalTransportUNet(skip_floor=.25,initial_gap_bias=-2.,widths=(32,24,16,12),
        radii=(1,4),detail_width=16,spatial=True)
    model.initialize_source(source.state_dict())
    model.freeze_source()
    model.train()
    assert not model.encoder.training and not model.up0.training and not model.bottleneck.training
    assert model.transport.training and model.rgb_detail.training
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == 71590
    optimizer = torch.optim.AdamW(trainable,lr=2e-4)
    output = model(torch.randn(1,3,128,128))
    output.square().mean().backward()
    optimizer.step()
    for name, value in source.state_dict().items():
        assert torch.equal(value,model.state_dict()[name]),name
    assert model.detail_fusion[-1].weight.abs().sum() > 0


def test_zero_epoch_reference_has_no_fake_trainable_parameters():
    from scripts.run_directional_transport import model_for, arms_for
    c = dict(experiment_kind='directional_transport_frozen_v2',
        architecture=dict(skip_floor=.25,initial_gap_bias=-2.),training=dict(freeze_source=True))
    model = model_for(c,'unet_reference')
    assert not any(p.requires_grad for p in model.parameters())
    assert arms_for(c) == ('unet_reference','global_detail','local_transport')
    # Explicit empty optimizer group is serialized only for the zero-update reference.
    optimizer = torch.optim.AdamW([dict(params=[],lr=1e-5)])
    assert optimizer.state_dict()['state'] == {}


def test_nonfinite_logits_cannot_silently_become_empty_predictions():
    from scripts.run_directional_transport import finite_probabilities
    import numpy as np
    assert np.array_equal(finite_probabilities(torch.zeros(1,2,4,4)),np.full((2,4,4),.5))
    with pytest.raises(FloatingPointError,match='Nonfinite'):
        finite_probabilities(torch.full((1,2,4,4),float('nan')))
    with pytest.raises(ValueError,match='two-channel'):
        finite_probabilities(torch.zeros(1,1,4,4))


def test_continuation_rejects_optimizer_target_or_data_drift():
    import copy,json
    from pathlib import Path
    from scripts.run_directional_transport import continuation_contract, arms_for
    root=Path(__file__).resolve().parents[2]
    parent=json.loads((root/'configs/approximate/directional_transport_frozen_v2.json').read_text())
    child=copy.deepcopy(parent);child['training']['epochs']=12
    child['experiment_kind']='directional_transport_continuation_v4'
    continuation_contract(child,parent)
    assert arms_for(child)==('unet_reference','global_detail')
    for key in ('new_learning_rate','batch_size'):
        bad=copy.deepcopy(child);bad['training'][key]*=2
        with pytest.raises(ValueError,match='invariant'):
            continuation_contract(bad,parent)
    bad=copy.deepcopy(child);bad['loss']['ownership_weight']=0.
    with pytest.raises(ValueError,match='invariant'):
        continuation_contract(bad,parent)
