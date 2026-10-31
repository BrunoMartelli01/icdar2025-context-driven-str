"""Scientific invariants for the three approximate context router proposals."""
import numpy as np
import pytest
import torch

from recognition.token_context_router import (
    BenefitRouter, DenseSupport, base_features, benefit_loss, context_views,
    dense_features, erased_views, likelihood_features, patch_geometry,
    transcript_chunks, unbalanced_transport, word_geometry,
)
from scripts.token_context_training import collate, point, ranked, targets


def row(word='CAFE', box=(2, 2, 6, 5), index=0):
    return dict(raw_text=word, bbox=list(box), candidate_id=str(index),
                confidence=.9, detector_score=.8, token_probabilities=[.8, .95])


def settings():
    return dict(clip_dim=8, query_dim=6, query_count=2, word_geometry_dim=12,
                hidden_dim=8, heads=2, mlp_dim=16, initialization_std=.02)


def feature(n):
    rng = np.random.default_rng(4+n)
    return dict(patches=rng.normal(size=(4, 8)).astype('f'),
                patch_geometry=np.array([[.25, .25, .5, .5], [.75, .25, .5, .5],
                                         [.25, .75, .5, .5], [.75, .75, .5, .5]], 'f'),
                patch_valid=np.array([True, True, True, False]),
                patch_mass=np.array([.25, .25, .25, 0], 'f'),
                queries=rng.normal(size=(2, 6)).astype('f'),
                word_text=rng.normal(size=(n, 8)).astype('f'),
                word_image=rng.normal(size=(n, 8)).astype('f'),
                word_geometry=np.tile(np.array([0, 0, .5, .5]+[.1]*8, 'f'), (n, 1)),
                scalars=np.arange(18, dtype='f'), likelihood=np.zeros(14, 'f'), eligible=n > 0)


def batch(n=3):
    return collate({'s': feature(n)}, ['s'], np.empty((1, 0), 'f'), np.zeros(18), np.ones(18), 'cpu')


def test_tiling_preserves_every_original_pixel_once_in_partition():
    rgb = np.arange(7*11*3, dtype=np.uint8).reshape(7, 11, 3)
    coverage = np.zeros((7, 11), int)
    views = context_views(rgb, [2, 2], [0, 0, 0])
    for v in views[1:]:
        a, b, c, d = v['bbox']; x, y = v['pad_xy']
        np.testing.assert_array_equal(v['rgb'][y:y+d-b, x:x+c-a], rgb[b:d, a:c])
        coverage[b:d, a:c] += 1
    assert np.all(coverage == 1)


def test_padding_geometry_masks_only_padding_centers():
    v = context_views(np.zeros((2, 4, 3), np.uint8), [1, 1], [0, 0, 0])[0]
    xy, valid, mass = patch_geometry(v, [4, 2], 4)
    assert valid.sum() == 8
    assert np.all((xy[valid, :2] >= 0)&(xy[valid, :2] < 1))
    assert mass.sum() == pytest.approx(.5)


def test_invalid_grid_cannot_create_empty_tiles():
    with pytest.raises(ValueError):
        context_views(np.zeros((1, 4, 3), np.uint8), [2, 2], [0, 0, 0])


def test_erasure_changes_only_declared_boxes():
    rgb = np.full((10, 12, 3), 57, np.uint8); rows = [row()]
    hidden, views = erased_views(rgb, rows, [128]*3, 1)
    outside = np.ones((10, 12), bool); outside[2:5, 2:6] = False
    np.testing.assert_array_equal(hidden[outside], rgb[outside])
    assert np.all(hidden[2:5, 2:6] == 128)
    assert len(views) == 1 and np.all(rgb == 57)


def test_chunks_keep_all_nonempty_words_and_eos_budget():
    rows = [row('three', (8, 3, 9, 5), 2), row('one', (0, 0, 2, 2), 0),
            row('two', (3, 0, 5, 2), 1), row('', (6, 0, 7, 2), 3)]
    tokenize = lambda text: text.split()+['EOS']
    chunks = transcript_chunks(rows, tokenize, 3)
    assert [c['text'] for c in chunks] == ['one two', 'three']
    assert [i for c in chunks for i in c['row_indices']] == [1, 2, 0]
    assert all(len(tokenize(c['text'])) <= 3 for c in chunks)


def test_oversized_word_raises_instead_of_truncating():
    with pytest.raises(ValueError):
        transcript_chunks([row('LONG')], lambda s: list(s)+['EOS'], 4)


def test_missing_likelihood_is_explicit_and_finite():
    x = likelihood_features([], [])
    assert len(x) == 14 and not any(x.values())
    values = [dict(visual_gain=2., eos_visual_gain=-1., original_logprob_per_token=-3.)]
    x = likelihood_features(values, [])
    assert x['global_available'] == 1 and x['word_available'] == 0
    assert x['global_visual_gain_min'] == 2


def test_scalar_and_geometry_schema_handles_empty_predictions():
    assert base_features([], [12, 10]).shape == (18,)
    assert word_geometry([], [12, 10]).shape == (0, 12)
    assert word_geometry([row()], [12, 10]).shape == (1, 12)


TRANSPORT = dict(epsilon=.1, rho=.5, iterations=80, mass_floor=1e-9, spatial_cost=2.)


def test_unbalanced_transport_is_finite_and_allows_unmatched_mass():
    t, meta = unbalanced_transport(np.array([[0., 8.], [8., 8.]]), [.5, .5], [.5, .5], TRANSPORT)
    assert np.isfinite(t).all() and t[0, 0] > 100*t[1, 1]
    assert t[1].sum() < .5 and meta['iterations'] == 80


def test_transport_permutation_equivariance():
    c = np.array([[.1, .4], [.8, .2], [.6, .7]])
    a = np.array([.2, .3, .5]); b = np.array([.4, .6]); p = [2, 0, 1]
    t, _ = unbalanced_transport(c, a, b, TRANSPORT)
    u, _ = unbalanced_transport(c[p][:, ::-1], a[p], b[::-1], TRANSPORT)
    np.testing.assert_allclose(u, t[p][:, ::-1])


@pytest.mark.parametrize('shape', [(0, 2), (3, 0)])
def test_transport_empty_dimensions(shape):
    t, _ = unbalanced_transport(np.empty(shape), np.ones(shape[0]), np.ones(shape[1]), TRANSPORT)
    assert t.shape == shape and not t.any()


@pytest.mark.parametrize('cost,a,b', [([[np.nan]], [1], [1]), ([[1]], [-1], [1]), ([[1]], [1], [np.inf])])
def test_transport_rejects_nonfinite_or_negative_inputs(cost, a, b):
    with pytest.raises(ValueError):
        unbalanced_transport(cost, a, b, TRANSPORT)


def test_dense_evidence_for_no_words_retains_unmatched_visual_signal():
    support = DenseSupport(8, 4)
    values, diagnostic = dense_features(feature(0), support, TRANSPORT, 'cpu')
    assert values.shape == (10,) and np.isfinite(values).all()
    assert values[3] == pytest.approx(1.) and diagnostic['transport_mass'] == 0


@pytest.mark.parametrize('arm', ['scalar', 'pooled_context', 'token_context', 'likelihood', 'dense', 'combined'])
def test_every_arm_handles_zero_words_and_has_finite_gradients(arm):
    model = BenefitRouter(settings(), 18, 0, arm)
    for n in (0, 3):
        output, _ = model(batch(n)); assert output.shape == (1, 2)
        assert torch.isfinite(output).all()
        output.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_word_order_is_equivariant_and_image_decision_invariant():
    torch.manual_seed(1); model = BenefitRouter(settings(), 18, 0, 'token_context').eval()
    x = batch(); a, aw = model(x); order = [2, 0, 1]
    y = {k: v.clone() for k, v in x.items()}
    for name in ('word_text', 'word_image', 'word_geometry', 'word_padding'):
        y[name] = y[name][:, order]
    b, bw = model(y)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(bw, aw[:, order], atol=1e-6, rtol=1e-6)


def test_padding_cannot_change_decision():
    model = BenefitRouter(settings(), 18, 0, 'token_context').eval(); x = batch(1)
    a, _ = model(x)
    for name in ('word_text', 'word_image', 'word_geometry'):
        x[name] = torch.cat([x[name], torch.full_like(x[name], 12345.)], dim=1)
    x['word_padding'] = torch.tensor([[False, True]])
    b, _ = model(x)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_cost_ranking_ties_and_correct_order():
    cfg = dict(huber_beta=1., ranking_weight=1.)
    y = torch.tensor([[1., 0.], [0., 0.]])
    right = y.clone().requires_grad_(); wrong = y.flip(0).requires_grad_()
    good, _ = benefit_loss(right, y, .8, cfg); bad, _ = benefit_loss(wrong, y, .8, cfg)
    assert good < bad
    tied, d = benefit_loss(right, torch.zeros_like(y), .8, cfg)
    assert d['ranking'] == 0.; tied.backward(); assert torch.isfinite(right.grad).all()


def metric(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn, detected_words=tp+fp, gt_words=tp+fn,
                all_outputs=tp+fp, ignored_outputs=0)


def test_branch_delta_target_and_top_budget_include_all_images():
    branches = {str(i): dict(light=metric(i, 1, 3-i), deepsolo=metric(2, 0, 1)) for i in range(4)}
    ids = list(branches)
    np.testing.assert_array_equal(targets(branches, ids), [[-2, -1], [-1, 0], [0, 1], [1, 2]])
    accepted = ranked({'0': None, '1': 4., '2': 2., '3': 1.}, ids, .5)
    assert accepted == ['1', '2']
    p = point(branches, ids, accepted)
    assert p['images'] == 4 and p['fallback_images'] == 2
    assert p['combined']['tp'] == 7 and p['combined']['gt_words'] == 12
