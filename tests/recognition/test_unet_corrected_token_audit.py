import numpy as np
from PIL import Image

from scripts.audit_unet_corrected_token_router import counts,square_mask
from recognition.clip_router_evaluation import native_mask


def test_independent_projection_preserves_odd_padding_and_nearest_pixels():
    rng=np.random.default_rng(23)
    for height,width in ((7,11),(8,8),(13,6),(29,41)):
        image=Image.fromarray((rng.integers(0,2,(height,width))*255).astype(np.uint8))
        for size in (7,32,256):
            assert np.array_equal(square_mask(image,size),native_mask(image,size))


def test_independent_counter_preserves_empty_and_symbol_denominators():
    rows=[dict(raw_text='CAFÉ!'),dict(raw_text='café'),dict(raw_text='?'),dict(raw_text='')]
    gt=[dict(text='Café'),dict(text='café'),dict(text='!')]
    actual=counts(rows,gt,[False,True,False,False])
    assert actual==dict(tp=1,fp=2,fn=2,detected_words=3,gt_words=3,all_outputs=4,ignored_outputs=1)
