from copy import deepcopy

import numpy as np
from PIL import Image
import torch

from segmentation.icdar13_word_refit import WordDataset
from segmentation.uniform_photometric_training import PhotometricWordDataset


def test_only_rgb_changes_and_validation_is_identical(tmp_path):
    config=dict(image_hw=[64,64],canvas_fill_rgb=[124,116,104],probability_threshold=.5,
        kernel_distance_fraction=.5,training=dict(seed=3,horizontal_flip_probability=0.,
        rotation_probability=0.,rotation_degrees=[0.,0.]),photometric_augmentation=dict(
        probability=1.,operations=['brightness','contrast','saturation','gamma','gaussian_blur','gaussian_noise'],
        magnitude=.2,blur_sigma_min=.1,blur_sigma_max=1.,noise_std_max=.015))
    rgb=np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3);mask=np.zeros((64,64),np.uint8);mask[10:40,12:50]=255
    record=dict(sample_id='fixture')
    for name,array in [('image',rgb),('mask',mask),('validity',np.full((64,64),255,np.uint8)),
                       ('separator',np.zeros((64,64),np.uint8)),('owners',(mask>0).astype(np.int32))]:
        path=tmp_path/(name+'.png');Image.fromarray(array).save(path);record[name+'_path']=str(path)
    base=WordDataset([record],config,augment=True);photo=PhotometricWordDataset([record],config,augment=True)
    a,b=base[0],photo[0]
    assert not torch.equal(a['image'],b['image'])
    assert torch.equal(photo[0]['image'],b['image'])
    for name in ('mask','validity','separation','kernel','owners'):assert torch.equal(a[name],b[name])
    photo.transform.set_epoch(1)
    assert not torch.equal(photo[0]['image'],b['image'])
    validation=PhotometricWordDataset([record],config,augment=False)
    reference=WordDataset([record],config,augment=False)
    assert torch.equal(validation[0]['image'],reference[0]['image'])
    disabled=deepcopy(config);disabled['photometric_augmentation']['probability']=0.
    assert torch.equal(PhotometricWordDataset([record],disabled,augment=True)[0]['image'],a['image'])
