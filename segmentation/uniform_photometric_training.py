"""RGB-only training perturbations over the qualified uniform word dataset."""

from segmentation.icdar13_word_refit import WordDataset
from segmentation.totaltext_image_frequency_augment import PracticalTotalTextImageTransform


class PhotometricWordDataset(WordDataset):
    def __init__(self,records,config,*,augment):
        super().__init__(records,config,augment=augment)
        self.photo=PracticalTotalTextImageTransform(self.transform.config,
            base_seed=config['training']['seed'],mode='photometric',augment_images=augment,
            parameters=config['photometric_augmentation'])

    def __getitem__(self,index):
        row=super().__getitem__(index)
        self.photo.set_epoch(self.transform.epoch)
        return dict(row,image=self.photo._image_only(row['image'],index))
