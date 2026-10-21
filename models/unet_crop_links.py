"""Independent crop-trained copies of the qualified approximate word U-Net."""

from models.unet_word_research_v2 import WordResearchV2AGUNet


class CropLinkAGUNet(WordResearchV2AGUNet):
    def __init__(self, variant, config):
        if variant not in ("crop_word", "crop_links"):
            raise ValueError("Unknown learned crop variant")
        self.crop_variant = variant
        super().__init__("links_isolated" if variant == "crop_links" else "control", config)

    @property
    def shared_trainable(self):
        # This is a separate model: the first-pass checkpoint is never updated.
        return True

    @property
    def architecture_identity(self):
        return "experimental-unet-crop-links-v1:" + self.crop_variant
