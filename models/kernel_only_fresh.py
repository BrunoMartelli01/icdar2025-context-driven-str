"""New Kernel U-Net: fresh decoder/head, local ImageNet encoder weights only."""
from models.unet_adv import PracticalTotalTextHighResAGUNet


class FreshKernelAGUNet(PracticalTotalTextHighResAGUNet):
    @property
    def architecture_identity(self):
        return 'experimental-fresh-kernel-only-agunet-v1'

    def initialize_encoder(self,state):
        expected=self.encoder.state_dict()
        if set(state)!=set(expected):raise ValueError('ImageNet encoder requires every original tensor')
        for name,value in state.items():
            if value.shape!=expected[name].shape or value.dtype!=expected[name].dtype:
                raise ValueError('Encoder shape/dtype mismatch: '+name)
        self.encoder.load_state_dict(state,strict=True)
        return {'encoder_load_strict':True,'decoder_initialization':'new_random',
                'head_initialization':'new_random','segmentation_checkpoint_loaded':False}
