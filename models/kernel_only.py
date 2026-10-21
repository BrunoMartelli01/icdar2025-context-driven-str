"""One-output inference extraction; source training remains joint Text+Kernel."""

from models.unet_adv import PracticalTotalTextHighResAGUNet


class KernelOnlyAGUNet(PracticalTotalTextHighResAGUNet):
    @property
    def architecture_identity(self):
        return "experimental-extracted-kernel-only-agunet-v1"

    def initialize_text_kernel_state(self, state):
        own = self.state_dict()
        if set(own) != set(state):
            raise ValueError("Extraction requires every Text+Kernel tensor, without auxiliary heads")
        extracted = {}
        for name, value in state.items():
            expected = own[name]
            if name in ("head.weight", "head.bias"):
                if value.shape != (2, *expected.shape[1:]):
                    raise ValueError("Expected exactly two source output channels")
                value = value[1:2]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("Source tensor shape/dtype mismatch: " + name)
            extracted[name] = value
        self.load_state_dict(extracted, strict=True)
        return {"source_channel": 1, "output_channels": 1, "strict_load": True,
                "training": "joint_Text_Kernel; no new training", "tensors": len(extracted)}
