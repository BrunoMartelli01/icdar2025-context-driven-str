# syntax=docker/dockerfile:1
#
# Reconstruction wrapper only. The exact original base-image recipe used to
# create the locally qualified image was not captured. BASE_IMAGE must already
# provide the qualified legacy userspace; this file validates it and never
# copies source code, checkpoints, or compiled extensions into an image layer.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEEPSOLO_PYTHON=/opt/context-str-deepsolo/bin/python
RUN test -x "${DEEPSOLO_PYTHON}" && \
    "${DEEPSOLO_PYTHON}" -c "import sys, torch, torchvision, detectron2; assert sys.version_info[:2] == (3, 8); assert torch.__version__ == '1.9.0+cu111'; assert torchvision.__version__ == '0.10.0+cu111'; assert torch.version.cuda == '11.1'"

ENV CUDA_HOME=/usr/local/cuda \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    TORCH_CUDA_ARCH_LIST=8.6 \
    MAX_JOBS=2 \
    FORCE_CUDA=1

WORKDIR /workspace/DeepSolo
CMD ["sleep", "infinity"]
