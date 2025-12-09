FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-dev \
    git \
    vim \
    libffi-dev \
    libglib2.0-dev \
    libgl1-mesa-glx \
    libglx-mesa0

RUN python3 -m venv tmu_venv

# Install Tsetlin Machine Unified with pip
RUN source tmu_venv/bin/activate && git clone https://github.com/cair/tmu.git && \
    cd tmu && pip --no-cache wheel . && pip --no-cache install . && \
    cd .. && rm -fdr tmu && pip install opencv-python pycuda 

ENV VIRTUAL_ENV=tmu_venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"