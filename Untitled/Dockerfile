# Docker file for PyTorch 

FROM nvidia/cuda:11.2.0-devel-ubuntu20.04

#
# USER root
#

USER root

ARG UID=1058
ARG GID=1060
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y -qq --no-install-recommends \
        tmux sudo nano htop wget less \
        iputils-ping net-tools \
        git curl \
        python3-dev python3-pip python3-tk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN apt install --yes build-essential libssl-dev cmake
RUN apt install --yes git git-lfs
RUN git lfs install
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
FROM condaforge/mambaforge:24.7.1-0
RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 neovim vim git ripgrep fd-find libffi-dev libglib2.0-dev libgl1-mesa-glx libglx-mesa0 
RUN mamba install -y -n base pip "gxx<14" "numpy<2" pandas matplotlib seaborn plotly python-kaleido scikit-learn scikit-image opencv scipy mlxtend nltk tqdm pycuda cuda "h5py>0=*openmpi*" numba sympy optuna optuna-integration pytorch torchvision -c nvidia -c pytorch 
RUN pip install git+https://github.com/cair/tmu.git
RUN pip install rdkit

COPY . /app
WORKDIR /app

CMD [ "tail", "-f", "/dev/null" ]
