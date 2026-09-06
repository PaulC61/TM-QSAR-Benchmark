# PIXI_IMAGE_TAG selects the pixi base image. The build script picks this
# automatically: the plain tag on a MacBook/laptop (no GPU), the CUDA-enabled
# tag on an H100/H200 server (nvidia-smi detected). Override manually if
# auto-detection guesses wrong.
ARG PIXI_IMAGE_TAG=0.41.4
FROM ghcr.io/prefix-dev/pixi:${PIXI_IMAGE_TAG} AS build

# Build tools for tmu/pyTsetlinMachineParallel's native C extensions
# (dev/*, built from source via `pixi install`; see pyproject.toml's [tool.pixi.*] tables), plus
# general dev quality-of-life packages.
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ clang make \
    vim nano git sudo curl \
    locales xdg-utils wget \
    htop tmux \
    libffi-dev libglib2.0-dev openssh-client \
    ca-certificates zstd

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install -y git-lfs

# Set the locale. Required for TMUX and neovim.
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Change user
ARG USER_ID
ARG GROUP_ID
ARG GROUP_NAME
ARG USER_NAME
RUN groupadd -f -g "${GROUP_ID}" "${GROUP_NAME}"
RUN useradd -m -l -s /bin/bash -d /home/"${USER_NAME}" -u "${USER_ID}" -g "${GROUP_ID}" "${USER_NAME}" && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER "${USER_NAME}"

# Dependencies (including the native dev/* extensions) are installed via
# postStartCommand, not baked in at build time: the submodules need the
# `scripts/patch_native_deps.py` portability fix applied first (see
# pyproject.toml's `fix-submodules` task), and this keeps the image reusable
# across submodule updates without a rebuild.
CMD [ "tail", "-f", "/dev/null" ]
