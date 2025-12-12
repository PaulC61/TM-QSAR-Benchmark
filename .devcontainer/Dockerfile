FROM ghcr.io/prefix-dev/pixi:0.41.4 AS build



# Important packages
RUN apt-get update && apt-get install -y \
    vim nano git sudo make curl \
    locales xdg-utils wget \
    nvtop htop tmux curl \
    libffi-dev libglib2.0-dev openssh-client

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs

# Set the locale. Required for TMUX and neovim
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Change user
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
RUN useradd -m -l -s /bin/bash -d /home/"${USER_NAME}" -u "${USER_ID}" -g "${GROUP_ID}" "${USER_NAME}" && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER "${USER_NAME}"

# install dependencies to `/app/.pixi/envs/prod`
# use `--locked` to ensure the lockfile is up to date with pixi.toml
RUN pixi install --locked -e prod
# create the shell-hook bash script to activate the environment
RUN pixi shell-hook -e prod -s bash > /shell-hook

CMD [ "tail", "-f", "/dev/null" ]




