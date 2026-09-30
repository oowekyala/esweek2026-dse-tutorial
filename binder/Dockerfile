# What mybinder.org builds. It wraps the tutorial image Cinnamon's CI publishes
# -- the compiler, the prebuilt LLVM and the pixi environment with Jupyter --
# in what Binder requires of an image: a user named jovyan with uid 1000, the
# repository in its home, and `jupyter` on PATH for the launcher.
#
# The image is expected to leave the pixi environment activated for every
# user (its bin on PATH, cinm-opt alongside), which is what makes the plain
# `jupyter` below resolve.
FROM ghcr.io/tud-ccc/cinnamon-tutorial:latest

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} \
    HOME=/home/${NB_USER}

USER root
RUN id -u ${NB_UID} >/dev/null 2>&1 \
 || useradd --create-home --uid ${NB_UID} --shell /bin/bash ${NB_USER}

COPY --chown=${NB_UID}:${NB_UID} . ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}
