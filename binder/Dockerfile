# What mybinder.org builds: the tutorial image Cinnamon's CI publishes (the
# compiler, the prebuilt LLVM, the pixi environment with Jupyter on PATH, and
# already the jovyan user Binder requires), with this repository in that
# user's home.
FROM ghcr.io/tud-ccc/cinnamon-tutorial:latest

ARG NB_USER=jovyan
ARG NB_UID=1000

USER root
COPY --chown=${NB_UID}:${NB_UID} . /home/${NB_USER}
# The notebooks' support package, so they import it wherever the kernel starts
RUN pip install --no-deps -e /home/${NB_USER}
USER ${NB_USER}
WORKDIR /home/${NB_USER}
