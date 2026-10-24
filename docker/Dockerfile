ARG ISAACLAB_BASE_IMAGE_ARG

# Isaac Lab main base image: uv venv at /opt/isaaclab-venv, runtime user isaaclab (uid 1000)
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

# The runtime home and the workspace use the host paths, so the `~/Datasets/...` defaults and the
# relative `logs` symlink resolve the same way inside and outside the container.
ARG DOCKER_USER_HOME_ARG
ARG DOCKER_INSTINCT_WS_PATH_ARG
ENV DOCKER_USER_HOME=${DOCKER_USER_HOME_ARG}
ENV HOME=${DOCKER_USER_HOME_ARG}
ENV DOCKER_INSTINCT_WS_PATH=${DOCKER_INSTINCT_WS_PATH_ARG}

USER root

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    htop \
    vim && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Move the base image's runtime home (.bashrc with the uv aliases) to the host-path home and
# pre-create every named-volume mount point, so fresh volumes come up owned by the runtime user.
RUN mkdir -p ${DOCKER_USER_HOME} && cp -rT /root ${DOCKER_USER_HOME} && \
    usermod -d ${DOCKER_USER_HOME} isaaclab && \
    mkdir -p \
        ${DOCKER_USER_HOME}/.cache/instinctlab \
        ${DOCKER_USER_HOME}/.cache/ov \
        ${DOCKER_USER_HOME}/.cache/pip \
        ${DOCKER_USER_HOME}/.cache/uv \
        ${DOCKER_USER_HOME}/.cache/warp \
        ${DOCKER_USER_HOME}/.cache/nvidia/GLCache \
        ${DOCKER_USER_HOME}/.nv/ComputeCache \
        ${DOCKER_USER_HOME}/.nvidia-omniverse/logs \
        ${DOCKER_USER_HOME}/.local/share/ov/data \
        ${DOCKER_USER_HOME}/Documents \
        ${DOCKER_INSTINCT_WS_PATH} && \
    chown -R isaaclab:isaaclab ${DOCKER_USER_HOME} && \
    chmod 750 ${DOCKER_USER_HOME}

# Editable installs at the host paths. The clones are bind-mounted over these copies at runtime,
# so the editable finders resolve into the mounted sources. Never run an exact `uv sync` in the
# container; it removes these editable installs.
COPY --from=instinct_rl --chown=isaaclab:isaaclab . ${DOCKER_INSTINCT_WS_PATH}/instinct_rl
COPY --chown=isaaclab:isaaclab source ${DOCKER_INSTINCT_WS_PATH}/instinctlab/source
USER isaaclab
# Add Isaac Lab's ``importers`` extra (the standalone ``isaacsim-asset-isolated`` URDF importer) to the base image's
# package set, from Isaac Lab's lockfile so its overrides keep the simulator stack pinned. URDF assets then convert
# without Kit on every backend. IMAGE_EXTRAS repeats the base image's list (docker/Dockerfile.base in Isaac Lab). Run
# before the Instinct-RL install, which pins onnx below the lockfile's version.
ARG IMAGE_EXTRAS="--extra sb3 --extra skrl --extra rl-games --extra rsl-rl --extra viser --extra rerun --extra mimic --extra teleop --extra test --extra tetrahedralization --extra ov"
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/uv,uid=1000,gid=1000 \
    cd ${ISAACLAB_PATH} && \
    uv sync --frozen --inexact ${IMAGE_EXTRAS} --extra importers --no-install-package imageio-ffmpeg
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/uv,uid=1000,gid=1000 \
    uv pip install -e ${DOCKER_INSTINCT_WS_PATH}/instinct_rl -e ${DOCKER_INSTINCT_WS_PATH}/instinctlab/source/instinctlab

WORKDIR ${DOCKER_INSTINCT_WS_PATH}/instinctlab
ENTRYPOINT []
CMD ["bash"]
