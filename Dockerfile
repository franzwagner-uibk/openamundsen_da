FROM mambaorg/micromamba:1.5.8
LABEL org.opencontainers.image.source="https://github.com/franzwagner-uibk/openamundsen_da"

# Allow `micromamba run -n <env>` directly as entrypoint
ARG MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/bin/bash", "-lc"]

# Create the conda-forge environment
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n openamundsen_da -f /tmp/environment.yml && \
    micromamba clean -a -y

# Work inside /workspace; mount your repo here at runtime
WORKDIR /workspace

# Build and run as root; entrypoint will restore /data ownership after the command
USER root

# Install openamundsen_da into the image so the `openamundsen-da` entrypoint is available
COPY . /workspace
RUN micromamba run -n openamundsen_da python -m pip install -e . --no-deps

# Lightweight entrypoint to clear stale mamba locks and run inside env
COPY scripts/oa_entrypoint.sh /usr/local/bin/oa_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/oa_entrypoint.sh"]
