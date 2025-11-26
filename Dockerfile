FROM mambaorg/micromamba:1.5.8

# Allow `micromamba run -n <env>` directly as entrypoint
ARG MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/bin/bash", "-lc"]

# Create the conda-forge environment
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n openamundsen -f /tmp/environment.yml && \
    micromamba clean -a -y

# Work inside /workspace; mount your repo here at runtime
WORKDIR /workspace

# Install openamundsen_da into the image so `oa-da-*` entrypoints are available
COPY . /workspace
RUN micromamba run -n openamundsen python -m pip install -e . --no-deps

# Default to executing inside env 'openamundsen'
ENTRYPOINT ["micromamba", "run", "-n", "openamundsen"]
