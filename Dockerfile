FROM mambaorg/micromamba:1.5.8

# Allow `micromamba run -n <env>` directly as entrypoint
ARG MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/bin/bash", "-lc"]

# Create the conda-forge environment
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n oa -f /tmp/environment.yml && \
    micromamba clean -a -y

# Work inside /workspace; mount your repo here at runtime
WORKDIR /workspace

# Optional convenience script to run the demo inside the container
COPY docker/oa-da-demo.sh /usr/local/bin/oa-da-demo
RUN chmod +x /usr/local/bin/oa-da-demo || true

# Default to executing inside env 'oa'
ENTRYPOINT ["micromamba", "run", "-n", "oa"]

