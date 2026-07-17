FROM mambaorg/micromamba:1.5.8@sha256:475730daef12ff9c0733e70092aeeefdf4c373a584c952dac3f7bdb739601990
LABEL org.opencontainers.image.source="https://github.com/franzwagner-uibk/openamundsen_da"

ARG VCS_REF="unknown"
ARG VERSION="0+unknown"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.version="${VERSION}"

# Allow `micromamba run -n <env>` directly as entrypoint
ARG MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/bin/bash", "-lc"]

# The pinned base predates two critical GnuTLS fixes. Keep the package version
# explicit so container builds fail instead of silently accepting a regression.
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends --only-upgrade \
        libgnutls30=3.7.9-2+deb12u7 && \
    rm -rf /var/lib/apt/lists/*
USER mambauser

# Create the conda-forge environment
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n openamundsen_da -f /tmp/environment.yml && \
    micromamba clean -a -y

# Named cache volumes must remain writable for the non-root image user too.
USER root
RUN mkdir -p /cache/xdg /cache/mamba/pkgs /cache/mpl && \
    chmod -R 0777 /cache

# Release execution uses only the mounted setup and the installed distribution.
WORKDIR /data

# Install exactly the wheel that passed the package gates. Source is never copied.
COPY dist/openamundsen_da-*.whl /tmp/dist/
RUN wheels=(/tmp/dist/openamundsen_da-*.whl) && \
    [[ "${#wheels[@]}" -eq 1 ]] && \
    micromamba run -n openamundsen_da python -m pip install --no-deps "${wheels[0]}" && \
    rm -rf /tmp/dist

# Preserve the documented example bootstraps without copying the source tree.
COPY examples/rofental /workspace/examples/rofental
COPY examples/subdomains /workspace/examples/subdomains

# Lightweight entrypoint to clear stale mamba locks and run inside env
COPY scripts/oa_entrypoint.sh /usr/local/bin/oa_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/oa_entrypoint.sh"]
