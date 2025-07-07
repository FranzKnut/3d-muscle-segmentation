
FROM mambaorg/micromamba:cuda12.4.1-ubuntu22.04

COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yml /tmp/env.yaml
RUN CONDA_OVERRIDE_CUDA="12.0" micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
    
