# syntax=docker/dockerfile:1.7
# jhu_ta1 evaluation image: the environment the JHU DKPS cards run in.
#
# Kitware evaluates a card by turning its `kwdagger:` block into a DAG and
# running every node of that DAG as one `docker run` of this image, with the
# checkout bind-mounted at its own absolute path and the node's cwd there. The
# copy of this repo baked below supplies the installed environment; the mount
# supplies the code that actually runs. See docs/containerized_evaluation.md.
#
#   docker build -t jhu-magnet-dkps-gpu .
#
# MAGNET_VERSION is the aiq-magnet release the evaluator runs against, from
# PyPI (0.1.0, released 2026-09-04; it also brings aiq-magnet-theory).
# `--build-arg MAGNET_VERSION=<version>` builds against another release.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
ARG MAGNET_VERSION=0.1.0

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        git \
        jq \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade uv

WORKDIR /opt/src

# MAGNET first, so the heavy layer is stable across edits to this repo. The
# [optional] extra (gcsfs) is what lets the materialize node download HELM
# lite runs from the public bucket when no precomputed root is given.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "aiq-magnet[optional]==${MAGNET_VERSION}"

# Runtime dependencies, mirroring pyproject.toml minus aiq-magnet. dkps is the
# DKPS library; sentence-transformers and einops back the local nomic embedder
# used by the text-embedding datasets.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        'dkps @ git+https://github.com/jataware/dkps@magnet' \
        'httpx>=0.28.1,<0.29' \
        'sentence-transformers>=3.1' \
        'einops>=0.8'

# This repo, without dependencies so it uses the magnet pinned above rather
# than re-resolving pyproject's git requirement.
COPY . /opt/src/jhu-magnet-dkps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps -e /opt/src/jhu-magnet-dkps

WORKDIR /opt/src/jhu-magnet-dkps
CMD ["bash"]
