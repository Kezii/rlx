# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Build environment for MLX's CUDA backend on a given NVIDIA CUDA image
# (`--build-arg BASE=nvidia/cuda:<ver>-cudnn-devel-ubuntu24.04`). Used by
# `cuda_matrix.py`; no GPU needed — it checks toolkit selection, the MLX CUDA
# compile and the link (against the driver stubs).
ARG BASE
FROM ${BASE}

ARG EXTRA_APT=""
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl git cmake build-essential pkg-config \
      libopenblas-dev liblapack-dev liblapacke-dev ${EXTRA_APT} \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:${PATH}
WORKDIR /work
