# TTIC Embeddings container.
#
# CUDA-enabled base for adaptor training; works fine on CPU for the
# smoke test and metric scoring. Built around `uv` for fast,
# reproducible installs.
#
# Build:  make docker-build  (or:  docker build -t ttic-embeddings:dev .)
# Smoke:  make docker-smoke  (CPU-only, no GPU needed)
# Train:  make docker-train  (mounts data dirs, attaches all GPUs)

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_CACHE_DIR=/root/.cache/uv

# System deps: Python 3.11, build tools for native wheels, git for HF
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        git build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# uv (Astral's fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /workspace

# --- Dependency layer -------------------------------------------------
# Copy only metadata + source first so the heavy install layer caches
# across changes to scripts/, configs/, and docs.
COPY pyproject.toml README.md ./
COPY src ./src

# Install with CUDA wheels for torch (matches the cuda:12.1 base above)
RUN uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -e .

# Pre-fetch NLP assets so the container is offline-ready for evaluation
RUN python -m spacy download en_core_web_lg \
    && python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# --- Source layer -----------------------------------------------------
COPY scripts ./scripts
COPY configs ./configs

CMD ["bash"]
