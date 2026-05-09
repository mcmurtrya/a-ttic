# TTIC Embeddings — common dev / run targets.
#
# Most local targets wrap `uv` (Astral's fast Python package manager).
# Docker targets use the local Dockerfile.
#
# Windows users: install GNU Make via git-for-windows (which bundles it
# in git-bash) or via Chocolatey. PowerShell does not ship with make;
# the README has the equivalent uv commands as a fallback.

.PHONY: help install install-dev sync lock smoke data lint format clean \
        docker-build docker-smoke docker-shell docker-train

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_NAME ?= ttic-embeddings
IMAGE_TAG  ?= dev

# Volume mounts for Docker — point these at your real data dirs via env.
COCO_HOST := $(if $(COCO_ROOT),$(COCO_ROOT),$(CURDIR)/data/coco)
VG_HOST   := $(if $(VG_ROOT),$(VG_ROOT),$(CURDIR)/data/vg)
HF_HOST   := $(if $(HF_HOME),$(HF_HOME),$(HOME)/.cache/huggingface)

DOCKER_RUN := docker run --rm -it \
    -v "$(CURDIR)":/workspace \
    -v "$(COCO_HOST)":/data/coco \
    -v "$(VG_HOST)":/data/vg \
    -v "$(HF_HOST)":/root/.cache/huggingface \
    -e COCO_ROOT=/data/coco \
    -e VG_ROOT=/data/vg \
    -w /workspace

# ---------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------
# Local (host) targets
# ---------------------------------------------------------------------

install:  ## Create venv and install the package via uv
	uv venv
	uv pip install -e .

install-dev:  ## Install package + dev/eval extras + NLP assets
	uv sync --extra dev --extra caption-quality
	uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

sync:  ## Re-sync deps from pyproject.toml (or uv.lock if present)
	uv sync

lock:  ## Generate or update uv.lock
	uv lock

smoke:  ## Run the Phase 0 smoke test
	uv run python scripts/00_smoke_test.py

test:  ## Run pytest (metric + stats unit tests)
	uv sync --extra dev --extra caption-quality
	uv run pytest

data:  ## Download COCO + VG into $COCO_ROOT and $VG_ROOT
	uv run python scripts/01_download_data.py

lint:  ## Run ruff check
	uv run ruff check .

format:  ## Run ruff format
	uv run ruff format .

clean:  ## Remove venv and Python caches
	rm -rf .venv build dist *.egg-info \
	       .ruff_cache .pytest_cache \
	       $(shell find . -type d -name __pycache__ 2>/dev/null)

# ---------------------------------------------------------------------
# Docker targets
# ---------------------------------------------------------------------

docker-build:  ## Build the Docker image (CUDA-enabled base)
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-smoke:  ## Run smoke test inside the container (CPU)
	$(DOCKER_RUN) $(IMAGE_NAME):$(IMAGE_TAG) python scripts/00_smoke_test.py

docker-shell:  ## Open an interactive shell in the container (CPU)
	$(DOCKER_RUN) $(IMAGE_NAME):$(IMAGE_TAG) bash

docker-train:  ## Open a shell with NVIDIA GPU access (needs nvidia-container-toolkit)
	$(DOCKER_RUN) --gpus all $(IMAGE_NAME):$(IMAGE_TAG) bash
