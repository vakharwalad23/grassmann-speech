# Downloads and caches stay under .cache/ so `make clean` removes them
ROOT := $(abspath .)
export HF_HOME := $(ROOT)/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY := 1
export UV_CACHE_DIR := $(ROOT)/.cache/uv
export XDG_CACHE_HOME := $(ROOT)/.cache/xdg
export TORCH_HOME := $(ROOT)/.cache/torch
export RUFF_CACHE_DIR := $(ROOT)/.cache/ruff
export PYTHONDONTWRITEBYTECODE := 1

UV := uv run --project python

.PHONY: help setup test lint probe-quick probe e01-quick e01-parity e01 clean

help:
	@echo "setup        python/.venv from uv.lock (torch 2.15.0, Python 3.15)"
	@echo "test         unit checks: Plucker identities, causality, receptive field, exactness, parameter counts, cost"
	@echo "lint         ruff on python/"
	@echo "probe-quick  e00 code paths at tiny sizes, seconds, no report"
	@echo "probe        e00 Apple GPU speed and memory probe; needs AC power, ~6 min"
	@echo "e01-quick    e01 code paths: 1 epoch of 5 batches per run, into work/"
	@echo "e01-parity   e01 Transformer seed 0 only, ~0.8 h on an M3 Pro"
	@echo "e01          e01 full sweep, 10 runs, ~9 h on an M3 Pro; finished runs are skipped"
	@echo "clean        delete .cache, work and python/.venv (keeps reports)"

setup:
	@uv sync --project python --locked

test:
	@$(UV) pytest python/tests

lint:
	@uv tool run ruff check python

probe-quick:
	@$(UV) python python/e00_mps_probe.py --quick

probe:
	@$(UV) python python/e00_mps_probe.py

e01-quick:
	@$(UV) python -I python/e01_layer_parity.py --quick

e01-parity:
	@$(UV) python -I python/e01_layer_parity.py --runs transformer:0

e01:
	@$(UV) python -I python/e01_layer_parity.py

clean:
	@rm -rf .cache work python/.venv
