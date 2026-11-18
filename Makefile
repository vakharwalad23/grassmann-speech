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
ARM ?= attention
SEED ?= 0
LIBRISPEECH_PARTS := dev-clean dev-other test-clean test-other train-clean-100

.PHONY: help setup test lint probe-quick probe e01-quick e01-parity e01 e02-data e02-quick e02-prepare e02-train e02-decode \
	e02-espnet-check e02-live-probe e02-espnet-setup e02-espnet-prepare e02-espnet clean

help:
	@echo "setup        python/.venv from uv.lock (torch 2.15.0, Python 3.15)"
	@echo "test         unit checks: Plucker identities, causality, receptive field, exactness, parameter counts, cost"
	@echo "lint         ruff on python/"
	@echo "probe-quick  e00 code paths at tiny sizes, seconds, no report"
	@echo "probe        e00 Apple GPU speed and memory probe; needs AC power, ~6 min"
	@echo "e01-quick    e01 code paths: 1 epoch of 5 batches per run, into work/"
	@echo "e01-parity   e01 Transformer seed 0 only, ~0.8 h on an M3 Pro"
	@echo "e01          e01 full sweep, 10 runs, ~9 h on an M3 Pro; finished runs are skipped"
	@echo "e02-data     LibriSpeech dev, test and train-clean-100 from OpenSLR into data/librispeech, 7.6 GB"
	@echo "e02-quick    e02 own trainer, every arm on a synthetic corpus, ~10 s"
	@echo "e02-prepare  e02 own trainer: split lists, sealed dev/test manifests in corpora/, tokenizer, MVN stats"
	@echo "e02-train    e02 own trainer, ARM=attention|grassmann|symmetric|symmetric_signed|no_mixer SEED=0; resumable"
	@echo "e02-decode   e02 own trainer: average the best checkpoints, greedy decode, score dev and test"
	@echo "e02-espnet-check    ESPnet track on CPU: arm sizes, frontend parity, 1-epoch train and decode per arm, ~2 min"
	@echo "e02-live-probe      e02 attention checkpoint on 96 synthetic macOS-speech clips x 9 capture distortions, CPU, ~6 min"
	@echo "e02-espnet-setup    pinned ESPnet recipe in vendor/espnet and e02_espnet/.venv; needs sox and flac"
	@echo "e02-espnet-prepare  ESPnet recipe stages 1-10 once: data prep, speed perturbation, BPE, stats"
	@echo "e02-espnet   ESPnet track ARM=attention|grassmann|symmetric|symmetric_signed|no_mixer SEED=0: train, greedy decode, score"
	@echo "clean        delete .cache, work, vendor and the virtual environments (keeps data and reports)"

setup:
	@uv sync --project python --locked

test:
	@$(UV) pytest python/tests

lint:
	@uv tool run ruff check python e02_espnet

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

e02-data:
	@mkdir -p data/librispeech
	@for part in $(LIBRISPEECH_PARTS); do \
		[ -d data/librispeech/LibriSpeech/$$part ] && continue; \
		curl -fL -C - -o data/librispeech/$$part.tar.gz https://www.openslr.org/resources/12/$$part.tar.gz \
			&& tar -xzf data/librispeech/$$part.tar.gz -C data/librispeech && rm data/librispeech/$$part.tar.gz || exit 1; \
	done

e02-quick:
	@$(UV) python -I python/e02_ctc.py quick

e02-prepare:
	@$(UV) python -I python/e02_ctc.py prepare

e02-train:
	@$(UV) python -I python/e02_ctc.py train --arm $(ARM) --seed $(SEED)

e02-decode:
	@$(UV) python -I python/e02_ctc.py decode --arm $(ARM) --seed $(SEED)

e02-espnet-check:
	@uv sync --project e02_espnet --locked -q
	@PATH="$(ROOT)/e02_espnet/.venv/bin:$$PATH" PYTHONPATH="$(ROOT)/python/src:$(ROOT)/e02_espnet" python3 e02_espnet/check.py

e02-live-probe:
	@uv sync --project e02_espnet --locked -q
	@PATH="$(ROOT)/e02_espnet/.venv/bin:$$PATH" PYTHONPATH="$(ROOT)/python/src:$(ROOT)/e02_espnet" python3 e02_espnet/live_probe.py

e02-espnet-setup:
	@bash e02_espnet/setup.sh

e02-espnet-prepare:
	@bash e02_espnet/run.sh prepare

e02-espnet:
	@bash e02_espnet/run.sh $(ARM) $(SEED)

clean:
	@rm -rf .cache work vendor python/.venv e02_espnet/.venv
