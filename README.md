# grassmann-speech

A tested reimplementation of the attention-free Causal Grassmann layer from Zhang Chong, "Attention Is Not What
You Need" ([arXiv 2512.19428](https://arxiv.org/abs/2512.19428)), a language-model parity check against the
paper, and two code paths for a speech recognition test of the layer in place of self-attention (Conformer-CTC on
LibriSpeech-100, ESPnet and a plain PyTorch trainer). The attention baseline reproduces the published WER (9.72 / 22.90
on test-clean / test-other against 9.8 / 23.3); the Grassmann runs are in progress ([results](docs/results.md)).

The layer reduces each token state to 32 dimensions, pairs it with states at fixed offsets {1, 2, 4, 8, 12, 16},
encodes every pair as the Plucker coordinates of the plane it spans, averages, and gates the result back into
the hidden state. Cost is linear in sequence length.

## Results

Wikitext-2, 6 layers, d 256, 3 seeds, best validation perplexity (lower is better):

| model | params | val PPL | paper |
|---|---|---|---|
| **Transformer** | 12,585,472 | **247.9 +- 2.3** | 248.4 |
| Grassmann | 12,607,168 | 330.0 +- 0.5 (+33 percent, 95% CI 30 to 36) | 275.7 (+11 percent) |
| Grassmann with symmetric products instead of Plucker coordinates | 12,607,168 | 327.0 +- 2.0 (tie with Grassmann) | - |
| Grassmann, one offset per layer (1 seed) | 12,607,168 | 312.6 | - |

- The Transformer baseline reproduces the paper, so the pipeline holds.
- The Grassmann model is 33 percent worse than the Transformer, three times the paper's gap.
- A same-size control without the Grassmann geometry does as well: the Plucker structure adds nothing
  measurable in this setting.
- A mean-first form of the layer, exact to 1e-12, trains 3.5x faster than the paper's form on an Apple GPU.

All numbers: [docs/results.md](docs/results.md).

## Quick start

Needs [uv](https://docs.astral.sh/uv/). Python 3.15 and torch 2.15.0 come from `python/uv.lock`.

```bash
make setup        # python/.venv from the lock file
make test         # 63 unit checks, CPU, ~5 s
make e01-quick    # every e01 run for 5 batches, ~25 s
make e01-parity   # Transformer seed 0, ~0.8 h on an M3 Pro; expect val PPL near 248
make e01          # the full sweep, 10 runs, ~9 h on an M3 Pro; resumable
make probe        # e00 speed probe on an Apple GPU, ~4 to 6 min, AC power
make e02-quick    # e02 own trainer, every arm on a synthetic corpus, ~10 s
make e02-espnet-check  # e02 ESPnet track on CPU: sizes, frontend parity, 1-epoch train and decode, ~2 min
make help         # every target
```

Runs pick CUDA, then MPS, then CPU. The e00 probe times Apple GPUs only. e02 training needs a CUDA GPU with 40 GB or
more; the full commands are in [docs/experiments.md](docs/experiments.md#e02-grassmann-vs-attention-in-a-conformer-ctc-on-librispeech-100-running).

## Layout

| path | holds |
|---|---|
| `python/src/grassmann_speech/plucker.py` | Plucker coordinates, symmetric products, norms from dot products |
| `python/src/grassmann_speech/mixer.py` | `GrassmannMixer`: causal or two-sided, paper form or mean-first, Plucker or symmetric |
| `python/src/grassmann_speech/blocks.py` | Grassmann block and the post-LN Transformer block |
| `python/src/grassmann_speech/lm.py` | language models for e01 |
| `python/src/grassmann_speech/conformer.py` | Conformer-CTC laid out as ESPnet's; arms attention, grassmann, symmetric, none |
| `python/src/grassmann_speech/features.py` | log-mel as ESPnet's frontend, speed perturbation, SpecAugment |
| `python/src/grassmann_speech/librispeech.py` | LibriSpeech lists, sealed test manifests, tokenizer, batching, dataset |
| `python/src/grassmann_speech/wer.py` | the scorer: WER, CER, paired bootstrap over speakers |
| `python/src/grassmann_speech/env.py` | device pick and the environment stamp stored with every run |
| `python/tests/` | unit checks: Plucker identities, causality, receptive field, exactness, parameter counts, FLOPs |
| `python/e00_mps_probe.py` | e00 speed and memory probe |
| `python/e01_layer_parity.py`, `.toml` | e01 language-model parity, config |
| `python/e02_ctc.py`, `.toml` | e02 own trainer: prepare, train, decode |
| `e02_espnet/` | e02 ESPnet track: encoder plug-in, launcher, arm configs, setup, run and check scripts, train.log summary (`summary.py`), its own uv env |
| `docs/` | [architecture](docs/architecture.md), [math](docs/math.md), [experiments](docs/experiments.md), [results](docs/results.md) |

Downloads and caches go to `.cache/`, intermediate data to `work/`, run reports to `reports/`; all three are
ignored by git. `make clean` deletes `.cache/`, `work/` and the virtual environment.

## Credits

- Layer: Zhang Chong, arXiv 2512.19428 (December 2025).
- e01 optimizer, schedule and init follow kanenorman's reimplementation,
  [github.com/kanenorman/grassmann](https://github.com/kanenorman/grassmann).
- Data: Wikitext-2 (CC BY-SA 3.0, GFDL) and the bert-base-uncased tokenizer (Apache-2.0), fetched at pinned
  revisions from the Hugging Face Hub; LibriSpeech (CC BY 4.0) from OpenSLR.
- e02 recipe: ESPnet (Apache-2.0), `egs2/librispeech_100/asr1` @ v.202611; the relative-position attention follows
  ESPnet's `RelPositionMultiHeadedAttention`.

## License

MIT, see [LICENSE](LICENSE). Data and third-party code keep their own licenses (Credits above).
