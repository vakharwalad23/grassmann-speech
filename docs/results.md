# Results

Every number measured with this code. Hardware for all of them: Apple M3 Pro (14-core GPU), 18 GB, macOS
27.0.1 (26A434), AC power, torch 2.15.0 on MPS, Python 3.15.0, 2026-11-13 to 2026-11-14. How each run was set
up: [experiments.md](experiments.md).

## e01 language-model parity

Wikitext-2-raw-v1, 6 layers, d 256, block 128. Best validation perplexity over 30 epochs (the paper's protocol);
test perplexity at that epoch, never used for selection.

| arm | params | val PPL, seeds 0 / 1 / 2 | val mean +- sd | test mean | vs Transformer (val) | wall per run |
|---|---|---|---|---|---|---|
| **Transformer** | 12,585,472 | 247.1 / 250.5 / 246.2 | **247.9 +- 2.3** | 260.2 | - | 0.79 h |
| Grassmann, all offsets per layer | 12,607,168 | 329.8 / 329.6 / 330.6 | 330.0 +- 0.5 | 341.5 | +33.1 percent | 0.94 h |
| symmetric control | 12,607,168 | 329.3 / 325.6 / 326.2 | 327.0 +- 2.0 | 337.4 | +31.9 percent | 0.96 h |
| Grassmann, one offset per layer | 12,607,168 | 312.6 / - / - | 312.6 (1 seed) | 334.6 | +26.1 percent | 0.88 h |
| paper, 2512.19428 Table 1 | 12.59M / 13.00M | 248.4 / 275.7 | - | - | +11.0 percent | - |

| comparison (val PPL ratio, 3 seeds each) | ratio | 95% CI | reading |
|---|---|---|---|
| Grassmann / Transformer | 1.331 | 1.301 to 1.362 | Grassmann worse by 30 to 36 percent |
| Grassmann / symmetric control | 1.009 | 0.994 to 1.025 | tie |

CI method: [math.md](math.md#statistics-e01).

### Per run

| arm | seed | best val PPL (epoch) | test PPL | wall |
|---|---|---|---|---|
| Transformer | 0 | 247.1 (20) | 255.2 | 0.79 h |
| Transformer | 1 | 250.5 (24) | 264.2 | 0.79 h |
| Transformer | 2 | 246.2 (23) | 261.4 | 0.79 h |
| Grassmann | 0 | 329.8 (24) | 342.1 | 0.94 h |
| Grassmann | 1 | 329.6 (27) | 340.9 | 0.94 h |
| Grassmann | 2 | 330.6 (27) | 341.6 | 0.94 h |
| symmetric control | 0 | 329.3 (26) | 343.1 | 0.96 h |
| symmetric control | 1 | 325.6 (27) | 334.3 | 0.96 h |
| symmetric control | 2 | 326.2 (26) | 334.8 | 0.96 h |
| Grassmann, one offset per layer | 0 | 312.6 (26) | 334.6 | 0.88 h |

### Transformer parity runs, seed 0

| recipe | epoch 1 | best (epoch) | epoch 30 | train loss at best |
|---|---|---|---|---|
| attempt 1: embeddings N(0, 0.02), gradient clip 1.0 (failed) | 1418.2 | 1388.4 (7) | 1702.2 | 7.208 |
| **attempt 2: default N(0, 1) embeddings, no clip** | 1774.9 | **247.1 (20)** | 253.5 | 4.736 |

Attempt 1 sat at the unigram entropy of the training tokens (7.190 nats) from epoch 1: the model ignored its
input. Parity target: val PPL 241.0 to 253.6.

### Trained pair geometry

sin of the angle between z_t-delta and z_t on 32 validation blocks, per layer and offset, after training.

| run | median sin, all layers and offsets | pairs below sin 0.05 |
|---|---|---|
| Grassmann s0 | 0.84 to 0.99 | at most 0.11 percent (layer 2) |
| Grassmann s1 | 0.90 to 0.99 | at most 0.05 percent (layer 2) |
| Grassmann s2 | 0.86 to 0.99 | 0 |
| symmetric control s0 | 0.80 to 0.99 | at most 0.11 percent (layer 2) |
| Grassmann one offset s0 | 0.98 to 0.99 | 0 |

Offset 1 has the lowest sin in every layer of Grassmann s0: neighbours are the most aligned.

## e00 speed probe

Step = forward + backward + AdamW, median of 20 interleaved runs. Synthetic inputs.

### Run 2, mean-first (anchored) Grassmann form

| setting | attention | Grassmann as written | Grassmann mean-first | Grassmann two-sided |
|---|---|---|---|---|
| one block, L 128 x batch 128 | **0.0449 s** | 0.2223 s | 0.0631 s | 0.0656 s |
| one block, L 512 x batch 32 | 0.0628 s | 0.2276 s | **0.0621 s** | 0.0645 s |
| one block, L 4096 x batch 4 | 0.2459 s | 0.2298 s | **0.0621 s** | 0.0645 s |
| e01 LM step, batch 32 x 128 | **0.1622 s**, 25.3k tokens/s | 0.4183 s | 0.1880 s, 21.8k tokens/s | - |
| Conformer-CTC 15 x 256 step, fp32, batch 16 x 12.7 s | **0.852 s** | - | - | 0.910 s |
| same, bf16 autocast | **0.744 s** | - | - | 0.903 s |
| one LibriSpeech-100 run, 70 epochs x 3 speed copies (projected) | 77 to 89 h | - | - | 94 to 95 h |

Wall 221 s, peak process RSS 2.2 GB.

### Run 1, before the mean-first rewrite

| setting | attention | Grassmann as written | Grassmann mean before projection, first version | Grassmann two-sided |
|---|---|---|---|---|
| one block, L 128 x batch 128 | **0.0445 s** (36 percent of 4.8 TFLOPS) | 0.2208 s | 0.2200 s | 0.2540 s |
| one block, L 1024 x batch 16 | 0.0897 s | 0.2287 s | 0.2266 s | 0.2617 s |
| one block, L 4096 x batch 4 | 0.2475 s | 0.2302 s | **0.2285 s** | 0.2638 s |
| e01 LM step, batch 32 x 128 | **0.1601 s** | 0.4193 s | 0.4165 s | - |
| Conformer-CTC step, fp32, batch 16 | **0.881 s** | - | - | 1.828 s |
| Conformer-CTC step, bf16, batch 16 | **0.757 s** | - | - | 2.277 s |

Grassmann time was flat across lengths and spent in elementwise pair work, not matmuls. Wall 368 s.

### CTC loss on MPS

| check | value |
|---|---|
| loss, CPU vs MPS (T 316, batch 4, vocab 5000) | 41.7267 vs 41.7267, difference 0.0 |
| largest gradient difference | 3.9e-6 |
