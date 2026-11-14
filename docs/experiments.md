# Experiments

What each run measured, how it was set up, and the command that repeats it. Numbers: [results.md](results.md).

```mermaid
flowchart LR
    E0["e00 speed probe"] --> Q["e01 quick"]
    Q --> P["e01 parity: Transformer seed 0"]
    P --> F["e01 full sweep: 10 runs"]
```

## e00: speed and memory probe

Question: how fast does each arm train on an Apple GPU, and does `ctc_loss` on MPS match CPU?

| section | arms | sizes |
|---|---|---|
| CTC loss parity | MPS vs CPU | T 316, batch 4, vocab 5000, 60 targets |
| blocks | attention (causal SDPA), Grassmann as written, mean-first, two-sided; d 256, r 32, FFN 1024 | length 128 to 4096, 16,384 tokens per step |
| e01 LM step | Transformer and Grassmann, 6 layers | batch 32 x 128, vocab 30,522 |
| Conformer-CTC step | 15 x 256, attention vs two-sided Grassmann; fp32 and bf16 autocast | batch 8 and 16 utterances of 12.7 s, vocab 5000 |

- Inputs: synthetic tensors, seed 0. No download.
- Timing: 3 warm-up steps per arm, then 20 rounds that time every arm once (interleaved), MPS synchronize around
  each step. Median reported.
- FLOPs: `torch.utils.flop_counter` on CPU at batch 1, scaled by batch; matmuls and convolutions only.
- Output: `reports/e00_mps_probe/<UTC time>.json`, written after every section.

```bash
make probe-quick   # code paths, tiny sizes, ~10 s
make probe         # the probe; refuses to start without AC power
```

Run 1 found the Grassmann layer 5x slower than attention at 128 tokens, with time spent in elementwise pair
work. The mean-first (anchored) form ([math.md](math.md#mean-first-form-is-exact)) replaced it; run 2 measured
3.5x faster Grassmann steps with identical outputs.

## e01 quick

Checks the code paths of all 10 e01 runs before any long run: 1 epoch of 5 batches per run, 5 evaluation
batches, output to `work/e01/quick/`. Not a result.

```bash
make e01-quick     # ~25 s on an M3 Pro
```

It confirms the exact parameter counts (12,585,472 and 12,607,168), the token counts of the splits (train
2,303,695, validation 238,656, test 273,178) and that every arm trains and evaluates.

## e01 parity: Transformer, seed 0

Question: does the Transformer baseline reproduce the paper (val PPL 248.4) before any Grassmann number is read?
Pass band: 241.0 to 253.6.

```bash
make e01-parity    # ~0.8 h on an M3 Pro
```

- Attempt 1 failed: embeddings at N(0, 0.02) with gradient clip 1.0 stalled at unigram loss (best val PPL 1388.4).
- Attempt 2 passed at 247.1 with the optimizer, schedule and init of kanenorman's reimplementation
  ([github.com/kanenorman/grassmann](https://github.com/kanenorman/grassmann), 248.7): PyTorch default N(0, 1)
  embeddings, embedding dropout 0.1, no gradient clip.

## e01 full sweep

Question: does this Causal Grassmann layer reproduce the paper's Wikitext-2 result at 13M parameters, and does
the Plucker structure beat a same-size control without it?

| arm | model | seeds |
|---|---|---|
| transformer | 6 layers, d 256, 4 heads, FFN 1024, post-LN | 0, 1, 2 |
| grassmann | 6 layers, r 32, all offsets {1, 2, 4, 8, 12, 16} per layer, unit Plucker, mean, gate, post-LN | 0, 1, 2 |
| control_symmetric | grassmann with a_i b_j + a_j b_i in place of the Plucker minors | 0, 1, 2 |
| grassmann_one_offset | layer l uses offset l of {1, 2, 4, 8, 12, 16} | 0 |

Shared settings (`python/e01_layer_parity.toml`):

| setting | value |
|---|---|
| data | Wikitext-2-raw-v1, `Salesforce/wikitext` @ b08601e04326c79dfdd32d625aee71d232d685c3 |
| tokenizer | bert-base-uncased WordPiece, `google-bert/bert-base-uncased` @ 86b5e0934494bd15c9632b12f734a8a67f723594, no special tokens |
| blocks | 128 tokens, contiguous, non-overlapping |
| training | batch 32, 30 epochs, AdamW lr 1e-3, weight decay 0.01, cosine over epochs, no clip, dropout 0.1 |
| selection | best validation PPL over epochs; test PPL read at that epoch |

```bash
make e01           # 10 runs in config order, ~9 h on an M3 Pro; finished runs are skipped
```

Each run writes `reports/e01_layer_parity/<arm>_s<seed>.json` after every epoch: history, best epoch,
environment and commit. Grassmann arms also log the pair geometry at the end.

Outcome: the Transformer reproduces the paper (247.9 +- 2.3); Grassmann lands 33 percent above it (paper: 11);
the symmetric control ties Grassmann, so the Plucker geometry adds nothing measurable here; one offset per layer
does better than all offsets (one seed).

Limits:
- The paper names no tokenizer, optimizer or schedule; the recipe above is tuned for neither arm.
- One seed for one offset per layer, so its edge is not yet a measured difference.
- The paper reports validation perplexity only; test perplexity here is extra.
