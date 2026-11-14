# Math

Symbols: d model width (256), r reduced width (32), C = r(r-1)/2 Plucker coordinates (496), W the offset set
{1, 2, 4, 8, 12, 16}, L sequence length, h_t the hidden state at position t.

## Plucker coordinates

Two vectors a, b in R^r span a plane. Its Plucker coordinates are the 2x2 minors

```math
p_{ij}(a, b) = a_i b_j - a_j b_i, \qquad 1 \le i < j \le r
```

Properties used by the layer (each one is a unit test in `python/tests/test_plucker.py`):

- **Dimension:** C = r(r-1)/2, so 496 at r = 32.
- **Antisymmetry:** p(b, a) = -p(a, b), and p(a, ca) = 0.
- **Change of basis:** for a' = alpha a + beta b, b' = gamma a + delta b,

```math
p(a', b') = (\alpha\delta - \beta\gamma)\, p(a, b)
```

- **Plane only:** after unit normalization p / |p|, any basis of the same plane with positive determinant
  gives the same vector. The coordinates describe the plane, not the two vectors.
- **Plucker relation** (r = 4): p_12 p_34 - p_13 p_24 + p_14 p_23 = 0. Not every vector in R^C is a plane.

## Norm without the coordinates

The Lagrange identity gives the norm from three dot products:

```math
|p(a, b)|^2 = |a|^2 |b|^2 - (a \cdot b)^2 = |a|^2 |b|^2 \sin^2\theta
```

theta is the angle between a and b. Worked example: a = (1, 0, 0), b = (1, 1, 0) gives p = (p_12, p_13, p_23)
= (1, 0, 0), so |p|^2 = 1; the identity gives 1 x 2 - 1^2 = 1.

The pair geometry logged in e01 is sin theta = |p| / (|a| |b|). Near 0 means a and b are almost parallel and
the plane is ill-defined.

## The layer

```math
z_t = W_{red} h_t + b_{red}, \qquad
\hat p_t = \frac{1}{|W_t|} \sum_{\delta \in W_t} \frac{p(z_{t-\delta}, z_t)}{\max(|p(z_{t-\delta}, z_t)|, \epsilon)}
```

```math
g_t = W_{plu} \hat p_t + b_{plu}, \qquad
\alpha_t = \sigma(W_{gate} [h_t ; g_t] + b_{gate}), \qquad
\tilde h_t = \mathrm{LN}(\alpha_t h_t + (1 - \alpha_t) g_t)
```

W_t = {delta in W : t - delta >= 0} holds the valid offsets at t; g_t = 0 when W_t is empty. eps = 1e-6. The
block output is LN(h~_t + FFN(h~_t)).

## Mean-first form is exact

The paper projects each unit pair vector and then averages:

```math
g_t = \frac{1}{|W_t|} \sum_{\delta \in W_t} \left( W_{plu} \frac{p(z_{t-\delta}, z_t)}{n_{t,\delta}} + b_{plu} \right),
\qquad n_{t,\delta} = \max(|p(z_{t-\delta}, z_t)|, \epsilon)
```

The projection is affine, so the average moves inside it. p is linear in its first argument, and every pair at
t shares the second argument z_t, so the average also moves inside p:

```math
g_t = W_{plu}\, p(s_t, z_t) + b_{plu}, \qquad s_t = \frac{1}{|W_t|} \sum_{\delta \in W_t} \frac{z_{t-\delta}}{n_{t,\delta}}
```

One Plucker vector and one projection per token instead of one per (token, offset). n comes from the Lagrange
identity, so no per-pair coordinates are formed. `test_mean_before_projection_is_exact` checks both forms agree
to 1e-12 in float64.

Two-sided variant (speech): t also pairs with z_t+delta, ordered (z_t, z_t+delta). Antisymmetry turns that
pair into p(-z_t+delta, z_t), so the partner sum takes those terms with sign -1:

```math
s_t = \frac{1}{|W_t^{-}| + |W_t^{+}|} \left( \sum_{\delta \in W_t^{-}} \frac{z_{t-\delta}}{n_{t-\delta,\,t}} - \sum_{\delta \in W_t^{+}} \frac{z_{t+\delta}}{n_{t,\,t+\delta}} \right)
```

## Symmetric control

```math
q_{ij}(a, b) = a_i b_j + a_j b_i, \qquad
|q|^2 = |a|^2 |b|^2 + (a \cdot b)^2 - 2 \sum_i a_i^2 b_i^2
```

Derivation: the sum over i < j of (a_i b_j + a_j b_i)^2 is the sum over i != j of a_i^2 b_j^2, which is
|a|^2 |b|^2 - sum_i a_i^2 b_i^2, plus 2 sum over i < j of a_i b_i a_j b_j, which is (a . b)^2 - sum_i a_i^2 b_i^2.
q is symmetric, q(b, a) = q(a, b), so the two-sided partner sum uses sign +1. q is bilinear too, so the
mean-first form holds unchanged. q changes under a change of basis of the plane: it is not a Grassmann feature.

## Cost

Matmul FLOPs of one mixer pass (forward), checked exactly by `test_mixer_matmul_flops_match_the_cost_formulas`:

```math
F_{mean\text{-}first} = 2Ldr + 2LCd, \qquad
F_{as\ written} = 2Ldr + \sum_{\delta \in W} 2(L - \delta)\, C d
```

Worked example, L = 128, d = 256, r = 32, C = 496: mean-first 2 x 128 x 256 x 32 + 2 x 128 x 496 x 256 =
2.10M + 32.5M = 34.6M; as written 2.10M + 2 x (127 + 126 + 124 + 120 + 116 + 112) x 496 x 256 = 2.10M + 184.1M
= 186.2M. Both are linear in L. Attention adds 4 L^2 d for QK^T and AV on top of its 8 L d^2 projections.

## Padding

With a mask v_t (1 for a real frame, 0 for padding), a pair counts only when both ends are real:

```math
s_t = \frac{\sum_{\delta} v_{t-\delta} v_t \, z_{t-\delta} / n_{t-\delta,\,t}}{\max\left(1, \sum_{\delta} v_{t-\delta} v_t\right)}
```

Real frames then get the same output as the unpadded sequence, and padded frames get g = 0
(`test_padding_changes_nothing_on_real_frames`).

## Receptive field

All offsets in every layer: each layer reaches back 16 positions, so 6 layers reach 6 x 16 = 96 tokens. One
offset per layer: 1 + 2 + 4 + 8 + 12 + 16 = 43 tokens. Both are unit tests
(`test_receptive_field_matches_the_offset_reading`). Two-sided speech layers reach 16 frames each way per layer:
15 layers x 16 x 40 ms = 9.6 s per side.

## Statistics (e01)

```math
\Delta = \overline{\ln \mathrm{PPL}_A} - \overline{\ln \mathrm{PPL}_B}, \qquad
\mathrm{CI}_{95} = \Delta \pm t_{0.975,\,2}\, \sqrt{s_A^2 / n_A + s_B^2 / n_B}
```

Delta is the mean log validation PPL difference over seeds; s the per-arm standard deviation of log PPL; n = 3
seeds; t = 4.303 (2 degrees of freedom, conservative for Welch df 2.1 to 2.3). Reported as ratios exp(Delta).
An interval that contains ratio 1 is a tie.

Worked example, Grassmann vs Transformer: Delta = 5.7991 - 5.5132 = 0.2859, SE 0.0053, CI 0.2631 to 0.3087,
ratio 1.301 to 1.362.

## Speech recognition (e02)

```math
\mathrm{WER} = \frac{S + D + I}{N}
```

S, D, I are substituted, deleted and inserted words after normalization (upper case, whitespace split); N is the
number of reference words. CER is the same over characters. Worked example: reference THE CAT SAT ON A MAT,
hypothesis THE BAT SAT ON MAT: S = 1, D = 1, I = 0, N = 6, so WER = 2 / 6 = 33.3 percent.

Paired bootstrap: speakers are resampled with replacement 1,000 times; each draw gives WER(A) - WER(B) over the
utterances of the drawn speakers. The 2.5 and 97.5 percentiles bound the difference; an interval that holds 0 is a tie.

Learning rate (ESPnet WarmupLR), step k counted from 1, peak lr eta = 2e-3 at w = 15,000:

```math
\eta_k = \eta \, w^{0.5} \min\left(k^{-0.5},\; k\, w^{-1.5}\right)
```

Worked example: k = 7,500 gives eta x 15000^0.5 x 7500 x 15000^-1.5 = eta x 0.5 = 1e-3; k = 60,000 gives
eta x (15000 / 60000)^0.5 = 1e-3.

## Run-time projection (e00)

```math
h_{run} = \frac{A}{\rho \cdot 3600}
```

A is the training audio of one run in seconds, rho the measured audio seconds per wall second. Worked example,
Conformer attention fp32 on an M3 Pro: A = 70 epochs x 3 speed copies x 100.6 h x 3600 = 7.61e7 s, rho = 16 x
12.7 / 0.852 = 238.5, so h = 7.61e7 / (238.5 x 3600) = 88.6 h.
