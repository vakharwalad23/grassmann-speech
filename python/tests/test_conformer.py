import pytest
import torch

from grassmann_speech.conformer import MIXERS, conformer_ctc, rel_shift

# Per block: two FFN 1,052,160 + conv 206,592 + final LN 512; subsampling 1,838,080, final LN 512, CTC head 1,285,000
SHARED = 15 * (1_052_160 + 206_592 + 512) + 1_838_080 + 512 + 1_285_000


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_parameter_counts_match_the_espnet_formulas():
    # Mixer with its LN: rel-pos attention 5d^2 + 6d + 2d = 329,728; gated Grassmann dr + r + C(r,2)d + d + 2d^2 + d + 2d = 267,296
    assert n_params(conformer_ctc(15, "attention")) == SHARED + 15 * 329_728 == 26_958_472
    assert n_params(conformer_ctc(15, "grassmann")) == n_params(conformer_ctc(15, "symmetric")) == SHARED + 15 * 267_296
    # No mixer keeps its LN (512) and widens the FFN to 1344 to stay within 0.1 percent of attention
    assert n_params(conformer_ctc(15, "none", d_ff=1344)) == 26_945_032


def small(mixer: str) -> torch.nn.Module:
    torch.manual_seed(0)
    return conformer_ctc(2, mixer, d=64, d_ff=128, r=8, channels=32, vocab=30).eval()


@pytest.mark.parametrize("mixer", MIXERS)
def test_output_is_ctc_log_probs_at_40_ms(mixer: str):
    out, lengths = small(mixer)(torch.randn(2, 1270, 80), torch.tensor([1270, 900]))
    assert out.shape == (2, 316, 30)
    assert lengths.tolist() == [316, 224]
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, 316), atol=1e-5)


@pytest.mark.parametrize("mixer", MIXERS)
def test_padding_changes_nothing_on_real_frames(mixer: str):
    model, feats = small(mixer), torch.randn(1, 900, 80)
    alone, n = model(feats, torch.tensor([900]))
    padded, _ = model(torch.cat([feats, 50 * torch.randn(1, 370, 80)], dim=1), torch.tensor([900]))
    assert torch.allclose(padded[:, : n.item()], alone, atol=1e-5)


def test_rel_shift_aligns_scores_to_relative_positions():
    t = 5
    scores = torch.randn(2, 3, t, 2 * t - 1)
    expected = torch.stack([torch.stack([scores[..., i, t - 1 - i + j] for j in range(t)], -1) for i in range(t)], -2)
    assert torch.equal(rel_shift(scores), expected)
