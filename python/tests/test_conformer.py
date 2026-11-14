import torch

from grassmann_speech.conformer import conformer_ctc


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_parameter_counts_match_the_spec_formulas():
    # docs/architecture.md formulas: 26,957,960 with rel-pos MHSA; plain MHSA drops d^2 + 2d per block (position projection and two biases)
    assert n_params(conformer_ctc(15, "attention")) == 26_957_960 - 15 * (256**2 + 2 * 256)
    assert n_params(conformer_ctc(15, "grassmann")) == 26_021_480


def test_output_is_ctc_log_probs_at_40_ms():
    model = conformer_ctc(2, "grassmann", vocab=30).eval()
    out = model(torch.randn(2, 1270, 80))
    assert out.shape == (2, 316, 30)
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, 316), atol=1e-5)
