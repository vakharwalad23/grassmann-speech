import pytest
import torch

from grassmann_speech.mixer import GrassmannMixer

D, R, OFFSETS, L = 16, 8, (1, 2, 4), 40


def mixer(two_sided: bool, **kw) -> GrassmannMixer:
    torch.manual_seed(0)
    return GrassmannMixer(D, R, OFFSETS, two_sided, **kw).double()


def x(length: int = L) -> torch.Tensor:
    return torch.randn(2, length, D, dtype=torch.float64)


def influencing_positions(m: GrassmannMixer, t: int) -> set[int]:
    inp = x().requires_grad_()
    m(inp)[:, t].sum().backward()
    return set(torch.nonzero(inp.grad.abs().sum(dim=(0, 2))).flatten().tolist())


@pytest.mark.parametrize("two_sided", [False, True])
def test_shape(two_sided: bool):
    assert mixer(two_sided)(x()).shape == (2, L, D)


def test_causal_output_ignores_the_future():
    m, a = mixer(False), x()
    for t in (0, 5, 20, L - 2):
        b = a.clone()
        b[:, t + 1 :] = torch.randn_like(b[:, t + 1 :])
        assert torch.equal(m(a)[:, : t + 1], m(b)[:, : t + 1])


def test_receptive_field_is_the_offsets():
    t = 20
    assert influencing_positions(mixer(False), t) == {t} | {t - o for o in OFFSETS}
    assert influencing_positions(mixer(True), t) == {t} | {t - o for o in OFFSETS} | {t + o for o in OFFSETS}


@pytest.mark.parametrize("two_sided", [False, True])
@pytest.mark.parametrize("features", ["plucker", "symmetric"])
def test_mean_before_projection_is_exact(two_sided: bool, features: str):
    paper, fast = mixer(two_sided, mean_first=False, features=features), mixer(two_sided, mean_first=True, features=features)
    fast.load_state_dict(paper.state_dict())
    a = x()
    assert torch.allclose(paper(a), fast(a), atol=1e-12)


def test_position_without_pairs_is_zero():
    assert torch.equal(mixer(False)(x())[:, 0], torch.zeros(2, D, dtype=torch.float64))
    assert torch.equal(mixer(True)(x(1)), torch.zeros(2, 1, D, dtype=torch.float64))


def test_offsets_longer_than_the_sequence_are_skipped():
    out = mixer(False)(x(3))
    assert out.shape == (2, 3, D) and torch.isfinite(out).all()


def test_control_has_the_same_parameter_count():
    count = [sum(p.numel() for p in m.parameters()) for m in (mixer(False), mixer(False, features="symmetric"))]
    assert count[0] == count[1] == D * R + R + (R * (R - 1) // 2) * D + D


@pytest.mark.parametrize("two_sided", [False, True])
def test_mean_first_stays_close_in_float32(two_sided: bool):
    paper, fast = mixer(two_sided, mean_first=False).float(), mixer(two_sided).float()
    fast.load_state_dict(paper.state_dict())
    a = x().float()
    assert torch.allclose(paper(a), fast(a), atol=1e-5)


@pytest.mark.parametrize("two_sided", [False, True])
@pytest.mark.parametrize("mean_first", [False, True])
@pytest.mark.parametrize("features", ["plucker", "symmetric"])
def test_padding_changes_nothing_on_real_frames(two_sided: bool, mean_first: bool, features: str):
    m, a = mixer(two_sided, mean_first=mean_first, features=features), x(25)
    padded = torch.cat([a, 100 * torch.randn(2, L - 25, D, dtype=torch.float64)], dim=1)
    valid = torch.arange(L) < 25
    out = m(padded, valid.expand(2, L))
    assert torch.allclose(out[:, :25], m(a), atol=1e-12)
    assert torch.equal(out[:, 25:], torch.zeros_like(out[:, 25:]))


def test_mix_stays_fp32_under_bf16_autocast():
    m, a = mixer(True).float(), x().float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(a)
    assert out.dtype == torch.float32
    assert torch.allclose(out, m(a), atol=1e-6)
