import pytest
import torch
import torch.nn.functional as F

from grassmann_speech.lm import grassmann_lm, transformer_lm

PAPER_6L = [(1, 2, 4, 8, 12, 16)] * 6
SMALL = {"vocab": 50, "d": 32, "d_ff": 64}


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def small(kind: str, layer_offsets=PAPER_6L) -> torch.nn.Module:
    torch.manual_seed(0)
    model = grassmann_lm(layer_offsets, r=8, **SMALL) if kind == "grassmann" else transformer_lm(len(layer_offsets), heads=4, **SMALL)
    return model.eval()


def test_parameter_counts_match_the_reimplementations():
    # Paper reports 13.00M / 12.59M (6L) and 18.16M / 17.32M (12L, block 256); Grassmann is short by d^2+d per layer (docs/architecture.md, A6)
    assert n_params(grassmann_lm(PAPER_6L)) == 12_607_168
    assert n_params(transformer_lm(6)) == 12_585_472
    assert n_params(grassmann_lm([(1, 2, 4, 8, 12, 16)] * 12)) == 17_367_424
    assert n_params(transformer_lm(12)) == 17_324_032


def test_offset_schedule_does_not_change_parameter_count():
    one_per_layer = [(o,) for o in (1, 2, 4, 8, 12, 16)]
    assert n_params(small("grassmann", one_per_layer)) == n_params(small("grassmann"))


@pytest.mark.parametrize("kind", ["grassmann", "transformer"])
def test_logits_ignore_future_tokens(kind: str):
    model, tokens = small(kind), torch.randint(0, 50, (2, 64))
    for t in (0, 10, 40):
        changed = tokens.clone()
        changed[:, t + 1 :] = torch.randint(0, 50, changed[:, t + 1 :].shape)
        assert torch.allclose(model(tokens)[:, : t + 1], model(changed)[:, : t + 1], atol=1e-6)


@pytest.mark.parametrize(("layer_offsets", "reach"), [(PAPER_6L, 96), ([(o,) for o in (1, 2, 4, 8, 12, 16)], 43)])
def test_receptive_field_matches_the_offset_reading(layer_offsets, reach: int):
    model, tokens, t = small("grassmann", layer_offsets), torch.randint(0, 50, (1, 128)), 120
    base = model(tokens)[:, t]

    def moved(s: int) -> bool:
        changed = tokens.clone()
        changed[:, s] = (changed[:, s] + 1) % 50
        return not torch.equal(model(changed)[:, t], base)

    assert moved(t - reach) and not moved(t - reach - 1)


@pytest.mark.parametrize("kind", ["grassmann", "transformer"])
def test_overfits_one_batch(kind: str):
    model, tokens = small(kind).train(), torch.randint(0, 50, (4, 33))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss = F.cross_entropy(model(tokens[:, :-1]).flatten(0, 1), tokens[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]

