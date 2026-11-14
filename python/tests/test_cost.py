import torch
from torch.utils.flop_counter import FlopCounterMode

from grassmann_speech.blocks import GrassmannBlock, TransformerBlock

D = 64


def flops(module: torch.nn.Module, length: int) -> int:
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        module(torch.randn(1, length, D))
    return counter.get_total_flops()


def grassmann(mean_first: bool) -> GrassmannBlock:
    return GrassmannBlock(D, 16, (1, 2, 4, 8, 12, 16), 4 * D, 0.0, two_sided=False, mean_first=mean_first).eval()


def test_grassmann_cost_is_linear_in_length():
    block = grassmann(mean_first=True)
    assert flops(block, 2048) == 2 * flops(block, 1024)


def test_attention_cost_grows_faster_than_linear():
    block = TransformerBlock(D, 4, 4 * D, 0.0).eval()
    assert flops(block, 2048) > 2.5 * flops(block, 1024)


def test_mixer_matmul_flops_match_the_cost_formulas():
    # docs/math.md: mean-first projects once per token; as written, once per valid (token, offset) pair
    length, r, offsets = 512, 16, (1, 2, 4, 8, 12, 16)
    c = r * (r - 1) // 2
    reduce = length * 2 * D * r
    assert flops(grassmann(mean_first=True).mixer, length) == reduce + length * 2 * c * D
    assert flops(grassmann(mean_first=False).mixer, length) == reduce + sum(length - o for o in offsets) * 2 * c * D
