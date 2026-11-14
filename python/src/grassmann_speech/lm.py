from collections.abc import Sequence

import torch
from torch import nn

from grassmann_speech.blocks import GrassmannBlock, TransformerBlock


class LanguageModel(nn.Module):
    """Token + learned absolute position embeddings, a block stack, final LayerNorm, head tied to the embedding."""

    def __init__(self, blocks: Sequence[nn.Module], vocab: int, d: int, max_len: int, dropout: float):
        super().__init__()
        # PyTorch's N(0, 1) embedding init on purpose: N(0, 0.02) collapsed the post-LN Transformer to unigram loss (e01 parity attempt 1, docs/experiments.md)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device)))
        for block in self.blocks:
            h = block(h)
        return self.ln(h) @ self.tok.weight.T


def grassmann_lm(
    layer_offsets: Sequence[Sequence[int]], vocab: int = 30522, d: int = 256, r: int = 32, d_ff: int = 1024,
    max_len: int = 128, dropout: float = 0.1, **mixer_kw,
) -> LanguageModel:
    """layer_offsets[l] is the offset set of layer l; the paper leaves all-offsets vs one-offset-per-layer open (docs/architecture.md, A2)."""
    blocks = [GrassmannBlock(d, r, offsets, d_ff, dropout, two_sided=False, **mixer_kw) for offsets in layer_offsets]
    return LanguageModel(blocks, vocab, d, max_len, dropout)


def transformer_lm(
    layers: int, vocab: int = 30522, d: int = 256, heads: int = 4, d_ff: int = 1024, max_len: int = 128, dropout: float = 0.1,
) -> LanguageModel:
    return LanguageModel([TransformerBlock(d, heads, d_ff, dropout) for _ in range(layers)], vocab, d, max_len, dropout)
