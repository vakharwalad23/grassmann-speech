"""ESPnet's Conformer encoder with the self-attention of every block swapped for this repo's mixer; all else stays ESPnet's."""

import torch
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.tasks.asr import encoder_choices
from grassmann_speech.conformer import mixer_module
from torch import nn


class MixerAsAttention(nn.Module):
    """Takes ESPnet's attention call (query, key, value, [pos_emb,] mask) and runs a mask-aware mixer on the normed query."""

    def __init__(self, mixer: nn.Module):
        super().__init__()
        self.mixer = mixer

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *pos_and_mask: torch.Tensor) -> torch.Tensor:
        mask, pos = pos_and_mask[-1], pos_and_mask[0] if len(pos_and_mask) == 2 else None
        valid = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device) if mask is None else mask[:, 0].bool()
        return self.mixer(query, valid, pos)


class MixerConformerEncoder(ConformerEncoder):
    def __init__(self, input_size: int, mixer: str, grassmann_rank: int = 32, grassmann_offsets: tuple[int, ...] = (1, 2, 4, 8, 12, 16),
                 **conformer):
        super().__init__(input_size, **conformer)
        if mixer not in ("grassmann", "symmetric", "symmetric_signed", "none"):
            raise ValueError(f"mixer must be grassmann, symmetric, symmetric_signed or none; attention is ESPnet's own conformer: {mixer}")
        for layer in self.encoders:
            layer.self_attn = MixerAsAttention(mixer_module(mixer, self.output_size(), 0, grassmann_rank, grassmann_offsets, 0.0))


def register() -> None:
    encoder_choices.classes["mixer_conformer"] = MixerConformerEncoder
