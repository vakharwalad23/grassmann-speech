import torch


def pair_indices(r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Index pairs (i, j) with i < j, in the order of the r(r-1)/2 Plucker coordinates."""
    i, j = torch.triu_indices(r, r, offset=1)
    return i, j


def plucker(a: torch.Tensor, b: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """2x2 minors a_i b_j - a_j b_i: Plucker coordinates of the plane span(a, b) (2512.19428 Sec 3.2 step 3)."""
    return a[..., i] * b[..., j] - a[..., j] * b[..., i]


def symmetric_products(a: torch.Tensor, b: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Symmetric counterpart a_i b_j + a_j b_i: same size as plucker, but not a function of the plane alone."""
    return a[..., i] * b[..., j] + a[..., j] * b[..., i]


def dots(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (a * a).sum(-1, keepdim=True), (b * b).sum(-1, keepdim=True), (a * b).sum(-1, keepdim=True)


def plucker_norm_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Lagrange identity: |a|^2 |b|^2 - (a.b)^2, without forming the coordinates."""
    aa, bb, ab = dots(a, b)
    return aa * bb - ab * ab


def symmetric_norm_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa, bb, ab = dots(a, b)
    return aa * bb + ab * ab - 2 * (a * a * b * b).sum(-1, keepdim=True)


def unit(p: torch.Tensor, eps: float) -> torch.Tensor:
    return p / p.norm(dim=-1, keepdim=True).clamp_min(eps)
