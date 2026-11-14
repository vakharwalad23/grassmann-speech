import math

import torch

from grassmann_speech.plucker import pair_indices, plucker, plucker_norm_sq, symmetric_norm_sq, symmetric_products, unit

torch.manual_seed(0)
R = 6
I, J = pair_indices(R)


def rand(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float64)


def test_dimension_is_r_choose_2():
    assert plucker(rand(3, R), rand(3, R), I, J).shape == (3, math.comb(R, 2))


def test_antisymmetric_and_zero_on_parallel_vectors():
    a, b = rand(4, R), rand(4, R)
    assert torch.allclose(plucker(a, b, I, J), -plucker(b, a, I, J))
    assert torch.allclose(plucker(a, 2.5 * a, I, J), torch.zeros(4, I.numel(), dtype=torch.float64))


def test_norms_from_dot_products_match_the_coordinates():
    a, b = rand(5, R), rand(5, R)
    assert torch.allclose(plucker(a, b, I, J).norm(dim=-1, keepdim=True) ** 2, plucker_norm_sq(a, b))
    assert torch.allclose(symmetric_products(a, b, I, J).norm(dim=-1, keepdim=True) ** 2, symmetric_norm_sq(a, b))


def test_change_of_basis_scales_by_determinant():
    a, b = rand(R), rand(R)
    alpha, beta, gamma, delta = 0.7, -1.3, 2.1, 0.4
    p_new = plucker(alpha * a + beta * b, gamma * a + delta * b, I, J)
    assert torch.allclose(p_new, (alpha * delta - beta * gamma) * plucker(a, b, I, J))


def test_unit_coordinates_depend_only_on_the_plane():
    a, b = rand(R), rand(R)
    theta = 0.9
    rotated = (math.cos(theta) * a - math.sin(theta) * b, math.sin(theta) * a + math.cos(theta) * b)
    assert torch.allclose(unit(plucker(*rotated, I, J), 1e-12), unit(plucker(a, b, I, J), 1e-12))
    assert not torch.allclose(unit(symmetric_products(*rotated, I, J), 1e-12), unit(symmetric_products(a, b, I, J), 1e-12))


def test_plucker_relation_holds_for_r4():
    i, j = pair_indices(4)
    p = plucker(rand(4), rand(4), i, j)
    p01, p02, p03, p12, p13, p23 = p
    assert torch.allclose(p01 * p23 - p02 * p13 + p03 * p12, torch.zeros((), dtype=torch.float64))
