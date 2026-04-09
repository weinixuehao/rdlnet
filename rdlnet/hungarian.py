"""Bipartite matching (Jonker–Volgenant via scipy if available, else brute force for small problems)."""

from __future__ import annotations

from typing import Tuple

import torch

try:
    from scipy.optimize import linear_sum_assignment as _lsa_scipy
except ImportError:
    _lsa_scipy = None


def linear_sum_assignment(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    cost: [n, m] float tensor.
    Returns (row_ind, col_ind) as long tensors on same device.
    """
    if _lsa_scipy is not None:
        r, c = _lsa_scipy(cost.detach().cpu().numpy())
        device = cost.device
        return torch.as_tensor(r, dtype=torch.long, device=device), torch.as_tensor(
            c, dtype=torch.long, device=device
        )

    n, m = cost.shape
    if n <= m:
        return _brute_min(cost, n, m)
    r, c = _brute_min(cost.T, m, n)
    return c, r


def _brute_min(cost: torch.Tensor, n: int, m: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enumerate injections when n<=m and n is tiny (num_queries <= 8)."""
    import itertools

    device = cost.device
    best = None
    best_perm = None
    cols = range(m)
    for comb in itertools.combinations(cols, n):
        for perm in itertools.permutations(comb):
            s = sum(float(cost[i, perm[i]]) for i in range(n))
            if best is None or s < best:
                best = s
                best_perm = perm
    assert best_perm is not None
    r = torch.arange(n, device=device, dtype=torch.long)
    c = torch.tensor(best_perm, device=device, dtype=torch.long)
    return r, c
