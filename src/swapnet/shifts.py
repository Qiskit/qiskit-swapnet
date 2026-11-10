# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Stage 1: split the couplings into an analytically-implementable part and a residual.

Routing is expensive and truncation is lossy, so the cheapest way to cover more of
``sum J^2`` is to *not route* the part of ``J`` that some other module can implement for
free. Two such parts are available, and both reduce to **one layer of RZ rotations plus a
global phase** -- no two-qubit gates, no routing, no depth budget:

    rank 0   J_ij ~ c                sum_{i<j} z_i z_j        = ((sum_i z_i)^2 - n) / 2
    rank 1   J_ij ~ b_i b_j          sum_{i<j} b_i b_j z_iz_j = ((b . z)^2 - |b|^2) / 2

Both right-hand sides are functions of a single weighted sum of ``z``, which is why they
cost one rotation layer rather than a network. An LCU or phase-oracle module consumes the
term; swapnet routes only :meth:`Shift.residual`.

WHAT "OPTIMISING THE SHIFT" MEANS
---------------------------------

Fit the shift to minimise what is left over, in the same measure the depth budget is
scored in::

    rank 0    min_c    sum_{i<j} (J_ij - c)^2
    rank 1    min_b    sum_{i<j} (J_ij - b_i b_j)^2

**Rank 0** has the closed form ``c* = mean_{i<j} J_ij``, and the split is exactly
orthogonal::

    sum J^2 = m c*^2 + sum d^2          m = number of pairs, d_ij = J_ij - c*

because ``sum d = 0`` kills the cross term. So the absorbed fraction is ``m c*^2 / sum
J^2`` with no approximation.

**Rank 1** is a symmetric rank-one fit with a **free diagonal** -- the objective runs over
``i < j`` only, so ``b_i^2`` is unconstrained and must not be fitted to anything. Solved
by alternating: take the top eigenpair of the current matrix, set ``b`` from it, then
reset the diagonal to ``b_i^2`` and repeat. That is strictly better than one plain
eigendecomposition, which implicitly fits a zero diagonal it was never scored on.

MEASURED, ON THE F2 100-ASSET MATRICES
--------------------------------------

    matrix                rank 0    rank 1    plain top eigenpair
    objective_low_vol     29.68%    99.75%    96.63%
    v2_high_vol            0.52%    99.98%    89.77%

Two things to read off. Rank 1 is worth far more than rank 0 on both, and the alternation
earns real ground over the plain eigendecomposition. And ``v2_high_vol`` shows rank 0 can
be nearly worthless -- its mass sits in a few huge entries that no constant can match --
so the fit is worth *measuring* per problem rather than assuming.

A CAUTION WORTH STATING
-----------------------

``absorbed_fraction`` is a Frobenius-type measure on the couplings. Absorbing 99.75% of
``sum J^2`` does **not** mean the shifted term reproduces the energy landscape to 99.75%,
and it does not mean the residual is negligible to the optimiser: the residual still has
every pair in it, just with small individual weights. What the number does say is exactly
what the depth budget is scored on, which is why it is the number reported here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .problem import Problem

__all__ = ["Shift", "rank0", "rank1"]


@dataclass(frozen=True)
class Shift:
    """A fitted coupling shift: what to implement analytically, and what is left to route.

    ``kind`` is ``"rank0"`` or ``"rank1"``. ``c`` is set for rank 0, ``b`` for rank 1.
    ``operator`` is the identity a consumer needs in order to implement the shifted term,
    as a human-readable string -- the module docstring gives both.
    """

    kind: str
    n: int
    absorbed_fraction: float
    total_j_squared: float
    residual_j_squared: float
    operator: str
    c: float | None = None
    b: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def shifted_couplings(self) -> dict[tuple[int, int], float]:
        """``{(i, j): the shifted-out J_ij}``, for every pair ``i < j``.

        This is what the analytic module must realise. Its sum against ``z_i z_j``
        equals the closed form in :attr:`operator`.
        """
        if self.kind == "rank0":
            value = float(self.c)
            return {(i, j): value for i in range(self.n) for j in range(i + 1, self.n)}
        b = np.asarray(self.b, dtype=float)
        return {
            (i, j): float(b[i] * b[j])
            for i in range(self.n)
            for j in range(i + 1, self.n)
        }

    def residual(self, problem: Problem) -> Problem:
        """``problem`` with the shift removed from every coupling. Route this.

        ``fields`` and ``offset`` are carried through untouched: the shift moves two-body
        weight into a separately-implemented term, it does not change the one-body part or
        the constant. Whoever implements the shifted term is responsible for its own
        constant (``-n/2`` or ``-|b|^2/2`` from the identities above).
        """
        shifted = self.shifted_couplings()
        return problem.with_couplings(
            {pair: value - shifted[pair] for pair, value in problem.couplings.items()},
            label=(
                f"{problem.label}-{self.kind}residual"
                if problem.label
                else f"{self.kind}residual"
            ),
            metadata={
                "shift_kind": self.kind,
                "shift_absorbed_fraction": self.absorbed_fraction,
            },
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "n": self.n,
            "absorbed_fraction": self.absorbed_fraction,
            "total_j_squared": self.total_j_squared,
            "residual_j_squared": self.residual_j_squared,
            "operator": self.operator,
            "c": None if self.c is None else float(self.c),
            "b": None if self.b is None else [float(x) for x in self.b],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Shift:
        b = payload.get("b")
        return cls(
            kind=str(payload["kind"]),
            n=int(payload["n"]),
            absorbed_fraction=float(payload["absorbed_fraction"]),
            total_j_squared=float(payload["total_j_squared"]),
            residual_j_squared=float(payload["residual_j_squared"]),
            operator=str(payload["operator"]),
            c=None if payload.get("c") is None else float(payload["c"]),
            b=None if b is None else np.asarray(b, dtype=float),
            metadata=dict(payload.get("metadata", {})),
        )

    def __repr__(self) -> str:
        return (
            f"<Shift {self.kind} n={self.n} "
            f"absorbs={self.absorbed_fraction:.2%} of sum J^2>"
        )


def _pair_vector(problem: Problem) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pairs = problem.pairs()
    return np.array([problem.couplings[p] for p in pairs], dtype=float), pairs


def rank0(problem: Problem) -> Shift:
    """Fit ``min_c sum_{i<j} (J_ij - c)^2``. Closed form: ``c* = mean J_ij``.

    Requires the complete graph to be meaningful, since the shifted operator
    ``sum_{i<j} z_i z_j`` runs over *every* pair: on a sparse problem the shift would add
    couplings that were not there. Raises in that case rather than silently densifying.
    """
    if not problem.is_complete:
        raise ValueError(
            f"rank0 needs a complete interaction graph -- the operator "
            f"sum_{{i<j}} z_i z_j covers all {problem.n * (problem.n - 1) // 2} pairs, "
            f"but the problem has {problem.num_pairs}. Shifting would invent couplings."
        )
    j, _ = _pair_vector(problem)
    total = float(j @ j)
    c = float(j.mean())
    residual = j - c
    residual_mass = float(residual @ residual)
    m = len(j)

    # The split is exact, not approximate: sum d = 0 kills the cross term. Assert it, so a
    # future change to the fit cannot quietly break the accounting.
    assert abs(total - (m * c * c + residual_mass)) <= 1e-9 * max(
        total, 1.0
    ), "the rank-0 split is not orthogonal; c* is not the mean"

    return Shift(
        kind="rank0",
        n=problem.n,
        absorbed_fraction=0.0 if total <= 0 else 1.0 - residual_mass / total,
        total_j_squared=total,
        residual_j_squared=residual_mass,
        operator="sum_{i<j} z_i z_j = ((sum_i z_i)^2 - n) / 2",
        c=c,
        metadata={"num_pairs": m},
    )


def rank1(problem: Problem, *, iterations: int = 200, tolerance: float = 1e-14) -> Shift:
    """Fit ``min_b sum_{i<j} (J_ij - b_i b_j)^2``, diagonal free.

    Alternates between taking the top eigenpair and resetting the diagonal to ``b_i^2``.
    The diagonal is not part of the objective, so leaving it at zero -- what a single
    eigendecomposition does -- fits a value that is never scored and costs real accuracy;
    the measured gap is in the module docstring.

    ``metadata`` records ``iterations_used`` and the plain top-eigenpair fraction, so the
    alternation's contribution is visible rather than asserted.
    """
    if not problem.is_complete:
        raise ValueError(
            f"rank1 needs a complete interaction graph -- the operator "
            f"sum_{{i<j}} b_i b_j z_i z_j covers all "
            f"{problem.n * (problem.n - 1) // 2} pairs, but the problem has "
            f"{problem.num_pairs}. Shifting would invent couplings."
        )
    j, pairs = _pair_vector(problem)
    total = float(j @ j)
    upper = tuple(np.array(pairs).T)  # (rows, cols) for indexing the fitted outer product

    matrix = problem.matrix()
    np.fill_diagonal(matrix, 0.0)

    # The plain top eigenpair, kept for comparison so the alternation is measurable.
    values, vectors = np.linalg.eigh(matrix)
    plain = np.sqrt(max(values[-1], 0.0)) * vectors[:, -1]
    plain_residual = j - np.outer(plain, plain)[upper]
    plain_fraction = (
        0.0 if total <= 0 else 1.0 - float(plain_residual @ plain_residual) / total
    )

    b = plain
    used = 0
    for iteration in range(1, iterations + 1):
        used = iteration
        values, vectors = np.linalg.eigh(matrix)
        top = values[-1]
        if top <= 0.0:
            break
        b = np.sqrt(top) * vectors[:, -1]
        updated = matrix.copy()
        np.fill_diagonal(updated, b * b)
        if np.allclose(updated, matrix, rtol=0.0, atol=tolerance):
            matrix = updated
            break
        matrix = updated

    residual = j - np.outer(b, b)[upper]
    residual_mass = float(residual @ residual)
    return Shift(
        kind="rank1",
        n=problem.n,
        absorbed_fraction=0.0 if total <= 0 else 1.0 - residual_mass / total,
        total_j_squared=total,
        residual_j_squared=residual_mass,
        operator="sum_{i<j} b_i b_j z_i z_j = ((b . z)^2 - |b|^2) / 2",
        b=np.asarray(b, dtype=float),
        metadata={
            "num_pairs": len(j),
            "iterations_used": used,
            "plain_top_eigenpair_fraction": plain_fraction,
        },
    )
