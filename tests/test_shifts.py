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

"""The rank-0 and rank-1 coupling shifts, and the accounting behind them."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import factor_covariance

from swapnet import shifts
from swapnet.problem import Problem


def demo_problem(n: int = 60, seed: int = 7) -> Problem:
    return Problem.from_covariance(
        factor_covariance(n=n, seed=seed), sigma_scale=0.5, label="factor"
    )


# -- rank 0 ------------------------------------------------------------------


def test_rank0_is_the_mean_and_the_split_is_exactly_orthogonal():
    problem = demo_problem()
    shift = shifts.rank0(problem)

    j = np.array([problem.couplings[p] for p in problem.pairs()])
    assert shift.c == pytest.approx(j.mean())

    # sum J^2 = m c^2 + sum d^2, with no cross term, because sum d = 0.
    residual = j - shift.c
    assert float(residual @ residual) == pytest.approx(shift.residual_j_squared)
    assert len(j) * shift.c**2 + shift.residual_j_squared == pytest.approx(
        shift.total_j_squared
    )
    assert shift.absorbed_fraction == pytest.approx(
        len(j) * shift.c**2 / shift.total_j_squared
    )


def test_rank0_is_optimal_over_c():
    """It is a fit, so perturbing c must make the residual worse. Otherwise it is not."""
    problem = demo_problem(n=40)
    shift = shifts.rank0(problem)
    j = np.array([problem.couplings[p] for p in problem.pairs()])

    for delta in (-0.3, -0.01, 0.01, 0.3):
        worse = j - (shift.c + delta * abs(shift.c or 1.0))
        assert float(worse @ worse) > shift.residual_j_squared


# -- rank 1 ------------------------------------------------------------------


def test_rank1_beats_the_plain_top_eigenpair():
    """The free diagonal is the point; a single eigendecomposition fits a zero it is
    never scored on."""
    problem = demo_problem()
    shift = shifts.rank1(problem)
    plain = shift.metadata["plain_top_eigenpair_fraction"]
    assert shift.absorbed_fraction >= plain - 1e-12
    assert shift.metadata["iterations_used"] >= 1


def test_rank1_residual_matches_the_reported_mass():
    problem = demo_problem()
    shift = shifts.rank1(problem)
    b = np.asarray(shift.b)
    j = np.array([problem.couplings[p] for p in problem.pairs()])
    fitted = np.array([b[a] * b[c] for a, c in problem.pairs()])
    residual = j - fitted
    assert float(residual @ residual) == pytest.approx(shift.residual_j_squared)


def test_rank1_recovers_an_exactly_rank_one_matrix():
    """A ground truth case: if J really is b b', the residual must be ~0."""
    rng = np.random.default_rng(3)
    b = rng.uniform(0.5, 2.0, size=25)
    J = np.outer(b, b)
    np.fill_diagonal(J, 0.0)
    problem = Problem.from_matrix(J)

    shift = shifts.rank1(problem)
    assert shift.absorbed_fraction > 1 - 1e-9
    # b is determined only up to a global sign, so compare the outer product.
    assert np.allclose(np.outer(shift.b, shift.b), np.outer(b, b), atol=1e-6)


def test_rank1_absorbs_more_than_rank0_on_a_dispersed_covariance():
    """The measured story: a scalar cannot follow the volatility scaling, a rank-1 can."""
    problem = demo_problem()
    assert (
        shifts.rank1(problem).absorbed_fraction > shifts.rank0(problem).absorbed_fraction
    )


# -- the residual as a Problem ----------------------------------------------


@pytest.mark.parametrize("kind", ["rank0", "rank1"])
def test_residual_is_a_problem_carrying_the_reported_mass(kind):
    problem = demo_problem(n=30)
    shift = getattr(shifts, kind)(problem)
    residual = shift.residual(problem)

    assert isinstance(residual, Problem)
    assert residual.n == problem.n
    assert residual.num_pairs == problem.num_pairs
    assert residual.total_j_squared == pytest.approx(shift.residual_j_squared)
    assert residual.metadata["shift_kind"] == kind
    # The one-body part and the constant are the shift's business, not changed by it.
    assert np.allclose(residual.fields, problem.fields)
    assert residual.offset == problem.offset


@pytest.mark.parametrize("kind", ["rank0", "rank1"])
def test_shifted_plus_residual_reconstructs_the_original(kind):
    """The split has to be a split: adding the two parts back must give J."""
    problem = demo_problem(n=30)
    shift = getattr(shifts, kind)(problem)
    residual = shift.residual(problem)
    shifted = shift.shifted_couplings()

    for pair, value in problem.couplings.items():
        assert residual.couplings[pair] + shifted[pair] == pytest.approx(value)


# -- the analytic operator ---------------------------------------------------


def test_the_rank0_operator_identity_holds():
    """sum_{i<j} z_i z_j == ((sum z)^2 - n)/2, which is what makes it one RZ layer."""
    rng = np.random.default_rng(1)
    n = 12
    for _ in range(20):
        z = rng.choice([-1.0, 1.0], size=n)
        direct = sum(z[i] * z[j] for i in range(n) for j in range(i + 1, n))
        closed = (z.sum() ** 2 - n) / 2
        assert direct == pytest.approx(closed)


def test_the_rank1_operator_identity_holds():
    """sum_{i<j} b_i b_j z_i z_j == ((b.z)^2 - |b|^2)/2."""
    rng = np.random.default_rng(2)
    n = 12
    b = rng.normal(size=n)
    for _ in range(20):
        z = rng.choice([-1.0, 1.0], size=n)
        direct = sum(b[i] * b[j] * z[i] * z[j] for i in range(n) for j in range(i + 1, n))
        closed = ((b @ z) ** 2 - b @ b) / 2
        assert direct == pytest.approx(closed)


def test_shifted_couplings_sum_to_the_closed_form():
    """Tie the artifact to the identity: the term it describes is the term it sums to."""
    problem = demo_problem(n=20)
    rng = np.random.default_rng(5)
    for kind in ("rank0", "rank1"):
        shift = getattr(shifts, kind)(problem)
        shifted = shift.shifted_couplings()
        for _ in range(10):
            z = rng.choice([-1.0, 1.0], size=problem.n)
            direct = sum(v * z[i] * z[j] for (i, j), v in shifted.items())
            if kind == "rank0":
                closed = shift.c * (z.sum() ** 2 - problem.n) / 2
            else:
                b = np.asarray(shift.b)
                closed = ((b @ z) ** 2 - b @ b) / 2
            assert direct == pytest.approx(closed)


# -- guardrails --------------------------------------------------------------


@pytest.mark.parametrize("kind", ["rank0", "rank1"])
def test_a_sparse_problem_is_refused_rather_than_densified(kind):
    """Both operators run over every pair, so shifting a sparse problem would invent
    couplings that were not in it."""
    problem = Problem(n=4, couplings={(0, 1): 1.0, (2, 3): 2.0})
    with pytest.raises(ValueError, match="complete interaction graph"):
        getattr(shifts, kind)(problem)


@pytest.mark.parametrize("kind", ["rank0", "rank1"])
def test_round_trips_through_json(kind):
    shift = getattr(shifts, kind)(demo_problem(n=20))
    again = shifts.Shift.from_dict(json.loads(json.dumps(shift.to_dict())))
    assert again.kind == shift.kind
    assert again.absorbed_fraction == pytest.approx(shift.absorbed_fraction)
    if shift.b is None:
        assert again.b is None and again.c == pytest.approx(shift.c)
    else:
        assert np.allclose(again.b, shift.b)
