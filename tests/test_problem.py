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

"""The Problem artifact, and above all the coupling-power asymmetry it exists to own."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import factor_covariance

from swapnet.problem import DEFAULT_TILT, Problem


def test_from_matrix_takes_j_directly():
    J = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    problem = Problem.from_matrix(J)
    assert problem.n == 3
    assert problem.couplings == {(0, 1): 1.0, (0, 2): 2.0, (1, 2): 3.0}
    assert problem.total_j_squared == pytest.approx(1 + 4 + 9)
    assert problem.is_complete
    assert np.array_equal(problem.matrix(), J)


def test_from_matrix_rejects_asymmetry_rather_than_guessing():
    J = np.array([[0.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        Problem.from_matrix(J)


def test_couplings_are_canonicalised_and_duplicates_rejected():
    problem = Problem(n=3, couplings={(1, 0): 5.0})
    assert problem.couplings == {(0, 1): 5.0}
    with pytest.raises(ValueError, match="given twice"):
        Problem(n=3, couplings={(0, 1): 1.0, (1, 0): 2.0})
    with pytest.raises(ValueError, match="self-coupling"):
        Problem(n=3, couplings={(1, 1): 1.0})
    with pytest.raises(ValueError, match="outside"):
        Problem(n=3, couplings={(0, 7): 1.0})


def test_drop_zeros_is_opt_in_because_a_zero_is_still_a_term():
    J = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    assert Problem.from_matrix(J).num_pairs == 3  # the zero is kept
    sparse = Problem.from_matrix(J, drop_zeros=True)
    assert sparse.num_pairs == 2 and not sparse.is_complete


# -- the reason this class exists --------------------------------------------


def test_mapper_gets_j_squared_and_router_gets_the_root():
    """The mapper's objective is linear in couplings, the router's is quadratic.

    Both target sum J^2, so the powers handed to them must differ. This is the invariant
    that a caller passing raw J would break silently.
    """
    problem = Problem.from_matrix(np.array([[0.0, 2.0], [2.0, 0.0]]))
    assert problem.mapper_couplings() == {(0, 1): 4.0}  # J^2

    untilted = problem.router_couplings(tilt=None)
    assert untilted == {(0, 1): 2.0}  # |J|
    # squared by the router -> J^2, which is the objective
    assert untilted[(0, 1)] ** 2 == pytest.approx(4.0)


def test_untilted_router_couplings_square_to_the_scored_objective():
    """For every pair: router_couplings(None)^2 == J^2. The whole point of the choice."""
    problem = Problem.from_covariance(factor_covariance(n=20, seed=3), sigma_scale=0.5)
    untilted = problem.router_couplings(tilt=None)
    for pair, value in problem.couplings.items():
        assert untilted[pair] ** 2 == pytest.approx(value * value)


def test_the_tilt_compresses_the_range_but_keeps_the_order():
    problem = Problem.from_covariance(factor_covariance(n=30, seed=5), sigma_scale=0.5)
    mass = problem.mapper_couplings()
    tilted = problem.router_couplings(DEFAULT_TILT)

    # Reward is the square. Every pair pays at least 1; the heaviest gets 1 + tilt.
    rewards = {p: v * v for p, v in tilted.items()}
    assert min(rewards.values()) >= 1.0
    assert max(rewards.values()) == pytest.approx(1.0 + DEFAULT_TILT)
    # Compression is real: the raw J^2 spread is orders of magnitude wider.
    raw_spread = max(mass.values()) / min(mass.values())
    assert max(rewards.values()) / min(rewards.values()) < raw_spread

    # Ordering is preserved, which is what makes the tilt a proxy and not a distortion.
    by_mass = sorted(mass, key=mass.get)
    by_reward = sorted(rewards, key=rewards.get)
    assert by_mass == by_reward


def test_tilt_must_be_positive_or_none():
    problem = Problem.from_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="positive or None"):
        problem.router_couplings(tilt=0.0)


def test_all_zero_couplings_do_not_divide_by_zero():
    problem = Problem(n=3, couplings={(0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0})
    assert set(problem.router_couplings().values()) == {1.0}
    assert problem.covered_fraction([(0, 1)]) == 0.0


# -- coverage accounting -----------------------------------------------------


def test_covered_fraction_is_invariant_to_rescaling_j():
    """The reason coverage *fraction* is the number to compare across conventions."""
    sigma = factor_covariance(n=25, seed=11)
    half = Problem.from_covariance(sigma, sigma_scale=0.5)
    full = Problem.from_covariance(sigma, sigma_scale=1.0)
    assert full.total_j_squared == pytest.approx(4.0 * half.total_j_squared)

    pairs = sorted(half.couplings)[:40]
    assert half.covered_fraction(pairs) == pytest.approx(full.covered_fraction(pairs))


def test_j_squared_rejects_a_pair_that_is_not_in_the_problem():
    problem = Problem.from_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    with pytest.raises(KeyError):
        problem.j_squared([(0, 5)])


# -- the substitutions -------------------------------------------------------


def test_from_qubo_reproduces_the_binary_objective():
    rng = np.random.default_rng(4)
    Q = rng.normal(size=(6, 6))
    Q = 0.5 * (Q + Q.T)
    problem = Problem.from_qubo(Q, scale=1.0)
    assert problem.verify(Q) < 1e-12


def test_from_qubo_symmetrises_and_keeps_the_diagonal_out_of_the_couplings():
    Q = np.array([[3.0, 2.0], [0.0, 5.0]])  # asymmetric, with a linear part
    problem = Problem.from_qubo(Q)
    # Only Q_ij + Q_ji is observable, so J_01 = (2+0)/2 / 2.
    assert problem.couplings == {(0, 1): pytest.approx(0.5)}
    assert problem.verify(Q) < 1e-12  # the diagonal landed in fields/offset


def test_from_covariance_reproduces_the_portfolio_objective():
    sigma = factor_covariance(n=8, seed=2)
    mu = np.random.default_rng(9).normal(size=8)
    problem = Problem.from_covariance(sigma, mu, sigma_scale=0.5, mu_scale=-1.0)
    assert problem.verify(sigma, mu) < 1e-12
    assert problem.metadata["quadratic_scale"] == 0.5
    assert problem.metadata["linear_scale"] == -1.0


def test_verify_would_catch_a_broken_one_body_part():
    """The check has to be sensitive, or running it proves nothing."""
    sigma = factor_covariance(n=8, seed=2)
    problem = Problem.from_covariance(sigma, sigma_scale=0.5)
    broken = Problem(
        n=problem.n,
        couplings=problem.couplings,
        fields=problem.fields * 1.5,
        offset=problem.offset,
        metadata=problem.metadata,
    )
    assert broken.verify(sigma) > 1e-6


# -- plumbing ----------------------------------------------------------------


def test_round_trips_through_json():
    problem = Problem.from_covariance(
        factor_covariance(n=12, seed=6), sigma_scale=0.5, label="demo"
    )
    again = Problem.from_dict(json.loads(json.dumps(problem.to_dict())))
    assert again.to_dict() == problem.to_dict()
    assert again.couplings == problem.couplings
    assert np.allclose(again.fields, problem.fields)


def test_interaction_graph_matches_the_couplings():
    problem = Problem.from_matrix(
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]]), drop_zeros=True
    )
    graph = problem.interaction_graph()
    assert graph.number_of_nodes() == 3  # isolated vertices are kept
    assert {frozenset(e) for e in graph.edges()} == {frozenset((0, 1)), frozenset((1, 2))}


def test_with_couplings_preserves_the_one_body_part():
    problem = Problem.from_covariance(factor_covariance(n=6, seed=1), sigma_scale=0.5)
    replaced = problem.with_couplings(dict.fromkeys(problem.couplings, 0.0))
    assert replaced.total_j_squared == 0.0
    assert np.allclose(replaced.fields, problem.fields)
    assert replaced.offset == problem.offset


def test_concentration_reports_where_the_mass_sits():
    problem = Problem.from_covariance(factor_covariance(n=40, seed=7), sigma_scale=0.5)
    facts = problem.concentration()
    assert facts["num_pairs"] == 40 * 39 // 2
    assert facts["total_j_squared"] == pytest.approx(problem.total_j_squared)
    # A covariance from a positive market factor has heavy concentration.
    assert facts["pairs_for_50pct_of_j2"] < facts["pairs_for_90pct_of_j2"]
    assert facts["pairs_for_99pct_of_j2"] <= facts["num_pairs"]
