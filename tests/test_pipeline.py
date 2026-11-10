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

"""All six stages, end to end, and the invariants that hold between them."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FAST_MAP, factor_covariance, grid

import swapnet

FAST = dict(map_params=swapnet.MapParams(**FAST_MAP), map_seconds=10.0)


def grid_hardware(rows: int, cols: int, seed: int = 0) -> swapnet.Hardware:
    """A rows x cols grid with varied per-link errors, so W_ij is not uniform."""
    edges, num_qubits = grid(rows, cols)
    rng = np.random.default_rng(seed)
    errors = {(min(a, b), max(a, b)): float(rng.uniform(0.001, 0.02)) for a, b in edges}
    return swapnet.Hardware.from_error_map(errors, num_qubits=num_qubits, name="grid")


def small_setup(n: int = 12, seed: int = 3):
    problem = swapnet.Problem.from_covariance(
        factor_covariance(n=n, seed=seed), sigma_scale=0.5, label="factor"
    )
    return problem, grid_hardware(4, 4)


# -- the sequence ------------------------------------------------------------


def test_all_six_stages_run_and_the_ansatz_verifies():
    problem, hardware = small_setup()

    shift = swapnet.shifts.rank1(problem)
    residual = shift.residual(problem)

    schedule = swapnet.plan_schedule(residual, hardware, k=2, seed=5, **FAST)
    swapnet.schedule.validate(schedule, residual.n, hardware, k=2)

    layer = swapnet.ansatz.build(schedule, residual, hardware, k=2)
    one, two = swapnet.ansatz.angles_from_problem(layer, residual, gamma=0.1)
    summary = swapnet.ansatz.verify_structure(layer, one, two)

    assert summary["num_rz"] == residual.n
    assert summary["num_rzz"] == len(layer.couplings)
    assert summary["output_layout_matches"]
    bound = layer.bind(one, two)
    assert not bound.parameters
    assert bound.count_ops()["rzz"] == len(layer.couplings)


def test_plan_schedule_respects_the_budget():
    problem, hardware = small_setup()
    for k in (0, 1, 3):
        schedule = swapnet.plan_schedule(problem, hardware, k=k, seed=5, **FAST)
        assert schedule.num_swap_layers <= k
    assert swapnet.plan_schedule(problem, hardware, k=0, seed=5, **FAST).num_swaps == 0


def test_k_zero_still_executes_whatever_the_layout_supports():
    """A budget of zero swaps is not a budget of zero terms."""
    problem, hardware = small_setup()
    schedule = swapnet.plan_schedule(problem, hardware, k=0, seed=5, **FAST)
    assert len(schedule.executed_pairs()) > 0


def test_more_budget_never_covers_less():
    problem, hardware = small_setup()
    plans = swapnet.plan_schedule_sweep(problem, hardware, [0, 1, 2, 3], seed=5, **FAST)
    covered = [problem.covered_fraction(plans[k].executed_pairs()) for k in sorted(plans)]
    assert covered == sorted(covered), f"coverage is not monotone in k: {covered}"


def test_the_sweep_is_the_prefix_of_one_run():
    """The property that turns 11 routing runs into 1. Checked, not assumed."""
    problem, hardware = small_setup()
    budgets = [0, 1, 2, 3]
    plans = swapnet.plan_schedule_sweep(problem, hardware, budgets, seed=5, **FAST)
    biggest = plans[max(budgets)]
    for k in budgets:
        assert biggest.truncate(k).content() == plans[k].content()


def test_truncating_a_plan_equals_re_planning_at_that_budget():
    """The stronger claim behind the sweep: swapnet really is prefix-greedy in k."""
    problem, hardware = small_setup()
    full = swapnet.plan_schedule(problem, hardware, k=3, seed=5, **FAST)
    for k in (0, 1, 2):
        direct = swapnet.plan_schedule(problem, hardware, k=k, seed=5, **FAST)
        assert full.truncate(k).content() == direct.content(), (
            f"truncating to k={k} differs from planning at k={k}; the sweep would be "
            "reporting plans nobody would have produced"
        )


# -- the invariant that catches a coupling-power mistake ---------------------


def test_untilted_realized_fraction_equals_the_scored_coverage():
    """With tilt=None the router's own accounting IS sum-J^2 coverage.

    This is the check that catches couplings handed over at the wrong power -- a mistake
    that is silent in every other observable, because the plan still validates and the
    gate counts still look reasonable.
    """
    problem, hardware = small_setup()
    schedule, plan = swapnet.plan_schedule(
        problem, hardware, k=2, seed=5, tilt=None, return_plan=True, **FAST
    )
    covered = problem.covered_fraction(schedule.executed_pairs())
    assert plan.realized_fraction == pytest.approx(covered, abs=1e-9)


def test_feeding_the_router_j_squared_breaks_that_equality():
    """The negative control: show the invariant is sensitive to the actual mistake.

    Passing J**2 to the router makes it optimise and report sum J^4. Without this test,
    the assertion above could be vacuous.
    """
    problem, hardware = small_setup()
    interaction = problem.interaction_graph()
    plan = swapnet.route(
        hardware.graph,
        interaction,
        swapnet.map_layout(
            hardware.graph,
            interaction,
            params=swapnet.MapParams(**FAST_MAP),
            seed=5,
            max_seconds=10.0,
            edge_weights=hardware.edge_weights,
            couplings=problem.mapper_couplings(),
        ),
        seed=5,
        edge_weights=hardware.edge_weights,
        couplings=problem.mapper_couplings(),  # J**2: the wrong power for routing
        max_depth=2,
    )
    covered = problem.covered_fraction([tuple(sorted(g)) for g in plan.scheduled_gates()])
    assert abs(plan.realized_fraction - covered) > 1e-6, (
        "J**2 into the router reported the same fraction as sum-J^2 coverage, so the "
        "equality above cannot detect the wrong power"
    )


# -- the shift interacts with routing as advertised --------------------------


def test_a_shift_leaves_a_problem_that_still_routes():
    """The residual has to be a first-class Problem, not a special case."""
    problem, hardware = small_setup()
    for shift in (swapnet.shifts.rank0(problem), swapnet.shifts.rank1(problem)):
        residual = shift.residual(problem)
        schedule = swapnet.plan_schedule(residual, hardware, k=2, seed=5, **FAST)
        swapnet.schedule.validate(schedule, residual.n, hardware, k=2)
        layer = swapnet.ansatz.build(schedule, residual, hardware, k=2)
        one, two = swapnet.ansatz.angles_from_problem(layer, residual, gamma=0.1)
        swapnet.ansatz.verify_structure(layer, one, two)


# -- artifacts survive the boundary -----------------------------------------


def test_every_artifact_round_trips_through_json_at_the_stage_boundary():
    """A stage boundary must survive being a file, since that is how modules will meet."""
    import json

    problem, hardware = small_setup()
    shift = swapnet.shifts.rank0(problem)
    schedule = swapnet.plan_schedule(problem, hardware, k=2, seed=5, **FAST)

    for artifact in (problem, hardware, shift, schedule):
        revived = type(artifact).from_dict(json.loads(json.dumps(artifact.to_dict())))
        assert revived.to_dict() == artifact.to_dict(), type(artifact).__name__

    # And the revived artifacts are still usable, not just equal.
    revived_problem = swapnet.Problem.from_dict(problem.to_dict())
    revived_schedule = swapnet.Schedule.from_dict(schedule.to_dict())
    revived_hardware = swapnet.Hardware.from_dict(hardware.to_dict())
    layer = swapnet.ansatz.build(revived_schedule, revived_problem, revived_hardware, k=2)
    one, two = swapnet.ansatz.angles_from_problem(layer, revived_problem, gamma=0.1)
    swapnet.ansatz.verify_structure(layer, one, two)


def test_the_schedule_records_its_producer():
    """Provenance travels with the artifact, so a consumer can say what made it."""
    problem, hardware = small_setup()
    schedule = swapnet.plan_schedule(problem, hardware, k=2, seed=5, **FAST)
    assert schedule.info["producer"] == "swapnet.plan_schedule"
    assert schedule.info["seed"] == 5
    assert schedule.info["k"] == 2
    assert schedule.info["max_detour"] == swapnet.plan.DEFAULT_MAX_DETOUR


def test_capacity_is_checked_before_any_work_happens():
    problem = swapnet.Problem.from_covariance(
        factor_covariance(n=30, seed=1), sigma_scale=0.5
    )
    hardware = grid_hardware(3, 3)  # 9 qubits for a 30-variable problem
    with pytest.raises(ValueError, match="needs 30"):
        swapnet.plan_schedule(problem, hardware, k=1, seed=5, **FAST)


# -- against a real device --------------------------------------------------


@pytest.mark.slow
def test_a_real_backend_runs_the_whole_sequence(fake_hardware):
    problem = swapnet.Problem.from_covariance(
        factor_covariance(n=40, seed=7), sigma_scale=0.5, label="factor40"
    )
    schedule = swapnet.plan_schedule(problem, fake_hardware, k=2, seed=12345, **FAST)
    swapnet.schedule.validate(schedule, problem.n, fake_hardware, k=2)

    layer = swapnet.ansatz.build(schedule, problem, fake_hardware, k=2)
    one, two = swapnet.ansatz.angles_from_problem(layer, problem, gamma=0.1)
    swapnet.ansatz.verify_structure(layer, one, two)
    assert layer.num_device_qubits == 156
    assert layer.metadata["device"] == "fake_boston"
    assert 0.0 < layer.metadata["covered_j2_fraction"] <= 1.0
