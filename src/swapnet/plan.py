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

"""swapnet's implementation of the stage-4 producer contract.

    def plan(problem, hardware, k, seed) -> Schedule

:func:`plan_schedule` wires stages 3 and 4 -- :func:`swapnet.map_layout` then
:func:`swapnet.route` -- and adapts the result to the neutral
:class:`~swapnet.schedule.Schedule`. It is *one implementation* of the contract, not a
super-API: the README runs the same two stages by hand, and every knob is reachable there.
Use this when you want the default wiring, and go stage by stage when you want to inspect
or replace a stage.

It gets the coupling powers right by construction, because it asks
:meth:`~swapnet.problem.Problem.mapper_couplings` and
:meth:`~swapnet.problem.Problem.router_couplings` rather than deciding itself -- see
:mod:`swapnet.problem` for why that distinction is load-bearing.

THE PREFIX PROPERTY
-------------------

swapnet's routing loop is greedy per round and ``max_depth`` only stops it, so the plan
for budget ``k`` is the ``k``-layer prefix of the plan for any larger budget. Sweeping
``k`` therefore costs **one** routing run plus
:meth:`~swapnet.schedule.Schedule.truncate`, not one run per ``k``.
:func:`plan_schedule_sweep` does that. An algorithm that reorders its rounds under a
smaller budget does not have this property and must be re-invoked per ``k``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import _core
from .api import map_layout, route
from .milp import SolverOptions
from .problem import DEFAULT_TILT, Problem
from .schedule import Schedule, from_routing_plan
from .schedule import validate as validate_schedule

__all__ = [
    "plan_schedule",
    "plan_schedule_sweep",
    "DEFAULT_MAX_DETOUR",
    "DEFAULT_MAX_COMMODITIES",
]

# Without a detour cap the per-round MILP on a dense problem does not solve inside the
# time limit, and a round that times out routes nothing at all -- the budget is spent and
# buys zero terms. Measured on the F2 100-asset instance: 0.1 s with the cap, 60 s
# (timeout) without. This is not a tuning preference, it is what makes dense routing
# terminate.
DEFAULT_MAX_DETOUR = 2
# Offering every pending pair to each round costs solve time without buying reservations:
# terminal capacity allows one routed commodity per qubit per round, so only a handful are
# ever selected.
DEFAULT_MAX_COMMODITIES = 120


def plan_schedule(
    problem: Problem,
    hardware,
    k: int | None = None,
    seed: int = 0,
    *,
    tilt: float | None = DEFAULT_TILT,
    map_params: Any = None,
    map_seconds: float | None = 60.0,
    max_detour: int | None = DEFAULT_MAX_DETOUR,
    max_commodities_per_round: int | None = DEFAULT_MAX_COMMODITIES,
    max_swaps_per_commodity: int | None = None,
    route_options: Any = None,
    solver: Any = None,
    solver_options: SolverOptions | None = None,
    validate: bool = True,
    return_plan: bool = False,
):
    """Map and route ``problem`` onto ``hardware`` within ``k`` SWAP layers.

    ``k`` is the budget in parallel SWAP layers: ``0`` forbids every swap, ``None`` means
    unlimited. ``seed`` makes the run reproducible -- both the annealing schedule and the
    MILP tie-breaking take it.

    Returns a :class:`~swapnet.schedule.Schedule`. With ``return_plan=True`` returns
    ``(schedule, routing_plan)``, the second being swapnet's own diagnostic object.

    ``route_options`` and ``map_params`` override the assembled defaults wholesale; the
    named arguments are the knobs worth reaching for. See the README's knob table.
    """
    hardware.check_capacity(problem.n)

    interaction = problem.interaction_graph()
    edge_weights = hardware.edge_weights

    if map_params is None:
        map_params = _core.QapParams()
    mapping = map_layout(
        hardware.graph,
        interaction,
        params=map_params,
        seed=seed,
        max_seconds=map_seconds,
        edge_weights=edge_weights,
        couplings=problem.mapper_couplings(),
    )

    if route_options is None:
        route_options = _core.RouteParams()
        if max_commodities_per_round is not None:
            route_options.max_commodities_per_round = int(max_commodities_per_round)
        if max_detour is not None:
            route_options.routing.max_detour = int(max_detour)
        if max_swaps_per_commodity is not None:
            route_options.max_swaps_per_commodity = int(max_swaps_per_commodity)

    routing_plan = route(
        hardware.graph,
        interaction,
        mapping,
        seed=seed,
        edge_weights=edge_weights,
        couplings=problem.router_couplings(tilt),
        max_depth=-1 if k is None else int(k),
        options=route_options,
        solver=solver,
        solver_options=solver_options,
    )

    info = {
        "producer": "swapnet.plan_schedule",
        "seed": seed,
        "k": k,
        "tilt": tilt,
        "max_detour": max_detour,
        "max_commodities_per_round": max_commodities_per_round,
        "max_swaps_per_commodity": max_swaps_per_commodity,
        "mapping_cost": mapping.cost,
        "mapping_restarts_run": mapping.restarts_run,
        "converged": routing_plan.converged,
        "hit_depth_limit": routing_plan.hit_depth_limit,
        "realized_fraction": routing_plan.realized_fraction,
    }
    schedule = from_routing_plan(routing_plan, problem.n, info=info)

    if validate:
        validate_schedule(schedule, problem.n, hardware, k=k)
        # With tilt=None the router's own accounting IS the sum-J^2 coverage, so the two
        # must agree exactly. When they disagree the couplings went in at the wrong power,
        # which is silent in every other observable -- see swapnet.problem.
        if tilt is None:
            covered = problem.covered_fraction(schedule.executed_pairs())
            if abs(routing_plan.realized_fraction - covered) > 1e-9:
                raise AssertionError(
                    f"realized_fraction {routing_plan.realized_fraction:.6f} != "
                    f"sum-J^2 coverage {covered:.6f}; the router received couplings at "
                    "the wrong power"
                )

    return (schedule, routing_plan) if return_plan else schedule


def plan_schedule_sweep(
    problem: Problem, hardware, budgets: Iterable[int], seed: int = 0, **kwargs
) -> dict[int, Schedule]:
    """One :class:`~swapnet.schedule.Schedule` per budget, from a single routing run.

    Exploits the prefix property described in the module docstring: plans once at
    ``max(budgets)`` and truncates. Returns ``{k: schedule}``.

    Only valid for a producer whose schedule is a prefix under ``k``, which swapnet's is.
    Verify it for a new producer by planning at each ``k`` separately and comparing.
    """
    wanted = sorted({int(k) for k in budgets})
    if not wanted:
        return {}
    if wanted[0] < 0:
        raise ValueError(f"budgets must be non-negative, got {wanted[0]}")
    full = plan_schedule(problem, hardware, wanted[-1], seed, **kwargs)
    return {k: full.truncate(k) for k in wanted}
