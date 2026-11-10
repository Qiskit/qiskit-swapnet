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

"""The public entry points.

Three calls, in increasing scope:

* :func:`map_layout` -- pick an initial layout by quadratic assignment.
* :func:`route` -- given a layout, produce the SWAP plan that executes every
  interaction.
* :func:`map_and_route` -- both, which is what the original ``swap.py`` did.

All of them take networkx or rustworkx graphs (or plain edge lists) and return the
dataclasses in :mod:`swapnet.results`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import _core
from .graphs import GraphView, as_graph_view
from .milp import MilpBackend, SolverOptions, get_backend
from .results import (
    Ancilla,
    IterationStats,
    MappingResult,
    RoutingPlan,
    Stage,
    Swap,
)
from .weights import mean_normalized, normalize

__all__ = [
    "map_layout",
    "route",
    "map_and_route",
    "MapParams",
    "RouteOptions",
    "RoutingOptions",
]

# Re-exported so callers can tune the search without importing the extension.
# All three accept their fields as keyword arguments.
MapParams = _core.QapParams
RouteOptions = _core.RouteParams
# The MILP sub-parameters of RouteOptions, as `RouteOptions(routing=...)`.
RoutingOptions = _core.RoutingParams


def _hardware_view(
    hardware: Any,
    edge_weights: Mapping[Any, float] | Sequence[float] | None = None,
    weight_attr: str | None = None,
    rescale: bool = True,
) -> GraphView:
    view = as_graph_view(
        hardware,
        name="hardware graph",
        weight_attr=weight_attr,
        weights=edge_weights,
    )
    view = _rescaled(view, rescale, "edge_weights")
    core = view.core
    if view.num_vertices == 0:
        raise ValueError("the hardware graph has no qubits")
    if not core.is_connected():
        raise ValueError(
            "the hardware graph is disconnected; every pair of physical qubits "
            "needs a connecting path for routing to be possible"
        )
    return view


def _rescaled(view: GraphView, rescale: bool, argument: str) -> GraphView:
    """Normalise a view's weights to mean 1, or check that they already are.

    The search parameters are calibrated against unit-cost edges, so raw
    calibration numbers (error rates near 1e-2, say) would defeat the routing
    model's path-length pruning and flatten the annealing acceptance. Rescaling
    preserves every ratio, so it changes the cost units but not the ordering of
    layouts or routes.
    """
    if view.weights is None:
        return view
    if rescale:
        return view.with_weights(normalize(view.weights))
    if not mean_normalized(view.weights):
        raise ValueError(
            f"{argument} do not have mean 1 and rescale=False was requested; "
            "the annealing schedule and the routing reward are calibrated "
            "against unit-cost edges, so pass rescale=True (the default) or "
            "normalise them yourself with swapnet.weights.normalize"
        )
    return view


def _interaction_indices(
    interaction: Any,
    hardware: GraphView,
    couplings: Mapping[Any, float] | Sequence[float] | None = None,
    coupling_attr: str | None = None,
    rescale: bool = True,
) -> tuple[GraphView, list[tuple[int, int]], list[Any]]:
    """Index the interaction graph into hardware-sized logical slots.

    Slots ``0..n2-1`` hold the interaction-graph vertices; the remaining slots up
    to the hardware size are ancillas. The layout is a permutation over all
    slots, because unused qubits are still moved by SWAPs.
    """
    view = as_graph_view(
        interaction,
        name="interaction graph",
        weight_attr=coupling_attr,
        weights=couplings,
    )
    view = _rescaled(view, rescale, "couplings")
    _ = view.core  # validates; the view caches it for the calls below

    if view.num_vertices > hardware.num_vertices:
        raise ValueError(
            f"the interaction graph has {view.num_vertices} vertices but the "
            f"hardware graph has only {hardware.num_vertices}"
        )

    slot_labels: list[Any] = list(view.labels)
    slot_labels.extend(
        Ancilla(slot) for slot in range(view.num_vertices, hardware.num_vertices)
    )
    return view, list(view.edges), slot_labels


def _layout_dicts(
    layout: Sequence[int], slot_labels: Sequence[Any], hardware: GraphView
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Turn an index permutation into (interaction-only, full) label dicts."""
    full = {
        slot_labels[slot]: hardware.labels[physical]
        for slot, physical in enumerate(layout)
    }
    partial = {
        label: physical
        for label, physical in full.items()
        if not isinstance(label, Ancilla)
    }
    return partial, full


def map_layout(
    hardware: Any,
    interaction: Any,
    *,
    params: _core.QapParams | None = None,
    seed: int | None = None,
    max_seconds: float | None = None,
    progress: Callable[[int, int, float], None] | None = None,
    edge_weights: Mapping[Any, float] | Sequence[float] | None = None,
    edge_weight_attr: str | None = None,
    couplings: Mapping[Any, float] | Sequence[float] | None = None,
    coupling_attr: str | None = None,
    rescale_weights: bool = True,
) -> MappingResult:
    """Choose an initial layout minimising total interaction distance.

    The objective is

        sum over interaction edges (l, m) of  J_lm * D(layout[l], layout[m])

    where ``D`` is the hardware distance -- hop count on an unweighted device, or
    accumulated edge cost when ``edge_weights`` are given. Solved by
    multi-restart simulated annealing followed by tabu search, so results improve
    with ``params.num_restarts`` and are only reproducible when ``seed`` is set.

    Parameters
    ----------
    hardware, interaction:
        networkx or rustworkx graphs, or iterables of ``(u, v)`` or
        ``(u, v, weight)`` items.
    params:
        A :class:`MapParams` for full control of the search schedule.
    seed:
        Overrides ``params.seed``. Any non-zero value makes the search
        reproducible.
    max_seconds:
        Wall-clock budget for the search, checked between restarts.
    progress:
        Called as ``progress(restart, num_restarts, best_cost)``.
    edge_weights, edge_weight_attr:
        Per-coupling traversal costs ``W_ij``, steering the layout onto the
        high-fidelity edges. Give a mapping keyed by qubit pair, a sequence in
        edge order, or the name of a networkx edge attribute to read. Lower means
        better, so pass an error metric -- see :mod:`swapnet.weights` to build
        them from fidelities or error rates.
    couplings, coupling_attr:
        Per-interaction strengths ``J_lm``, so that strongly coupled pairs are
        placed closer together. Same three input forms.
    rescale_weights:
        Normalise both weight sets to mean 1 (the default). The annealing
        schedule is calibrated against unit-cost hops, so raw calibration data
        would distort the search; rescaling preserves all ratios. Pass False only
        if the weights are already mean-1.
    """
    hardware_view = _hardware_view(
        hardware, edge_weights, edge_weight_attr, rescale_weights
    )
    interaction_view, edges, slot_labels = _interaction_indices(
        interaction, hardware_view, couplings, coupling_attr, rescale_weights
    )

    search = params if params is not None else _core.QapParams()
    if seed is not None:
        search.seed = seed
    if max_seconds is not None:
        search.max_seconds = max_seconds

    distances = _distance_matrix(hardware_view)
    result = _core.solve_qap(
        distances,
        edges,
        search,
        interaction_view.weight_list(),
        progress,
    )

    partial, full = _layout_dicts(result.layout, slot_labels, hardware_view)
    return MappingResult(
        layout=partial,
        full_layout=full,
        cost=result.cost,
        restarts_run=result.restarts_run,
        hit_time_limit=result.hit_time_limit,
    )


def _distance_matrix(hardware: GraphView) -> list[list[float]]:
    """All-pairs hardware distances, weighted when the device is.

    Dijkstra reproduces the BFS hop counts exactly on an unweighted graph, but
    BFS is cheaper, so keep using it there. The integer hop matrix needs no cast:
    pybind11 converts it to the double matrix the solver takes.
    """
    if hardware.is_weighted:
        return _core.dijkstra_all_pairs(hardware.core)
    return _core.bfs_all_pairs(hardware.core)


def route(
    hardware: Any,
    interaction: Any,
    layout: MappingResult | dict[Any, Any] | Sequence[int],
    *,
    options: _core.RouteParams | None = None,
    solver: str | MilpBackend | None = None,
    solver_options: SolverOptions | None = None,
    seed: int | None = None,
    progress: Callable[[IterationStats], None] | None = None,
    edge_weights: Mapping[Any, float] | Sequence[float] | None = None,
    edge_weight_attr: str | None = None,
    couplings: Mapping[Any, float] | Sequence[float] | None = None,
    coupling_attr: str | None = None,
    rescale_weights: bool = True,
    max_depth: int | None = None,
) -> RoutingPlan:
    """Produce the SWAP plan that makes every interaction executable.

    Each iteration executes the interactions the current layout already supports,
    then solves a vertex-disjoint multi-commodity flow to find parallel paths for
    as many of the rest as possible, and swaps along those paths. The plan is
    returned as alternating gate batches and SWAP layers.

    Parameters
    ----------
    layout:
        A :class:`MappingResult`, a ``{interaction vertex: hardware qubit}`` dict,
        or a raw slot-indexed permutation. Dicts need not mention ancilla qubits;
        unassigned hardware qubits are filled in automatically.
    options:
        A :class:`RouteOptions` controlling the iteration cap, swaps per
        interaction per round, the routing reward, and model-size limits.
    solver:
        Backend name (``"highs"``, ``"scipy"``, ``"cplex"``) or a callable taking
        a ``MilpProblem`` and returning a ``MilpSolution``. Defaults to the first
        available.
    solver_options:
        Per-solve time limit, MIP gap, thread count and verbosity.
    progress:
        Called with an :class:`IterationStats` after each iteration.
    edge_weights, edge_weight_attr:
        Per-coupling traversal costs ``W_ij``. Routed paths are then priced by the
        total error they accumulate, so the solver prefers high-fidelity edges
        even at the cost of a longer path.
    couplings, coupling_attr:
        Per-interaction strengths ``J_lm``. The reward for realising an
        interaction scales with ``J^2``, so under a depth truncation the router
        spends its budget on the interactions carrying the most weight. Read
        ``plan.realized_weight`` to see how much of the total it captured.
    rescale_weights:
        Normalise both weight sets to mean 1 (the default); see
        :func:`map_layout`.
    max_depth:
        Hard cap on emitted SWAP layers, overriding ``options.max_depth``. When it
        binds, the plan is truncated and the interactions left over appear in
        ``plan.unrouted``; ``plan.hit_depth_limit`` records that this happened.
        ``0`` forbids every SWAP, leaving only the interactions the initial layout
        already supports; a negative value means unlimited.
    """
    hardware_view = _hardware_view(
        hardware, edge_weights, edge_weight_attr, rescale_weights
    )
    interaction_view, edges, slot_labels = _interaction_indices(
        interaction, hardware_view, couplings, coupling_attr, rescale_weights
    )

    permutation = _resolve_layout(layout, interaction_view, hardware_view, slot_labels)

    route_options = options if options is not None else _core.RouteParams()
    if seed is not None:
        route_options.seed = seed
        if route_options.routing.seed == 0:
            route_options.routing.seed = seed
    if max_depth is not None:
        route_options.max_depth = max_depth

    backend = get_backend(solver, solver_options)

    def _forward_progress(stats):
        return progress(_stats_to_dataclass(stats))

    core_progress = None if progress is None else _forward_progress

    result = _core.route(
        hardware_view.core,
        edges,
        permutation,
        route_options,
        backend,
        interaction_view.weight_list(),
        core_progress,
    )

    return _plan_from_core(
        result,
        hardware_view,
        slot_labels,
        mapping_cost=layout.cost if isinstance(layout, MappingResult) else None,
    )


def map_and_route(
    hardware: Any,
    interaction: Any,
    *,
    params: _core.QapParams | None = None,
    options: _core.RouteParams | None = None,
    solver: str | MilpBackend | None = None,
    solver_options: SolverOptions | None = None,
    seed: int | None = None,
    map_seconds: float | None = None,
    map_progress: Callable[[int, int, float], None] | None = None,
    route_progress: Callable[[IterationStats], None] | None = None,
    edge_weights: Mapping[Any, float] | Sequence[float] | None = None,
    edge_weight_attr: str | None = None,
    couplings: Mapping[Any, float] | Sequence[float] | None = None,
    coupling_attr: str | None = None,
    rescale_weights: bool = True,
    max_depth: int | None = None,
) -> RoutingPlan:
    """Map an interaction graph onto a topology, then route it.

    This is the whole pipeline: the equivalent of running the original ``map``
    and ``swap`` executables in sequence, without the intermediate text files.
    The weight arguments are forwarded to both stages, so ``W_ij`` steers the
    layout and the routing consistently, as does ``J_lm``.

    Returns a :class:`RoutingPlan` holding the initial layout, the final layout,
    and the alternating gate/SWAP timeline.
    """
    mapping = map_layout(
        hardware,
        interaction,
        params=params,
        seed=seed,
        max_seconds=map_seconds,
        progress=map_progress,
        edge_weights=edge_weights,
        edge_weight_attr=edge_weight_attr,
        couplings=couplings,
        coupling_attr=coupling_attr,
        rescale_weights=rescale_weights,
    )
    return route(
        hardware,
        interaction,
        mapping,
        options=options,
        solver=solver,
        solver_options=solver_options,
        seed=seed,
        progress=route_progress,
        edge_weights=edge_weights,
        edge_weight_attr=edge_weight_attr,
        couplings=couplings,
        coupling_attr=coupling_attr,
        rescale_weights=rescale_weights,
        max_depth=max_depth,
    )


def _resolve_layout(
    layout: MappingResult | dict[Any, Any] | Sequence[int],
    interaction: GraphView,
    hardware: GraphView,
    slot_labels: Sequence[Any],
) -> list[int]:
    """Normalise any layout form into a slot-indexed permutation of qubits."""
    if isinstance(layout, MappingResult):
        layout = layout.full_layout

    if isinstance(layout, dict):
        slot_of = {label: slot for slot, label in enumerate(slot_labels)}
        permutation = [-1] * hardware.num_vertices
        taken = set()

        for label, physical in layout.items():
            if label not in slot_of:
                raise ValueError(
                    f"layout key {label!r} is not an interaction-graph vertex "
                    "or ancilla slot"
                )
            if physical not in hardware.index_of:
                raise ValueError(
                    f"layout maps {label!r} to {physical!r}, which is not a "
                    "hardware qubit"
                )
            index = hardware.index_of[physical]
            if index in taken:
                raise ValueError(
                    f"two logical qubits are mapped to hardware qubit " f"{physical!r}"
                )
            taken.add(index)
            permutation[slot_of[label]] = index

        # Fill the slots the caller left out with the unused hardware qubits, so
        # that the result is always a full permutation.
        spare = [i for i in range(hardware.num_vertices) if i not in taken]
        for slot, value in enumerate(permutation):
            if value == -1:
                permutation[slot] = spare.pop(0)

        missing = [
            slot_labels[slot]
            for slot in range(interaction.num_vertices)
            if slot not in {slot_of[label] for label in layout}
        ]
        if missing:
            raise ValueError(
                f"layout is missing an assignment for interaction vertices "
                f"{missing!r}"
            )
        return permutation

    permutation = [int(value) for value in layout]
    _core.validate_layout(permutation, hardware.num_vertices)
    return permutation


def _stats_to_dataclass(stats: Any) -> IterationStats:
    return IterationStats(
        iteration=stats.iteration,
        executable_before=stats.executable_before,
        commodities=stats.commodities,
        deferred=stats.deferred,
        routed=stats.routed,
        objective=stats.objective,
        solver_optimal=stats.solver_optimal,
        solver_status=stats.solver_status,
        swaps=stats.swaps,
        layers=stats.layers,
        satisfied_after_swaps=stats.satisfied_after_swaps,
        remaining=stats.remaining,
        realized_weight=stats.realized_weight,
        swap_weight=stats.swap_weight,
    )


def _plan_from_core(
    result: Any,
    hardware: GraphView,
    slot_labels: Sequence[Any],
    mapping_cost: float | None,
) -> RoutingPlan:
    physical = hardware.labels
    logical = tuple(slot_labels)

    stages = tuple(
        Stage(
            gates=tuple((logical[u], logical[v]) for u, v in stage.gates),
            swaps=tuple(
                Swap(
                    physical=(physical[op.physical_u], physical[op.physical_v]),
                    logical=(logical[op.logical_a], logical[op.logical_b]),
                )
                for op in stage.swaps
            ),
        )
        for stage in result.stages
    )

    initial_partial, initial_full = _layout_dicts(
        result.initial_layout, logical, hardware
    )
    final_partial, final_full = _layout_dicts(result.final_layout, logical, hardware)

    return RoutingPlan(
        initial_layout=initial_partial,
        final_layout=final_partial,
        initial_full_layout=initial_full,
        final_full_layout=final_full,
        stages=stages,
        unrouted=tuple((logical[u], logical[v]) for u, v in result.unrouted_edges),
        total_swaps=result.total_swaps,
        num_swap_layers=result.num_layers,
        converged=result.converged,
        mapping_cost=mapping_cost,
        iterations=tuple(_stats_to_dataclass(stats) for stats in result.iterations),
        hit_depth_limit=result.hit_depth_limit,
        realized_weight=result.realized_weight,
        unrealized_weight=result.unrealized_weight,
        swap_weight=result.swap_weight,
    )
