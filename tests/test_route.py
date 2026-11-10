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

"""The routing stage and the full pipeline.

Almost every assertion here routes through :func:`swapnet.verify_plan`, which
replays the plan from scratch against the hardware graph. That is the property
that matters: a plan whose SWAPs are all on real couplings, whose layers are
internally disjoint, and whose gates are on adjacent qubits when their turn
comes is a plan a compiler can emit.
"""

from __future__ import annotations

import json

import pytest
from conftest import grid

import swapnet

networkx = pytest.importorskip("networkx")


TOPOLOGIES = {
    "path": lambda: networkx.path_graph(12),
    "grid": lambda: networkx.grid_2d_graph(4, 4),
    "hex": lambda: networkx.hexagonal_lattice_graph(2, 2),
    "heavy_hex_ish": lambda: networkx.hexagonal_lattice_graph(3, 2),
}


@pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
def test_pipeline_produces_a_verifiable_plan(topology, fast_params):
    hardware = TOPOLOGIES[topology]()
    size = hardware.number_of_nodes()
    interaction = networkx.random_regular_graph(3, size - (size % 2), seed=7)

    plan = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=13)

    swapnet.verify_plan(hardware, interaction, plan)
    assert plan.converged
    assert plan.unrouted == ()


def test_every_interaction_is_scheduled_exactly_once(fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.random_regular_graph(3, 14, seed=21)

    plan = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=2)

    scheduled = [frozenset(gate) for gate in plan.scheduled_gates()]
    assert len(scheduled) == len(set(scheduled))
    assert set(scheduled) | {frozenset(e) for e in plan.unrouted} == {
        frozenset(e) for e in interaction.edges()
    }


def test_swaps_lie_on_hardware_couplings(fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.random_regular_graph(3, 12, seed=5)

    plan = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=3)

    couplings = {frozenset(e) for e in hardware.edges()}
    for stage in plan.stages:
        for swap in stage.swaps:
            assert frozenset(swap.physical) in couplings


def test_each_swap_layer_is_internally_disjoint(fast_params):
    hardware = networkx.grid_2d_graph(5, 5)
    interaction = networkx.random_regular_graph(3, 20, seed=9)

    plan = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=4)

    for layer in plan.swap_layers:
        touched = [qubit for swap in layer for qubit in swap.physical]
        assert len(touched) == len(set(touched))


def test_an_already_embeddable_interaction_graph_needs_no_swaps(fast_params):
    # A cycle maps onto a cycle exactly, so the layout alone satisfies every
    # interaction and the plan should be a single gate batch.
    hardware = networkx.cycle_graph(10)
    interaction = networkx.cycle_graph(10)

    plan = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=1)

    swapnet.verify_plan(hardware, interaction, plan)
    assert plan.total_swaps == 0
    assert plan.num_swap_layers == 0
    assert len(plan.stages) == 1
    assert plan.initial_layout == plan.final_layout


def test_route_accepts_a_plain_layout_dict():
    hardware = networkx.path_graph(6)
    interaction = networkx.Graph([(0, 1), (1, 2), (0, 2)])

    # Deliberately spread the triangle out; ancillas are left unspecified.
    layout = {0: 0, 1: 3, 2: 5}
    plan = swapnet.route(hardware, interaction, layout, seed=1)

    swapnet.verify_plan(hardware, interaction, plan)
    assert plan.initial_layout == layout
    assert plan.total_swaps > 0


def test_route_accepts_a_mapping_result(fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.cycle_graph(12)

    mapping = swapnet.map_layout(hardware, interaction, params=fast_params, seed=6)
    plan = swapnet.route(hardware, interaction, mapping, seed=6)

    swapnet.verify_plan(hardware, interaction, plan)
    assert plan.initial_layout == mapping.layout
    assert plan.mapping_cost == mapping.cost


def test_route_rejects_a_layout_that_double_books_a_qubit():
    with pytest.raises(ValueError, match="two logical qubits"):
        swapnet.route(networkx.path_graph(4), networkx.Graph([(0, 1)]), {0: 2, 1: 2})


def test_route_rejects_an_incomplete_layout():
    with pytest.raises(ValueError, match="missing an assignment"):
        swapnet.route(
            networkx.path_graph(4),
            networkx.Graph([(0, 1), (1, 2)]),
            {0: 0, 1: 1},
        )


def test_route_rejects_an_unknown_qubit():
    with pytest.raises(ValueError, match=r"not a\n?.*hardware qubit"):
        swapnet.route(networkx.path_graph(4), networkx.Graph([(0, 1)]), {0: 0, 1: 99})


def test_backends_agree_on_the_plan(backend, fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.random_regular_graph(3, 12, seed=11)

    mapping = swapnet.map_layout(hardware, interaction, params=fast_params, seed=8)
    plan = swapnet.route(hardware, interaction, mapping, solver=backend, seed=8)

    swapnet.verify_plan(hardware, interaction, plan)
    assert plan.converged


def test_a_custom_solver_callable_is_accepted(fast_params):
    from swapnet.milp import HighsBackend

    inner = HighsBackend()
    calls = []

    def solver(problem):
        calls.append((problem.num_vars, problem.num_rows))
        return inner(problem)

    hardware = networkx.grid_2d_graph(3, 3)
    interaction = networkx.cycle_graph(8)
    mapping = swapnet.map_layout(hardware, interaction, params=fast_params, seed=1)
    plan = swapnet.route(hardware, interaction, mapping, solver=solver, seed=1)

    swapnet.verify_plan(hardware, interaction, plan)
    if plan.total_swaps:
        assert calls


def test_a_time_limit_still_yields_a_verifiable_plan(fast_params):
    hardware = networkx.grid_2d_graph(5, 5)
    interaction = networkx.random_regular_graph(4, 24, seed=12)

    mapping = swapnet.map_layout(hardware, interaction, params=fast_params, seed=1)
    plan = swapnet.route(
        hardware,
        interaction,
        mapping,
        solver_options=swapnet.SolverOptions(time_limit=0.05),
        seed=1,
    )

    # A cut-short solve routes fewer interactions per round and may leave some
    # unrouted, which is a documented outcome rather than an error. What must
    # still hold is that everything it did emit is valid.
    swapnet.verify_plan(hardware, interaction, plan, require_converged=False)


def test_max_detour_shrinks_the_model_and_still_verifies(fast_params):
    hardware = networkx.grid_2d_graph(5, 5)
    interaction = networkx.random_regular_graph(3, 20, seed=14)

    mapping = swapnet.map_layout(hardware, interaction, params=fast_params, seed=1)
    options = swapnet.RouteOptions(routing=swapnet.RoutingOptions(max_detour=2))
    plan = swapnet.route(hardware, interaction, mapping, options=options, seed=1)

    swapnet.verify_plan(hardware, interaction, plan)


def test_max_commodities_per_round_is_respected(fast_params):
    hardware = networkx.grid_2d_graph(5, 5)
    interaction = networkx.random_regular_graph(3, 20, seed=15)

    mapping = swapnet.map_layout(hardware, interaction, params=fast_params, seed=1)
    options = swapnet.RouteOptions(max_commodities_per_round=4)
    plan = swapnet.route(hardware, interaction, mapping, options=options, seed=1)

    swapnet.verify_plan(hardware, interaction, plan)
    assert all(stats.commodities <= 4 for stats in plan.iterations)


def test_progress_callback_reports_each_iteration(fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.random_regular_graph(3, 12, seed=16)

    seen = []
    mapping = swapnet.map_layout(hardware, interaction, params=fast_params, seed=1)
    plan = swapnet.route(hardware, interaction, mapping, seed=1, progress=seen.append)

    assert len(seen) == len(plan.iterations)
    assert [s.iteration for s in seen] == [s.iteration for s in plan.iterations]


def test_the_same_seed_gives_the_same_plan(fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.random_regular_graph(3, 12, seed=17)

    first = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=42)
    second = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=42)

    assert first.to_dict() == second.to_dict()


def test_plan_is_json_serialisable(fast_params):
    hardware = networkx.grid_2d_graph(3, 3)
    interaction = networkx.cycle_graph(8)

    plan = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=1)
    restored = json.loads(json.dumps(plan.to_dict()))

    assert restored["converged"] is plan.converged
    assert restored["total_swaps"] == plan.total_swaps
    assert len(restored["stages"]) == len(plan.stages)


def test_totals_match_the_stage_contents(fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.random_regular_graph(3, 14, seed=18)

    plan = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=1)

    assert plan.total_swaps == sum(len(s.swaps) for s in plan.stages)
    assert plan.num_swap_layers == len(plan.swap_layers)
    assert plan.depth == plan.num_swap_layers


def test_integer_edge_lists_work_end_to_end(fast_params):
    edges, _n = grid(4, 4)
    interaction = [(0, 1), (2, 3), (0, 5), (7, 11), (4, 9)]

    plan = swapnet.map_and_route(edges, interaction, params=fast_params, seed=1)

    swapnet.verify_plan(edges, interaction, plan)
    assert plan.converged


def test_an_interaction_graph_with_isolated_vertices_is_fine(fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.Graph()
    interaction.add_nodes_from(range(10))
    interaction.add_edges_from([(0, 1), (2, 3), (4, 5)])

    plan = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=1)

    swapnet.verify_plan(hardware, interaction, plan)
    assert plan.converged


def test_an_empty_interaction_graph_produces_an_empty_plan(fast_params):
    hardware = networkx.grid_2d_graph(3, 3)
    interaction = networkx.Graph()
    interaction.add_nodes_from(range(4))

    plan = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=1)

    assert plan.total_swaps == 0
    assert plan.scheduled_gates() == []
    assert plan.converged
