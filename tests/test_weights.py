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

"""Weighted mapping and routing.

Two independent extensions are covered here:

* ``W_ij`` on the hardware graph, which should steer both stages away from
  low-fidelity couplings;
* ``J_lm`` on the interaction graph, which should decide *which* interactions
  survive when a depth truncation means not all of them can.

The controlling property throughout is that uniform weights must reproduce the
unweighted pipeline exactly -- otherwise the weighting has changed the algorithm
rather than parameterised it.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest
from conftest import FAST_MAP, grid

import swapnet
from swapnet import _core

# --------------------------------------------------------------- weights module


def test_normalize_preserves_ratios_and_sets_mean_one():
    raw = [0.002, 0.004, 0.008, 0.002]
    scaled = swapnet.normalize(raw)

    assert math.isclose(sum(scaled) / len(scaled), 1.0)
    for a, b in zip(raw, scaled, strict=True):
        assert math.isclose(b / a, scaled[0] / raw[0])


def test_normalize_rejects_degenerate_input():
    with pytest.raises(ValueError, match="negative"):
        swapnet.normalize([1.0, -1.0])
    with pytest.raises(ValueError, match="not finite"):
        swapnet.normalize([1.0, float("inf")])
    with pytest.raises(ValueError, match="every weight is zero"):
        swapnet.normalize([0.0, 0.0])


def test_fidelities_compose_multiplicatively():
    """-log(fidelity) is the cost that makes path cost additive."""
    costs = swapnet.from_fidelities([0.99, 0.99, 0.9801])
    # 0.9801 == 0.99**2, so its cost must be twice a single 0.99 hop.
    assert math.isclose(costs[2], 2 * costs[0], rel_tol=1e-9)


def test_perfect_fidelity_still_costs_something():
    """A reported F = 1 must not become a zero-cost edge.

    The routing model derives its path-length bound from the cheapest edge, so a
    single free edge makes no finite hop bound safe and silently disables the
    pruning for every commodity. Vendor tables do report rounded 1.000.
    """
    costs = swapnet.from_fidelities({(0, 1): 1.0, (1, 2): 0.99, (2, 3): 0.98})
    assert min(costs.values()) > 0.0
    # The cap must not disturb the ordering of the real couplings.
    assert costs[(0, 1)] < costs[(1, 2)] < costs[(2, 3)]


def test_zero_cost_edge_is_reported_as_unpruned():
    """A hand-built zero weight is legal but must not silently lose the pruning."""
    view = swapnet.as_graph_view([(0, 1, 0.0), (1, 2, 1.0), (2, 3, 1.0)])
    model = _core.build_routing_model(
        view.core,
        [(0, 3)],
        _core.bfs_all_pairs(view.core),
        _core.RoutingParams(tie_break_noise=0.0, seed=1),
    )
    assert not model.pruning_applied

    healthy = swapnet.as_graph_view([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
    assert _core.build_routing_model(
        healthy.core,
        [(0, 3)],
        _core.bfs_all_pairs(healthy.core),
        _core.RoutingParams(tie_break_noise=0.0, seed=1),
    ).pruning_applied


def test_exhausted_depth_budget_skips_the_solver():
    """An exhausted budget must not pay for a MILP whose layers get discarded."""
    hardware = [(i, i + 1) for i in range(11)]
    interaction = nx.random_regular_graph(3, 12, seed=3)

    solves = []

    def counting_solver(problem):
        backend = swapnet.get_backend(
            "highs" if "highs" in swapnet.available_backends() else None
        )
        solves.append(problem.num_vars)
        return backend(problem)

    layout = swapnet.map_layout(
        hardware, interaction, params=swapnet.MapParams(**FAST_MAP), seed=3
    )
    plan = swapnet.route(
        hardware,
        interaction,
        layout,
        seed=3,
        max_depth=0,
        solver=counting_solver,
    )

    assert plan.total_swaps == 0
    assert plan.hit_depth_limit
    assert solves == [], "solved a MILP despite having no depth to spend"
    swapnet.verify_plan(hardware, interaction, plan, require_converged=False)


def test_error_rates_agree_with_fidelities():
    errors = [0.01, 0.02, 0.05]
    assert swapnet.from_error_rates(errors) == pytest.approx(
        swapnet.from_fidelities([1 - e for e in errors])
    )


def test_fidelity_helpers_keep_mapping_keys():
    fid = {("a", "b"): 0.99, ("b", "c"): 0.95}
    costs = swapnet.from_fidelities(fid)
    assert set(costs) == set(fid)
    # The worse coupling must cost more.
    assert costs[("b", "c")] > costs[("a", "b")]


# ------------------------------------------------------------- weighted distance


def test_dijkstra_matches_bfs_when_unweighted():
    edges, n = grid(4, 4)
    view = swapnet.as_graph_view(edges, num_vertices=n)

    hops = _core.bfs_all_pairs(view.core)
    costs = _core.dijkstra_all_pairs(view.core)

    assert costs == [[float(d) for d in row] for row in hops]


def test_unit_weights_reproduce_hop_counts():
    edges, n = grid(3, 4)
    view = swapnet.as_graph_view(edges, num_vertices=n).with_weights([1.0] * len(edges))

    assert _core.dijkstra_all_pairs(view.core) == [
        [float(d) for d in row] for row in _core.bfs_all_pairs(view.core)
    ]


def test_expensive_edge_is_routed_around():
    """A costly coupling should be bypassed even at the price of extra hops."""
    # Square: the (1, 2) edge is the direct route, the other three the detour.
    view = swapnet.as_graph_view([(0, 1, 1.0), (1, 2, 10.0), (2, 3, 1.0), (0, 3, 1.0)])
    costs = _core.dijkstra_all_pairs(view.core)

    assert costs[1][2] == pytest.approx(3.0)  # 1 -> 0 -> 3 -> 2
    assert _core.bfs_all_pairs(view.core)[1][2] == 1  # one hop, ignoring cost


def test_disconnected_pairs_are_infinite():
    view = swapnet.as_graph_view([(0, 1, 1.0), (2, 3, 1.0)])
    costs = _core.dijkstra_all_pairs(view.core)

    assert costs[0][1] == pytest.approx(1.0)
    assert costs[0][2] == math.inf


# ------------------------------------------------------------------ graph inputs


def test_weights_from_edge_list_tuples():
    view = swapnet.as_graph_view([(0, 1, 2.5), (1, 2, 0.5)])
    assert view.is_weighted
    assert view.weights == (2.5, 0.5)


def test_weights_from_networkx_attribute():
    graph = nx.Graph()
    graph.add_edge(0, 1, err=0.01)
    graph.add_edge(1, 2, err=0.02)

    view = swapnet.as_graph_view(graph, weight_attr="err")
    assert view.weights == (0.01, 0.02)


def test_missing_attribute_is_an_error_not_a_default():
    """A silently defaulted weight would quietly misreport a coupling."""
    graph = nx.Graph()
    graph.add_edge(0, 1, err=0.01)
    graph.add_edge(1, 2)  # no annotation

    with pytest.raises(ValueError, match="has no 'err' attribute"):
        swapnet.as_graph_view(graph, weight_attr="err")


def test_weight_attr_on_a_graphview_is_rejected_not_ignored():
    """Silently returning an unweighted view would route as if uniform."""
    graph = nx.Graph()
    graph.add_edge(0, 1, err=0.01)
    graph.add_edge(1, 2, err=0.02)
    view = swapnet.as_graph_view(graph)

    with pytest.raises(ValueError, match="cannot be read from a GraphView"):
        swapnet.as_graph_view(view, weight_attr="err")


def test_mixing_weighted_and_unweighted_entries_is_rejected():
    with pytest.raises(ValueError, match="mixes weighted"):
        swapnet.as_graph_view([(0, 1, 1.0), (1, 2)])


def test_weights_by_mapping_accept_either_orientation():
    view = swapnet.as_graph_view([(0, 1), (1, 2)], weights={(1, 0): 3.0, (1, 2): 4.0})
    assert view.weights == (3.0, 4.0)


def test_incomplete_weight_mapping_is_rejected():
    with pytest.raises(KeyError, match="no weight given"):
        swapnet.as_graph_view([(0, 1), (1, 2)], weights={(0, 1): 1.0})


def test_negative_and_nonfinite_weights_are_rejected():
    with pytest.raises(ValueError, match="negative edge weight"):
        _ = swapnet.as_graph_view([(0, 1, -1.0)], name="hw").core
    with pytest.raises(ValueError, match="not finite"):
        _ = swapnet.as_graph_view([(0, 1, float("inf"))], name="hw").core


def test_weights_survive_networkx_round_trip():
    view = swapnet.as_graph_view([(0, 1, 2.0), (1, 2, 3.0)])
    rebuilt = swapnet.as_graph_view(swapnet.to_networkx(view), weight_attr="weight")
    assert rebuilt.weights == view.weights


def test_weights_survive_rustworkx_round_trip():
    pytest.importorskip("rustworkx")
    view = swapnet.as_graph_view([(0, 1, 2.0), (1, 2, 3.0)])
    rebuilt = swapnet.as_graph_view(swapnet.to_rustworkx(view), weight_attr="weight")
    assert rebuilt.weights == view.weights


# ------------------------------------------------------------------- weighted map


def test_uniform_weights_reproduce_the_unweighted_layout():
    """The controlling equivalence: weighting must parameterise, not change."""
    edges, n = grid(4, 4)
    interaction = nx.cycle_graph(8)
    params = swapnet.MapParams(**FAST_MAP)

    plain = swapnet.map_layout((edges, n)[0], interaction, params=params, seed=5)
    weighted = swapnet.map_layout(
        [(u, v, 1.0) for u, v in edges],
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=5,
    )

    assert weighted.full_layout == plain.full_layout
    assert weighted.cost == pytest.approx(plain.cost)


def test_uniform_couplings_reproduce_the_unweighted_layout():
    edges, _n = grid(4, 4)
    interaction = nx.cycle_graph(8)

    plain = swapnet.map_layout(
        edges, interaction, params=swapnet.MapParams(**FAST_MAP), seed=5
    )
    weighted = swapnet.map_layout(
        edges,
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=5,
        couplings={tuple(e): 1.0 for e in interaction.edges},
    )

    assert weighted.full_layout == plain.full_layout
    assert weighted.cost == pytest.approx(plain.cost)


def test_reported_cost_matches_independent_recomputation():
    edges, n = grid(4, 4)
    interaction = nx.cycle_graph(8)
    couplings = {tuple(sorted(e)): 1.0 + i for i, e in enumerate(interaction.edges)}

    result = swapnet.map_layout(
        edges,
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=11,
        couplings=couplings,
    )

    # Rebuild the exact inputs the core saw, including the mean-1 rescaling.
    hardware = swapnet.as_graph_view(edges, num_vertices=n)
    interaction_view = swapnet.as_graph_view(
        interaction, name="interaction"
    ).with_weights(couplings)
    normalised = swapnet.normalize(interaction_view.weights)

    # full_layout is keyed by logical label, with Ancilla objects for the slots
    # the interaction graph does not use; the core wants a slot-indexed list.
    layout = [
        result.full_layout[
            slot if slot < interaction.number_of_nodes() else swapnet.Ancilla(slot)
        ]
        for slot in range(n)
    ]
    expected = _core.layout_cost(
        _core.dijkstra_all_pairs(hardware.core),
        list(interaction_view.edges),
        layout,
        list(normalised),
    )

    assert result.cost == pytest.approx(expected)


def test_strong_coupling_is_placed_closer_than_weak_ones():
    """A dominant J should buy adjacency for its own pair."""
    # A star interaction graph cannot be embedded in a path, so the mapper must
    # choose which arm to keep short.
    edges, n = grid(1, 6)  # a 6-vertex path
    interaction = nx.Graph([(0, 1), (0, 2), (0, 3)])
    couplings = {(0, 1): 100.0, (0, 2): 1.0, (0, 3): 1.0}

    result = swapnet.map_layout(
        edges,
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=3,
        couplings=couplings,
    )

    hardware = swapnet.as_graph_view(edges, num_vertices=n)
    hops = _core.bfs_all_pairs(hardware.core)
    layout = result.full_layout

    assert hops[layout[0]][layout[1]] == 1


def test_layout_avoids_a_bad_coupling():
    """With one coupling made very noisy, the mapper should route around it.

    A 6-vertex path embeds in a 6-cycle by using 5 of its 6 couplings, so the
    mapper gets to choose exactly which one to leave unused. Unweighted it has no
    reason to care; weighted it should leave out the noisy one.
    """
    hardware = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    interaction = nx.path_graph(6)

    bad = (2, 3)
    weights = dict.fromkeys(hardware, 1.0)
    weights[bad] = 50.0

    result = swapnet.map_layout(
        hardware,
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=9,
        edge_weights=weights,
    )

    # Which hardware couplings do the interaction edges land on?
    layout = result.full_layout
    used = {frozenset((layout[u], layout[v])) for u, v in interaction.edges}
    assert frozenset(bad) not in used


def test_rescale_false_rejects_unnormalised_weights():
    edges, _ = grid(3, 3)
    with pytest.raises(ValueError, match="do not have mean 1"):
        swapnet.map_layout(
            edges,
            nx.cycle_graph(4),
            params=swapnet.MapParams(**FAST_MAP),
            seed=1,
            edge_weights=[0.01] * len(edges),
            rescale_weights=False,
        )


def test_rescale_false_accepts_already_normalised_weights():
    edges, _ = grid(3, 3)
    result = swapnet.map_layout(
        edges,
        nx.cycle_graph(4),
        params=swapnet.MapParams(**FAST_MAP),
        seed=1,
        edge_weights=[1.0] * len(edges),
        rescale_weights=False,
    )
    assert result.cost >= 0


# ----------------------------------------------------------------- weighted route


def test_uniform_weights_reproduce_the_unweighted_plan():
    edges, _n = grid(4, 4)
    interaction = nx.random_regular_graph(3, 10, seed=4)

    plain = swapnet.map_and_route(
        edges, interaction, params=swapnet.MapParams(**FAST_MAP), seed=7
    )
    weighted = swapnet.map_and_route(
        [(u, v, 1.0) for u, v in edges],
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=7,
        couplings={tuple(e): 1.0 for e in interaction.edges},
    )

    assert weighted.to_dict()["stages"] == plain.to_dict()["stages"]
    assert weighted.total_swaps == plain.total_swaps


def test_weighted_plan_verifies():
    edges, n = grid(4, 4)
    interaction = nx.random_regular_graph(3, 10, seed=4)

    weights = {edge: 1.0 + (i % 5) for i, edge in enumerate(edges)}
    couplings = {tuple(sorted(e)): 1.0 + (i % 3) for i, e in enumerate(interaction.edges)}

    plan = swapnet.map_and_route(
        edges,
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=7,
        edge_weights=weights,
        couplings=couplings,
    )

    hardware_view = swapnet.as_graph_view(edges, num_vertices=n).with_weights(weights)
    interaction_view = swapnet.as_graph_view(interaction).with_weights(couplings)

    # Weighted verification recomputes the J^2 split and the swap cost.
    swapnet.verify_plan(
        hardware_view, interaction_view, plan, require_converged=plan.converged
    )


def test_arc_costs_reach_the_milp_objective():
    """The per-arc objective must be W_ij, not 1."""
    view = swapnet.as_graph_view([(0, 1, 2.0), (1, 2, 5.0)])
    model = _core.build_routing_model(
        view.core,
        [(0, 2)],
        _core.bfs_all_pairs(view.core),
        _core.RoutingParams(tie_break_noise=0.0, seed=1),
    )

    for arc_index, arc in enumerate(model.arcs):
        coefficient = model.problem.objective[model.x_index(arc_index, 0)]
        assert coefficient == pytest.approx(arc.cost)

    # Both directions of an edge share its cost.
    assert {arc.cost for arc in model.arcs} == {2.0, 5.0}


def test_commodity_reward_scales_with_j_squared():
    edges, n = grid(3, 3)
    view = swapnet.as_graph_view(edges, num_vertices=n)
    commodities = [(0, 2), (3, 5)]

    model = _core.build_routing_model(
        view.core,
        commodities,
        _core.bfs_all_pairs(view.core),
        _core.RoutingParams(route_reward=30.0, tie_break_noise=0.0, seed=1),
        [1.0, 3.0],
    )

    # mean(J^2) = (1 + 9) / 2 = 5, so the rewards are 30/5 and 270/5.
    assert model.rewards == pytest.approx([6.0, 54.0])
    assert model.rewards[1] / model.rewards[0] == pytest.approx(9.0)
    # The mean reward is preserved, which keeps the pruning bound calibrated.
    assert sum(model.rewards) / len(model.rewards) == pytest.approx(30.0)


def test_uniform_couplings_leave_rewards_flat():
    edges, n = grid(3, 3)
    view = swapnet.as_graph_view(edges, num_vertices=n)
    params = _core.RoutingParams(route_reward=30.0, tie_break_noise=0.0, seed=1)

    without = _core.build_routing_model(
        view.core, [(0, 2), (3, 5)], _core.bfs_all_pairs(view.core), params
    )
    with_uniform = _core.build_routing_model(
        view.core,
        [(0, 2), (3, 5)],
        _core.bfs_all_pairs(view.core),
        params,
        [2.0, 2.0],
    )

    assert without.rewards == pytest.approx(with_uniform.rewards)


# ------------------------------------------------------------- depth truncation


def _depth_hungry_instance():
    """A dense interaction graph on a path, which needs many SWAP layers."""
    hardware = [(i, i + 1) for i in range(9)]
    interaction = nx.random_regular_graph(3, 10, seed=3)
    return hardware, interaction


def test_max_depth_caps_the_emitted_layers():
    hardware, interaction = _depth_hungry_instance()

    full = swapnet.map_and_route(
        hardware, interaction, params=swapnet.MapParams(**FAST_MAP), seed=11
    )
    assert full.num_swap_layers > 2, "instance is not depth-hungry enough"

    for budget in (1, 2):
        plan = swapnet.map_and_route(
            hardware,
            interaction,
            params=swapnet.MapParams(**FAST_MAP),
            seed=11,
            max_depth=budget,
        )
        assert plan.num_swap_layers <= budget
        assert plan.hit_depth_limit
        assert not plan.converged


def test_truncated_plan_is_still_executable():
    hardware, interaction = _depth_hungry_instance()

    plan = swapnet.map_and_route(
        hardware,
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=11,
        max_depth=2,
    )

    # The whole point of truncation is that what remains is a valid prefix.
    swapnet.verify_plan(hardware, interaction, plan, require_converged=False)
    assert plan.unrouted


def test_zero_depth_allows_only_already_executable_gates():
    hardware, interaction = _depth_hungry_instance()

    plan = swapnet.map_and_route(
        hardware,
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=11,
        max_depth=0,
    )

    assert plan.total_swaps == 0
    assert plan.num_swap_layers == 0
    swapnet.verify_plan(hardware, interaction, plan, require_converged=False)


def test_couplings_steer_which_interactions_survive_truncation():
    """The headline property: J^2 decides what a truncated plan keeps."""
    hardware, interaction = _depth_hungry_instance()
    edges = sorted(tuple(sorted(e)) for e in interaction.edges)

    # Make one interaction overwhelmingly the most valuable.
    favoured = edges[0]
    couplings = dict.fromkeys(edges, 0.1)
    couplings[favoured] = 10.0

    weighted = swapnet.map_and_route(
        hardware,
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=11,
        max_depth=1,
        couplings=couplings,
    )

    realised = {tuple(sorted(g)) for g in weighted.scheduled_gates()}
    assert favoured in realised

    # And it captures essentially all of the J^2 mass, since 10^2 dominates.
    assert weighted.realized_fraction > 0.9


def test_couplings_change_which_commodities_the_router_buys():
    """Against one budget and one layout, J must change the routing choice.

    Both runs start from the same layout, so any difference is the router's
    doing rather than the mapper's. The interaction graph is far denser than the
    path can satisfy at once, so vertex-disjointness forces the solver to pick a
    subset -- and the reward being proportional to J^2 should make it pick the
    valuable ones.
    """
    hardware = [(i, i + 1) for i in range(9)]
    interaction = nx.random_regular_graph(3, 10, seed=3)
    edges = sorted(tuple(sorted(e)) for e in interaction.edges)

    # One layout, reused, so the comparison isolates routing.
    layout = swapnet.map_layout(
        hardware, interaction, params=swapnet.MapParams(**FAST_MAP), seed=11
    )

    def routed_with(couplings):
        plan = swapnet.route(
            hardware,
            interaction,
            layout,
            seed=11,
            max_depth=1,
            couplings=couplings,
        )
        return {tuple(sorted(g)) for g in plan.scheduled_gates()}

    blind = routed_with(None)

    # Favour, in turn, an interaction the J-blind run left unrouted. If the
    # couplings are reaching the objective at all, at least one such nudge must
    # change the outcome.
    missed = [edge for edge in edges if edge not in blind]
    assert missed, "instance is not constrained enough to force a choice"

    changed = False
    for target in missed:
        couplings = dict.fromkeys(edges, 0.05)
        couplings[target] = 20.0
        if target in routed_with(couplings):
            changed = True
            break

    assert changed, (
        "no coupling assignment changed which interactions were routed, so J is "
        "not influencing the routing decision"
    )


def test_realized_and_unrealized_masses_account_for_everything():
    hardware, interaction = _depth_hungry_instance()
    edges = sorted(tuple(sorted(e)) for e in interaction.edges)
    couplings = {edge: 1.0 + i for i, edge in enumerate(edges)}

    plan = swapnet.map_and_route(
        hardware,
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=11,
        max_depth=2,
        couplings=couplings,
    )

    # Normalised J^2 over every interaction, which is what the core accumulated.
    normalised = swapnet.normalize([couplings[e] for e in edges])
    total = math.fsum(j * j for j in normalised)

    assert plan.realized_weight + plan.unrealized_weight == pytest.approx(total)
    assert 0.0 <= plan.realized_fraction <= 1.0


def test_fully_routed_plan_reports_full_realization():
    edges, _n = grid(4, 4)
    interaction = nx.cycle_graph(6)

    plan = swapnet.map_and_route(
        edges,
        interaction,
        params=swapnet.MapParams(**FAST_MAP),
        seed=7,
        couplings={tuple(sorted(e)): 1.0 + i for i, e in enumerate(interaction.edges)},
    )

    assert plan.converged
    assert plan.unrealized_weight == pytest.approx(0.0)
    assert plan.realized_fraction == pytest.approx(1.0)


def test_swap_weight_is_reported_and_zero_without_swaps():
    edges, _n = grid(4, 4)
    # A cycle on a grid needs no routing at all.
    plan = swapnet.map_and_route(
        edges,
        nx.cycle_graph(4),
        params=swapnet.MapParams(**FAST_MAP),
        seed=7,
        edge_weights=[1.0 + (i % 3) for i in range(len(edges))],
    )
    if plan.total_swaps == 0:
        assert plan.swap_weight == pytest.approx(0.0)
    else:
        assert plan.swap_weight > 0.0


# ------------------------------------------------------------------ upstream I/O
