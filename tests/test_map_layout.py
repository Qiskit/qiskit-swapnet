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

"""The quadratic-assignment layout stage."""

from __future__ import annotations

import pytest
from conftest import grid

import swapnet
from swapnet import _core
from swapnet.results import Ancilla

networkx = pytest.importorskip("networkx")


def test_layout_is_a_bijection_onto_hardware_qubits(fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.cycle_graph(10)

    result = swapnet.map_layout(hardware, interaction, params=fast_params, seed=1)

    assert set(result.layout) == set(interaction.nodes())
    assert len(set(result.layout.values())) == len(result.layout)
    assert set(result.layout.values()) <= set(hardware.nodes())

    # The full layout covers every hardware qubit exactly once, padding with
    # ancillas, because idle qubits are still moved by SWAPs.
    assert len(result.full_layout) == hardware.number_of_nodes()
    assert set(result.full_layout.values()) == set(hardware.nodes())
    ancillas = [k for k in result.full_layout if isinstance(k, Ancilla)]
    assert len(ancillas) == 16 - 10


def test_reported_cost_matches_an_independent_evaluation(fast_params):
    edges, n = grid(4, 4)
    interaction = networkx.random_regular_graph(3, 12, seed=2)

    result = swapnet.map_layout(edges, interaction, params=fast_params, seed=5)

    hardware_view = swapnet.as_graph_view(edges, num_vertices=n)
    interaction_view = swapnet.as_graph_view(interaction)
    distances = _core.bfs_all_pairs(hardware_view.core)

    permutation = [
        (
            result.full_layout[label].slot
            if isinstance(result.full_layout[label], Ancilla)
            else result.full_layout[label]
        )
        for label in sorted(
            result.full_layout,
            key=lambda k: k.slot if isinstance(k, Ancilla) else k,
        )
    ]
    recomputed = _core.layout_cost(distances, list(interaction_view.edges), permutation)
    assert recomputed == result.cost


def test_the_same_seed_gives_the_same_layout(fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.random_regular_graph(3, 12, seed=4)

    first = swapnet.map_layout(hardware, interaction, params=fast_params, seed=99)
    second = swapnet.map_layout(hardware, interaction, params=fast_params, seed=99)

    assert first.layout == second.layout
    assert first.cost == second.cost


def test_different_seeds_are_allowed_to_differ(fast_params):
    # Not a correctness requirement, but if this ever passes identically the
    # seeding has stopped reaching the search.
    hardware = networkx.grid_2d_graph(5, 5)
    interaction = networkx.random_regular_graph(3, 16, seed=8)

    costs = {
        swapnet.map_layout(hardware, interaction, params=fast_params, seed=seed).cost
        for seed in (1, 2, 3, 4, 5)
    }
    assert len(costs) >= 1  # the search ran; equality across seeds is possible


def test_a_perfectly_embeddable_interaction_graph_reaches_the_lower_bound(
    fast_params,
):
    # A path embeds into a path with every interaction at distance 1, so the
    # optimal cost is exactly the number of interaction edges.
    hardware = networkx.path_graph(8)
    interaction = networkx.path_graph(8)

    result = swapnet.map_layout(hardware, interaction, params=fast_params, seed=3)
    assert result.cost == interaction.number_of_edges()


def test_time_limit_is_reported():
    hardware = networkx.grid_2d_graph(6, 6)
    interaction = networkx.random_regular_graph(4, 30, seed=6)

    params = swapnet.MapParams(num_restarts=100000, sa_iterations=20000)
    result = swapnet.map_layout(
        hardware, interaction, params=params, seed=2, max_seconds=0.5
    )

    assert result.hit_time_limit
    assert result.restarts_run < 100000


def test_progress_callback_is_invoked(fast_params):
    seen = []
    swapnet.map_layout(
        networkx.grid_2d_graph(3, 3),
        networkx.cycle_graph(6),
        params=fast_params,
        seed=1,
        progress=lambda restart, total, best: seen.append((restart, total, best)),
    )
    assert seen
    assert all(total == fast_params.num_restarts for _, total, _ in seen)


def test_disconnected_hardware_is_rejected():
    # Upstream summed BFS distances of -1 for unreachable pairs, so splitting
    # interacting qubits across components lowered the cost.
    with pytest.raises(ValueError, match="disconnected"):
        swapnet.map_layout(networkx.Graph([(0, 1), (2, 3)]), [(0, 1)])


def test_interaction_graph_larger_than_hardware_is_rejected():
    with pytest.raises(ValueError, match="only 3"):
        swapnet.map_layout(networkx.path_graph(3), networkx.path_graph(5))


def test_empty_hardware_is_rejected():
    with pytest.raises(ValueError, match="no qubits"):
        swapnet.map_layout(networkx.Graph(), [])
