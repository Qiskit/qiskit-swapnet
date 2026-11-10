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

"""The graph adapters: label preservation, normalisation, and rejection."""

from __future__ import annotations

import pytest

import swapnet
from swapnet.graphs import GraphView, as_graph_view

networkx = pytest.importorskip("networkx")
rustworkx = pytest.importorskip("rustworkx")


def test_networkx_tuple_labels_survive_the_round_trip():
    graph = networkx.grid_2d_graph(2, 3)
    view = as_graph_view(graph)

    assert view.num_vertices == 6
    assert set(view.labels) == set(graph.nodes())
    assert {frozenset(e) for e in view.edge_labels()} == {
        frozenset(e) for e in graph.edges()
    }


def test_networkx_labels_are_indexed_in_sorted_order():
    # Sorting makes the index assignment independent of insertion order, so two
    # graphs with the same nodes added differently give the same indices.
    forward = as_graph_view(networkx.Graph([("a", "b"), ("b", "c")]))
    backward = networkx.Graph()
    backward.add_edges_from([("c", "b"), ("b", "a")])

    assert forward.labels == ("a", "b", "c")
    assert as_graph_view(backward).labels == ("a", "b", "c")


def test_rustworkx_generators_with_repeated_pairs_are_accepted():
    # PyGraph is a multigraph by default and this generator emits both (u, v)
    # and (v, u); the duplicate is redundant, not an error.
    graph = rustworkx.undirected_gnm_random_graph(15, 25, seed=3)
    view = as_graph_view(graph)

    normalised = {frozenset(e) for e in graph.edge_list()}
    assert len(view.edges) == len(normalised)
    assert {frozenset(e) for e in view.edges} == normalised


def test_rustworkx_node_index_holes_are_compacted():
    graph = rustworkx.PyGraph()
    a, b, c = graph.add_nodes_from(["a", "b", "c"])
    graph.add_edges_from([(a, b, None), (b, c, None)])
    graph.remove_node(b)

    view = as_graph_view(graph)
    assert view.num_vertices == 2
    assert set(view.labels) == {"a", "c"}
    assert view.edges == ()


def test_rustworkx_falls_back_to_indices_when_payloads_cannot_be_labels():
    graph = rustworkx.PyGraph()
    graph.add_nodes_from([None, None, None])
    graph.add_edges_from([(0, 1, None), (1, 2, None)])

    view = as_graph_view(graph)
    assert view.labels == (0, 1, 2)


def test_self_loops_and_duplicates_are_dropped_from_edge_lists():
    view = as_graph_view([(0, 1), (1, 0), (1, 1), (1, 2)])
    assert view.edges == ((0, 1), (1, 2))


def test_integer_edge_lists_can_declare_isolated_vertices():
    view = as_graph_view([(0, 1)], num_vertices=4)
    assert view.num_vertices == 4
    assert view.labels == (0, 1, 2, 3)


def test_num_vertices_too_small_is_rejected():
    with pytest.raises(ValueError, match="too small"):
        as_graph_view([(0, 5)], num_vertices=3)


def test_directed_graphs_are_rejected():
    with pytest.raises(TypeError, match="directed"):
        as_graph_view(networkx.DiGraph([(0, 1)]))
    with pytest.raises(TypeError, match="directed"):
        as_graph_view(rustworkx.PyDiGraph())


def test_networkx_multigraph_is_rejected():
    with pytest.raises(TypeError, match="multigraph"):
        as_graph_view(networkx.MultiGraph([(0, 1)]))


def test_validation_errors_name_the_offending_graph():
    # The name is what turns a bare "graph: ..." into something that says which
    # of the two inputs was at fault.
    view = GraphView(
        num_vertices=2,
        edges=((0, 9),),
        labels=(0, 1),
        index_of={0: 0, 1: 1},
        name="interaction graph",
    )
    with pytest.raises(ValueError, match="interaction graph"):
        _ = view.core


def test_unsupported_types_are_rejected():
    with pytest.raises(TypeError, match="unsupported graph type"):
        as_graph_view(42)


def test_to_networkx_and_to_rustworkx_reproduce_the_view():
    original = networkx.cycle_graph(6)
    view = as_graph_view(original)

    rebuilt = swapnet.to_networkx(view)
    assert networkx.is_isomorphic(rebuilt, original)

    as_rx = swapnet.to_rustworkx(view)
    assert as_rx.num_nodes() == view.num_vertices
    assert as_rx.num_edges() == len(view.edges)
