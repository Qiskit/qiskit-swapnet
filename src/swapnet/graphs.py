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

"""Conversion between user graph types and the dense integer graphs the core uses.

The native core indexes vertices as ``0..n-1``. Real inputs are networkx or
rustworkx graphs whose node labels are arbitrary (strings, tuples, sparse ints
after node removals), so every entry point funnels through :class:`GraphView`,
which owns the index/label correspondence and is used again on the way out to
translate results back into the caller's labels.

networkx and rustworkx are optional: the adapters are selected by duck typing and
lazy imports, so neither is needed unless you pass that graph type.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from . import _core

__all__ = ["GraphView", "as_graph_view", "to_networkx", "to_rustworkx"]


def _sorted_if_possible(labels: list[Any]) -> list[Any]:
    """Sort node labels so that indexing is stable across runs.

    Falls back to iteration order when labels are not mutually comparable, which
    is still stable for a given graph object.
    """
    try:
        return sorted(labels)
    except TypeError:
        return labels


def _simple_edges(
    pairs: Iterable[Sequence[Any]],
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...] | None]:
    """Drop self-loops and repeated undirected pairs, keeping first occurrence.

    Redundant pairs are normal input rather than user error: a ``rustworkx``
    ``PyGraph`` is a multigraph unless constructed otherwise, and several of its
    generators emit both ``(u, v)`` and ``(v, u)``. Neither graph here carries
    edge multiplicity -- a coupling either exists or it does not, and an
    interaction is satisfied once -- so collapsing repeats is faithful rather
    than lossy.

    Accepts ``(u, v)`` or ``(u, v, weight)`` items. Returns the edges plus the
    weights, or ``None`` for the weights when no item carried one. Mixing the two
    forms is an error: a silently defaulted weight would quietly bias the
    routing.
    """
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    weighted: bool | None = None

    for item in pairs:
        item = tuple(item)
        if len(item) == 2:
            (u, v), weight = item, None
        elif len(item) == 3:
            u, v, weight = item
        else:
            raise ValueError(f"expected (u, v) or (u, v, weight), got {item!r}")

        if weighted is None:
            weighted = weight is not None
        elif weighted != (weight is not None):
            raise ValueError(
                "edge list mixes weighted (u, v, w) and unweighted (u, v) "
                "entries; give every edge a weight or none of them"
            )

        if u == v:
            continue
        key = (u, v) if u <= v else (v, u)
        if key not in seen:
            seen.add(key)
            edges.append((u, v))
            if weight is not None:
                weights.append(float(weight))

    return tuple(edges), (tuple(weights) if weighted else None)


@dataclass(frozen=True)
class GraphView:
    """A graph as consecutive integer indices plus the labels they stand for."""

    num_vertices: int
    edges: tuple[tuple[int, int], ...]
    labels: tuple[Any, ...]
    index_of: Mapping[Any, int]
    # Used only to name this graph in validation errors.
    name: str = "graph"
    # One non-negative cost per edge, parallel to `edges`, or None for an
    # unweighted graph. On a hardware graph these are the W_ij traversal costs;
    # on an interaction graph they are the J_lm couplings.
    weights: tuple[float, ...] | None = None

    @cached_property
    def core(self) -> _core.Graph:
        """The native graph, validated. Built once per view."""
        if self.weights is None:
            graph = _core.Graph(self.num_vertices, list(self.edges))
        else:
            graph = _core.Graph(self.num_vertices, list(self.edges), list(self.weights))
        graph.validate(self.name)
        return graph

    @property
    def is_weighted(self) -> bool:
        return self.weights is not None

    def label(self, index: int) -> Any:
        return self.labels[index]

    def edge_labels(self) -> tuple[tuple[Any, Any], ...]:
        return tuple((self.labels[u], self.labels[v]) for u, v in self.edges)

    def weight_list(self) -> list[float]:
        """Weights as a list, empty when unweighted -- the core's convention."""
        return [] if self.weights is None else list(self.weights)

    def with_weights(
        self, weights: Mapping[Any, float] | Sequence[float] | None
    ) -> GraphView:
        """A copy carrying `weights`, given per edge-label-pair or per index.

        A mapping may be keyed by either orientation of the label pair. Passing
        None returns an unweighted copy.
        """
        if weights is None:
            resolved: tuple[float, ...] | None = None
        elif isinstance(weights, Mapping):
            resolved = tuple(
                _lookup_edge_weight(weights, self.labels[u], self.labels[v])
                for u, v in self.edges
            )
        else:
            values = [float(w) for w in weights]
            if len(values) != len(self.edges):
                raise ValueError(
                    f"{self.name}: got {len(values)} weights for "
                    f"{len(self.edges)} edges"
                )
            resolved = tuple(values)

        return GraphView(
            self.num_vertices,
            self.edges,
            self.labels,
            self.index_of,
            self.name,
            resolved,
        )


def _lookup_edge_weight(weights: Mapping[Any, float], u: Any, v: Any) -> float:
    """Find (u, v) in a weight mapping, accepting either orientation."""
    for key in ((u, v), (v, u), frozenset((u, v))):
        try:
            if key in weights:
                return float(weights[key])
        except TypeError:
            continue
    raise KeyError(
        f"no weight given for edge ({u!r}, {v!r}); the mapping must cover every "
        "edge of the graph"
    )


def _from_networkx(graph: Any, name: str, weight_attr: str | None = None) -> GraphView:
    if graph.is_directed():
        raise TypeError(
            "directed graphs are not supported; both the hardware topology and "
            "the interaction graph are undirected"
        )
    if graph.is_multigraph():
        # Unlike a rustworkx PyGraph, a networkx MultiGraph is a deliberate
        # choice to model multiplicity, which this algorithm has no meaning for.
        raise TypeError(
            "multigraphs are not supported; collapse parallel edges first, "
            "e.g. networkx.Graph(multigraph)"
        )

    labels = _sorted_if_possible(list(graph.nodes()))
    index_of = {label: i for i, label in enumerate(labels)}

    if weight_attr is None:
        edges, weights = _simple_edges(
            (index_of[u], index_of[v]) for u, v in graph.edges()
        )
    else:
        # Every edge must carry the attribute; a default would silently make an
        # unannotated coupling look average.
        items = []
        for u, v, data in graph.edges(data=True):
            if weight_attr not in data:
                raise ValueError(
                    f"{name}: edge ({u!r}, {v!r}) has no {weight_attr!r} "
                    "attribute; annotate every edge or pass weights explicitly"
                )
            items.append((index_of[u], index_of[v], data[weight_attr]))
        edges, weights = _simple_edges(items)

    return GraphView(len(labels), edges, tuple(labels), index_of, name, weights)


def _from_rustworkx(graph: Any, name: str, weight_attr: str | None = None) -> GraphView:
    import rustworkx as rx

    if isinstance(graph, rx.PyDiGraph):
        raise TypeError(
            "directed graphs are not supported; both the hardware topology and "
            "the interaction graph are undirected"
        )

    # rustworkx node indices can have holes after removals, so compact them.
    node_indices = list(graph.node_indices())
    index_of_node = {node: i for i, node in enumerate(node_indices)}
    labels = tuple(graph[node] for node in node_indices)

    # Payloads are often None or unhashable; fall back to the node index as the
    # label when payloads cannot serve as keys.
    usable_payloads = all(isinstance(label, Hashable) for label in labels)
    if (
        usable_payloads
        and len(set(labels)) == len(labels)
        and not any(label is None for label in labels)
    ):
        index_of = {label: i for i, label in enumerate(labels)}
    else:
        labels = tuple(node_indices)
        index_of = {node: i for i, node in enumerate(node_indices)}

    if weight_attr is None:
        edges, weights = _simple_edges(
            (index_of_node[u], index_of_node[v]) for u, v in graph.edge_list()
        )
    else:
        # rustworkx edge payloads are arbitrary objects: accept a mapping keyed
        # by weight_attr, or a bare number when the payload *is* the weight.
        items = []
        for u, v, payload in graph.weighted_edge_list():
            if isinstance(payload, Mapping):
                if weight_attr not in payload:
                    raise ValueError(
                        f"{name}: edge ({u}, {v}) payload has no " f"{weight_attr!r} key"
                    )
                value = payload[weight_attr]
            elif isinstance(payload, (int, float)):
                value = payload
            else:
                # ValueError, not TypeError: this is the documented public contract and
                # callers (and tests) match on it. Not worth breaking for the lint.
                raise ValueError(  # noqa: TRY004
                    f"{name}: edge ({u}, {v}) payload {payload!r} is neither a "
                    f"number nor a mapping containing {weight_attr!r}"
                )
            items.append((index_of_node[u], index_of_node[v], value))
        edges, weights = _simple_edges(items)

    return GraphView(len(node_indices), edges, labels, index_of, name, weights)


def _from_edge_list(
    edges: Iterable[Sequence[Any]], num_vertices: int | None, name: str
) -> GraphView:
    items = [tuple(edge) for edge in edges]
    for edge in items:
        if len(edge) not in (2, 3):
            raise ValueError(f"expected (u, v) or (u, v, weight), got {edge!r}")

    labels_seen: list[Any] = []
    for edge in items:
        for label in edge[:2]:
            if label not in labels_seen:
                labels_seen.append(label)

    all_ints = all(isinstance(label, int) for label in labels_seen)

    if all_ints:
        # Integer labels are taken as vertex indices directly, so that an
        # explicit num_vertices can declare isolated vertices.
        implied = (max(labels_seen) + 1) if labels_seen else 0
        size = implied if num_vertices is None else num_vertices
        if size < implied:
            raise ValueError(
                f"num_vertices={num_vertices} is too small for vertex index "
                f"{implied - 1}"
            )
        labels = tuple(range(size))
        index_of = {label: label for label in labels}
        simple, weights = _simple_edges(items)
        return GraphView(size, simple, labels, index_of, name, weights)

    labels = tuple(_sorted_if_possible(labels_seen))
    index_of = {label: i for i, label in enumerate(labels)}
    if num_vertices is not None and num_vertices != len(labels):
        raise ValueError("num_vertices cannot be combined with non-integer vertex labels")
    simple, weights = _simple_edges(
        (index_of[edge[0]], index_of[edge[1]], *edge[2:]) for edge in items
    )
    return GraphView(len(labels), simple, labels, index_of, name, weights)


def as_graph_view(
    graph: Any,
    *,
    num_vertices: int | None = None,
    name: str = "graph",
    weight_attr: str | None = None,
    weights: Mapping[Any, float] | Sequence[float] | None = None,
) -> GraphView:
    """Normalise any supported graph input into a :class:`GraphView`.

    Accepts a networkx ``Graph``, a rustworkx ``PyGraph``, an existing
    ``GraphView``, or an iterable of ``(u, v)`` or ``(u, v, weight)`` items.
    Self-loops and repeated undirected pairs are dropped. ``num_vertices`` only
    applies to integer-labelled edge lists, where it declares vertices that no
    edge touches. ``name`` labels this graph in validation errors.

    ``weight_attr`` reads per-edge costs from a networkx edge attribute (or a
    rustworkx payload); ``weights`` supplies them directly, either as a mapping
    keyed by label pair or as one value per edge, and takes precedence.
    """
    if isinstance(graph, GraphView):
        # A view has already discarded whatever edge attributes it was built
        # from, so an attribute name cannot be honoured here. Silently ignoring
        # it would hand back an unweighted view and route as if the device were
        # uniform, which is exactly the mistake worth being loud about.
        if weight_attr is not None:
            raise ValueError(
                f"weight_attr={weight_attr!r} cannot be read from a GraphView, "
                "which no longer carries edge attributes; pass weight_attr when "
                "first converting the graph, or use weights=/with_weights() here"
            )
        view = graph
    elif hasattr(graph, "is_directed") and hasattr(graph, "nodes"):
        view = _from_networkx(graph, name, weight_attr)
    elif hasattr(graph, "node_indices") and hasattr(graph, "edge_list"):
        view = _from_rustworkx(graph, name, weight_attr)
    elif isinstance(graph, _core.Graph):
        labels = tuple(range(graph.num_vertices))
        simple, _ = _simple_edges(graph.edges)
        core_weights = tuple(graph.weights) if graph.is_weighted else None
        if core_weights is not None and len(simple) != len(graph.edges):
            raise ValueError(
                f"{name}: cannot carry weights through a graph whose edge list "
                "contains self-loops or duplicates"
            )
        view = GraphView(
            graph.num_vertices,
            simple,
            labels,
            {label: label for label in labels},
            name,
            core_weights,
        )
    elif isinstance(graph, Iterable):
        view = _from_edge_list(graph, num_vertices, name)
    else:
        raise TypeError(
            f"unsupported graph type {type(graph).__name__}; pass a networkx "
            "Graph, a rustworkx PyGraph, or an iterable of (u, v) pairs"
        )

    if weights is not None:
        view = view.with_weights(weights)
    return view


def to_networkx(view: GraphView, weight_attr: str = "weight") -> Any:
    """Rebuild a networkx graph from a view, for inspection and plotting.

    A weighted view writes its costs to the ``weight_attr`` edge attribute.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(view.labels)
    if view.weights is None:
        graph.add_edges_from(view.edge_labels())
    else:
        graph.add_edges_from(
            (view.labels[u], view.labels[v], {weight_attr: w})
            for (u, v), w in zip(view.edges, view.weights, strict=True)
        )
    return graph


def to_rustworkx(view: GraphView) -> Any:
    """Rebuild a rustworkx graph from a view, for inspection and plotting.

    A weighted view stores each cost as the edge payload.
    """
    import rustworkx as rx

    graph = rx.PyGraph(multigraph=False)
    nodes = graph.add_nodes_from(list(view.labels))
    weights = view.weights if view.weights is not None else [None] * len(view.edges)
    graph.add_edges_from(
        [(nodes[u], nodes[v], w) for (u, v), w in zip(view.edges, weights, strict=True)]
    )
    return graph
