// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// Graph container plus the validation and all-pairs shortest-path helpers used
// by both the mapping (QAP) and the routing stages.
//
// Derived from the upstream MAP/Sh_path.{hpp,cpp} and SWAP/GraphDataset.{hpp,cpp},
// with file I/O removed: graphs arrive as vertex counts plus edge lists.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace swapnet {

using Edge = std::pair<int, int>;

// An undirected edge in canonical (low, high) order, so it can be used as a set
// or map key regardless of the orientation it was written in.
inline Edge normalized(int u, int v) {
    return (u < v) ? Edge{u, v} : Edge{v, u};
}

// Sentinel distance for vertex pairs with no connecting path.
constexpr int kUnreachable = -1;

struct Graph {
    int num_vertices = 0;
    std::vector<Edge> edges;
    // Per-edge traversal cost W_ij, parallel to `edges`. Empty means every edge
    // costs 1, which reproduces the unweighted behaviour exactly; otherwise the
    // length must match `edges`.
    //
    // This is a *cost*, so a lower value marks a better edge. For noise-aware
    // routing the natural choice is an error metric such as -log(fidelity) or a
    // two-qubit error rate: the mapper and router then prefer the high-fidelity
    // couplings. Normalise to mean 1 (see swapnet.weights) before use, because
    // the routing reward and the annealing temperatures are calibrated against
    // unit-cost edges.
    std::vector<double> weights;

    Graph() = default;
    Graph(int n, std::vector<Edge> e)
        : num_vertices(n), edges(std::move(e)) {}
    Graph(int n, std::vector<Edge> e, std::vector<double> w)
        : num_vertices(n), edges(std::move(e)), weights(std::move(w)) {}

    int num_edges() const { return static_cast<int>(edges.size()); }

    bool is_weighted() const { return !weights.empty(); }

    // Cost of edge `index`, which is 1 for an unweighted graph.
    double weight(int index) const {
        return weights.empty() ? 1.0 : weights[static_cast<std::size_t>(index)];
    }

    // Throws std::invalid_argument on out-of-range endpoints, self-loops,
    // duplicated undirected edges, or a weight vector that is the wrong length,
    // negative, or not finite. `name` is used in the message only.
    void validate(const std::string& name = "graph") const;

    // Undirected adjacency lists.
    std::vector<std::vector<int>> adjacency() const;

    // Undirected adjacency lists carrying the edge cost alongside the neighbour.
    std::vector<std::vector<std::pair<int, double>>> weighted_adjacency() const;

    // True when every vertex is reachable from vertex 0 (vacuously true for an
    // empty graph). Weights do not affect reachability.
    bool is_connected() const;
};

// Unweighted all-pairs shortest paths by repeated BFS, in hops. Unreachable
// pairs are kUnreachable, so callers that feed the result into a cost function
// must reject disconnected graphs first (see Graph::is_connected).
std::vector<std::vector<int>> bfs_all_pairs(const Graph& g);

// Weighted all-pairs shortest paths by repeated Dijkstra, in units of the edge
// costs. Unreachable pairs are +infinity rather than kUnreachable, because a
// weighted distance has no spare negative range to borrow a sentinel from.
//
// On an unweighted graph this returns exactly the bfs_all_pairs hop counts as
// doubles, so callers may always use it; bfs_all_pairs is kept because the
// path-length pruning in the routing model counts hops, not cost.
std::vector<std::vector<double>> dijkstra_all_pairs(const Graph& g);

// Validates an interaction-graph edge list against a host graph of `num_hosts`
// vertices: endpoints in range, no self-loops, no duplicates, and no more
// distinct vertices than the host can hold.
void validate_interaction_edges(const std::vector<Edge>& edges, int num_hosts);

// Validates interaction-edge couplings J_lm: either empty (all couplings 1) or
// one finite, non-negative entry per edge. `name` is used in the message only.
void validate_interaction_weights(const std::vector<double>& weights,
                                  std::size_t num_edges,
                                  const std::string& name = "interaction graph");

// Validates that `layout` is a permutation of [0, num_vertices).
// layout[logical] == physical.
void validate_layout(const std::vector<int>& layout, int num_vertices);

}  // namespace swapnet
