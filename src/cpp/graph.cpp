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

#include "graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>

namespace swapnet {

namespace {

// Shared predicates for both weight vectors. `noun` names the quantity and
// `why` explains the domain reason a bad value is rejected, because a hardware
// traversal cost and an interaction coupling fail for different reasons and the
// message is the useful half of the check.
void check_weight_vector(const std::vector<double>& weights,
                         std::size_t expected, const std::string& name,
                         const std::string& noun, const std::string& why) {
    if (weights.empty()) {
        return;
    }
    if (weights.size() != expected) {
        throw std::invalid_argument(
            name + ": has " + std::to_string(expected) + " edges but " +
            std::to_string(weights.size()) + " " + noun +
            "s; pass one " + noun + " per edge or none at all");
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const double w = weights[i];
        if (!std::isfinite(w)) {
            throw std::invalid_argument(
                name + ": " + noun + " " + std::to_string(i) +
                " is not finite");
        }
        if (w < 0.0) {
            throw std::invalid_argument(
                name + ": negative " + noun + " " + std::to_string(w) +
                " at edge " + std::to_string(i) + "; " + why);
        }
    }
}

}  // namespace

void Graph::validate(const std::string& name) const {
    if (num_vertices < 0) {
        throw std::invalid_argument(
            name + ": negative vertex count " + std::to_string(num_vertices));
    }

    std::set<Edge> seen;

    for (const auto& [u, v] : edges) {
        if (u < 0 || u >= num_vertices || v < 0 || v >= num_vertices) {
            throw std::invalid_argument(
                name + ": edge (" + std::to_string(u) + ", " +
                std::to_string(v) + ") has an endpoint outside [0, " +
                std::to_string(num_vertices) + ")");
        }
        if (u == v) {
            throw std::invalid_argument(
                name + ": self-loop at vertex " + std::to_string(u));
        }
        if (!seen.insert(normalized(u, v)).second) {
            throw std::invalid_argument(
                name + ": duplicate undirected edge (" + std::to_string(u) +
                ", " + std::to_string(v) + ")");
        }
    }

    check_weight_vector(
        weights, edges.size(), name, "edge weight",
        "weights are traversal costs, so they must be non-negative; an unusable "
        "coupling should be left out of the graph rather than priced highly");
}

std::vector<std::vector<int>> Graph::adjacency() const {
    std::vector<std::vector<int>> adj(num_vertices);
    for (const auto& [u, v] : edges) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    return adj;
}

std::vector<std::vector<std::pair<int, double>>> Graph::weighted_adjacency()
    const {
    std::vector<std::vector<std::pair<int, double>>> adj(num_vertices);
    for (int i = 0; i < num_edges(); ++i) {
        const auto [u, v] = edges[static_cast<std::size_t>(i)];
        const double w = weight(i);
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }
    return adj;
}

bool Graph::is_connected() const {
    if (num_vertices <= 1) {
        return true;
    }

    const auto adj = adjacency();
    std::vector<bool> seen(num_vertices, false);
    std::queue<int> q;

    seen[0] = true;
    q.push(0);
    int reached = 1;

    while (!q.empty()) {
        const int u = q.front();
        q.pop();
        for (const int v : adj[u]) {
            if (!seen[v]) {
                seen[v] = true;
                ++reached;
                q.push(v);
            }
        }
    }

    return reached == num_vertices;
}

std::vector<std::vector<int>> bfs_all_pairs(const Graph& g) {
    const int n = g.num_vertices;
    const auto adj = g.adjacency();

    std::vector<std::vector<int>> dist_matrix(
        n, std::vector<int>(n, kUnreachable));

    for (int start = 0; start < n; ++start) {
        auto& dist = dist_matrix[start];
        dist[start] = 0;

        std::queue<int> q;
        q.push(start);

        while (!q.empty()) {
            const int u = q.front();
            q.pop();
            for (const int v : adj[u]) {
                if (dist[v] == kUnreachable) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
    }

    return dist_matrix;
}

std::vector<std::vector<double>> dijkstra_all_pairs(const Graph& g) {
    const int n = g.num_vertices;
    const double inf = std::numeric_limits<double>::infinity();
    const auto adj = g.weighted_adjacency();

    std::vector<std::vector<double>> dist_matrix(n, std::vector<double>(n, inf));

    // Min-heap of (tentative distance, vertex).
    using Entry = std::pair<double, int>;

    for (int start = 0; start < n; ++start) {
        auto& dist = dist_matrix[start];
        dist[start] = 0.0;

        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
        pq.emplace(0.0, start);

        while (!pq.empty()) {
            const auto [d, u] = pq.top();
            pq.pop();
            // Stale heap entry: a shorter route to u was settled already.
            if (d > dist[u]) {
                continue;
            }
            for (const auto& [v, w] : adj[u]) {
                const double candidate = d + w;
                if (candidate < dist[v]) {
                    dist[v] = candidate;
                    pq.emplace(candidate, v);
                }
            }
        }
    }

    return dist_matrix;
}

void validate_interaction_edges(const std::vector<Edge>& edges, int num_hosts) {
    std::set<Edge> seen;
    std::set<int> vertices;

    for (const auto& [u, v] : edges) {
        if (u < 0 || v < 0) {
            throw std::invalid_argument(
                "interaction graph: negative vertex index in edge (" +
                std::to_string(u) + ", " + std::to_string(v) + ")");
        }
        if (u >= num_hosts || v >= num_hosts) {
            throw std::invalid_argument(
                "interaction graph: edge (" + std::to_string(u) + ", " +
                std::to_string(v) + ") refers to a vertex outside [0, " +
                std::to_string(num_hosts) +
                "); the interaction graph must not be larger than the "
                "hardware graph");
        }
        if (u == v) {
            throw std::invalid_argument(
                "interaction graph: self-loop at vertex " + std::to_string(u));
        }
        if (!seen.insert(normalized(u, v)).second) {
            throw std::invalid_argument(
                "interaction graph: duplicate undirected edge (" +
                std::to_string(u) + ", " + std::to_string(v) + ")");
        }
        vertices.insert(u);
        vertices.insert(v);
    }

    if (static_cast<int>(vertices.size()) > num_hosts) {
        throw std::invalid_argument(
            "interaction graph has " + std::to_string(vertices.size()) +
            " vertices but the hardware graph has only " +
            std::to_string(num_hosts));
    }
}

void validate_interaction_weights(const std::vector<double>& weights,
                                  std::size_t num_edges,
                                  const std::string& name) {
    check_weight_vector(
        weights, num_edges, name, "coupling",
        "the truncation objective maximises a sum of squares, so a sign on the "
        "coupling would be meaningless");
}

void validate_layout(const std::vector<int>& layout, int num_vertices) {
    if (static_cast<int>(layout.size()) != num_vertices) {
        throw std::invalid_argument(
            "layout has " + std::to_string(layout.size()) +
            " entries but the hardware graph has " +
            std::to_string(num_vertices) +
            " vertices; the layout must be a full permutation");
    }

    std::vector<bool> used(num_vertices, false);

    for (const int p : layout) {
        if (p < 0 || p >= num_vertices) {
            throw std::invalid_argument(
                "layout entry " + std::to_string(p) + " is outside [0, " +
                std::to_string(num_vertices) + ")");
        }
        if (used[p]) {
            throw std::invalid_argument(
                "layout maps two logical qubits to physical qubit " +
                std::to_string(p));
        }
        used[p] = true;
    }
}

}  // namespace swapnet
