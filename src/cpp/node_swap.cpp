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

#include "node_swap.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace swapnet {

SwapRoundResult apply_node_swaps(const std::vector<int>& layout,
                                 const std::vector<Edge>& commodity_edges,
                                 const std::vector<SolutionArc>& solution_arcs,
                                 int max_swaps_per_commodity) {
    SwapRoundResult result;
    result.layout = layout;

    // adjacency[k][u] = path neighbours of hardware vertex u for commodity k.
    // std::map keeps the commodity order deterministic.
    std::map<int, std::unordered_map<int, std::vector<int>>> adjacency;
    for (const auto& arc : solution_arcs) {
        adjacency[arc.k][arc.u].push_back(arc.v);
        adjacency[arc.k][arc.v].push_back(arc.u);
    }

    // The two endpoints of each routed path are its degree-one vertices. A
    // commodity whose flow does not form a simple path is skipped.
    std::map<int, Edge> terminals;
    for (const auto& [k, adj] : adjacency) {
        std::vector<int> ends;
        for (const auto& [vertex, neighbours] : adj) {
            if (neighbours.size() == 1) {
                ends.push_back(vertex);
            }
        }
        if (ends.size() == 2) {
            std::sort(ends.begin(), ends.end());
            terminals[k] = Edge{ends[0], ends[1]};
        }
    }

    std::vector<int>& physical_of = result.layout;  // logical -> physical
    std::unordered_map<int, int> logical_at;        // physical -> logical
    for (int logical = 0; logical < static_cast<int>(physical_of.size()); ++logical) {
        logical_at[physical_of[logical]] = logical;
    }

    // Logical qubit currently sitting on `vertex`, or -1 if that is `except`
    // itself or the vertex is unoccupied.
    const auto occupant_of = [&](int vertex, int except) -> int {
        const auto it = logical_at.find(vertex);
        if (it == logical_at.end() || it->second == except) {
            return -1;
        }
        return it->second;
    };

    const auto adjacent_in_path = [&](int k, int from, int to) {
        const auto& adj = adjacency[k];
        const auto it = adj.find(from);
        if (it == adj.end()) {
            return false;
        }
        return std::find(it->second.begin(), it->second.end(), to) !=
               it->second.end();
    };

    std::set<int> satisfied;
    int max_swaps_used = 0;

    for (const auto& [k, ends] : terminals) {
        const int s0 = ends.first;
        const int t0 = ends.second;

        // A path of a single arc means the two qubits are already adjacent.
        if (adjacent_in_path(k, s0, t0)) {
            satisfied.insert(k);
            continue;
        }

        const auto it_s = logical_at.find(s0);
        const auto it_t = logical_at.find(t0);
        if (it_s == logical_at.end() || it_t == logical_at.end()) {
            continue;
        }

        const int logical_s = it_s->second;
        const int logical_t = it_t->second;

        int current_s = s0;
        int current_t = t0;
        int previous_s = -1;
        int previous_t = -1;
        int swaps_used = 0;
        bool move_from_s = true;

        while (swaps_used < max_swaps_per_commodity) {
            if (adjacent_in_path(k, current_s, current_t)) {
                break;
            }

            // Advance one side by one hop along the path, carrying its qubit.
            int& current = move_from_s ? current_s : current_t;
            int& previous = move_from_s ? previous_s : previous_t;
            const int carried = move_from_s ? logical_s : logical_t;

            bool moved = false;

            for (const int next : adjacency[k][current]) {
                if (next == previous) {
                    continue;
                }

                const int displaced = occupant_of(next, carried);
                if (displaced == -1) {
                    throw std::logic_error(
                        "node swap: hardware vertex " + std::to_string(next) +
                        " on a routed path is unoccupied; the layout must be a "
                        "full permutation of the hardware vertices");
                }

                std::swap(physical_of[carried], physical_of[displaced]);
                logical_at[current] = displaced;
                previous = current;
                current = next;
                logical_at[current] = carried;

                const std::size_t layer = static_cast<std::size_t>(swaps_used / 2);
                if (result.layers.size() <= layer) {
                    result.layers.resize(layer + 1);
                }
                result.layers[layer].push_back(
                    SwapOp{carried, displaced, previous, current});

                ++swaps_used;
                ++result.total_swaps;
                moved = true;
                break;
            }

            if (!moved) {
                break;
            }

            move_from_s = !move_from_s;
        }

        max_swaps_used = std::max(max_swaps_used, swaps_used);

        if (adjacent_in_path(k, current_s, current_t)) {
            satisfied.insert(k);
        }
    }

    // Layer count implied by the deepest commodity, kept for parity with the
    // upstream depth accounting. resize() above already produced exactly this
    // many layers, so this is an assertion in effect.
    const std::size_t expected_layers =
        static_cast<std::size_t>((max_swaps_used + 1) / 2);
    if (result.layers.size() > expected_layers) {
        throw std::logic_error("node swap: inconsistent layer accounting");
    }

    for (int k = 0; k < static_cast<int>(commodity_edges.size()); ++k) {
        if (satisfied.count(k) != 0) {
            result.satisfied_edges.push_back(commodity_edges[k]);
            result.satisfied_indices.push_back(k);
        } else {
            result.remaining_edges.push_back(commodity_edges[k]);
        }
    }

    return result;
}

}  // namespace swapnet
