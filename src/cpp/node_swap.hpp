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

// Turns routed disjoint paths into concrete SWAP layers.
//
// Port of upstream SWAP/NodeSwap.{hpp,cpp}. For each routed commodity the two
// path endpoints are walked towards each other, alternating sides, exchanging
// the qubits on consecutive path vertices until the commodity's two logical
// qubits sit on adjacent hardware vertices. Because the routed paths are
// vertex-disjoint and the two sides of one path move away from each other, all
// swaps recorded at the same index are on disjoint hardware edges and form one
// parallel layer.
//
// Differences from upstream:
//   * commodities are visited in ascending index order rather than in
//     unordered_set order, so a given input produces a repeatable result,
//   * each swap records the hardware edge it acts on, not only the pair of
//     logical qubits that changed places, which is what a circuit needs,
//   * the layer bookkeeping is local to one call instead of a mutable member.

#pragma once

#include <utility>
#include <vector>

#include "graph.hpp"
#include "milp.hpp"

namespace swapnet {

// One SWAP gate: logical qubits `logical_a` and `logical_b` exchange places, and
// they sit on the adjacent hardware vertices `physical_u` and `physical_v` at the
// moment the gate is applied.
struct SwapOp {
    int logical_a = 0;
    int logical_b = 0;
    int physical_u = 0;
    int physical_v = 0;
};

struct SwapRoundResult {
    // Layout after this round's swaps; layout[logical] == physical.
    std::vector<int> layout;
    // Swap layers produced by this round, in execution order.
    std::vector<std::vector<SwapOp>> layers;
    // Commodity edges whose endpoints became adjacent, and those still pending.
    std::vector<Edge> satisfied_edges;
    std::vector<Edge> remaining_edges;
    // Indices into the input `commodity_edges` for the satisfied entries.
    std::vector<int> satisfied_indices;
    int total_swaps = 0;
};

// `layout` must be a full permutation of the hardware vertices.
// `commodity_edges[k]` is the logical interaction edge behind commodity k, so it
// must be ordered exactly like the commodity list passed to
// build_routing_model. `solution_arcs` come from extract_solution_arcs.
SwapRoundResult apply_node_swaps(const std::vector<int>& layout,
                                 const std::vector<Edge>& commodity_edges,
                                 const std::vector<SolutionArc>& solution_arcs,
                                 int max_swaps_per_commodity = 2);

}  // namespace swapnet
