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

// The routing loop: repeatedly execute whatever interactions the current layout
// already supports, then route and swap to enable more, until nothing is left.
//
// Port of the driver loop in upstream SWAP/main.cpp, with the MILP solver
// injected as a callable so the C++ side stays solver-agnostic. Additions:
//   * the interaction edges are grouped into an explicit execution timeline of
//     alternating gate batches and SWAP layers, which is the routing plan a
//     circuit compiler actually needs; upstream only wrote out the swap layers,
//   * interaction edges that never became executable are reported instead of
//     silently dropped when the iteration cap is reached,
//   * per-iteration statistics are collected, and an iteration that makes no
//     progress is detected instead of burning the remaining iteration budget,
//   * the shuffle that decides which pending interactions get offered to the
//     solver is seeded.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "graph.hpp"
#include "milp.hpp"
#include "node_swap.hpp"

namespace swapnet {

struct RouteParams {
    int max_iterations = 100;
    // Hard cap on emitted SWAP layers. Negative means unlimited, matching
    // RoutingParams::max_detour; zero is a real budget that forbids every SWAP,
    // leaving only the interactions the initial layout already supports. When the
    // cap binds, routing stops and whatever is left appears in
    // RouteResult::unrouted_edges. Combined with per-interaction couplings this
    // is the depth-truncation knob: the router spends the layers it is allowed on
    // the interactions carrying the most J^2 mass.
    int max_depth = -1;
    // Swaps one commodity may contribute per iteration. Two means each routed
    // interaction moves both of its qubits one hop closer, in a single layer.
    int max_swaps_per_commodity = 2;
    // Cap on interactions offered to the solver per iteration; <= 0 offers all
    // of them. Vertex-disjointness already limits how many can be routed at
    // once, so a cap near half the hardware size loses little and can shrink
    // the model a lot.
    int max_commodities_per_round = 0;
    // Consecutive iterations with no progress before giving up.
    int max_stagnant_iterations = 3;
    // Shuffle pending interactions each iteration, as upstream did, so a capped
    // or solver-limited round does not keep retrying the same subset.
    bool shuffle_pending = true;
    RoutingParams routing;
    // 0 draws a nondeterministic seed from std::random_device.
    std::uint64_t seed = 0;
};

// One step of the execution plan: the interactions that can be executed at this
// point, followed by a layer of SWAP gates applied in parallel.
struct Stage {
    std::vector<Edge> gates;
    std::vector<SwapOp> swaps;
};

struct IterationStats {
    int iteration = 0;
    int executable_before = 0;
    int commodities = 0;
    int deferred = 0;
    int routed = 0;
    double objective = 0.0;
    bool solver_optimal = false;
    std::string solver_status;
    int swaps = 0;
    int layers = 0;
    int satisfied_after_swaps = 0;
    int remaining = 0;
    // J^2 mass this iteration made executable, and the hardware weight its
    // swaps cost.
    double realized_weight = 0.0;
    double swap_weight = 0.0;
};

struct RouteResult {
    std::vector<int> initial_layout;
    std::vector<int> final_layout;
    std::vector<Stage> stages;
    // Interactions still not executable when the loop stopped.
    std::vector<Edge> unrouted_edges;
    std::vector<IterationStats> iterations;
    int total_swaps = 0;
    int num_layers = 0;
    bool converged = false;
    // True when the loop stopped because max_depth was reached rather than
    // because everything was routed or progress stalled.
    bool hit_depth_limit = false;
    // Sum of J^2 over the interactions the plan executes, and over those it does
    // not. With no couplings supplied every J is 1, so these are plain counts.
    // realized_weight / (realized_weight + unrealized_weight) is the fraction of
    // the requested interaction strength the truncated plan achieves.
    double realized_weight = 0.0;
    double unrealized_weight = 0.0;
    // Total hardware weight W_ij accumulated by every emitted SWAP, in the units
    // of the weights this call was given. Note the Python layer normalises the
    // weights to mean 1 before calling in, so a plan produced through
    // swapnet.route reports this on that normalised footing rather than in the
    // caller's original units; ratios between plans are unaffected.
    double swap_weight = 0.0;
};

// Solves one MILP. Must return has_solution = false only when no primal
// solution at all was found.
using MilpSolveFn = std::function<MilpSolution(const MilpProblem&)>;

// Called after each iteration. May be empty.
using RouteProgressFn = std::function<void(const IterationStats&)>;

// `hardware` must be validated and connected. `interaction_edges` are logical
// vertex pairs. `layout` is a full permutation with layout[logical] == physical.
// `couplings` is either empty (every interaction equally valuable) or one
// non-negative J per interaction edge.
//
// When the couplings are non-uniform, the pending interactions offered to the
// solver each round are ordered by descending J, so that a round capped by
// max_commodities_per_round offers the most valuable ones first. Within the
// solver the reward is proportional to J^2 (see build_routing_model).
RouteResult route(const Graph& hardware,
                  const std::vector<Edge>& interaction_edges,
                  const std::vector<double>& couplings,
                  const std::vector<int>& layout, const RouteParams& params,
                  const MilpSolveFn& solve,
                  const RouteProgressFn& progress = {});

}  // namespace swapnet
