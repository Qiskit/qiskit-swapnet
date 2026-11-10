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

// Initial-layout search: a quadratic assignment problem solved by multi-restart
// simulated annealing followed by tabu search.
//
// Port of upstream MAP/QAP.{hpp,cpp}. Behavioural changes:
//   * the RNG is seedable (upstream used std::random_device unconditionally),
//   * an optional wall-clock budget can stop the search early,
//   * progress is reported through a callback instead of std::cout,
//   * all tuning constants are parameters rather than literals.
//
// The objective is sum over interaction edges (i, j) of
// J_ij * D[layout[i]][layout[j]], i.e. the total hardware distance spanned by
// the requested two-qubit interactions, each weighted by its coupling strength.
// layout is a permutation of the hardware vertices, so
// layout[logical] == physical.
//
// D may be hop counts (bfs_all_pairs) or weighted path costs
// (dijkstra_all_pairs); passing the latter steers the layout onto the
// high-fidelity couplings. J defaults to all ones, which recovers the
// unweighted objective. Both are doubles, so a weighted cost is exact rather
// than rounded.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "graph.hpp"

namespace swapnet {

struct QapParams {
    int num_restarts = 70;
    int sa_iterations = 50000;
    int tabu_iterations = 20000;
    int tabu_tenure = 25;
    // Tabu/perturb cycles run after each annealing restart.
    int cycles_per_restart = 10;
    // Number of random transpositions applied between cycles.
    int perturb_strength = 5;
    // Initial annealing temperature is drawn uniformly from [t0_min, t0_max]
    // and the per-iteration cooling factor from [cooling_min, cooling_max].
    //
    // These are calibrated against a cost built from unit-weight hop counts. If
    // you scale the edge weights or the couplings far from mean 1, the
    // acceptance probability exp(-delta / T) saturates or freezes and the
    // annealing stops exploring; normalise the weights instead of retuning
    // (see swapnet.weights.normalize).
    double t0_min = 50.0;
    double t0_max = 150.0;
    double cooling_min = 0.9995;
    double cooling_max = 0.99995;
    // Wall-clock budget in seconds; <= 0 means unlimited. Checked between
    // restarts and cycles, so a single cycle always runs to completion.
    double max_seconds = 0.0;
    // 0 draws a nondeterministic seed from std::random_device.
    std::uint64_t seed = 0;
};

struct QapResult {
    std::vector<int> layout;
    double cost = 0.0;
    int restarts_run = 0;
    bool hit_time_limit = false;
};

// Called as (restart_index, num_restarts, best_cost_so_far). May be empty.
using QapProgressFn =
    std::function<void(int, int, double)>;

// `dist` must be a square matrix of finite non-negative distances (typically
// dijkstra_all_pairs output on a connected hardware graph, or bfs_all_pairs
// converted to double). `interaction_edges` indexes into the same vertex range.
// `couplings` is either empty (all J = 1) or one non-negative finite entry per
// interaction edge. Throws std::invalid_argument on a ragged matrix, a negative
// or non-finite entry, or out-of-range edges.
QapResult solve_qap(const std::vector<std::vector<double>>& dist,
                    const std::vector<Edge>& interaction_edges,
                    const std::vector<double>& couplings,
                    const QapParams& params,
                    const QapProgressFn& progress = {});

// Objective value of an explicit layout, for verification and for comparing
// against externally produced layouts.
double layout_cost(const std::vector<std::vector<double>>& dist,
                   const std::vector<Edge>& interaction_edges,
                   const std::vector<double>& couplings,
                   const std::vector<int>& layout);

}  // namespace swapnet
