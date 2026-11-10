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

// Solver-agnostic mixed-integer program plus the builder for the disjoint-path
// routing model.
//
// Upstream (SWAP/MCF.{hpp,cpp}) built the model directly against the CPLEX
// Concert C++ API, which made CPLEX a hard build-time dependency of the whole
// project. Here the model is emitted as a plain sparse structure; choosing and
// running a solver is the caller's job (see swapnet/milp.py for the Python-side
// HiGHS / CPLEX backends).
//
// The model is a prize-collecting, vertex-disjoint multi-commodity flow on the
// hardware graph. One commodity per interaction edge that is not yet executable:
// y[k] = 1 iff commodity k is routed, x[a][k] = 1 iff commodity k uses directed
// arc a. Minimising (arcs used) - route_reward * (commodities routed) buys as
// many routed commodities as possible with as few arcs as possible.

#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "graph.hpp"

namespace swapnet {

// Row sense codes.
constexpr char kEqual = 'E';
constexpr char kLessEqual = 'L';
constexpr char kGreaterEqual = 'G';

// Variable type codes.
constexpr char kBinary = 'B';
constexpr char kInteger = 'I';
constexpr char kContinuous = 'C';

// Minimisation problem: min obj . x  s.t.  A x (sense) rhs,  lb <= x <= ub.
// A is stored row-major in compressed sparse row form.
struct MilpProblem {
    int num_vars = 0;
    std::vector<double> objective;
    std::vector<double> lower_bound;
    std::vector<double> upper_bound;
    std::vector<char> var_type;

    std::vector<int> row_start{0};
    std::vector<int> col_index;
    std::vector<double> coefficient;
    std::vector<char> sense;
    std::vector<double> rhs;

    int num_rows() const { return static_cast<int>(sense.size()); }
    int num_nonzeros() const { return static_cast<int>(col_index.size()); }

    // Appends a row from (column, coefficient) pairs. Duplicate columns are
    // summed by the caller's construction, not here.
    void add_row(const std::vector<std::pair<int, double>>& terms, char row_sense,
                 double row_rhs);

    // Two-sided form of the rows, which is what most solver APIs take:
    // 'E' gives rhs <= a.x <= rhs, 'L' gives -inf <= a.x <= rhs, 'G' gives
    // rhs <= a.x <= +inf. Computed here rather than in Python because a model
    // can have hundreds of thousands of rows.
    std::vector<double> row_lower() const;
    std::vector<double> row_upper() const;

    // 1 for integer and binary columns, 0 for continuous ones.
    std::vector<int> integrality() const;
};

struct MilpSolution {
    // True when the solver returned a usable (not necessarily proven optimal)
    // primal solution.
    bool has_solution = false;
    bool is_optimal = false;
    double objective = 0.0;
    std::vector<double> x;
    std::string status = "not solved";
};

struct RoutingParams {
    // Reward for routing one commodity, in units of "arcs used". Larger values
    // trade circuit depth for more parallel routing.
    //
    // With per-commodity couplings (see build_routing_model) the reward for
    // commodity k is route_reward * J_k^2 / mean(J^2), so this stays the average
    // reward and the pruning bound below stays meaningful.
    double route_reward = 30.0;
    // Upper bound of the per-arc uniform noise added to the objective to break
    // ties between equal-length paths.
    double tie_break_noise = 1e-3;
    // Hard cap on the number of arcs a routed commodity may use. 0 derives the
    // largest value that cannot exclude an optimal solution from route_reward.
    int max_path_length = 0;
    // Additionally restrict each commodity to paths at most
    // (hardware distance + max_detour) long. Negative means no extra limit.
    // This is a heuristic restriction: it can make the model much smaller on
    // large topologies but may exclude the true optimum.
    int max_detour = -1;
    // 0 draws a nondeterministic seed from std::random_device.
    std::uint64_t seed = 0;
};

struct Arc {
    int tail = 0;
    int head = 0;
    // Traversal cost of the underlying undirected hardware edge, W_ij. Both
    // directions of an edge carry the same cost.
    double cost = 1.0;
};

struct SolutionArc {
    int u = 0;
    int v = 0;
    int k = 0;
    double value = 0.0;
};

// A built routing model together with the index bookkeeping needed to read a
// solution back out.
struct RoutingModel {
    MilpProblem problem;
    // Directed arcs: the first num_edges entries follow the hardware edge list,
    // the next num_edges are their reverses.
    std::vector<Arc> arcs;
    // commodities[k] is a pair of physical vertices.
    std::vector<Edge> commodities;
    // Commodities fixed to y[k] = 0 because routing them cannot pay for itself.
    std::vector<char> active;
    // False when a zero-cost hardware edge made the exact path-length bound
    // unusable, so no arc could be pruned. The model is still correct, just
    // larger: a free edge means an arbitrarily long path can be free, so no
    // finite hop bound is safe. Callers building weights with
    // swapnet.weights.from_fidelities never hit this (fidelities are capped
    // below 1); a hand-built zero weight does.
    bool pruning_applied = true;
    // Reward actually placed on y[k], i.e. route_reward scaled by the
    // commodity's relative J^2. Kept for diagnostics and for explaining why one
    // commodity outbid another.
    std::vector<double> rewards;

    int num_arcs() const { return static_cast<int>(arcs.size()); }
    int num_commodities() const { return static_cast<int>(commodities.size()); }

    int x_index(int arc, int k) const { return arc * num_commodities() + k; }
    int y_index(int k) const { return num_arcs() * num_commodities() + k; }
};

// `hardware` must be validated and connected. `commodities` are pairs of
// physical vertices (origin, destination) with origin != destination.
// `distances` is the hardware all-pairs *hop* distance matrix, used for the
// path-length pruning; pass an empty matrix to skip pruning.
//
// `couplings` is either empty (every commodity equally valuable) or one J per
// commodity. The reward on y[k] is then
//
//     route_reward * J_k^2 / mean(J^2)
//
// so that a depth-truncated run -- one stopped by max_iterations or a max_depth
// budget before every interaction is realised -- spends its limited routing
// capacity on the interactions carrying the most J^2 mass. Squaring is the
// caller's requested objective (maximise sum of J_lm^2 over realised pairs);
// dividing by the mean keeps the average reward at route_reward, which is what
// the arc costs and the pruning bound are calibrated against.
//
// Hardware edge weights W_ij enter as the per-arc objective coefficient, so a
// path's price is the total error it accumulates rather than its hop count.
RoutingModel build_routing_model(
    const Graph& hardware, const std::vector<Edge>& commodities,
    const std::vector<std::vector<int>>& distances,
    const std::vector<double>& couplings, const RoutingParams& params);

// Reads the arcs carrying flow out of a solution vector. Arcs of commodities
// whose y[k] is zero are dropped, matching the upstream behaviour where only
// routed commodities contribute usable paths.
std::vector<SolutionArc> extract_solution_arcs(const RoutingModel& model,
                                              const MilpSolution& solution,
                                              double tolerance = 1e-6);

// Number of commodities with y[k] = 1.
int count_routed(const RoutingModel& model, const MilpSolution& solution,
                 double tolerance = 1e-6);

}  // namespace swapnet
