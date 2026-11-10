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

#include "milp.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

namespace swapnet {

void MilpProblem::add_row(const std::vector<std::pair<int, double>>& terms,
                          char row_sense, double row_rhs) {
    for (const auto& [col, coef] : terms) {
        col_index.push_back(col);
        coefficient.push_back(coef);
    }
    row_start.push_back(static_cast<int>(col_index.size()));
    sense.push_back(row_sense);
    rhs.push_back(row_rhs);
}

std::vector<double> MilpProblem::row_lower() const {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> lower(sense.size());
    for (std::size_t r = 0; r < sense.size(); ++r) {
        lower[r] = (sense[r] == kLessEqual) ? -inf : rhs[r];
    }
    return lower;
}

std::vector<double> MilpProblem::row_upper() const {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> upper(sense.size());
    for (std::size_t r = 0; r < sense.size(); ++r) {
        upper[r] = (sense[r] == kGreaterEqual) ? inf : rhs[r];
    }
    return upper;
}

std::vector<int> MilpProblem::integrality() const {
    std::vector<int> flags(var_type.size());
    for (std::size_t j = 0; j < var_type.size(); ++j) {
        flags[j] = (var_type[j] == kContinuous) ? 0 : 1;
    }
    return flags;
}

namespace {

// Largest number of arcs a commodity rewarded `reward` can use while still
// improving the objective. Every arc costs at least `min_arc_cost`, so a path of
// L arcs costs at least L * min_arc_cost; once that reaches the reward, routing
// is dominated by leaving the commodity unrouted. Returns 0 for "no profitable
// path exists" and -1 for "no bound can be derived", which happens when some
// edge is free and a path can therefore be arbitrarily long at no cost.
//
// The bound is in hops while the cost is weighted, which is what keeps it
// conservative: it can only ever be looser than the true cost bound, so it
// never excludes an optimal solution.
int exact_length_bound(double reward, double min_arc_cost) {
    if (min_arc_cost <= 0.0) {
        return -1;
    }
    if (reward <= min_arc_cost) {
        return 0;
    }
    return static_cast<int>(std::floor(reward / min_arc_cost - 1e-9));
}

}  // namespace

RoutingModel build_routing_model(
    const Graph& hardware, const std::vector<Edge>& commodities,
    const std::vector<std::vector<int>>& distances,
    const std::vector<double>& couplings, const RoutingParams& params) {
    const int n = hardware.num_vertices;
    const int num_edges = hardware.num_edges();

    RoutingModel model;
    model.commodities = commodities;

    // Directed arcs: forward pass over the edge list, then all reverses. The
    // arc-capacity rows below rely on arc a and arc a + num_edges being a pair,
    // and both directions carry the same weight.
    model.arcs.reserve(static_cast<std::size_t>(2) * num_edges);
    for (int i = 0; i < num_edges; ++i) {
        const auto [u, v] = hardware.edges[static_cast<std::size_t>(i)];
        model.arcs.push_back({u, v, hardware.weight(i)});
    }
    for (int i = 0; i < num_edges; ++i) {
        const auto [u, v] = hardware.edges[static_cast<std::size_t>(i)];
        model.arcs.push_back({v, u, hardware.weight(i)});
    }

    const int A = model.arcs.size();
    const int K = static_cast<int>(commodities.size());

    for (const auto& [o, d] : commodities) {
        if (o < 0 || o >= n || d < 0 || d >= n) {
            throw std::invalid_argument(
                "commodity endpoint (" + std::to_string(o) + ", " +
                std::to_string(d) + ") is outside [0, " + std::to_string(n) + ")");
        }
        if (o == d) {
            throw std::invalid_argument(
                "commodity has identical origin and destination " +
                std::to_string(o));
        }
    }

    validate_interaction_weights(couplings, commodities.size(), "commodities");

    // Per-commodity reward: route_reward * J_k^2 / mean(J^2). Squaring realises
    // the "maximise sum of J^2 over the realised interactions" objective, and
    // normalising by the mean keeps the average reward equal to route_reward so
    // that the arc costs and the length bound stay calibrated.
    model.rewards.assign(K, params.route_reward);
    if (!couplings.empty() && K > 0) {
        double mean_square = 0.0;
        for (const double j : couplings) {
            mean_square += j * j;
        }
        mean_square /= static_cast<double>(K);

        if (mean_square > 0.0) {
            for (int k = 0; k < K; ++k) {
                model.rewards[k] =
                    params.route_reward * couplings[k] * couplings[k] / mean_square;
            }
        } else {
            // Every coupling is zero: no interaction is worth any depth. Leave
            // the rewards at zero so nothing is routed.
            model.rewards.assign(K, 0.0);
        }
    }

    // Cheapest arc, which sets how long a profitable path can be.
    double min_arc_cost = 1.0;
    if (hardware.is_weighted() && num_edges > 0) {
        min_arc_cost = *std::min_element(hardware.weights.begin(),
                                         hardware.weights.end());
    }

    const bool have_distances = static_cast<int>(distances.size()) == n;

    // Decide which commodities can pay for themselves, and which arcs each of
    // them could possibly use. Both restrictions keep at least one optimal
    // solution when max_detour < 0; max_detour tightens it heuristically.
    model.active.assign(K, 1);
    std::vector<char> arc_allowed(static_cast<std::size_t>(A) * K, 1);

    if (have_distances) {
        for (int k = 0; k < K; ++k) {
            const int o = commodities[k].first;
            const int d = commodities[k].second;
            const int direct = distances[o][d];

            // The bound is per-commodity now, because the reward is.
            const int exact = exact_length_bound(model.rewards[k], min_arc_cost);
            int budget = params.max_path_length > 0 ? params.max_path_length
                                                    : exact;
            if (params.max_path_length <= 0 && exact < 0) {
                // A zero-cost edge exists, so no finite hop bound is safe. Record
                // it: the model is correct but unpruned, which is the difference
                // between a solvable and an unsolvable round on a large device.
                budget = std::numeric_limits<int>::max();
                model.pruning_applied = false;
            }
            if (params.max_detour >= 0 && direct >= 0) {
                budget = std::min(budget, direct + params.max_detour);
            }

            if (direct < 0 || direct > budget) {
                // Unreachable, or no profitable path can exist.
                model.active[k] = 0;
                for (int a = 0; a < A; ++a) {
                    arc_allowed[static_cast<std::size_t>(a) * K + k] = 0;
                }
                continue;
            }

            for (int a = 0; a < A; ++a) {
                const int to_tail = distances[o][model.arcs[a].tail];
                const int from_head = distances[model.arcs[a].head][d];
                const bool usable = to_tail >= 0 && from_head >= 0 &&
                                    to_tail + 1 + from_head <= budget;
                if (!usable) {
                    arc_allowed[static_cast<std::size_t>(a) * K + k] = 0;
                }
            }
        }
    }

    // Variables: x[a][k] laid out arc-major, then y[k].
    MilpProblem& p = model.problem;
    p.num_vars = A * K + K;
    p.objective.assign(p.num_vars, 0.0);
    p.lower_bound.assign(p.num_vars, 0.0);
    p.upper_bound.assign(p.num_vars, 1.0);
    p.var_type.assign(p.num_vars, kBinary);

    const std::uint64_t seed =
        params.seed != 0
            ? params.seed
            : (static_cast<std::uint64_t>(std::random_device{}()) << 32) ^
                  std::random_device{}();
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> noise(0.0, params.tie_break_noise);

    // Using an arc costs its hardware weight W_ij, so the solver prefers paths
    // through the high-fidelity couplings. The noise only breaks ties.
    for (int a = 0; a < A; ++a) {
        for (int k = 0; k < K; ++k) {
            const int idx = model.x_index(a, k);
            p.objective[idx] = model.arcs[a].cost + noise(rng);
            if (!arc_allowed[static_cast<std::size_t>(a) * K + k]) {
                p.upper_bound[idx] = 0.0;
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        const int idx = model.y_index(k);
        p.objective[idx] = -model.rewards[k];
        if (!model.active[k]) {
            p.upper_bound[idx] = 0.0;
        }
    }

    // Incidence lists over the directed arcs.
    std::vector<std::vector<int>> arcs_in(n);
    std::vector<std::vector<int>> arcs_out(n);
    for (int a = 0; a < A; ++a) {
        arcs_in[model.arcs[a].head].push_back(a);
        arcs_out[model.arcs[a].tail].push_back(a);
    }

    std::vector<std::pair<int, double>> terms;

    // Flow conservation: out - in = y at the origin, -y at the destination,
    // 0 elsewhere. Rows for inactive commodities are skipped; their variables
    // are already fixed to zero.
    for (int k = 0; k < K; ++k) {
        if (!model.active[k]) {
            continue;
        }
        for (int i = 0; i < n; ++i) {
            terms.clear();
            for (const int a : arcs_out[i]) {
                terms.emplace_back(model.x_index(a, k), 1.0);
            }
            for (const int a : arcs_in[i]) {
                terms.emplace_back(model.x_index(a, k), -1.0);
            }
            if (i == model.commodities[k].first) {
                terms.emplace_back(model.y_index(k), -1.0);
            }
            if (i == model.commodities[k].second) {
                terms.emplace_back(model.y_index(k), 1.0);
            }
            if (!terms.empty()) {
                p.add_row(terms, kEqual, 0.0);
            }
        }
    }

    // Vertex-disjointness: at most one unit of flow leaves, and at most one
    // enters, each hardware vertex across all commodities.
    for (int i = 0; i < n; ++i) {
        terms.clear();
        for (const int a : arcs_out[i]) {
            for (int k = 0; k < K; ++k) {
                terms.emplace_back(model.x_index(a, k), 1.0);
            }
        }
        if (!terms.empty()) {
            p.add_row(terms, kLessEqual, 1.0);
        }

        terms.clear();
        for (const int a : arcs_in[i]) {
            for (int k = 0; k < K; ++k) {
                terms.emplace_back(model.x_index(a, k), 1.0);
            }
        }
        if (!terms.empty()) {
            p.add_row(terms, kLessEqual, 1.0);
        }
    }

    // Terminal capacity: a vertex is an endpoint of at most one routed
    // commodity.
    for (int i = 0; i < n; ++i) {
        terms.clear();
        for (int k = 0; k < K; ++k) {
            if (model.commodities[k].first == i ||
                model.commodities[k].second == i) {
                terms.emplace_back(model.y_index(k), 1.0);
            }
        }
        if (!terms.empty()) {
            p.add_row(terms, kLessEqual, 1.0);
        }
    }

    // Arc capacity: each undirected hardware edge carries at most one unit, in
    // either direction, across all commodities.
    for (int a = 0; a < num_edges; ++a) {
        terms.clear();
        for (int k = 0; k < K; ++k) {
            terms.emplace_back(model.x_index(a, k), 1.0);
            terms.emplace_back(model.x_index(a + num_edges, k), 1.0);
        }
        if (!terms.empty()) {
            p.add_row(terms, kLessEqual, 1.0);
        }
    }

    return model;
}

std::vector<SolutionArc> extract_solution_arcs(const RoutingModel& model,
                                               const MilpSolution& solution,
                                               double tolerance) {
    std::vector<SolutionArc> arcs;

    if (!solution.has_solution) {
        return arcs;
    }
    if (static_cast<int>(solution.x.size()) != model.problem.num_vars) {
        throw std::invalid_argument(
            "solution vector has " + std::to_string(solution.x.size()) +
            " entries but the model has " +
            std::to_string(model.problem.num_vars) + " variables");
    }

    const int A = model.num_arcs();
    const int K = model.num_commodities();

    for (int k = 0; k < K; ++k) {
        if (solution.x[model.y_index(k)] <= 0.5) {
            continue;
        }
        for (int a = 0; a < A; ++a) {
            const double value = solution.x[model.x_index(a, k)];
            if (value > tolerance) {
                arcs.push_back(
                    {model.arcs[a].tail, model.arcs[a].head, k, value});
            }
        }
    }

    return arcs;
}

int count_routed(const RoutingModel& model, const MilpSolution& solution,
                 double tolerance) {
    if (!solution.has_solution) {
        return 0;
    }
    int routed = 0;
    for (int k = 0; k < model.num_commodities(); ++k) {
        if (solution.x[model.y_index(k)] > 1.0 - tolerance) {
            ++routed;
        }
    }
    return routed;
}

}  // namespace swapnet
