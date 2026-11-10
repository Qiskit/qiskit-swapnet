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

#include "pipeline.hpp"

#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <stdexcept>

namespace swapnet {

RouteResult route(const Graph& hardware,
                  const std::vector<Edge>& interaction_edges,
                  const std::vector<double>& couplings,
                  const std::vector<int>& layout, const RouteParams& params,
                  const MilpSolveFn& solve, const RouteProgressFn& progress) {
    hardware.validate("hardware graph");
    validate_interaction_edges(interaction_edges, hardware.num_vertices);
    validate_interaction_weights(couplings, interaction_edges.size());
    validate_layout(layout, hardware.num_vertices);

    if (!solve) {
        throw std::invalid_argument("no MILP solver callable was provided");
    }

    RouteResult result;
    result.initial_layout = layout;
    result.final_layout = layout;

    // Couplings keyed by normalised interaction edge. The loop reshuffles,
    // caps and partitions the pending edges every round, so carrying J along in
    // a parallel vector would mean re-deriving the permutation each time; the
    // edges are validated duplicate-free, so they are safe map keys.
    std::map<Edge, double> coupling_of;
    for (std::size_t e = 0; e < interaction_edges.size(); ++e) {
        const auto [u, v] = interaction_edges[e];
        coupling_of[normalized(u, v)] = couplings.empty() ? 1.0 : couplings[e];
    }
    const auto coupling_for = [&](const Edge& e) {
        return coupling_of.at(normalized(e.first, e.second));
    };
    // J^2 is the quantity the truncation objective accumulates.
    const auto mass_for = [&](const Edge& e) {
        const double j = coupling_for(e);
        return j * j;
    };

    // Hardware weight of the coupling a SWAP acts on, for the cost accounting.
    std::map<Edge, double> hardware_weight;
    for (int i = 0; i < hardware.num_edges(); ++i) {
        const auto [u, v] = hardware.edges[static_cast<std::size_t>(i)];
        hardware_weight[normalized(u, v)] = hardware.weight(i);
    }
    const auto swap_cost = [&](const std::vector<SwapOp>& layer) {
        double total = 0.0;
        for (const auto& op : layer) {
            total += hardware_weight.at(normalized(op.physical_u, op.physical_v));
        }
        return total;
    };

    if (interaction_edges.empty()) {
        result.converged = true;
        return result;
    }

    if (!hardware.is_connected()) {
        throw std::invalid_argument(
            "hardware graph is disconnected; routing needs a path between every "
            "pair of physical qubits");
    }

    // Hardware adjacency as a set, for the "already executable" test, and the
    // distance matrix used to prune the routing model.
    std::set<Edge> hardware_edges;
    for (const auto& [u, v] : hardware.edges) {
        hardware_edges.insert(normalized(u, v));
    }
    const auto distances = bfs_all_pairs(hardware);

    const std::uint64_t seed =
        params.seed != 0
            ? params.seed
            : (static_cast<std::uint64_t>(std::random_device{}()) << 32) ^
                  std::random_device{}();
    std::mt19937_64 rng(seed);

    std::vector<int> current_layout = layout;
    std::vector<Edge> remaining = interaction_edges;

    // Interactions that are executable but whose stage has not been emitted yet,
    // because no SWAP layer has closed the stage.
    std::vector<Edge> pending_gates;

    int stagnant = 0;
    // SWAP layers emitted so far, against which max_depth is charged.
    int emitted_layers = 0;

    for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
        IterationStats stats;
        stats.iteration = iteration;

        // Split the remaining interactions into those the current layout already
        // supports and those that still need swaps.
        std::vector<Edge> pending;
        std::vector<Edge> executable;
        for (const auto& e : remaining) {
            const Edge mapped =
                normalized(current_layout[e.first], current_layout[e.second]);
            if (hardware_edges.count(mapped) != 0) {
                executable.push_back(e);
            } else {
                pending.push_back(e);
            }
        }

        stats.executable_before = static_cast<int>(executable.size());
        pending_gates.insert(pending_gates.end(), executable.begin(),
                             executable.end());
        for (const auto& e : executable) {
            const double mass = mass_for(e);
            stats.realized_weight += mass;
            result.realized_weight += mass;
        }

        if (pending.empty()) {
            stats.remaining = 0;
            result.iterations.push_back(stats);
            if (progress) {
                progress(stats);
            }
            remaining.clear();
            result.converged = true;
            break;
        }

        // Out of depth: routing the rest would need a SWAP layer we cannot
        // afford. Stop before building and solving a model whose every layer
        // would be discarded. The epilogue below harvests anything the current
        // layout already supports, since executing a gate costs no depth.
        if (params.max_depth >= 0 && emitted_layers >= params.max_depth) {
            stats.remaining = static_cast<int>(pending.size());
            result.iterations.push_back(stats);
            if (progress) {
                progress(stats);
            }
            remaining = std::move(pending);
            result.hit_depth_limit = true;
            break;
        }

        if (params.shuffle_pending) {
            std::shuffle(pending.begin(), pending.end(), rng);
        }

        // With non-uniform couplings, sort the strongest interactions to the
        // front so that the per-round cap below defers the least valuable ones.
        //
        // Note this largely supersedes the shuffle above whenever the couplings
        // are distinct, which is the normal case: the sort is then a total order
        // and only ties keep their shuffled order. That is intended -- offering
        // the most valuable interactions first is the whole point -- and it does
        // not starve a capped round, because a routed interaction leaves
        // `remaining`, so the top of the order advances every iteration. Rounds
        // that route nothing at all are caught by max_stagnant_iterations.
        if (!couplings.empty()) {
            std::stable_sort(pending.begin(), pending.end(),
                             [&](const Edge& a, const Edge& b) {
                                 return coupling_for(a) > coupling_for(b);
                             });
        }

        std::vector<Edge> deferred;
        if (params.max_commodities_per_round > 0 &&
            static_cast<int>(pending.size()) > params.max_commodities_per_round) {
            deferred.assign(pending.begin() + params.max_commodities_per_round,
                            pending.end());
            pending.resize(params.max_commodities_per_round);
        }
        stats.deferred = static_cast<int>(deferred.size());
        stats.commodities = static_cast<int>(pending.size());

        // Commodities are the pending interactions expressed in physical
        // vertices under the current layout.
        std::vector<Edge> commodities;
        commodities.reserve(pending.size());
        for (const auto& e : pending) {
            commodities.emplace_back(current_layout[e.first],
                                     current_layout[e.second]);
        }

        RoutingParams routing = params.routing;
        if (routing.seed == 0) {
            // Derive a per-iteration seed so the tie-break noise is
            // reproducible without being identical every round.
            routing.seed = rng();
        }

        // Commodity couplings, in the same order as the commodity list.
        std::vector<double> commodity_couplings;
        if (!couplings.empty()) {
            commodity_couplings.reserve(pending.size());
            for (const auto& e : pending) {
                commodity_couplings.push_back(coupling_for(e));
            }
        }

        const RoutingModel model = build_routing_model(
            hardware, commodities, distances, commodity_couplings, routing);

        const MilpSolution solution = solve(model.problem);
        stats.objective = solution.objective;
        stats.solver_optimal = solution.is_optimal;
        stats.solver_status = solution.status;
        stats.routed = count_routed(model, solution);

        const auto arcs = extract_solution_arcs(model, solution);

        const SwapRoundResult round = apply_node_swaps(
            current_layout, pending, arcs, params.max_swaps_per_commodity);

        // Honour the depth budget. Layers beyond it are dropped, which means the
        // round's own layout and satisfied set no longer apply: replay only the
        // layers that fit and re-derive the state from there.
        std::size_t layers_to_emit = round.layers.size();
        bool truncated = false;
        if (params.max_depth >= 0) {
            const std::size_t budget =
                static_cast<std::size_t>(params.max_depth) -
                std::min(static_cast<std::size_t>(params.max_depth),
                         static_cast<std::size_t>(emitted_layers));
            if (layers_to_emit > budget) {
                layers_to_emit = budget;
                truncated = true;
            }
        }

        // Emit one stage per swap layer, flushing the gates that became
        // executable before it.
        for (std::size_t l = 0; l < layers_to_emit; ++l) {
            Stage stage;
            stage.gates = std::move(pending_gates);
            pending_gates.clear();
            stage.swaps = round.layers[l];
            const double cost = swap_cost(round.layers[l]);
            stats.swap_weight += cost;
            result.swap_weight += cost;
            stats.swaps += static_cast<int>(round.layers[l].size());
            result.total_swaps += static_cast<int>(round.layers[l].size());
            result.stages.push_back(std::move(stage));
            ++emitted_layers;
        }

        if (!truncated) {
            stats.layers = static_cast<int>(round.layers.size());
            stats.satisfied_after_swaps =
                static_cast<int>(round.satisfied_edges.size());

            // Interactions closed by this round's swaps become executable at the
            // layout the round ended on, so they belong to the next stage.
            pending_gates.insert(pending_gates.end(),
                                 round.satisfied_edges.begin(),
                                 round.satisfied_edges.end());
            for (const auto& e : round.satisfied_edges) {
                const double mass = mass_for(e);
                stats.realized_weight += mass;
                result.realized_weight += mass;
            }

            current_layout = round.layout;
            remaining = round.remaining_edges;
        } else {
            // Replay just the emitted layers to get the layout actually reached.
            for (std::size_t l = 0; l < layers_to_emit; ++l) {
                for (const auto& op : round.layers[l]) {
                    std::swap(current_layout[op.logical_a],
                              current_layout[op.logical_b]);
                }
            }
            stats.layers = static_cast<int>(layers_to_emit);
            // Everything offered this round is still outstanding; the loop-top
            // executability test below decides what the truncated layout closed.
            remaining = pending;
            result.hit_depth_limit = true;
        }

        remaining.insert(remaining.end(), deferred.begin(), deferred.end());
        stats.remaining = static_cast<int>(remaining.size());

        result.iterations.push_back(stats);
        if (progress) {
            progress(stats);
        }

        if (result.hit_depth_limit) {
            break;
        }

        const bool progressed = stats.executable_before > 0 ||
                                stats.satisfied_after_swaps > 0 ||
                                stats.swaps > 0;
        stagnant = progressed ? 0 : stagnant + 1;
        if (params.max_stagnant_iterations > 0 &&
            stagnant >= params.max_stagnant_iterations) {
            break;
        }

        if (remaining.empty()) {
            result.converged = true;
            break;
        }
    }

    // The loop can exit with interactions that the layout it stopped on already
    // supports: a depth-truncated round leaves everything it offered pending, and
    // a stagnant or iteration-capped exit never re-tests. Harvest them, since
    // executing a gate costs no depth.
    {
        std::vector<Edge> still_pending;
        for (const auto& e : remaining) {
            const Edge mapped =
                normalized(current_layout[e.first], current_layout[e.second]);
            if (hardware_edges.count(mapped) != 0) {
                pending_gates.push_back(e);
                result.realized_weight += mass_for(e);
            } else {
                still_pending.push_back(e);
            }
        }
        remaining = std::move(still_pending);
    }

    if (!pending_gates.empty()) {
        Stage stage;
        stage.gates = std::move(pending_gates);
        result.stages.push_back(std::move(stage));
    }

    result.final_layout = current_layout;
    result.unrouted_edges = remaining;
    for (const auto& e : remaining) {
        result.unrealized_weight += mass_for(e);
    }
    result.num_layers = 0;
    for (const auto& stage : result.stages) {
        if (!stage.swaps.empty()) {
            ++result.num_layers;
        }
    }
    if (remaining.empty()) {
        result.converged = true;
    }

    return result;
}

}  // namespace swapnet
