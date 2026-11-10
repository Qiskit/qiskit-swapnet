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

#include "qap.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace swapnet {

namespace {

using Clock = std::chrono::steady_clock;

// Cost improvements below this are treated as noise rather than progress, so
// that accumulated floating-point drift in the incremental cost cannot be
// mistaken for a better layout.
constexpr double kCostEpsilon = 1e-9;

double elapsed_since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void check_inputs(const std::vector<std::vector<double>>& dist,
                  const std::vector<Edge>& edges,
                  const std::vector<double>& couplings) {
    const int n = static_cast<int>(dist.size());

    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(dist[i].size()) != n) {
            throw std::invalid_argument(
                "distance matrix is not square (row " + std::to_string(i) +
                " has " + std::to_string(dist[i].size()) + " of " +
                std::to_string(n) + " entries)");
        }
        for (int j = 0; j < n; ++j) {
            if (dist[i][j] < 0.0) {
                throw std::invalid_argument(
                    "distance matrix entry (" + std::to_string(i) + ", " +
                    std::to_string(j) +
                    ") is negative; the hardware graph is probably "
                    "disconnected");
            }
            if (!std::isfinite(dist[i][j])) {
                throw std::invalid_argument(
                    "distance matrix entry (" + std::to_string(i) + ", " +
                    std::to_string(j) +
                    ") is infinite; the hardware graph is disconnected, so no "
                    "layout has a finite cost");
            }
        }
    }

    validate_interaction_edges(edges, n);
    validate_interaction_weights(couplings, edges.size());
}

// Multi-restart simulated annealing + tabu search over permutations.
class QapSolver {
public:
    QapSolver(const std::vector<std::vector<double>>& dist,
              const std::vector<Edge>& edges,
              const std::vector<double>& couplings,
              std::uint64_t seed)
        : d_(dist),
          edges_(edges),
          couplings_(couplings),
          n_(static_cast<int>(dist.size())),
          rng_(seed),
          pick_(0, std::max(0, n_ - 1)) {
        // Each neighbour entry carries the coupling of the edge it came from, so
        // that the incremental delta below stays O(degree) under weighting.
        neighbours_.assign(n_, {});
        for (std::size_t e = 0; e < edges_.size(); ++e) {
            const auto [i, j] = edges_[e];
            const double j_lm = coupling(e);
            neighbours_[i].emplace_back(j, j_lm);
            neighbours_[j].emplace_back(i, j_lm);
        }
    }

    double coupling(std::size_t edge_index) const {
        return couplings_.empty() ? 1.0 : couplings_[edge_index];
    }

    double cost(const std::vector<int>& p) const {
        double total = 0.0;
        for (std::size_t e = 0; e < edges_.size(); ++e) {
            const auto [i, j] = edges_[e];
            total += coupling(e) * d_[p[i]][p[j]];
        }
        return total;
    }

    // Change in cost from exchanging the physical slots of logical a and b.
    double delta_swap(const std::vector<int>& p, int a, int b) const {
        const int pa = p[a];
        const int pb = p[b];
        double delta = 0.0;

        for (const auto& [k, j_lm] : neighbours_[a]) {
            if (k != b) {
                delta += j_lm * (d_[pb][p[k]] - d_[pa][p[k]]);
            }
        }
        for (const auto& [k, j_lm] : neighbours_[b]) {
            if (k != a) {
                delta += j_lm * (d_[pa][p[k]] - d_[pb][p[k]]);
            }
        }
        return delta;
    }

    std::pair<std::vector<int>, double> anneal(int iterations, double t0,
                                               double cooling) {
        std::vector<int> p(n_);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), rng_);

        std::uniform_real_distribution<double> unit(0.0, 1.0);

        double current = cost(p);
        std::vector<int> best_p = p;
        double best = current;
        double t = t0;

        for (int it = 0; it < iterations; ++it) {
            const int a = pick_(rng_);
            const int b = pick_(rng_);
            if (a != b) {
                const double delta = delta_swap(p, a, b);
                // Guard the exponential: t decays geometrically and underflows
                // to zero on long runs, which would make exp() produce NaN.
                const bool accept =
                    delta < 0.0 ||
                    (t > 0.0 && unit(rng_) < std::exp(-delta / t));

                if (accept) {
                    std::swap(p[a], p[b]);
                    current += delta;
                    if (current < best - kCostEpsilon) {
                        best = current;
                        best_p = p;
                    }
                }
            }
            t *= cooling;
        }

        // Recompute rather than trust the incremental total: on a weighted
        // instance the running sum drifts over tens of thousands of updates.
        return {best_p, cost(best_p)};
    }

    std::pair<std::vector<int>, double> tabu(int iterations, int tenure,
                                             std::vector<int> p) {
        double current = cost(p);
        std::vector<int> best_p = p;
        double best = current;

        // tabu[a][b] holds the iteration until which the (a, b) exchange is
        // forbidden, for a < b.
        std::vector<std::vector<int>> tabu_until(n_, std::vector<int>(n_, 0));

        for (int it = 0; it < iterations; ++it) {
            int best_a = -1;
            int best_b = -1;
            double best_delta = std::numeric_limits<double>::infinity();

            // Sample n_ candidate exchanges rather than scanning all O(n^2).
            for (int t = 0; t < n_; ++t) {
                int a = pick_(rng_);
                int b = pick_(rng_);
                if (a == b) {
                    continue;
                }
                if (a > b) {
                    std::swap(a, b);
                }

                const double delta = delta_swap(p, a, b);

                // Tabu unless it would beat the incumbent (aspiration).
                if (tabu_until[a][b] > it && current + delta >= best) {
                    continue;
                }
                if (delta < best_delta) {
                    best_delta = delta;
                    best_a = a;
                    best_b = b;
                }
            }

            if (best_a == -1) {
                continue;
            }

            std::swap(p[best_a], p[best_b]);
            current += best_delta;
            tabu_until[best_a][best_b] = it + tenure;

            if (current < best - kCostEpsilon) {
                best = current;
                best_p = p;
            }
        }

        return {best_p, cost(best_p)};
    }

    void perturb(std::vector<int>& p, int k) {
        for (int t = 0; t < k; ++t) {
            const int a = pick_(rng_);
            const int b = pick_(rng_);
            if (a != b) {
                std::swap(p[a], p[b]);
            }
        }
    }

    std::mt19937_64& rng() { return rng_; }
    int size() const { return n_; }

private:
    const std::vector<std::vector<double>>& d_;
    const std::vector<Edge>& edges_;
    const std::vector<double>& couplings_;
    int n_;
    // neighbours_[logical] = list of (neighbour, coupling).
    std::vector<std::vector<std::pair<int, double>>> neighbours_;
    std::mt19937_64 rng_;
    std::uniform_int_distribution<int> pick_;
};

}  // namespace

double layout_cost(const std::vector<std::vector<double>>& dist,
                   const std::vector<Edge>& interaction_edges,
                   const std::vector<double>& couplings,
                   const std::vector<int>& layout) {
    check_inputs(dist, interaction_edges, couplings);
    validate_layout(layout, static_cast<int>(dist.size()));

    double total = 0.0;
    for (std::size_t e = 0; e < interaction_edges.size(); ++e) {
        const auto [i, j] = interaction_edges[e];
        const double j_lm = couplings.empty() ? 1.0 : couplings[e];
        total += j_lm * dist[layout[i]][layout[j]];
    }
    return total;
}

QapResult solve_qap(const std::vector<std::vector<double>>& dist,
                    const std::vector<Edge>& interaction_edges,
                    const std::vector<double>& couplings,
                    const QapParams& params,
                    const QapProgressFn& progress) {
    check_inputs(dist, interaction_edges, couplings);

    const int n = static_cast<int>(dist.size());
    QapResult result;

    if (n == 0) {
        return result;
    }

    // With no interactions every layout has cost 0; skip the search.
    if (interaction_edges.empty()) {
        result.layout.resize(n);
        std::iota(result.layout.begin(), result.layout.end(), 0);
        result.cost = 0.0;
        return result;
    }

    const std::uint64_t seed =
        params.seed != 0 ? params.seed
                         : (static_cast<std::uint64_t>(std::random_device{}()) << 32) ^
                               std::random_device{}();

    QapSolver solver(dist, interaction_edges, couplings, seed);

    std::uniform_real_distribution<double> t0_dist(params.t0_min, params.t0_max);
    std::uniform_real_distribution<double> cooling_dist(params.cooling_min,
                                                        params.cooling_max);

    double global_best = std::numeric_limits<double>::infinity();
    std::vector<int> global_best_p(n);
    std::iota(global_best_p.begin(), global_best_p.end(), 0);
    global_best = solver.cost(global_best_p);

    const auto start = Clock::now();

    for (int r = 0; r < params.num_restarts; ++r) {
        if (params.max_seconds > 0.0 && elapsed_since(start) > params.max_seconds) {
            result.hit_time_limit = true;
            break;
        }

        if (progress) {
            progress(r, params.num_restarts, global_best);
        }

        const double t0 = t0_dist(solver.rng());
        const double cooling = cooling_dist(solver.rng());

        auto [p, sa_cost] = solver.anneal(params.sa_iterations, t0, cooling);
        (void)sa_cost;

        for (int cycle = 0; cycle < params.cycles_per_restart; ++cycle) {
            auto [p2, cost2] =
                solver.tabu(params.tabu_iterations, params.tabu_tenure, p);

            if (cost2 < global_best - kCostEpsilon) {
                global_best = cost2;
                global_best_p = p2;
            }

            p = std::move(p2);
            solver.perturb(p, params.perturb_strength);

            if (params.max_seconds > 0.0 &&
                elapsed_since(start) > params.max_seconds) {
                result.hit_time_limit = true;
                break;
            }
        }

        result.restarts_run = r + 1;

        if (result.hit_time_limit) {
            break;
        }
    }

    result.layout = std::move(global_best_p);
    result.cost = global_best;
    return result;
}

}  // namespace swapnet
