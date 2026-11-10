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

// pybind11 bindings for the swapnet core.
//
// The extension deliberately knows nothing about MILP solvers, networkx or
// rustworkx: it exposes the algorithm as plain vertex counts, edge lists and
// sparse matrices, and takes the MILP solve as a Python callable. The
// user-facing conveniences live in the pure-Python layer.

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "graph.hpp"
#include "milp.hpp"
#include "node_swap.hpp"
#include "pipeline.hpp"
#include "qap.hpp"

namespace py = pybind11;
using namespace swapnet;

namespace {

template <typename T>
py::array_t<T> to_array(const std::vector<T>& v) {
    return py::array_t<T>(static_cast<py::ssize_t>(v.size()), v.data());
}

py::bytes to_bytes(const std::vector<char>& v) {
    return py::bytes(std::string(v.begin(), v.end()));
}

// Wraps a Python callable as a MilpSolveFn. The problem is handed over by
// reference rather than copied: models routinely have millions of nonzeros and
// the callable only needs read access for the duration of the call.
// Lets the tuning structs be built as `QapParams(num_restarts=8)` rather than
// default-constructed and then assigned field by field. Each field is already
// exposed via def_readwrite, so the setters do the type conversion and an
// unknown keyword is an error rather than a silently ignored typo.
template <typename T>
T params_from_kwargs(const py::kwargs& kwargs) {
    T params;
    {
        py::object view =
            py::cast(&params, py::return_value_policy::reference);
        for (auto item : kwargs) {
            const auto key = item.first.cast<std::string>();
            if (!py::hasattr(view, key.c_str())) {
                throw py::type_error("unexpected keyword argument '" + key +
                                     "'");
            }
            py::setattr(view, key.c_str(), item.second);
        }
    }
    return params;
}

MilpSolveFn wrap_solver(const py::object& fn) {
    return [fn](const MilpProblem& problem) -> MilpSolution {
        py::gil_scoped_acquire gil;
        py::object view = py::cast(const_cast<MilpProblem*>(&problem),
                                   py::return_value_policy::reference);
        py::object result = fn(view);
        return result.cast<MilpSolution>();
    };
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() =
        "Native core of the swapnet qubit mapping and routing pipeline. "
        "Use the swapnet package instead of these bindings directly.";

    m.attr("UNREACHABLE") = py::int_(kUnreachable);

    // ---------------------------------------------------------------- graphs

    py::class_<Graph>(m, "Graph",
                      "A simple undirected graph as a vertex count plus an edge "
                      "list of (u, v) pairs, optionally with one traversal cost "
                      "per edge.")
        .def(py::init<>())
        .def(py::init<int, std::vector<Edge>>(), py::arg("num_vertices"),
             py::arg("edges"))
        .def(py::init<int, std::vector<Edge>, std::vector<double>>(),
             py::arg("num_vertices"), py::arg("edges"), py::arg("weights"))
        .def_readwrite("num_vertices", &Graph::num_vertices)
        .def_readwrite("edges", &Graph::edges)
        .def_readwrite("weights", &Graph::weights,
                       "Per-edge traversal cost W_ij, parallel to `edges`. Empty "
                       "means every edge costs 1. Lower is better, so pass an "
                       "error metric rather than a fidelity.")
        .def_property_readonly("num_edges", &Graph::num_edges)
        .def_property_readonly("is_weighted", &Graph::is_weighted)
        .def("weight", &Graph::weight, py::arg("index"),
             "Cost of one edge, which is 1 on an unweighted graph.")
        .def("validate", &Graph::validate, py::arg("name") = "graph",
             "Raise ValueError on out-of-range endpoints, self-loops, duplicate "
             "undirected edges, or a bad weight vector.")
        .def("adjacency", &Graph::adjacency)
        .def("weighted_adjacency", &Graph::weighted_adjacency,
             "Adjacency lists of (neighbour, edge cost) pairs.")
        .def("is_connected", &Graph::is_connected)
        .def("__repr__", [](const Graph& g) {
            return "<Graph vertices=" + std::to_string(g.num_vertices) +
                   " edges=" + std::to_string(g.num_edges()) +
                   (g.is_weighted() ? " weighted" : "") + ">";
        });

    m.def("bfs_all_pairs", &bfs_all_pairs, py::arg("graph"),
          py::call_guard<py::gil_scoped_release>(),
          "Unweighted all-pairs shortest path lengths in hops; UNREACHABLE for "
          "pairs with no connecting path. Ignores edge weights.");

    m.def("dijkstra_all_pairs", &dijkstra_all_pairs, py::arg("graph"),
          py::call_guard<py::gil_scoped_release>(),
          "Weighted all-pairs shortest path costs; math.inf for pairs with no "
          "connecting path. Equals bfs_all_pairs on an unweighted graph.");

    m.def("validate_interaction_edges", &validate_interaction_edges,
          py::arg("edges"), py::arg("num_hosts"));

    m.def("validate_interaction_weights", &validate_interaction_weights,
          py::arg("weights"), py::arg("num_edges"),
          py::arg("name") = "interaction graph");

    m.def("validate_layout", &validate_layout, py::arg("layout"),
          py::arg("num_vertices"));

    // ------------------------------------------------------------------- QAP

    py::class_<QapParams>(m, "QapParams")
        .def(py::init<>())
        .def(py::init(&params_from_kwargs<QapParams>),
             "Construct with any subset of the fields as keyword arguments.")
        .def_readwrite("num_restarts", &QapParams::num_restarts)
        .def_readwrite("sa_iterations", &QapParams::sa_iterations)
        .def_readwrite("tabu_iterations", &QapParams::tabu_iterations)
        .def_readwrite("tabu_tenure", &QapParams::tabu_tenure)
        .def_readwrite("cycles_per_restart", &QapParams::cycles_per_restart)
        .def_readwrite("perturb_strength", &QapParams::perturb_strength)
        .def_readwrite("t0_min", &QapParams::t0_min)
        .def_readwrite("t0_max", &QapParams::t0_max)
        .def_readwrite("cooling_min", &QapParams::cooling_min)
        .def_readwrite("cooling_max", &QapParams::cooling_max)
        .def_readwrite("max_seconds", &QapParams::max_seconds)
        .def_readwrite("seed", &QapParams::seed);

    py::class_<QapResult>(m, "QapResult")
        .def_readonly("layout", &QapResult::layout)
        .def_readonly("cost", &QapResult::cost)
        .def_readonly("restarts_run", &QapResult::restarts_run)
        .def_readonly("hit_time_limit", &QapResult::hit_time_limit);

    m.def(
        "solve_qap",
        [](const std::vector<std::vector<double>>& dist,
           const std::vector<Edge>& interaction_edges, const QapParams& params,
           const std::vector<double>& couplings, const py::object& progress) {
            QapProgressFn callback;
            if (!progress.is_none()) {
                callback = [progress](int restart, int total, double best) {
                    py::gil_scoped_acquire gil;
                    progress(restart, total, best);
                };
            }
            py::gil_scoped_release release;
            return solve_qap(dist, interaction_edges, couplings, params,
                             callback);
        },
        py::arg("dist"), py::arg("interaction_edges"), py::arg("params"),
        py::arg("couplings") = std::vector<double>{},
        py::arg("progress") = py::none(),
        "Search for a layout minimising sum of J_lm * distance across the "
        "interaction edges. Pass an empty `couplings` for all J = 1.");

    m.def(
        "layout_cost",
        [](const std::vector<std::vector<double>>& dist,
           const std::vector<Edge>& interaction_edges,
           const std::vector<int>& layout,
           const std::vector<double>& couplings) {
            return layout_cost(dist, interaction_edges, couplings, layout);
        },
        py::arg("dist"), py::arg("interaction_edges"), py::arg("layout"),
        py::arg("couplings") = std::vector<double>{},
        "Weighted hardware distance spanned by the interaction edges under an "
        "explicit layout.");

    // ------------------------------------------------------------------ MILP

    py::class_<MilpProblem>(m, "MilpProblem",
                            "A minimisation MILP: min objective . x subject to "
                            "row_lower <= A x <= row_upper and bounds, with A "
                            "in compressed sparse row form.")
        .def(py::init<>())
        .def_readonly("num_vars", &MilpProblem::num_vars)
        .def_property_readonly("num_rows", &MilpProblem::num_rows)
        .def_property_readonly("num_nonzeros", &MilpProblem::num_nonzeros)
        .def_property_readonly(
            "objective",
            [](const MilpProblem& p) { return to_array(p.objective); })
        .def_property_readonly(
            "lower_bound",
            [](const MilpProblem& p) { return to_array(p.lower_bound); })
        .def_property_readonly(
            "upper_bound",
            [](const MilpProblem& p) { return to_array(p.upper_bound); })
        .def_property_readonly(
            "var_type", [](const MilpProblem& p) { return to_bytes(p.var_type); },
            "One byte per column: b'B' binary, b'I' integer, b'C' continuous.")
        .def_property_readonly(
            "row_start",
            [](const MilpProblem& p) { return to_array(p.row_start); },
            "CSR row pointer, length num_rows + 1.")
        .def_property_readonly(
            "col_index",
            [](const MilpProblem& p) { return to_array(p.col_index); })
        .def_property_readonly(
            "coefficient",
            [](const MilpProblem& p) { return to_array(p.coefficient); })
        .def_property_readonly(
            "sense", [](const MilpProblem& p) { return to_bytes(p.sense); },
            "One byte per row: b'E' equality, b'L' <=, b'G' >=.")
        .def_property_readonly(
            "rhs", [](const MilpProblem& p) { return to_array(p.rhs); })
        .def_property_readonly(
            "row_lower",
            [](const MilpProblem& p) { return to_array(p.row_lower()); })
        .def_property_readonly(
            "row_upper",
            [](const MilpProblem& p) { return to_array(p.row_upper()); })
        .def_property_readonly(
            "integrality",
            [](const MilpProblem& p) { return to_array(p.integrality()); })
        .def("__repr__", [](const MilpProblem& p) {
            return "<MilpProblem vars=" + std::to_string(p.num_vars) +
                   " rows=" + std::to_string(p.num_rows()) +
                   " nonzeros=" + std::to_string(p.num_nonzeros()) + ">";
        });

    py::class_<MilpSolution>(m, "MilpSolution")
        .def(py::init([](bool has_solution, bool is_optimal, double objective,
                         std::vector<double> x, std::string status) {
                 MilpSolution s;
                 s.has_solution = has_solution;
                 s.is_optimal = is_optimal;
                 s.objective = objective;
                 s.x = std::move(x);
                 s.status = std::move(status);
                 return s;
             }),
             py::arg("has_solution") = false, py::arg("is_optimal") = false,
             py::arg("objective") = 0.0,
             py::arg("x") = std::vector<double>{},
             py::arg("status") = std::string("not solved"))
        .def_readwrite("has_solution", &MilpSolution::has_solution)
        .def_readwrite("is_optimal", &MilpSolution::is_optimal)
        .def_readwrite("objective", &MilpSolution::objective)
        .def_readwrite("x", &MilpSolution::x)
        .def_readwrite("status", &MilpSolution::status)
        .def("__repr__", [](const MilpSolution& s) {
            return "<MilpSolution status=" + s.status +
                   " objective=" + std::to_string(s.objective) + ">";
        });

    py::class_<RoutingParams>(m, "RoutingParams")
        .def(py::init<>())
        .def(py::init(&params_from_kwargs<RoutingParams>),
             "Construct with any subset of the fields as keyword arguments.")
        .def_readwrite("route_reward", &RoutingParams::route_reward)
        .def_readwrite("tie_break_noise", &RoutingParams::tie_break_noise)
        .def_readwrite("max_path_length", &RoutingParams::max_path_length)
        .def_readwrite("max_detour", &RoutingParams::max_detour)
        .def_readwrite("seed", &RoutingParams::seed);

    py::class_<Arc>(m, "Arc")
        .def_readonly("tail", &Arc::tail)
        .def_readonly("head", &Arc::head)
        .def_readonly("cost", &Arc::cost,
                      "Traversal cost W_ij of the underlying hardware edge.");

    py::class_<SolutionArc>(m, "SolutionArc")
        .def(py::init([](int u, int v, int k, double value) {
                 return SolutionArc{u, v, k, value};
             }),
             py::arg("u"), py::arg("v"), py::arg("k"), py::arg("value") = 1.0)
        .def_readonly("u", &SolutionArc::u)
        .def_readonly("v", &SolutionArc::v)
        .def_readonly("k", &SolutionArc::k)
        .def_readonly("value", &SolutionArc::value)
        .def("__repr__", [](const SolutionArc& a) {
            return "<SolutionArc " + std::to_string(a.u) + "->" +
                   std::to_string(a.v) + " k=" + std::to_string(a.k) + ">";
        });

    py::class_<RoutingModel>(m, "RoutingModel")
        .def_readonly("problem", &RoutingModel::problem)
        .def_readonly("arcs", &RoutingModel::arcs)
        .def_readonly("commodities", &RoutingModel::commodities)
        .def_property_readonly("num_arcs", &RoutingModel::num_arcs)
        .def_property_readonly("num_commodities", &RoutingModel::num_commodities)
        .def_property_readonly(
            "active",
            [](const RoutingModel& mod) { return to_bytes(mod.active); },
            "One byte per commodity: zero when the commodity was fixed out "
            "because routing it cannot improve the objective.")
        .def_readonly("rewards", &RoutingModel::rewards,
                      "Reward placed on each y[k], i.e. route_reward scaled by "
                      "the commodity's relative J^2.")
        .def_readonly("pruning_applied", &RoutingModel::pruning_applied,
                      "False when a zero-cost hardware edge made the exact "
                      "path-length bound unusable, leaving the model unpruned.")
        .def("x_index", &RoutingModel::x_index, py::arg("arc"), py::arg("k"))
        .def("y_index", &RoutingModel::y_index, py::arg("k"));

    m.def(
        "build_routing_model",
        [](const Graph& hardware, const std::vector<Edge>& commodities,
           const std::vector<std::vector<int>>& distances,
           const RoutingParams& params, const std::vector<double>& couplings) {
            return build_routing_model(hardware, commodities, distances,
                                       couplings, params);
        },
        py::arg("hardware"), py::arg("commodities"), py::arg("distances"),
        py::arg("params"), py::arg("couplings") = std::vector<double>{},
        "Build the prize-collecting vertex-disjoint multi-commodity flow "
        "model for one routing round. Arc costs come from the hardware edge "
        "weights; commodity rewards scale with J^2.");

    m.def("extract_solution_arcs", &extract_solution_arcs, py::arg("model"),
          py::arg("solution"), py::arg("tolerance") = 1e-6);

    m.def("count_routed", &count_routed, py::arg("model"), py::arg("solution"),
          py::arg("tolerance") = 1e-6);

    // -------------------------------------------------------------- swapping

    py::class_<SwapOp>(m, "SwapOp")
        .def_readonly("logical_a", &SwapOp::logical_a)
        .def_readonly("logical_b", &SwapOp::logical_b)
        .def_readonly("physical_u", &SwapOp::physical_u)
        .def_readonly("physical_v", &SwapOp::physical_v)
        .def("__repr__", [](const SwapOp& s) {
            return "<SwapOp logical=(" + std::to_string(s.logical_a) + "," +
                   std::to_string(s.logical_b) + ") physical=(" +
                   std::to_string(s.physical_u) + "," +
                   std::to_string(s.physical_v) + ")>";
        });

    py::class_<SwapRoundResult>(m, "SwapRoundResult")
        .def_readonly("layout", &SwapRoundResult::layout)
        .def_readonly("layers", &SwapRoundResult::layers)
        .def_readonly("satisfied_edges", &SwapRoundResult::satisfied_edges)
        .def_readonly("remaining_edges", &SwapRoundResult::remaining_edges)
        .def_readonly("satisfied_indices", &SwapRoundResult::satisfied_indices)
        .def_readonly("total_swaps", &SwapRoundResult::total_swaps);

    m.def("apply_node_swaps", &apply_node_swaps, py::arg("layout"),
          py::arg("commodity_edges"), py::arg("solution_arcs"),
          py::arg("max_swaps_per_commodity") = 2,
          "Convert routed disjoint paths into parallel SWAP layers and the "
          "resulting layout.");

    // -------------------------------------------------------------- pipeline

    py::class_<RouteParams>(m, "RouteParams")
        .def(py::init<>())
        .def(py::init(&params_from_kwargs<RouteParams>),
             "Construct with any subset of the fields as keyword arguments; "
             "`routing=RoutingParams(...)` sets the MILP sub-parameters.")
        .def_readwrite("max_iterations", &RouteParams::max_iterations)
        .def_readwrite("max_depth", &RouteParams::max_depth,
                       "Hard cap on emitted SWAP layers. Negative means "
                       "unlimited; 0 forbids every SWAP.")
        .def_readwrite("max_swaps_per_commodity",
                       &RouteParams::max_swaps_per_commodity)
        .def_readwrite("max_commodities_per_round",
                       &RouteParams::max_commodities_per_round)
        .def_readwrite("max_stagnant_iterations",
                       &RouteParams::max_stagnant_iterations)
        .def_readwrite("shuffle_pending", &RouteParams::shuffle_pending)
        .def_readwrite("routing", &RouteParams::routing)
        .def_readwrite("seed", &RouteParams::seed);

    py::class_<Stage>(m, "Stage")
        .def_readonly("gates", &Stage::gates)
        .def_readonly("swaps", &Stage::swaps);

    py::class_<IterationStats>(m, "IterationStats")
        .def_readonly("iteration", &IterationStats::iteration)
        .def_readonly("executable_before", &IterationStats::executable_before)
        .def_readonly("commodities", &IterationStats::commodities)
        .def_readonly("deferred", &IterationStats::deferred)
        .def_readonly("routed", &IterationStats::routed)
        .def_readonly("objective", &IterationStats::objective)
        .def_readonly("solver_optimal", &IterationStats::solver_optimal)
        .def_readonly("solver_status", &IterationStats::solver_status)
        .def_readonly("swaps", &IterationStats::swaps)
        .def_readonly("layers", &IterationStats::layers)
        .def_readonly("satisfied_after_swaps",
                      &IterationStats::satisfied_after_swaps)
        .def_readonly("remaining", &IterationStats::remaining)
        .def_readonly("realized_weight", &IterationStats::realized_weight)
        .def_readonly("swap_weight", &IterationStats::swap_weight);

    py::class_<RouteResult>(m, "RouteResult")
        .def_readonly("initial_layout", &RouteResult::initial_layout)
        .def_readonly("final_layout", &RouteResult::final_layout)
        .def_readonly("stages", &RouteResult::stages)
        .def_readonly("unrouted_edges", &RouteResult::unrouted_edges)
        .def_readonly("iterations", &RouteResult::iterations)
        .def_readonly("total_swaps", &RouteResult::total_swaps)
        .def_readonly("num_layers", &RouteResult::num_layers)
        .def_readonly("converged", &RouteResult::converged)
        .def_readonly("hit_depth_limit", &RouteResult::hit_depth_limit)
        .def_readonly("realized_weight", &RouteResult::realized_weight,
                      "Sum of J^2 over the interactions the plan executes.")
        .def_readonly("unrealized_weight", &RouteResult::unrealized_weight,
                      "Sum of J^2 over the interactions it does not.")
        .def_readonly("swap_weight", &RouteResult::swap_weight,
                      "Total hardware weight accumulated by the emitted SWAPs.");

    m.def(
        "route",
        [](const Graph& hardware, const std::vector<Edge>& interaction_edges,
           const std::vector<int>& layout, const RouteParams& params,
           const py::object& solve, const std::vector<double>& couplings,
           const py::object& progress) {
            if (solve.is_none()) {
                throw std::invalid_argument(
                    "no MILP solver callable was provided");
            }
            RouteProgressFn callback;
            if (!progress.is_none()) {
                callback = [progress](const IterationStats& stats) {
                    py::gil_scoped_acquire gil;
                    progress(stats);
                };
            }
            return route(hardware, interaction_edges, couplings, layout, params,
                         wrap_solver(solve), callback);
        },
        py::arg("hardware"), py::arg("interaction_edges"), py::arg("layout"),
        py::arg("params"), py::arg("solve"),
        py::arg("couplings") = std::vector<double>{},
        py::arg("progress") = py::none(),
        "Run the routing loop, calling `solve(problem)` once per iteration.");
}
