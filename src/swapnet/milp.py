# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MILP solver backends.

The native core emits the routing model as a sparse matrix and never links a
solver, so the solver is chosen here at runtime. Upstream hard-wired IBM CPLEX
through its Concert C++ API, which made a commercial licence a build requirement
for the whole project; HiGHS is now the default and CPLEX is one option among
several.

A backend is any callable taking a ``_core.MilpProblem`` and returning a
``_core.MilpSolution``. Write your own if you have a solver we do not cover.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from . import _core

__all__ = [
    "MilpBackend",
    "SolverOptions",
    "HighsBackend",
    "ScipyBackend",
    "CplexBackend",
    "available_backends",
    "get_backend",
]


class MilpBackend(Protocol):
    """Callable that solves one MILP."""

    def __call__(self, problem: _core.MilpProblem) -> _core.MilpSolution: ...


@dataclass(frozen=True)
class SolverOptions:
    """Solver settings shared by all backends.

    ``time_limit`` is per MILP, not for the whole pipeline: the routing loop
    solves one model per iteration. A time limit that cuts a solve short is not
    an error -- the loop simply routes fewer interactions that round and retries
    the rest.
    """

    time_limit: float | None = None
    mip_rel_gap: float | None = None
    threads: int | None = None
    verbose: bool = False


class HighsBackend:
    """HiGHS via its own Python bindings. Requires ``highspy``."""

    name = "highs"

    def __init__(self, options: SolverOptions | None = None) -> None:
        import highspy  # noqa: F401  (fail early with a clear ImportError)

        self.options = options or SolverOptions()

    def __call__(self, problem: _core.MilpProblem) -> _core.MilpSolution:
        import highspy

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", bool(self.options.verbose))
        if self.options.time_limit is not None:
            highs.setOptionValue("time_limit", float(self.options.time_limit))
        if self.options.mip_rel_gap is not None:
            highs.setOptionValue("mip_rel_gap", float(self.options.mip_rel_gap))
        if self.options.threads is not None:
            highs.setOptionValue("threads", int(self.options.threads))

        num_vars = problem.num_vars
        num_rows = problem.num_rows

        lp = highspy.HighsLp()
        lp.num_col_ = num_vars
        lp.num_row_ = num_rows
        lp.sense_ = highspy.ObjSense.kMinimize
        lp.offset_ = 0.0
        lp.col_cost_ = np.asarray(problem.objective, dtype=np.float64)
        lp.col_lower_ = np.asarray(problem.lower_bound, dtype=np.float64)
        lp.col_upper_ = np.asarray(problem.upper_bound, dtype=np.float64)
        lp.row_lower_ = _replace_infinities(problem.row_lower, highspy.kHighsInf)
        lp.row_upper_ = _replace_infinities(problem.row_upper, highspy.kHighsInf)

        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.num_col_ = num_vars
        lp.a_matrix_.num_row_ = num_rows
        lp.a_matrix_.start_ = np.asarray(problem.row_start, dtype=np.int32)
        lp.a_matrix_.index_ = np.asarray(problem.col_index, dtype=np.int32)
        lp.a_matrix_.value_ = np.asarray(problem.coefficient, dtype=np.float64)

        integer = highspy.HighsVarType.kInteger
        continuous = highspy.HighsVarType.kContinuous
        lp.integrality_ = [
            integer if flag else continuous for flag in problem.integrality
        ]

        highs.passModel(lp)
        highs.run()

        status = highs.getModelStatus()
        feasible = (
            highs.getInfo().primal_solution_status
            == highspy.SolutionStatus.kSolutionStatusFeasible
        )
        is_optimal = status == highspy.HighsModelStatus.kOptimal

        if not feasible:
            return _core.MilpSolution(has_solution=False, status=_status_name(status))

        return _core.MilpSolution(
            has_solution=True,
            is_optimal=is_optimal,
            objective=float(highs.getObjectiveValue()),
            x=list(highs.getSolution().col_value),
            status=_status_name(status),
        )


class ScipyBackend:
    """HiGHS via ``scipy.optimize.milp``. Requires ``scipy``.

    Equivalent to :class:`HighsBackend` in results but usually a little slower to
    set up, since scipy rebuilds the model. Useful when scipy is already a
    dependency and ``highspy`` is not.
    """

    name = "scipy"

    def __init__(self, options: SolverOptions | None = None) -> None:
        import scipy.optimize  # noqa: F401

        self.options = options or SolverOptions()

    def __call__(self, problem: _core.MilpProblem) -> _core.MilpSolution:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_array

        matrix = csr_array(
            (
                np.asarray(problem.coefficient, dtype=np.float64),
                np.asarray(problem.col_index, dtype=np.int32),
                np.asarray(problem.row_start, dtype=np.int32),
            ),
            shape=(problem.num_rows, problem.num_vars),
        )

        options: dict[str, Any] = {"disp": bool(self.options.verbose)}
        if self.options.time_limit is not None:
            options["time_limit"] = float(self.options.time_limit)
        if self.options.mip_rel_gap is not None:
            options["mip_rel_gap"] = float(self.options.mip_rel_gap)

        result = milp(
            c=np.asarray(problem.objective, dtype=np.float64),
            constraints=LinearConstraint(
                matrix,
                np.asarray(problem.row_lower, dtype=np.float64),
                np.asarray(problem.row_upper, dtype=np.float64),
            ),
            bounds=Bounds(
                np.asarray(problem.lower_bound, dtype=np.float64),
                np.asarray(problem.upper_bound, dtype=np.float64),
            ),
            integrality=np.asarray(problem.integrality, dtype=np.int32),
            options=options,
        )

        if result.x is None:
            return _core.MilpSolution(has_solution=False, status=str(result.message))

        return _core.MilpSolution(
            has_solution=True,
            # status 0 is "optimal"; 1 is "time limit reached" with an incumbent.
            is_optimal=result.status == 0,
            objective=float(result.fun),
            x=[float(value) for value in result.x],
            status=str(result.message),
        )


class CplexBackend:
    """IBM CPLEX via its Python API. Requires the ``cplex`` module.

    Present for parity with the original implementation and for benchmarking.
    Install it from a CPLEX Studio tree, e.g.
    ``pip install /Applications/CPLEX_Studio2211/python``.
    """

    name = "cplex"

    def __init__(self, options: SolverOptions | None = None) -> None:
        import cplex  # noqa: F401

        self.options = options or SolverOptions()

    def __call__(self, problem: _core.MilpProblem) -> _core.MilpSolution:
        import cplex

        model = cplex.Cplex()
        if not self.options.verbose:
            model.set_log_stream(None)
            model.set_results_stream(None)
            model.set_warning_stream(None)
            model.set_error_stream(None)
        if self.options.time_limit is not None:
            model.parameters.timelimit.set(float(self.options.time_limit))
        if self.options.mip_rel_gap is not None:
            model.parameters.mip.tolerances.mipgap.set(float(self.options.mip_rel_gap))
        if self.options.threads is not None:
            model.parameters.threads.set(int(self.options.threads))

        model.objective.set_sense(model.objective.sense.minimize)

        var_type = problem.var_type.decode("ascii")
        model.variables.add(
            obj=[float(value) for value in problem.objective],
            lb=[float(value) for value in problem.lower_bound],
            ub=[float(value) for value in problem.upper_bound],
            types=var_type,
        )

        row_start = np.asarray(problem.row_start)
        col_index = np.asarray(problem.col_index)
        coefficient = np.asarray(problem.coefficient)

        rows = [
            cplex.SparsePair(
                ind=col_index[row_start[r] : row_start[r + 1]].tolist(),
                val=coefficient[row_start[r] : row_start[r + 1]].tolist(),
            )
            for r in range(problem.num_rows)
        ]
        model.linear_constraints.add(
            lin_expr=rows,
            senses=problem.sense.decode("ascii"),
            rhs=[float(value) for value in problem.rhs],
        )

        model.solve()

        solution = model.solution
        status = solution.get_status_string()
        if not solution.is_primal_feasible():
            return _core.MilpSolution(has_solution=False, status=status)

        return _core.MilpSolution(
            has_solution=True,
            is_optimal=solution.get_status() in (101, 102),
            objective=float(solution.get_objective_value()),
            x=[float(value) for value in solution.get_values()],
            status=status,
        )


_BACKENDS: dict[str, type] = {
    "highs": HighsBackend,
    "scipy": ScipyBackend,
    "cplex": CplexBackend,
}

# Order in which a backend is picked when the caller does not name one.
_PREFERENCE = ("highs", "scipy", "cplex")


def available_backends() -> list[str]:
    """Names of the backends whose solver package is importable."""
    found = []
    for name in _PREFERENCE:
        try:
            _BACKENDS[name](SolverOptions())
        except ImportError:
            continue
        found.append(name)
    return found


def get_backend(
    solver: str | MilpBackend | Callable[..., Any] | None = None,
    options: SolverOptions | None = None,
) -> MilpBackend:
    """Resolve ``solver`` into a callable backend.

    ``solver`` may be a backend name, an already-built callable, or ``None`` to
    pick the first importable backend in HiGHS, scipy, CPLEX order.
    """
    if solver is not None and not isinstance(solver, str):
        return solver

    if solver is None:
        for name in _PREFERENCE:
            try:
                return _BACKENDS[name](options)
            except ImportError:
                continue
        raise ImportError(
            "no MILP solver found. Install one of: highspy (pip install "
            "highspy), scipy, or the cplex Python API."
        )

    key = solver.lower()
    if key not in _BACKENDS:
        raise ValueError(
            f"unknown solver {solver!r}; choose from "
            f"{sorted(_BACKENDS)} or pass a callable"
        )
    try:
        return _BACKENDS[key](options)
    except ImportError as error:
        raise ImportError(f"solver backend {key!r} is unavailable: {error}") from error


def _replace_infinities(values: Any, infinity: float) -> np.ndarray:
    """Swap numpy infinities for the solver's own infinity marker."""
    array = np.asarray(values, dtype=np.float64).copy()
    array[np.isposinf(array)] = infinity
    array[np.isneginf(array)] = -infinity
    return array


def _status_name(status: Any) -> str:
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    return str(status)
