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

"""Qubit layout and SWAP routing for a QAOA cost layer, as a pipeline of stages.

swapnet takes a problem's couplings ``J_ij`` and a backend, and returns one QAOA cost
layer that the device can actually run within a budget of ``k`` parallel SWAP layers. The
layout is a quadratic assignment and the routing is a prize-collecting vertex-disjoint
multi-commodity flow, both from the original C++ implementation by Francisco Barahona.

THE STAGES, AND THE ARTIFACTS BETWEEN THEM
------------------------------------------

Each stage is a separate call, and each hands on a declared, serialisable artifact. There
is deliberately no single entry point that does all of it: the interesting decisions live
between the stages.

    0  Problem.from_covariance / from_qubo / from_matrix  ->  Problem
    1  shifts.rank0 / shifts.rank1                        ->  Shift  (+ residual Problem)
    2  Hardware.from_backend                              ->  Hardware
    3  map_layout                                         ->  MappingResult
    4  route  ->  schedule.from_routing_plan              ->  Schedule
    5  ansatz.build                                       ->  Ansatz

``Schedule`` is the pivot. It is an initial layout plus rounds of "realise these pairs,
then apply this disjoint swap layer" -- a description any routing algorithm can produce,
not just this one. Stage 5 consumes only a ``Schedule``, so a different
circuit-optimisation method plugs into the rest of the pipeline by emitting one. See
:mod:`swapnet.schedule` for the producer contract, and :func:`plan_schedule` for swapnet's
implementation of it.

The default wiring of stages 3-4, with the README's worked example in full::

    import swapnet
    from qiskit_ibm_runtime.fake_provider import FakeBoston

    hardware = swapnet.Hardware.from_backend(FakeBoston())
    problem = swapnet.Problem.from_matrix(J)            # or .from_covariance / .from_qubo
    schedule = swapnet.plan_schedule(problem, hardware, k=4, seed=12345)
    layer = swapnet.ansatz.build(schedule, problem, hardware, k=4)
    one_body, two_body = swapnet.ansatz.angles_from_problem(layer, problem, gamma=0.1)
    circuit = layer.bind(one_body, two_body)

THE TWO WEIGHT SETS
-------------------

Both stages 3 and 4 accept two independent sets of weights, which is what lets a plan be
steered by real device data rather than by hop counts:

* ``edge_weights`` are per-coupling traversal costs ``W_ij`` on the hardware graph. Lower
  is better, so pass an error metric -- :attr:`Hardware.edge_weights` builds them. The
  mapper then measures distance in accumulated error and the router prices each path by
  the error it incurs, both preferring the high-fidelity couplings.
* ``couplings`` are per-interaction strengths on the interaction graph, and decide which
  interactions a truncated budget spends its layers on.

**The two stages consume ``couplings`` at different powers** -- the mapper linearly, the
router quadratically -- so use :meth:`Problem.mapper_couplings` and
:meth:`Problem.router_couplings` rather than passing ``J`` yourself.
:mod:`swapnet.problem` explains why, and why getting it wrong is silent.

Weights are normalised to mean 1 internally, because the search schedule and the routing
reward are calibrated against unit-cost hops. That preserves every ratio, so it changes
the cost units but not which layout or route wins.

THE ANGLE CONVENTION
--------------------

    U(theta) = exp(-i G(theta) / 2)        G(theta) = sum_alpha theta_alpha Z^alpha

Qiskit's rotations carry the same sign and half-angle, so the gate angle *is* ``theta``.
:mod:`swapnet.conventions` is the authority; ``tests/test_ansatz.py`` pins it against
``expm``.
"""

from __future__ import annotations

from . import _core, ansatz, conventions, plan, schedule, shifts
from .ansatz import Ansatz
from .api import (
    MapParams,
    RouteOptions,
    RoutingOptions,
    map_and_route,
    map_layout,
    route,
)
from .conventions import gamma_to_theta, rz_angle, rzz_angle
from .graphs import GraphView, as_graph_view, to_networkx, to_rustworkx
from .hardware import Hardware
from .milp import (
    CplexBackend,
    HighsBackend,
    MilpBackend,
    ScipyBackend,
    SolverOptions,
    available_backends,
    get_backend,
)
from .plan import plan_schedule, plan_schedule_sweep
from .problem import DEFAULT_TILT, Problem
from .results import (
    Ancilla,
    IterationStats,
    MappingResult,
    RoutingPlan,
    Stage,
    Swap,
)
from .schedule import Round, Schedule, ScheduleError
from .shifts import Shift
from .verify import PlanError, verify_plan
from .weights import (
    from_error_rates,
    from_fidelities,
    mean_normalized,
    normalize,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # stage 0: the problem
    "Problem",
    "DEFAULT_TILT",
    # stage 1: the analytic shift
    "shifts",
    "Shift",
    # stage 2: the device
    "Hardware",
    # stages 3-4: layout and routing
    "map_layout",
    "route",
    "map_and_route",
    "plan",
    "plan_schedule",
    "plan_schedule_sweep",
    "MapParams",
    "RouteOptions",
    "RoutingOptions",
    # stage 4's artifact: the algorithm-neutral routing contract
    "schedule",
    "Schedule",
    "Round",
    "ScheduleError",
    # stage 5: the circuit
    "ansatz",
    "Ansatz",
    "conventions",
    "rz_angle",
    "rzz_angle",
    "gamma_to_theta",
    # swapnet's own richer routing result
    "MappingResult",
    "RoutingPlan",
    "Stage",
    "Swap",
    "Ancilla",
    "IterationStats",
    # MILP solvers
    "SolverOptions",
    "MilpBackend",
    "HighsBackend",
    "ScipyBackend",
    "CplexBackend",
    "available_backends",
    "get_backend",
    # graphs
    "GraphView",
    "as_graph_view",
    "to_networkx",
    "to_rustworkx",
    # weights
    "normalize",
    "from_fidelities",
    "from_error_rates",
    "mean_normalized",
    # verification
    "verify_plan",
    "PlanError",
    # escape hatch to the native layer
    "_core",
]
