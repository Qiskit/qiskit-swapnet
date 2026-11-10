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

"""Plain-Python result types.

Everything the public API returns is a frozen dataclass of built-in types, with a
``to_dict`` that produces JSON-serialisable output. Node labels are the caller's
own, not the internal indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Ancilla",
    "Swap",
    "Stage",
    "IterationStats",
    "MappingResult",
    "RoutingPlan",
]


@dataclass(frozen=True, order=True)
class Ancilla:
    """A hardware qubit that carries no interaction-graph vertex.

    The layout is a full permutation of the hardware qubits, so the qubits not
    used by the interaction graph still occupy slots and still get moved by
    SWAPs. Those slots are labelled with ``Ancilla`` so they cannot be confused
    with real interaction-graph labels.
    """

    slot: int

    def __repr__(self) -> str:
        return f"Ancilla({self.slot})"


@dataclass(frozen=True)
class Swap:
    """One SWAP gate.

    ``physical`` is the pair of adjacent hardware qubits the gate acts on, which
    is what a circuit needs. ``logical`` is the pair of interaction-graph
    vertices (or :class:`Ancilla` slots) that exchange places.
    """

    physical: tuple[Any, Any]
    logical: tuple[Any, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical": list(self.physical),
            "logical": [_label_to_json(x) for x in self.logical],
        }


@dataclass(frozen=True)
class Stage:
    """One step of the execution plan.

    ``gates`` are the interaction edges that become executable at this point,
    given all swaps emitted by earlier stages. ``swaps`` is the layer of SWAP
    gates applied afterwards; they act on disjoint hardware edges and so can run
    in parallel. The final stage usually has gates and no swaps.
    """

    gates: tuple[tuple[Any, Any], ...]
    swaps: tuple[Swap, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates": [list(gate) for gate in self.gates],
            "swaps": [swap.to_dict() for swap in self.swaps],
        }


@dataclass(frozen=True)
class IterationStats:
    """Diagnostics for one iteration of the routing loop."""

    iteration: int
    executable_before: int
    commodities: int
    deferred: int
    routed: int
    objective: float
    solver_optimal: bool
    solver_status: str
    swaps: int
    layers: int
    satisfied_after_swaps: int
    remaining: int
    # Sum of J^2 over the interactions this iteration made executable, and the
    # hardware weight W_ij its SWAPs cost. Both are plain counts when no weights
    # were supplied.
    realized_weight: float = 0.0
    swap_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "executable_before": self.executable_before,
            "commodities": self.commodities,
            "deferred": self.deferred,
            "routed": self.routed,
            "objective": self.objective,
            "solver_optimal": self.solver_optimal,
            "solver_status": self.solver_status,
            "swaps": self.swaps,
            "layers": self.layers,
            "satisfied_after_swaps": self.satisfied_after_swaps,
            "remaining": self.remaining,
            "realized_weight": self.realized_weight,
            "swap_weight": self.swap_weight,
        }


@dataclass(frozen=True)
class MappingResult:
    """Output of the quadratic-assignment layout search."""

    # Interaction-graph vertex -> hardware qubit.
    layout: dict[Any, Any]
    # Every logical slot, including ancillas, in slot order.
    full_layout: dict[Any, Any]
    # Total hardware distance spanned by the interaction edges, each weighted by
    # its coupling J_lm. An integer hop count on an unweighted problem.
    cost: float
    restarts_run: int
    hit_time_limit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": {_label_to_json(k): v for k, v in self.layout.items()},
            "full_layout": {_label_to_json(k): v for k, v in self.full_layout.items()},
            "cost": self.cost,
            "restarts_run": self.restarts_run,
            "hit_time_limit": self.hit_time_limit,
        }


@dataclass(frozen=True)
class RoutingPlan:
    """Initial layout plus the routing plan that executes every interaction."""

    # Interaction-graph vertex -> hardware qubit, before and after the plan.
    initial_layout: dict[Any, Any]
    final_layout: dict[Any, Any]
    # Same, including ancilla slots.
    initial_full_layout: dict[Any, Any]
    final_full_layout: dict[Any, Any]
    # Alternating gate batches and SWAP layers, in execution order.
    stages: tuple[Stage, ...]
    # Interaction edges the loop never made executable. Empty when converged.
    unrouted: tuple[tuple[Any, Any], ...]
    total_swaps: int
    num_swap_layers: int
    converged: bool
    # Cost of the initial layout, when the layout came from the mapper.
    mapping_cost: float | None = None
    iterations: tuple[IterationStats, ...] = field(default_factory=tuple)
    # True when routing stopped because the depth budget ran out rather than
    # because every interaction was realised.
    hit_depth_limit: bool = False
    # Sum of J_lm^2 over the interactions the plan executes, and over those left
    # unrouted. With no couplings supplied every J is 1, so these are counts.
    #
    # These are on the mean-1 normalised footing the solver used, not in the
    # caller's original units, so compare them to each other (or read
    # realized_fraction) rather than treating them as absolute physical
    # quantities. Same for swap_weight below.
    realized_weight: float = 0.0
    unrealized_weight: float = 0.0
    # Total hardware weight W_ij accumulated by every emitted SWAP.
    swap_weight: float = 0.0

    @property
    def realized_fraction(self) -> float:
        """Share of the requested interaction weight the plan achieves.

        ``sum J^2`` over the executed interactions divided by the same over all
        of them, so 1.0 means fully routed. This is the quantity a depth-truncated
        run is trying to maximise.
        """
        total = self.realized_weight + self.unrealized_weight
        return 1.0 if total <= 0.0 else self.realized_weight / total

    @property
    def swap_layers(self) -> list[list[Swap]]:
        """Just the SWAP layers, in order, skipping empty ones."""
        return [list(stage.swaps) for stage in self.stages if stage.swaps]

    @property
    def gate_layers(self) -> list[list[tuple[Any, Any]]]:
        """Just the gate batches, in order, skipping empty ones."""
        return [list(stage.gates) for stage in self.stages if stage.gates]

    @property
    def depth(self) -> int:
        """Number of SWAP layers, i.e. the routing overhead in circuit depth."""
        return self.num_swap_layers

    def scheduled_gates(self) -> list[tuple[Any, Any]]:
        """Every interaction edge the plan executes, in execution order."""
        return [gate for stage in self.stages for gate in stage.gates]

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_layout": {
                _label_to_json(k): v for k, v in self.initial_layout.items()
            },
            "final_layout": {_label_to_json(k): v for k, v in self.final_layout.items()},
            "initial_full_layout": {
                _label_to_json(k): v for k, v in self.initial_full_layout.items()
            },
            "final_full_layout": {
                _label_to_json(k): v for k, v in self.final_full_layout.items()
            },
            "stages": [stage.to_dict() for stage in self.stages],
            "unrouted": [list(edge) for edge in self.unrouted],
            "total_swaps": self.total_swaps,
            "num_swap_layers": self.num_swap_layers,
            "converged": self.converged,
            "mapping_cost": self.mapping_cost,
            "iterations": [stats.to_dict() for stats in self.iterations],
            "hit_depth_limit": self.hit_depth_limit,
            "realized_weight": self.realized_weight,
            "unrealized_weight": self.unrealized_weight,
            "realized_fraction": self.realized_fraction,
            "swap_weight": self.swap_weight,
        }

    def summary(self) -> str:
        lines = [
            f"converged: {self.converged}",
            f"gates scheduled: {len(self.scheduled_gates())}",
            f"unrouted interactions: {len(self.unrouted)}",
            f"swaps: {self.total_swaps} in {self.num_swap_layers} layers",
        ]
        if self.mapping_cost is not None:
            lines.append(f"initial layout cost: {self.mapping_cost}")
        if self.hit_depth_limit:
            lines.append("stopped at the depth limit")
        if self.unrealized_weight > 0.0:
            lines.append(
                f"realized interaction weight: {self.realized_weight:.4g} of "
                f"{self.realized_weight + self.unrealized_weight:.4g} "
                f"({self.realized_fraction:.1%})"
            )
        if self.swap_weight > 0.0:
            lines.append(f"swap cost: {self.swap_weight:.4g}")
        return "\n".join(lines)


def _label_to_json(label: Any) -> Any:
    """Make a node label usable as a JSON object key."""
    if isinstance(label, Ancilla):
        return f"ancilla:{label.slot}"
    if isinstance(label, (str, int, float, bool)) or label is None:
        return label
    return str(label)
