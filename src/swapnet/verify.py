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

"""Independent replay of a routing plan.

The routing loop and the plan builder are separate enough that a bug in either
would be easy to miss, and a wrong plan produces a silently wrong circuit. This
module re-derives everything from the plan alone: it walks the timeline, applies
each SWAP to its own copy of the layout, and asserts that every scheduled gate is
on adjacent hardware qubits when its turn comes.

Cheap enough to run on every plan; :func:`swapnet.api.map_and_route` does not call
it automatically, but the tests and the CLI do.
"""

from __future__ import annotations

import math
from typing import Any

from .graphs import as_graph_view
from .results import RoutingPlan
from .weights import normalize

__all__ = ["PlanError", "verify_plan"]


class PlanError(AssertionError):
    """A routing plan is not executable on the given hardware."""


def _normalized(a: Any, b: Any) -> frozenset:
    return frozenset((a, b))


def verify_plan(
    hardware: Any,
    interaction: Any,
    plan: RoutingPlan,
    *,
    require_converged: bool = True,
) -> None:
    """Check that ``plan`` executes ``interaction`` on ``hardware``.

    Raises :class:`PlanError` describing the first problem found. Returns ``None``
    on success.

    Verified properties:

    * the initial layout is a bijection between logical slots and hardware qubits,
    * every SWAP acts on an adjacent pair of hardware qubits,
    * every SWAP's logical pair really occupies those qubits at that moment,
    * SWAPs inside one layer touch disjoint qubits, so the layer is executable in
      one time step,
    * every scheduled gate is on adjacent qubits when its stage is reached,
    * replaying all SWAPs reproduces the reported final layout,
    * no edge is both scheduled and reported unrouted,
    * the scheduled gates plus the unrouted edges are exactly the interaction
      edges, each appearing once,
    * on a weighted interaction graph, the reported share of realised ``J^2``
      matches the plan's own gate/unrouted split,
    * on a weighted hardware graph, the reported swap cost matches the ``W_ij``
      of the couplings its SWAP layers actually use.

    Pass the same weighted graphs you routed with for the last two checks to
    apply; an unweighted graph simply skips them.
    """
    hardware_view = as_graph_view(hardware, name="hardware graph")
    interaction_view = as_graph_view(interaction, name="interaction graph")

    hardware_edges = {
        _normalized(hardware_view.labels[u], hardware_view.labels[v])
        for u, v in hardware_view.edges
    }
    hardware_qubits = set(hardware_view.labels)

    # logical label -> hardware qubit, mutated as we replay.
    position = dict(plan.initial_full_layout)

    if len(position) != len(hardware_qubits):
        raise PlanError(
            f"initial layout covers {len(position)} slots but the hardware has "
            f"{len(hardware_qubits)} qubits"
        )
    if set(position.values()) != hardware_qubits:
        raise PlanError("initial layout is not a bijection onto the hardware qubits")

    for stage_index, stage in enumerate(plan.stages):
        for gate in stage.gates:
            a, b = gate
            for label in gate:
                if label not in position:
                    raise PlanError(
                        f"stage {stage_index}: gate {gate!r} names unknown "
                        f"logical qubit {label!r}"
                    )
            pair = _normalized(position[a], position[b])
            if pair not in hardware_edges:
                raise PlanError(
                    f"stage {stage_index}: gate {gate!r} is scheduled on "
                    f"non-adjacent qubits {tuple(sorted(map(str, pair)))}"
                )

        touched: set[Any] = set()
        for swap in stage.swaps:
            u, v = swap.physical
            if _normalized(u, v) not in hardware_edges:
                raise PlanError(
                    f"stage {stage_index}: swap on {swap.physical!r} is not a "
                    "hardware edge"
                )

            a, b = swap.logical
            for label in swap.logical:
                if label not in position:
                    raise PlanError(
                        f"stage {stage_index}: swap names unknown logical qubit "
                        f"{label!r}"
                    )
            if _normalized(position[a], position[b]) != _normalized(u, v):
                raise PlanError(
                    f"stage {stage_index}: swap claims logical {swap.logical!r} "
                    f"sit on {swap.physical!r}, but they are on "
                    f"{(position[a], position[b])!r}"
                )

            if u in touched or v in touched:
                raise PlanError(
                    f"stage {stage_index}: swap layer reuses qubit "
                    f"{u if u in touched else v!r}, so the layer cannot run in "
                    "parallel"
                )
            touched.update((u, v))

            position[a], position[b] = position[b], position[a]

    if position != plan.final_full_layout:
        differing = {
            label: (position[label], plan.final_full_layout.get(label))
            for label in position
            if position[label] != plan.final_full_layout.get(label)
        }
        raise PlanError(
            "replaying the swaps did not reproduce the reported final layout; "
            f"{len(differing)} slots differ, e.g. {dict(list(differing.items())[:3])}"
        )

    expected = {
        _normalized(interaction_view.labels[u], interaction_view.labels[v])
        for u, v in interaction_view.edges
    }

    scheduled: list[frozenset] = [
        _normalized(*gate) for stage in plan.stages for gate in stage.gates
    ]
    if len(scheduled) != len(set(scheduled)):
        raise PlanError("some interaction edges are scheduled more than once")

    unrouted = {_normalized(*edge) for edge in plan.unrouted}

    # An edge cannot be both executed and reported as never executable. The set
    # union below would hide that, so check the overlap first.
    contradictory = unrouted & set(scheduled)
    if contradictory:
        raise PlanError(
            f"{len(contradictory)} interaction edges are both scheduled and "
            f"reported unrouted, e.g. {sorted(map(tuple, contradictory))[:3]}"
        )

    covered = set(scheduled) | unrouted

    if covered != expected:
        missing = expected - covered
        extra = covered - expected
        parts = []
        if missing:
            parts.append(f"{len(missing)} interaction edges are unaccounted for")
        if extra:
            parts.append(f"{len(extra)} scheduled edges are not interactions")
        raise PlanError("; ".join(parts))

    # Weight accounting, when the caller passed a weighted interaction graph. The
    # reported J^2 split must match the plan's own gate/unrouted partition, since
    # a truncated run is judged on exactly that number.
    if interaction_view.is_weighted:
        mass_of = {
            _normalized(interaction_view.labels[u], interaction_view.labels[v]): w * w
            for (u, v), w in zip(
                interaction_view.edges, interaction_view.weights, strict=True
            )
        }
        realized = math.fsum(mass_of[edge] for edge in scheduled)
        total = realized + math.fsum(mass_of[edge] for edge in unrouted)

        # The pipeline normalises the couplings to mean 1 before solving, so the
        # absolute J^2 mass it reports is on a rescaled footing. The *share* is
        # invariant under that rescaling, so that is what can be checked here.
        comparable = total > 0.0 and (plan.realized_weight + plan.unrealized_weight) > 0.0
        if comparable and not math.isclose(
            plan.realized_fraction, realized / total, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise PlanError(
                f"plan reports realized_fraction={plan.realized_fraction!r} "
                f"but its stages imply {realized / total!r}"
            )

    # Likewise for the routing cost: every emitted SWAP pays its coupling's W_ij.
    # The weights are mean-1 normalised inside the pipeline, so compare against
    # the caller's weights put on the same footing.
    if hardware_view.is_weighted and plan.total_swaps > 0:
        # Reuse the package's normalisation rather than repeating the arithmetic:
        # it rejects the all-zero and non-finite cases with an actionable message
        # instead of dividing by zero.
        scaled = normalize(hardware_view.weights)
        cost_of = {
            _normalized(hardware_view.labels[u], hardware_view.labels[v]): w
            for (u, v), w in zip(hardware_view.edges, scaled, strict=True)
        }
        recomputed = math.fsum(
            cost_of[_normalized(*swap.physical)]
            for stage in plan.stages
            for swap in stage.swaps
        )
        if not math.isclose(plan.swap_weight, recomputed, rel_tol=1e-6, abs_tol=1e-9):
            raise PlanError(
                f"plan reports swap_weight={plan.swap_weight!r} but its swap "
                f"layers imply {recomputed!r} (comparing mean-1 normalised "
                "weights)"
            )

    if require_converged:
        if plan.unrouted:
            raise PlanError(
                f"{len(plan.unrouted)} interaction edges were never made " "executable"
            )
        if not plan.converged:
            raise PlanError("the routing loop stopped before converging")
