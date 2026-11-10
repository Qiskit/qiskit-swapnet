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

"""Stage 4: the routing artifact -- and the one interface a replacement algorithm
implements.

THIS IS THE PIVOT OF THE PIPELINE
---------------------------------

A :class:`Schedule` is an initial layout plus ordered rounds of *"execute these variable
pairs, then apply this qubit-disjoint layer of swaps"*. That description is not specific
to swapnet's quadratic-assignment-plus-multi-commodity-flow method -- a linear swap
network, a SABRE-style heuristic, a hand-built pattern or a future solver can all express
their output as one.

So :mod:`swapnet.ansatz` consumes **only** a ``Schedule``. Any algorithm that emits one
plugs into the rest of the pipeline unchanged, and the contract to implement is::

    def plan(problem: Problem, hardware: Hardware, k: int, seed: int) -> Schedule

swapnet's own implementation of that contract is
:func:`swapnet.plan_schedule`. Its :class:`~swapnet.results.RoutingPlan` is richer --
per-round MILP statistics, reservation counts, realized weight -- and is deliberately
*not* part of the contract: it is a diagnostic of one method.
:func:`from_routing_plan` is the adapter down to the neutral form.

THE SCHEDULE
------------

    initial_layout   label -> physical qubit, injective. Problem variables are ``0..n-1``
                     and must all appear. Any other label is an idle qubit the algorithm
                     wants to swap *through*; include as many as needed. Qubits the device
                     cannot use (:attr:`~swapnet.hardware.Hardware.unusable_qubits`) stay
                     out of the map.
    rounds           ordered. Round ``r`` applies its ``pairs`` at the layout reached so
                     far, then applies its ``swaps``.

    Round.pairs      unordered variable pairs to realise now. Each must sit on a live link
                     at that moment.
    Round.swaps      ``(label_a, label_b)`` to exchange. Each on a live link, and one
                     round's swaps must touch disjoint qubits, because one round is
                     one parallel SWAP layer.

THE RULES, ALL CHECKED BY validate()
------------------------------------

1. ``initial_layout`` is injective into ``0..num_qubits-1`` and places every variable.
2. Every executed pair is on a live link at the moment it executes.
3. Every swap is on a live link, and one round's swaps are qubit-disjoint.
4. No pair is executed twice -- a QAOA cost layer applies each term once.
5. At most ``k`` rounds carry swaps.

A schedule that breaks a rule is rejected, never repaired and never scored: a broken
schedule is not a slower circuit, it is not a circuit.

WHY SWAP *LAYERS* ARE THE BUDGET
--------------------------------

The depth cost of routing is the number of parallel SWAP layers, not the number of SWAP
gates, because swaps on disjoint qubits run simultaneously. Executing a term costs no
budget at all -- which is why :meth:`Schedule.truncate` lets the round whose swaps get cut
still apply its pairs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Round", "Schedule", "ScheduleError", "validate", "from_routing_plan"]


class ScheduleError(ValueError):
    """A schedule that breaks one of the five rules. Not scored."""


@dataclass
class Round:
    """One batch of terms to realise, then one parallel layer of swaps."""

    pairs: list[tuple[int, int]] = field(default_factory=list)
    swaps: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pairs": [list(p) for p in self.pairs],
            "swaps": [list(s) for s in self.swaps],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Round:
        return cls(
            pairs=[(int(a), int(b)) for a, b in payload.get("pairs", [])],
            swaps=[(int(a), int(b)) for a, b in payload.get("swaps", [])],
        )


@dataclass
class Schedule:
    """A complete routing plan. See the module docstring for the rules."""

    initial_layout: dict[int, int]
    rounds: list[Round] = field(default_factory=list)
    # Free-form provenance from the producing algorithm: its name, parameters, timings.
    info: dict = field(default_factory=dict)

    @property
    def num_swap_layers(self) -> int:
        return sum(1 for r in self.rounds if r.swaps)

    @property
    def num_swaps(self) -> int:
        return sum(len(r.swaps) for r in self.rounds)

    def executed_pairs(self) -> list[tuple[int, int]]:
        """Every executed pair, normalised to ``(low, high)``, in execution order."""
        out = []
        for round_ in self.rounds:
            for a, b in round_.pairs:
                out.append((a, b) if a < b else (b, a))
        return out

    def truncate(self, k: int) -> Schedule:
        """The prefix using at most ``k`` swap layers.

        Terms cost no depth budget, so the round whose swaps are cut still executes its
        pairs; rounds after it are dropped.

        This is only equivalent to re-planning at ``k`` when the producing algorithm is a
        prefix under the budget -- swapnet's routing loop is greedy per round and the
        budget only stops it, so it is. An algorithm that reorders its rounds when given a
        smaller budget must be re-invoked per ``k`` rather than truncated.
        """
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        kept: list[Round] = []
        used = 0
        for round_ in self.rounds:
            if round_.swaps and used >= k:
                kept.append(Round(pairs=list(round_.pairs), swaps=[]))
                break
            kept.append(Round(pairs=list(round_.pairs), swaps=list(round_.swaps)))
            if round_.swaps:
                used += 1
        # `info` still describes the run that produced the full plan -- its `k`, its
        # timings, its realized_weight all refer to that run, not to this prefix. Record
        # the truncation so a consumer reading `info` is not misled about which is which.
        info = dict(self.info)
        info["truncated_to"] = k
        return Schedule(initial_layout=dict(self.initial_layout), rounds=kept, info=info)

    def content(self) -> dict:
        """The routing itself, without provenance. Two schedules that route identically
        compare equal here even when they were produced by different runs."""
        return {
            "initial_layout": dict(self.initial_layout),
            "rounds": [r.to_dict() for r in self.rounds],
        }

    def to_dict(self) -> dict:
        return {
            "initial_layout": {str(k): int(v) for k, v in self.initial_layout.items()},
            "rounds": [r.to_dict() for r in self.rounds],
            "info": dict(self.info),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Schedule:
        return cls(
            initial_layout={int(k): int(v) for k, v in payload["initial_layout"].items()},
            rounds=[Round.from_dict(r) for r in payload.get("rounds", [])],
            info=dict(payload.get("info", {})),
        )

    def __repr__(self) -> str:
        return (
            f"<Schedule rounds={len(self.rounds)} "
            f"swap_layers={self.num_swap_layers} swaps={self.num_swaps} "
            f"terms={len(self.executed_pairs())}>"
        )


def validate(schedule: Schedule, n: int, hardware, k: int | None = None) -> dict:
    """Replay a schedule against the hardware. Raises :class:`ScheduleError` on violation.

    ``n`` is the number of problem variables; ``hardware`` is a
    :class:`~swapnet.hardware.Hardware` or anything with ``num_qubits`` and
    ``is_live(a, b)``. Returns a summary: the final layout, the executed pairs, the layer
    counts.

    Call this on any schedule you did not produce yourself, and on your own before
    building a circuit from it. It is the only thing standing between a subtly wrong
    routing algorithm and a circuit that runs but computes the wrong generator.
    """
    layout = dict(schedule.initial_layout)

    missing = [v for v in range(n) if v not in layout]
    if missing:
        raise ScheduleError(f"initial_layout has no qubit for variables {missing[:8]}")
    if len(set(layout.values())) != len(layout):
        raise ScheduleError("initial_layout puts two labels on the same qubit")
    out_of_range = sorted(q for q in layout.values() if not 0 <= q < hardware.num_qubits)
    if out_of_range:
        raise ScheduleError(
            f"initial_layout uses qubits {out_of_range[:8]}, outside "
            f"0..{hardware.num_qubits - 1}"
        )

    executed: set[tuple[int, int]] = set()
    swap_layers = 0

    for index, round_ in enumerate(schedule.rounds):
        for a, b in round_.pairs:
            if not (0 <= a < n and 0 <= b < n):
                raise ScheduleError(
                    f"round {index}: pair ({a},{b}) is not a pair of problem variables"
                )
            if a == b:
                raise ScheduleError(f"round {index}: pair ({a},{b}) is a self-pair")
            key = (a, b) if a < b else (b, a)
            if key in executed:
                raise ScheduleError(
                    f"round {index}: pair {key} is executed twice; a cost layer applies "
                    "each term once"
                )
            pa, pb = layout[a], layout[b]
            if not hardware.is_live(pa, pb):
                raise ScheduleError(
                    f"round {index}: pair {key} sits on qubits ({pa},{pb}), which is not "
                    "a live link of this device"
                )
            executed.add(key)

        touched: set[int] = set()
        for a, b in round_.swaps:
            if a not in layout or b not in layout:
                raise ScheduleError(
                    f"round {index}: swap ({a},{b}) names a label that is not placed"
                )
            pa, pb = layout[a], layout[b]
            if not hardware.is_live(pa, pb):
                raise ScheduleError(
                    f"round {index}: swap ({a},{b}) sits on ({pa},{pb}), not a live link"
                )
            if pa in touched or pb in touched:
                clash = pa if pa in touched else pb
                raise ScheduleError(
                    f"round {index}: swap ({a},{b}) reuses qubit {clash} inside one "
                    "layer, so the layer is not parallel"
                )
            touched.add(pa)
            touched.add(pb)
            layout[a], layout[b] = layout[b], layout[a]

        if round_.swaps:
            swap_layers += 1

    if k is not None and swap_layers > k:
        raise ScheduleError(
            f"schedule uses {swap_layers} swap layers but the budget is k={k}"
        )

    return {
        "final_layout": layout,
        "executed_pairs": sorted(executed),
        "num_executed_pairs": len(executed),
        "num_swap_layers": swap_layers,
        "num_swaps": schedule.num_swaps,
        "num_rounds": len(schedule.rounds),
    }


def from_routing_plan(plan, n: int, *, info: Mapping[str, Any] | None = None) -> Schedule:
    """Adapt a swapnet :class:`~swapnet.results.RoutingPlan` to the neutral artifact.

    swapnet labels a problem variable by its own integer and an idle slot by
    :class:`~swapnet.results.Ancilla`. Both become integer labels here, with idle slots
    shifted past the variable range so the two can never collide -- the schedule needs
    them because routing swaps *through* idle qubits, and a downstream stage that saw only
    the variables would mis-replay the layout.
    """
    layout = {
        _label_key(label, n): int(physical)
        for label, physical in plan.initial_full_layout.items()
    }
    if len(set(layout.values())) != len(layout):
        raise ScheduleError("the routing plan puts two labels on one qubit")
    unplaced = [v for v in range(n) if v not in layout]
    if unplaced:
        raise ScheduleError(f"the routing plan does not place variables {unplaced[:8]}")

    rounds = [
        Round(
            pairs=[(int(a), int(b)) for a, b in stage.gates],
            swaps=[
                (_label_key(op.logical[0], n), _label_key(op.logical[1], n))
                for op in stage.swaps
            ],
        )
        for stage in plan.stages
    ]
    return Schedule(initial_layout=layout, rounds=rounds, info=dict(info or {}))


def _label_key(label, n: int) -> int:
    """A swapnet label as an int: a variable stays itself, ``Ancilla(slot)`` shifts by n.

    Keeps the two namespaces from colliding.
    """
    return int(label) if isinstance(label, int) else n + int(label.slot)
