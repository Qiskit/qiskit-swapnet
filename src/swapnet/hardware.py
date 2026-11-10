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

"""Stage 2: the hardware artifact -- the coupling graph to plan on, and its costs W_ij.

A :class:`Hardware` is the device reduced to the two things layout and routing need: an
undirected graph of couplings that can actually run the circuit, and a traversal cost
``W_ij`` per coupling. Everything else about a backend is provenance.

    >>> from qiskit_ibm_runtime.fake_provider import FakeBoston
    >>> hw = Hardware.from_backend(FakeBoston())
    >>> hw.num_links, len(hw.dead_links)      # 175 live of 176 couplings
    (175, 1)

DEAD LINKS ARE PER GATE, NOT PER DEVICE
---------------------------------------

A link is unusable for a *gate*, and different gates on the same link fail
independently. The circuit needs more than one two-qubit gate -- the cost terms and the
SWAP network are not the same operation -- so a link qualifies only when **every** gate
the circuit needs is live on it, and its cost is the **worst** of them. Pricing only one
gate would let a cost term land on a link that cannot run it. On the frozen
``ibm_boston`` fractional snapshot this is not academic: ``cz`` has 0 dead links while
``rzz`` has 2.

Dead links are removed from :attr:`graph`, so a plan cannot schedule a gate the device
cannot execute. That is deliberate rather than pricing them at fidelity 0: a circuit
containing one dead gate scores exactly 0, which discards the information in the rest of
it, whereas refusing to plan on the link keeps the comparison meaningful.

WHY THE GRAPH IS THE LARGEST CONNECTED COMPONENT
------------------------------------------------

Dead links can isolate a qubit. Routing needs one connected component, and a problem
needs only ``n`` of the qubits, so the largest component is the right domain. The qubits
left out are reported as :attr:`unusable_qubits` -- never silently dropped, because
"there were fewer qubits than you thought" is exactly the kind of thing that turns into a
wrong conclusion three stages later.

THE COST W_ij
-------------

``-log(1 - error)``, normalised to mean 1, via :func:`swapnet.from_error_rates`. That
transform is the one that makes a path's cost *additive* exactly when its fidelity is
*multiplicative*, so minimising summed cost maximises the fidelity of the route. Mean-1
normalisation preserves every ratio and only changes the unit, which matters because the
search schedule and the routing reward are calibrated against unit-cost hops.

THE BACKEND OBJECT IS NOT STORED
--------------------------------

Only its name and the numbers read off it. A Qiskit ``Target`` is large, version-coupled
and unpicklable across releases; keeping one on the artifact would pin every downstream
consumer to this Qiskit build. Rebuild it from the backend when a transpiler pass needs
it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from .weights import from_error_rates

__all__ = ["Hardware"]


@dataclass(frozen=True)
class Hardware:
    """A device's coupling graph and per-link traversal costs.

    ``errors`` holds only the **live** links, keyed by ``(low, high)``, each value the
    worst error over the gates the circuit needs there. ``dead_links`` are the ones
    excluded and why is in ``metadata["dead_links_by_gate"]``.
    """

    name: str
    num_qubits: int
    errors: dict[tuple[int, int], float]
    dead_links: tuple[tuple[int, int], ...] = ()
    priced_gates: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical: dict[tuple[int, int], float] = {}
        for pair, value in self.errors.items():
            a, b = int(pair[0]), int(pair[1])
            if a == b:
                raise ValueError(f"({a}, {b}) is a self-loop, not a coupling")
            if not (0 <= a < self.num_qubits and 0 <= b < self.num_qubits):
                raise ValueError(f"link ({a}, {b}) is outside 0..{self.num_qubits - 1}")
            error = float(value)
            if not 0.0 <= error < 1.0:
                raise ValueError(
                    f"link ({a}, {b}) has error {error}, which is not a live error rate; "
                    "links with error >= 1 belong in dead_links"
                )
            key = (a, b) if a < b else (b, a)
            # Both directions of one link may be reported; keep the worse.
            canonical[key] = max(canonical.get(key, 0.0), error)
        object.__setattr__(self, "errors", canonical)
        object.__setattr__(
            self,
            "dead_links",
            tuple(sorted((min(a, b), max(a, b)) for a, b in self.dead_links)),
        )
        object.__setattr__(self, "priced_gates", tuple(self.priced_gates))
        if not canonical:
            raise ValueError(f"{self.name}: no live links at all")

    # -- the graph -----------------------------------------------------------

    @property
    def num_links(self) -> int:
        """Live links only."""
        return len(self.errors)

    @cached_property
    def full_graph(self):
        """Every qubit, live links only. May be disconnected."""
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_qubits))
        graph.add_edges_from(self.errors)
        return graph

    @cached_property
    def graph(self):
        """The graph to plan on: the largest connected live component."""
        import networkx as nx

        components = sorted(
            nx.connected_components(self.full_graph), key=len, reverse=True
        )
        if not components:
            return self.full_graph
        return self.full_graph.subgraph(components[0]).copy()

    @property
    def usable_qubits(self) -> list[int]:
        return sorted(self.graph.nodes())

    @property
    def unusable_qubits(self) -> list[int]:
        """Qubits outside the largest live component. Reported, never silently dropped."""
        return sorted(set(range(self.num_qubits)) - set(self.graph.nodes()))

    @cached_property
    def edge_weights(self) -> dict[tuple[int, int], float]:
        """``W_ij``: ``-log(1 - error)`` normalised to mean 1, over the planning graph.

        Restricted to :attr:`graph`, so it matches exactly the edges a plan may use.
        Pass this as ``edge_weights=`` to :func:`swapnet.map_layout` and
        :func:`swapnet.route`.
        """
        live = {
            (min(a, b), max(a, b)): self.errors[(min(a, b), max(a, b))]
            for a, b in self.graph.edges()
        }
        return from_error_rates(live)

    def is_live(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.errors

    def check_capacity(self, n: int) -> None:
        """Raise when the usable component cannot hold an ``n``-variable problem."""
        usable = self.graph.number_of_nodes()
        if usable < n:
            raise ValueError(
                f"{self.name}: the largest live component has {usable} qubits but the "
                f"problem needs {n}. Dead links {[list(p) for p in self.dead_links]} "
                "are what removed the rest."
            )

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "errors": {f"{a},{b}": v for (a, b), v in sorted(self.errors.items())},
            "dead_links": [list(p) for p in self.dead_links],
            "priced_gates": list(self.priced_gates),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Hardware:
        errors = {}
        for key, value in payload["errors"].items():
            a, b = key.split(",")
            errors[(int(a), int(b))] = float(value)
        return cls(
            name=str(payload["name"]),
            num_qubits=int(payload["num_qubits"]),
            errors=errors,
            dead_links=tuple(tuple(p) for p in payload.get("dead_links", ())),
            priced_gates=tuple(payload.get("priced_gates", ())),
            metadata=dict(payload.get("metadata", {})),
        )

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_backend(
        cls, backend: Any, *, gates: Sequence[str] | None = None
    ) -> Hardware:
        """From a Qiskit ``BackendV2``. Reads ``backend.target``; no network call.

        ``gates`` names the two-qubit gates that must be live on a link. The default is
        every two-qubit gate the target defines, which is the conservative choice -- see
        the module docstring on why one gate is not enough.
        """
        target = getattr(backend, "target", None)
        if target is None:
            raise TypeError(
                f"{type(backend).__name__} has no .target; pass a BackendV2, a Target, "
                "or use Hardware.from_error_map"
            )
        name = getattr(backend, "name", None) or type(backend).__name__
        return cls.from_target(target, gates=gates, name=str(name))

    @classmethod
    def from_target(
        cls, target: Any, *, gates: Sequence[str] | None = None, name: str = "target"
    ) -> Hardware:
        """From a Qiskit transpiler ``Target``."""
        two_qubit = _two_qubit_operations(target)
        if not two_qubit:
            raise ValueError(f"{name}: the target defines no two-qubit operation")

        if gates is None:
            needed = sorted(two_qubit)
        else:
            missing = [g for g in gates if g not in two_qubit]
            if missing:
                raise ValueError(
                    f"{name}: no two-qubit gate {missing} in the target; it has "
                    f"{sorted(two_qubit)}"
                )
            needed = list(gates)

        # Per gate: {(low, high): error}, taking the worse of the two directions.
        per_gate: dict[str, dict[tuple[int, int], float]] = {}
        for gate in needed:
            table: dict[tuple[int, int], float] = {}
            for qargs, properties in target[gate].items():
                if qargs is None or len(qargs) != 2:
                    continue
                a, b = int(qargs[0]), int(qargs[1])
                key = (min(a, b), max(a, b))
                error = getattr(properties, "error", None) if properties else None
                # An unpriced link is not a free one. Treat it as dead: planning on a
                # coupling whose error nobody reported is planning on a guess.
                value = float("inf") if error is None else float(error)
                table[key] = max(table.get(key, 0.0), value)
            per_gate[gate] = table

        every_link = sorted(set().union(*(set(t) for t in per_gate.values())))
        live: dict[tuple[int, int], float] = {}
        dead: list[tuple[int, int]] = []
        for link in every_link:
            worst = 0.0
            usable = True
            for gate in needed:
                error = per_gate[gate].get(link)
                if error is None or error >= 1.0:
                    usable = False
                    break
                worst = max(worst, error)
            if usable:
                live[link] = worst
            else:
                dead.append(link)

        metadata = {
            "source": "qiskit_target",
            "two_qubit_operations": sorted(two_qubit),
            "dead_links_by_gate": {
                gate: sorted([list(p) for p, e in table.items() if e >= 1.0])
                for gate, table in sorted(per_gate.items())
            },
        }
        return cls(
            name=name,
            num_qubits=int(target.num_qubits),
            errors=live,
            dead_links=tuple(dead),
            priced_gates=tuple(needed),
            metadata=metadata,
        )

    @classmethod
    def from_error_map(
        cls,
        errors: Mapping[tuple[int, int], float],
        *,
        num_qubits: int | None = None,
        name: str = "hardware",
        metadata: Mapping[str, Any] | None = None,
    ) -> Hardware:
        """From ``{(i, j): two-qubit error rate}`` directly.

        The escape hatch, and the way to use a frozen calibration snapshot or a
        hand-built topology without constructing a Qiskit ``Target``. Links priced at
        ``>= 1.0`` are recorded as dead rather than rejected, so a caller can hand over a
        full table and let this method do the filtering.
        """
        live: dict[tuple[int, int], float] = {}
        dead: list[tuple[int, int]] = []
        highest = -1
        for pair, value in errors.items():
            a, b = int(pair[0]), int(pair[1])
            highest = max(highest, a, b)
            key = (min(a, b), max(a, b))
            error = float(value)
            if error >= 1.0:
                dead.append(key)
            else:
                live[key] = max(live.get(key, 0.0), error)
        # A link dead in one direction is dead: drop it from live if it appears in both.
        for key in dead:
            live.pop(key, None)

        size = highest + 1 if num_qubits is None else int(num_qubits)
        info = {"source": "error_map"}
        info.update(metadata or {})
        return cls(
            name=name,
            num_qubits=size,
            errors=live,
            dead_links=tuple(dead),
            priced_gates=(),
            metadata=info,
        )

    def __repr__(self) -> str:
        return (
            f"<Hardware {self.name} qubits={self.num_qubits} "
            f"live_links={self.num_links} dead={len(self.dead_links)} "
            f"usable={len(self.usable_qubits)}>"
        )


def _two_qubit_operations(target: Any) -> list[str]:
    """The names of every two-qubit operation the target defines."""
    out = []
    for name in target.operation_names:
        try:
            qargs = target.qargs_for_operation_name(name)
        except KeyError:
            # We iterate the target's own operation_names, so this should not happen;
            # tolerate it rather than fail on a Target that reports an op it cannot
            # describe.
            continue
        if not qargs:
            continue
        if any(entry is not None and len(entry) == 2 for entry in qargs):
            out.append(name)
    return out
