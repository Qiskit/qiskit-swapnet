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

"""Stage 5: one QAOA cost layer as a self-describing object.

:class:`Ansatz` is the handover point between routing and everything downstream. It
carries the parametrized unitary, the generator that unitary exponentiates, the two
layouts, and the mask of which problem terms survived the depth budget -- so a consumer
needs nothing beside it.

:func:`build` is the builder. It consumes a :class:`~swapnet.schedule.Schedule`, which is
the algorithm-neutral artifact, so it works for any routing method that emits one.

THE CONVENTION, ONCE
--------------------

    U(theta) = exp(-i G(theta) / 2)        G(theta) = sum_alpha theta_alpha Z^alpha

The ``/2`` is part of the definition of the generator. Qiskit's rotations carry the same
sign and the same half-angle, so the gate angle *is* ``theta`` and there is no conversion
factor -- see :mod:`swapnet.conventions`, which is the authority, and
``tests/test_ansatz.py``, which pins it against ``expm``.

For QAOA at a given ``gamma``, use :func:`angles_from_problem`: it applies
``theta_alpha = 2 * gamma * coeff_alpha`` so a parameter-setting module never handles the
factor itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .conventions import gamma_to_theta, rz_angle, rzz_angle

__all__ = ["Ansatz", "build", "angles_from_problem", "verify_structure", "edge_coloring"]


@dataclass
class Ansatz:
    """One QAOA cost layer: the parametrized unitary, its generator, and the two layouts.

    ``circuit``
        ``U(theta) = exp(-i G(theta)/2)`` for *one* layer, on device qubits, including
        whatever SWAP network the schedule implies. It carries no initial state, no mixer
        and no measurement: each depends on something a single layer cannot know -- the
        bias, this layer's ``beta``, and the parity of the layer count -- so all three are
        added by the consumer.

        Read exactly: the circuit implements ``exp(-i G/2)`` when the input is interpreted
        through :attr:`input_layout` and the output through :attr:`output_layout`. As an
        operator on device qubits it is that exponential *followed by* the relabelling
        permutation the SWAP network performs, which is why both layouts are here.
    ``generator``
        ``G(theta)`` on ``n`` problem qubits, as a ``SparsePauliOp`` whose coefficients
        **are** the circuit's own ``Parameter`` objects. Named ``generator`` and not
        ``hamiltonian`` on purpose: under ``U = exp(-i G/2)`` the word "Hamiltonian"
        invites the reading ``exp(-i t H)``, which is wrong by a factor 2.
        :attr:`couplings`, :attr:`theta_two_body` and :attr:`theta_one_body` are read off
        it rather than stored beside it.
    ``input_layout``, ``output_layout``
        ``pi_0`` and ``pi_fin``, problem qubit -> device qubit. ``pi_fin`` is ``pi_0``
        composed with this layer's SWAP network, and is what an odd number of layers is
        read out under.
    ``mask``
        ``(n, n)`` symmetric boolean, ``True`` where the two-body term ``Z_i Z_j`` is in
        the circuit. The depth budget drops terms, and a downstream stage that does not
        know which were dropped will compare against the wrong objective. The diagonal is
        ``False``; one-body terms are all kept, so they need no mask.
    ``backend``
        Left ``None`` deliberately. A Qiskit ``Target`` is large and version-coupled, so
        storing one would pin every consumer to this Qiskit build; the device is recorded
        in ``metadata`` instead. :attr:`target` accepts either a backend or a ``Target``
        for the caller that does set it.
    """

    circuit: Any
    generator: Any
    input_layout: Sequence[int]  # or a {problem qubit: device qubit} mapping
    output_layout: Sequence[int]
    backend: Any = None
    mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.input_layout = _as_layout_list(
            self.input_layout, self.num_problem_qubits, "input_layout"
        )
        self.output_layout = _as_layout_list(
            self.output_layout, self.num_problem_qubits, "output_layout"
        )
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)
            n = self.num_problem_qubits
            if self.mask.shape != (n, n):
                raise ValueError(f"mask must be ({n}, {n}), got {self.mask.shape}")
            if not np.array_equal(self.mask, self.mask.T):
                raise ValueError("mask must be symmetric")
            if self.mask.diagonal().any():
                raise ValueError("mask must have a False diagonal")

    # -- shape ---------------------------------------------------------------

    @property
    def num_problem_qubits(self) -> int:
        return int(self.generator.num_qubits)

    @property
    def num_device_qubits(self) -> int:
        return int(self.circuit.num_qubits)

    # -- read off the generator ---------------------------------------------

    @property
    def couplings(self) -> tuple[tuple[int, int], ...]:
        """The two-body terms present, as sorted ``(i, j)`` with ``i < j``."""
        return tuple(sorted(self._terms()[1]))

    @property
    def theta_one_body(self) -> dict[int, Any]:
        """``{problem qubit: Parameter}`` for the one-body terms."""
        return dict(self._terms()[0])

    @property
    def theta_two_body(self) -> dict[tuple[int, int], Any]:
        """``{(i, j): Parameter}`` for the two-body terms."""
        return dict(self._terms()[1])

    def _terms(self):
        """Split the generator into its one- and two-body parts, by inspection."""
        one: dict[int, Any] = {}
        two: dict[tuple[int, int], Any] = {}
        for pauli, coeff in zip(
            self.generator.paulis, self.generator.coeffs, strict=True
        ):
            if pauli.x.any():
                raise ValueError("the generator must be diagonal: found an X or Y factor")
            support = np.flatnonzero(pauli.z)
            if support.size == 1:
                one[int(support[0])] = _as_parameter(coeff)
            elif support.size == 2:
                two[(int(support[0]), int(support[1]))] = _as_parameter(coeff)
            elif support.size == 0:
                continue  # a constant offset carries no parameter of interest
            else:
                raise ValueError(f"unexpected {support.size}-body term at {support}")
        return one, two

    @property
    def target(self):
        """The transpiler ``Target`` to translate into, or ``None``.

        Accepts either a backend or a ``Target`` in :attr:`backend` and returns the
        ``Target`` either way, so a translation-only pass has one thing to ask for.
        """
        if self.backend is None:
            return None
        if hasattr(self.backend, "operation_names") and hasattr(
            self.backend, "qubit_properties"
        ):
            return self.backend  # already a Target
        return getattr(self.backend, "target", None)

    # -- binding -------------------------------------------------------------

    def bind(
        self,
        one_body: Mapping[int, float] | Sequence[float] | None = None,
        two_body: Mapping[tuple[int, int], float] | None = None,
    ):
        """Bind the generator's coefficients and return the concrete circuit.

        ``one_body`` is a mapping from problem qubit, or a length-``n`` sequence.
        ``two_body`` is a mapping keyed by ``(i, j)``; either orientation is accepted.
        Both hold ``theta`` values, not ``gamma`` and not raw ``J`` -- use
        :func:`angles_from_problem` to get them.

        Every parameter of the circuit must be covered. A missing term is an error rather
        than an implicit zero, because an implicit zero silently drops a term from the
        cost layer and nothing downstream would notice.
        """
        one_map, two_map = self._terms()
        values: dict[Any, float] = {}

        if one_body is not None:
            if isinstance(one_body, Mapping):
                items = {int(k): float(v) for k, v in one_body.items()}
            else:
                sequence = list(one_body)
                if len(sequence) != self.num_problem_qubits:
                    raise ValueError(
                        f"one_body has {len(sequence)} entries, expected "
                        f"{self.num_problem_qubits}"
                    )
                items = {i: float(v) for i, v in enumerate(sequence)}
            for qubit, value in items.items():
                if qubit not in one_map:
                    raise KeyError(f"no one-body term on qubit {qubit}")
                values[one_map[qubit]] = value

        if two_body is not None:
            for pair, value in two_body.items():
                a, b = int(pair[0]), int(pair[1])
                key = (a, b) if a < b else (b, a)
                if key not in two_map:
                    raise KeyError(
                        f"term {key} is not in this ansatz; the depth budget dropped it "
                        "(check `mask`)"
                    )
                values[two_map[key]] = float(value)

        unbound = set(self.circuit.parameters) - set(values)
        if unbound:
            names = sorted(str(p) for p in unbound)
            raise ValueError(
                f"{len(names)} parameters were left unbound, e.g. {names[:5]}. Binding "
                "must be complete: an implicit zero would silently drop a term."
            )

        return self.circuit.assign_parameters(values)

    # -- the mask ------------------------------------------------------------

    @property
    def kept_pairs(self) -> tuple[tuple[int, int], ...]:
        """The two-body terms in the circuit. Same as :attr:`couplings`."""
        return self.couplings

    def dropped_pairs(self, all_pairs) -> tuple[tuple[int, int], ...]:
        """The terms of ``all_pairs`` that the depth budget left out."""
        kept = set(self.couplings)
        out = []
        for a, b in all_pairs:
            key = (a, b) if a < b else (b, a)
            if key not in kept:
                out.append(key)
        return tuple(sorted(out))

    def __repr__(self) -> str:
        return (
            f"<Ansatz n={self.num_problem_qubits} "
            f"device_qubits={self.num_device_qubits} "
            f"terms={len(self.couplings)} "
            f"params={len(self.circuit.parameters)} "
            f"swap_layers={self.metadata.get('swap_layers_used', '?')}>"
        )


def _as_parameter(coeff):
    """The bare ``Parameter`` inside a generator coefficient.

    ``SparsePauliOp`` stores what it is given as a ``ParameterExpression``, even when that
    expression is a single unmodified ``Parameter``. The documented contract is that the
    generator's coefficients *are* the circuit's own ``Parameter`` objects, and
    ``assign_parameters`` only accepts the bare object, so unwrap it here. Anything with 0
    or 2+ parameters is returned unchanged.
    """
    parameters = getattr(coeff, "parameters", None)
    if parameters is not None and len(parameters) == 1:
        return next(iter(parameters))
    return coeff


def _as_layout_list(layout, n: int, name: str) -> list[int]:
    """Accept a sequence or a ``{problem qubit: device qubit}`` mapping; store a list."""
    if isinstance(layout, Mapping):
        missing = [q for q in range(n) if q not in layout]
        if missing:
            raise ValueError(f"{name} has no device qubit for {missing[:8]}")
        return [int(layout[q]) for q in range(n)]
    values = [int(q) for q in layout]
    if len(values) != n:
        raise ValueError(f"{name} has {len(values)} entries, expected {n}")
    return values


def edge_coloring(pairs: Sequence[tuple[int, int]]) -> dict[frozenset, int]:
    """Partition ``pairs`` into classes no two of which share a qubit.

    ``{frozenset((u, v)): class index}``, and ``{}`` for no pairs. Each class is a set of
    gates that can run in one layer, so the number of classes is the two-qubit depth an
    emission ordered by it achieves.

    A layer of two-qubit gates is a set acting on disjoint qubits, that is a matching, so
    a depth-``d`` schedule of these gates *is* a proper ``d``-edge-colouring: the minimum
    achievable depth is exactly the edge chromatic number. Both branches are rustworkx
    routines -- already a hard dependency of qiskit, which this module needs anyway:

    * **bipartite** -- ``graph_bipartite_edge_color`` reaches ``Delta``, the maximum
      degree, which Konig's theorem makes equal to the chromatic number, so the result is
      exactly optimal. Every IBM heavy-hex coupling graph is bipartite and a subgraph of a
      bipartite graph is bipartite, so this is the branch a round takes on hardware.
    * **anything else** -- ``graph_misra_gries_edge_color``, which Vizing bounds at
      ``Delta + 1``: within one layer of optimal, and no better guarantee is cheap, since
      deciding ``Delta`` against ``Delta + 1`` in general is NP-complete.

    Repeated pairs are ignored; a pair and its reverse are the same gate. The caller is
    responsible for only passing gates that commute -- for a diagonal cost layer they all
    do, exactly.
    """
    import rustworkx as rx

    graph = rx.PyGraph(multigraph=False)
    index_of: dict[int, int] = {}
    edge_pair: dict[int, frozenset] = {}
    for u, v in pairs:
        u, v = int(u), int(v)
        key = frozenset((u, v))
        for qubit in (u, v):
            if qubit not in index_of:
                index_of[qubit] = graph.add_node(qubit)
        edge = graph.add_edge(index_of[u], index_of[v], None)
        edge_pair[edge] = key
    if graph.num_edges() == 0:
        return {}

    try:
        by_edge = rx.graph_bipartite_edge_color(graph)
    except rx.GraphNotBipartite:
        by_edge = rx.graph_misra_gries_edge_color(graph)
    return {edge_pair[edge]: int(colour) for edge, colour in by_edge.items()}


def build(
    schedule,
    problem,
    hardware,
    *,
    backend: Any = None,
    validate: bool = True,
    k: int | None = None,
    name: str = "qaoa_cost_layer",
) -> Ansatz:
    """Build the one-layer :class:`Ansatz` a schedule implies.

    The one-body terms are emitted first, at the input layout, which is exact because
    every term is diagonal and therefore commutes with all the others. The two-body terms
    then follow the schedule round by round, with the SWAP network between them.

    Within a round the two-body terms are emitted grouped by :func:`edge_coloring`, so
    terms sharing no qubit go out together and the round costs as few layers as its
    interaction subgraph allows. The order is free to choose: the terms are diagonal, so
    they commute exactly and no ordering can change the unitary. It is not free for depth,
    and nothing downstream recovers it -- Qiskit's depth is the DAG longest path, and
    neither the transpiler nor a scheduling pass reorders commuting gates.

    The circuit is left in ``rz``/``rzz``/``swap`` rather than translated to a device
    basis, so the convention stays checkable without a basis translation in the way.
    Translate it afterwards; ``hardware`` is used only to validate the schedule and to
    size the register.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp

    from .schedule import validate as validate_schedule

    n = problem.n
    if validate:
        validate_schedule(schedule, n, hardware, k=k)

    executed = schedule.executed_pairs()
    if len(set(executed)) != len(executed):
        raise ValueError("the schedule executes a pair twice")
    kept = sorted(set(executed))

    one = {i: Parameter(f"theta_z[{i}]") for i in range(n)}
    two = {pair: Parameter(f"theta_zz[{pair[0]},{pair[1]}]") for pair in kept}

    register = QuantumRegister(hardware.num_qubits, "q")
    circuit = QuantumCircuit(register, name=name)

    layout = dict(schedule.initial_layout)
    input_layout = [layout[q] for q in range(n)]

    for qubit in range(n):
        circuit.rz(rz_angle(one[qubit]), layout[qubit])

    for round_ in schedule.rounds:
        pairs = list(round_.pairs)
        if pairs:
            # Per round, on this round's own terms. Both matter: a SWAP does not
            # commute with an rzz, so no term may cross a swap layer, and a class that
            # is internally disjoint in one round need not be in the next. Colouring
            # the device instead would size every round's palette by the device's
            # degree rather than its own.
            #
            # Which layout this uses is immaterial -- two terms collide exactly when
            # they share a variable, and a layout is injective, so any consistent one
            # induces the same partition. Physical is what actually collides.
            physical = [(layout[a], layout[b]) for a, b in pairs]
            colour = edge_coloring(physical)
            pairs = [
                pair
                for _, pair in sorted(
                    zip(
                        (colour[frozenset(edge)] for edge in physical),
                        pairs,
                        strict=True,
                    ),
                    key=lambda item: item[0],
                )
            ]
        for a, b in pairs:
            key = (a, b) if a < b else (b, a)
            circuit.rzz(rzz_angle(two[key]), layout[a], layout[b])
        for a, b in round_.swaps:
            circuit.swap(layout[a], layout[b])
            layout[a], layout[b] = layout[b], layout[a]

    output_layout = [layout[q] for q in range(n)]

    # SparsePauliOp.from_sparse_list / from_list reject Parameter coefficients in
    # Qiskit 2.x (coeffs array is cast to numeric at construction time).
    # Build with PauliList + object-dtype coeffs array so that:
    #   1. the coeff array preserves insertion order (one-body first, then two-body),
    #   2. adaptive_qaoa's Ansatz.parameters iterates coeffs in that order, and
    #   3. theta[k] aligns with term k in the circuit's own parameter order.
    from qiskit.quantum_info import PauliList

    pauli_labels: list[str] = []
    coeffs_list: list = []

    for q in range(n):
        label = "I" * q + "Z" + "I" * (n - q - 1)
        pauli_labels.append(label[::-1])  # qubit 0 = rightmost char
        coeffs_list.append(one[q])

    for a, b in kept:
        lo, hi = min(a, b), max(a, b)
        label = "I" * lo + "Z" + "I" * (hi - lo - 1) + "Z" + "I" * (n - hi - 1)
        pauli_labels.append(label[::-1])
        coeffs_list.append(two[(a, b)])

    import numpy as _np
    coeffs_arr = _np.empty(len(coeffs_list), dtype=object)
    for _i, _c in enumerate(coeffs_list):
        coeffs_arr[_i] = _c
    generator = SparsePauliOp(PauliList(pauli_labels), coeffs=coeffs_arr, copy=False)

    mask = np.zeros((n, n), dtype=bool)
    for a, b in kept:
        mask[a, b] = mask[b, a] = True

    metadata = {
        "problem_label": problem.label,
        "num_problem_qubits": n,
        "device": getattr(hardware, "name", None),
        "device_qubits": hardware.num_qubits,
        "k_swap_layers_allowed": k,
        "swap_layers_used": schedule.num_swap_layers,
        "num_swaps": schedule.num_swaps,
        "num_terms_kept": len(kept),
        "num_terms_total": problem.num_pairs,
        "covered_j2": problem.j_squared(kept),
        "total_j2": problem.total_j_squared,
        "covered_j2_fraction": problem.covered_fraction(kept),
        "convention": "U(theta) = exp(-i G/2), G = sum_alpha theta_alpha Z^alpha",
        "producer": schedule.info.get("producer"),
    }
    metadata.update(problem.metadata.get("ansatz_metadata", {}))

    return Ansatz(
        circuit=circuit,
        generator=generator,
        input_layout=input_layout,
        output_layout=output_layout,
        backend=backend,
        mask=mask,
        metadata=metadata,
    )


def angles_from_problem(
    ansatz: Ansatz, problem, gamma: float = 1.0
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """The ``(one_body, two_body)`` theta values realising ``exp(-i gamma H_C)``.

    Applies ``theta_alpha = 2 * gamma * coeff_alpha`` per :mod:`swapnet.conventions`, with
    ``coeff`` the problem's own ``h_i`` and ``J_ij``, restricted to the terms the ansatz
    kept. Pass the results straight to :meth:`Ansatz.bind`.

    CAUTION. The dropped terms are simply absent. That is what :attr:`Ansatz.mask` is for:
    an energy computed from this generator differs from one computed from the full ``J``
    by exactly the dropped mass, and that difference is a result, not an error.
    """
    fields = problem.fields
    one = {
        q: gamma_to_theta(0.0 if fields is None else float(fields[q]), gamma)
        for q in range(problem.n)
    }
    two = {
        pair: gamma_to_theta(float(problem.couplings[pair]), gamma)
        for pair in ansatz.couplings
    }
    return one, two


def verify_structure(
    ansatz: Ansatz,
    one_body: Mapping[int, float],
    two_body: Mapping[tuple[int, int], float],
    tolerance: float = 1e-12,
) -> dict:
    """Check the circuit's *structure* by walking it, at any size.

    Replays the circuit while tracking the layout, and confirms that every ``rz`` on
    problem qubit ``i`` carries ``rz_angle(one_body[i])``, every ``rzz`` on the pair
    ``(i, j)`` carries ``rzz_angle(two_body[i, j])``, every term appears exactly once, and
    the SWAP network really lands on :attr:`Ansatz.output_layout`.

    CAUTION, AND IT MATTERS. This does **not** verify the angle convention. It computes
    what it expects using the same :mod:`swapnet.conventions` helpers the builder used, so
    the two move together: hand it values already off by a factor and it passes. What it
    catches is a *structural* fault -- a term on the wrong qubits, a missing or duplicated
    term, a layout that drifts from the advertised one -- which is what can go wrong at
    n=100 and cannot be checked with ``expm``.

    The convention itself is pinned only by ``tests/test_ansatz.py``, which compares a
    bound circuit against ``expm(-i G / 2)`` computed independently of
    :mod:`swapnet.conventions`, and includes a negative control. Those are the tests that
    fail if the factor is wrong; keep them.

    Returns a summary. Raises ``AssertionError`` on the first disagreement -- as an
    explicit ``raise``, never a bare ``assert``, because ``python -O`` strips asserts and
    a verification routine that silently becomes a no-op is worse than none at all.
    """
    bound = ansatz.bind(one_body, two_body)

    # device qubit -> problem qubit, at the start. Every device qubit must be a key, not
    # only the ones holding a problem qubit: a schedule routes *through* idle qubits, so a
    # swap can touch one that carries nothing. None marks those.
    occupant: dict[int, int | None] = dict.fromkeys(range(ansatz.num_device_qubits))
    for problem_q, device_q in enumerate(ansatz.input_layout):
        occupant[device_q] = problem_q
    seen_one: set[int] = set()
    seen_two: set[tuple[int, int]] = set()

    for instruction in bound.data:
        name = instruction.operation.name
        qubits = [bound.find_bit(q).index for q in instruction.qubits]

        if name == "rz":
            (device_q,) = qubits
            problem_q = occupant[device_q]
            if problem_q is None:
                raise AssertionError(
                    f"an rz landed on device qubit {device_q}, which holds no problem "
                    "qubit at that point"
                )
            want = float(rz_angle(one_body[problem_q]))
            got = float(instruction.operation.params[0])
            if abs(got - want) > tolerance:
                raise AssertionError(
                    f"rz on problem qubit {problem_q} carries {got}, expected {want}"
                )
            seen_one.add(problem_q)

        elif name == "rzz":
            a, b = occupant[qubits[0]], occupant[qubits[1]]
            if a is None or b is None:
                raise AssertionError(
                    f"an rzz landed on device qubits {qubits}, one of which holds no "
                    "problem qubit at that point"
                )
            key = (a, b) if a < b else (b, a)
            want = float(rzz_angle(two_body[key]))
            got = float(instruction.operation.params[0])
            if abs(got - want) > tolerance:
                raise AssertionError(f"rzz on pair {key} carries {got}, expected {want}")
            seen_two.add(key)

        elif name == "swap":
            first, second = qubits
            occupant[first], occupant[second] = occupant[second], occupant[first]

        else:
            raise AssertionError(f"unexpected gate {name!r} in the ansatz circuit")

    if seen_one != set(range(ansatz.num_problem_qubits)):
        raise AssertionError("not every problem qubit received a one-body term")
    if seen_two != set(ansatz.couplings):
        raise AssertionError("the circuit's two-body terms differ from the generator's")

    final: list[int | None] = [None] * ansatz.num_problem_qubits
    for device_q, problem_q in occupant.items():
        if problem_q is not None and problem_q < ansatz.num_problem_qubits:
            final[problem_q] = device_q
    if final != list(ansatz.output_layout):
        raise AssertionError(
            "replaying the SWAP network does not reach the advertised output_layout"
        )

    return {
        "num_rz": len(seen_one),
        "num_rzz": len(seen_two),
        "output_layout_matches": True,
    }
