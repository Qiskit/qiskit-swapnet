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

"""The convention, checked against expm(-i G / 2) entry by entry.

The contract is ``U(theta) = exp(-i G(theta)/2)`` for
``G(theta) = sum_alpha theta_alpha Z^alpha``. A wrong factor still produces a running
circuit and a converging optimiser, so only an explicit comparison catches it. These tests
are small enough to build the full unitary, and they compute the expectation **without**
using :mod:`swapnet.conventions` -- otherwise they would move with the bug.

``test_a_rescaled_theta_is_caught`` is the negative control. Without it, a test that only
ever confirms agreement proves nothing about sensitivity.

The circuit contains a SWAP network, so as an operator on device qubits it is the
exponential *followed by* the relabelling the network performs. Both facts are checked:
the phase on every basis state, and the permutation.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import line_hardware

import swapnet
from swapnet import ansatz as ansatz_mod
from swapnet.problem import Problem
from swapnet.schedule import Round, Schedule


def small_problem(n: int = 4, seed: int = 7) -> Problem:
    rng = np.random.default_rng(seed)
    couplings = {(i, j): float(rng.normal()) for i in range(n) for j in range(i + 1, n)}
    return Problem(n=n, couplings=couplings, fields=rng.normal(size=n), label="small")


def generator_diagonal(one_body, two_body, n: int) -> np.ndarray:
    """The diagonal of ``G = sum theta_i Z_i + sum theta_ij Z_i Z_j``.

    Basis index ``b`` has bit ``q`` of qubit ``q`` as ``(b >> q) & 1``, matching Qiskit's
    little-endian ordering, and ``z_q = +1`` for bit 0.
    """
    diagonal = np.zeros(1 << n, dtype=float)
    for basis in range(1 << n):
        z = [1.0 - 2.0 * ((basis >> q) & 1) for q in range(n)]
        value = sum(one_body[q] * z[q] for q in one_body)
        value += sum(w * z[i] * z[j] for (i, j), w in two_body.items())
        diagonal[basis] = value
    return diagonal


def test_the_convention_is_exp_minus_i_theta_over_two():
    """rz(t) is exp(-i t Z/2) and rzz(t) is exp(-i t ZZ/2). Everything builds on this."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from scipy.linalg import expm

    z = np.diag([1.0, -1.0]).astype(complex)
    circuit = QuantumCircuit(1)
    circuit.rz(swapnet.rz_angle(0.37), 0)
    assert np.allclose(Operator(circuit).data, expm(-1j * 0.37 * z / 2))

    zz = np.diag([1.0, -1.0, -1.0, 1.0]).astype(complex)
    circuit = QuantumCircuit(2)
    circuit.rzz(swapnet.rzz_angle(0.21), 0, 1)
    assert np.allclose(Operator(circuit).data, expm(-1j * 0.21 * zz / 2))


def test_the_angle_helpers_are_the_identity():
    """Stated explicitly, because the whole point is that there is no factor to lose."""
    assert swapnet.rz_angle(0.37) == 0.37
    assert swapnet.rzz_angle(-1.25) == -1.25
    # And the QAOA relation carries the 2, so a caller never writes it themselves.
    assert swapnet.gamma_to_theta(0.5, gamma=0.3) == pytest.approx(2 * 0.3 * 0.5)


def test_bound_circuit_equals_exp_minus_i_g_over_two_with_no_swaps():
    """With no SWAP network the circuit is exactly exp(-i G/2) on the mapped qubits."""
    from qiskit.quantum_info import Operator

    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    pairs = [(i, i + 1) for i in range(n - 1)]  # what a line supports at identity layout
    schedule = Schedule(
        initial_layout={i: i for i in range(n)}, rounds=[Round(pairs=pairs)]
    )

    ansatz = ansatz_mod.build(schedule, problem, hardware)
    assert ansatz.input_layout == ansatz.output_layout  # no swaps

    rng = np.random.default_rng(1)
    one = {q: float(v) for q, v in zip(range(n), rng.normal(size=n), strict=True)}
    two = {
        p: float(v)
        for p, v in zip(
            ansatz.couplings, rng.normal(size=len(ansatz.couplings)), strict=True
        )
    }

    got = Operator(ansatz.bind(one, two)).data
    want = np.diag(np.exp(-0.5j * generator_diagonal(one, two, n)))
    assert np.allclose(got, want), "the bound circuit is not exp(-i G/2)"


def test_a_rescaled_theta_is_caught():
    """The negative control: agreement must be sensitive to the factor.

    A test that only ever confirms the builder against itself would pass with any
    convention. This one asserts that doubling theta -- the exact error a consumer makes
    by applying a half-angle factor twice -- produces a *different* unitary.
    """
    from qiskit.quantum_info import Operator

    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    schedule = Schedule(
        initial_layout={i: i for i in range(n)},
        rounds=[Round(pairs=[(i, i + 1) for i in range(n - 1)])],
    )
    ansatz = ansatz_mod.build(schedule, problem, hardware)

    one = dict.fromkeys(range(n), 0.3)
    two = dict.fromkeys(ansatz.couplings, 0.4)
    want = np.diag(np.exp(-0.5j * generator_diagonal(one, two, n)))

    doubled_one = {q: 2 * v for q, v in one.items()}
    doubled_two = {p: 2 * v for p, v in two.items()}
    got = Operator(ansatz.bind(doubled_one, doubled_two)).data
    assert not np.allclose(got, want), (
        "doubling theta produced the same unitary, so this comparison cannot detect a "
        "wrong half-angle factor and proves nothing"
    )


def test_bound_circuit_is_exp_then_the_swap_relabelling():
    """With a SWAP network: the same phases, composed with the layout permutation."""
    from qiskit.quantum_info import Operator

    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    # Execute the neighbours, then one disjoint SWAP layer. That layer sends the layout to
    # 0->1, 1->0, 2->3, 3->2, so device link (1,2) now holds problem qubits 0 and 3: the
    # one newly available pair is (0,3), not (0,2).
    schedule = Schedule(
        initial_layout={i: i for i in range(n)},
        rounds=[
            Round(pairs=[(0, 1), (1, 2), (2, 3)], swaps=[(0, 1), (2, 3)]),
            Round(pairs=[(0, 3)]),
        ],
    )
    ansatz = ansatz_mod.build(schedule, problem, hardware)
    assert ansatz.input_layout == [0, 1, 2, 3]
    assert ansatz.output_layout == [1, 0, 3, 2]
    assert ansatz.couplings == ((0, 1), (0, 3), (1, 2), (2, 3))

    rng = np.random.default_rng(2)
    one = {q: float(v) for q, v in zip(range(n), rng.normal(size=n), strict=True)}
    two = {
        p: float(v)
        for p, v in zip(
            ansatz.couplings, rng.normal(size=len(ansatz.couplings)), strict=True
        )
    }
    got = Operator(ansatz.bind(one, two)).data

    # The phase on a basis state depends on the problem-qubit values, which sit on device
    # qubits via input_layout. The output sits on them via output_layout.
    phases = np.zeros(1 << n, dtype=float)
    permutation = np.zeros(1 << n, dtype=int)
    for basis in range(1 << n):
        z = {q: 1.0 - 2.0 * ((basis >> ansatz.input_layout[q]) & 1) for q in range(n)}
        phases[basis] = sum(one[q] * z[q] for q in range(n)) + sum(
            w * z[i] * z[j] for (i, j), w in two.items()
        )
        out = 0
        for q in range(n):
            if z[q] < 0:  # this problem qubit is |1>
                out |= 1 << ansatz.output_layout[q]
        permutation[basis] = out

    want = np.zeros_like(got)
    for basis in range(1 << n):
        want[permutation[basis], basis] = np.exp(-0.5j * phases[basis])
    assert np.allclose(got, want), "the circuit is not exp(-i G/2) then the relabelling"


def test_angles_from_problem_realises_exp_minus_i_gamma_h():
    """The QAOA relation: binding angles_from_problem gives exp(-i gamma H_C) exactly."""
    from qiskit.quantum_info import Operator

    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    # A schedule covering every pair, so H_C here is the full problem with nothing
    # dropped. On the line 0-1-2-3: the identity layout supports the 3 neighbours;
    # swapping (0,1) and (2,3) puts variables 0 and 3 on the middle link; swapping those
    # two then brings (0,2) and (1,3) together on the two end links.
    schedule = Schedule(
        initial_layout={i: i for i in range(n)},
        rounds=[
            Round(pairs=[(0, 1), (1, 2), (2, 3)], swaps=[(0, 1), (2, 3)]),
            Round(pairs=[(0, 3)], swaps=[(0, 3)]),
            Round(pairs=[(0, 2), (1, 3)]),
        ],
    )
    ansatz = ansatz_mod.build(schedule, problem, hardware)
    assert set(ansatz.couplings) == set(problem.couplings), "expected full coverage"

    gamma = 0.23
    one, two = ansatz_mod.angles_from_problem(ansatz, problem, gamma)
    got = Operator(ansatz.bind(one, two)).data

    # exp(-i gamma H_C), built from the problem's own J and h, then the relabelling.
    want = np.zeros_like(got)
    for basis in range(1 << n):
        z = {q: 1.0 - 2.0 * ((basis >> ansatz.input_layout[q]) & 1) for q in range(n)}
        energy = sum(float(problem.fields[q]) * z[q] for q in range(n)) + sum(
            v * z[i] * z[j] for (i, j), v in problem.couplings.items()
        )
        out = 0
        for q in range(n):
            if z[q] < 0:
                out |= 1 << ansatz.output_layout[q]
        want[out, basis] = np.exp(-1j * gamma * energy)
    assert np.allclose(got, want), "binding angles_from_problem is not exp(-i gamma H_C)"


def test_generator_is_read_off_not_supplied():
    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    schedule = Schedule(
        initial_layout={i: i for i in range(n)},
        rounds=[Round(pairs=[(0, 1), (1, 2), (2, 3)])],
    )
    ansatz = ansatz_mod.build(schedule, problem, hardware)

    assert ansatz.couplings == ((0, 1), (1, 2), (2, 3))
    assert sorted(ansatz.theta_one_body) == list(range(n))
    assert set(ansatz.theta_two_body) == set(ansatz.couplings)
    # The generator's coefficients are the circuit's own parameters, not copies.
    assert set(ansatz.circuit.parameters) == (
        set(ansatz.theta_one_body.values()) | set(ansatz.theta_two_body.values())
    )


def test_mask_marks_exactly_the_kept_terms():
    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    kept = [(0, 1), (1, 2), (2, 3)]
    schedule = Schedule(
        initial_layout={i: i for i in range(n)}, rounds=[Round(pairs=kept)]
    )
    ansatz = ansatz_mod.build(schedule, problem, hardware)

    assert ansatz.mask.shape == (n, n)
    assert np.array_equal(ansatz.mask, ansatz.mask.T)
    assert not ansatz.mask.diagonal().any()
    assert {
        tuple(sorted(p)) for p in zip(*np.where(np.triu(ansatz.mask)), strict=True)
    } == set(kept)
    assert ansatz.dropped_pairs(problem.couplings) == ((0, 2), (0, 3), (1, 3))
    assert ansatz.metadata["covered_j2_fraction"] == pytest.approx(
        problem.covered_fraction(kept)
    )


def test_incomplete_binding_is_an_error_not_an_implicit_zero():
    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    schedule = Schedule(
        initial_layout={i: i for i in range(n)}, rounds=[Round(pairs=[(0, 1), (1, 2)])]
    )
    ansatz = ansatz_mod.build(schedule, problem, hardware)

    with pytest.raises(ValueError, match="left unbound"):
        ansatz.bind(one_body=dict.fromkeys(range(n), 0.1), two_body={(0, 1): 0.2})

    with pytest.raises(KeyError, match="dropped it"):
        ansatz.bind(
            one_body=dict.fromkeys(range(n), 0.1),
            two_body={(0, 1): 0.2, (1, 2): 0.3, (0, 3): 0.4},
        )


def test_layouts_accept_a_mapping_and_store_a_list():
    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    schedule = Schedule(
        initial_layout={i: i for i in range(n)}, rounds=[Round(pairs=[(0, 1)])]
    )
    ansatz = ansatz_mod.build(schedule, problem, hardware)
    rebuilt = ansatz_mod.Ansatz(
        circuit=ansatz.circuit,
        generator=ansatz.generator,
        input_layout={q: ansatz.input_layout[q] for q in range(n)},
        output_layout={q: ansatz.output_layout[q] for q in range(n)},
    )
    assert rebuilt.input_layout == ansatz.input_layout
    assert isinstance(rebuilt.input_layout, list)


def test_verify_structure_catches_a_structural_fault_but_not_the_convention():
    """State the tool's reach explicitly, so nobody mistakes it for the expm check."""
    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    schedule = Schedule(
        initial_layout={i: i for i in range(n)},
        rounds=[Round(pairs=[(0, 1), (1, 2), (2, 3)])],
    )
    ansatz = ansatz_mod.build(schedule, problem, hardware)
    one = dict.fromkeys(range(n), 0.3)
    two = dict.fromkeys(ansatz.couplings, 0.4)

    assert ansatz_mod.verify_structure(ansatz, one, two)["num_rzz"] == 3

    # It is self-consistent by construction, so a globally rescaled theta still passes.
    # Only the expm comparisons above pin the factor.
    assert ansatz_mod.verify_structure(
        ansatz, {q: 2 * v for q, v in one.items()}, {p: 2 * v for p, v in two.items()}
    )

    # A structural fault IS caught: claim an output layout the network does not reach.
    broken = ansatz_mod.Ansatz(
        circuit=ansatz.circuit,
        generator=ansatz.generator,
        input_layout=ansatz.input_layout,
        output_layout=list(reversed(ansatz.output_layout)),
        mask=ansatz.mask,
    )
    with pytest.raises(AssertionError, match="output_layout"):
        ansatz_mod.verify_structure(broken, one, two)


def test_swaps_through_spare_qubits_are_tracked():
    """A schedule may route through qubits that hold no problem variable.

    Real instances hit this immediately -- 40 variables on a 156-qubit device leaves 116
    spare -- and the occupant map has to cover every device qubit, not only the placed
    ones, or the first idle-qubit swap raises KeyError.
    """
    from qiskit.quantum_info import Operator

    n = 3
    hardware = line_hardware(6)  # 3 spare qubits on the far end of the line
    problem = small_problem(n)

    # Variables on 0,1,2; labels 3,4,5 are the idle qubits. Swapping variable 2 out to
    # qubit 3 puts an idle qubit into the occupied region.
    schedule = Schedule(
        initial_layout={i: i for i in range(6)},
        rounds=[Round(pairs=[(0, 1), (1, 2)], swaps=[(2, 3)])],
    )
    ansatz = ansatz_mod.build(schedule, problem, hardware)

    assert ansatz.input_layout == [0, 1, 2]
    assert ansatz.output_layout == [0, 1, 3]  # variable 2 moved onto an idle qubit

    one = {q: 0.2 * (q + 1) for q in range(n)}
    two = dict.fromkeys(ansatz.couplings, 0.3)
    assert ansatz_mod.verify_structure(ansatz, one, two)["output_layout_matches"]

    # And the circuit is still exp(-i G/2) then the relabelling, on 6 device qubits.
    got = Operator(ansatz.bind(one, two)).data
    want = np.zeros_like(got)
    for basis in range(1 << hardware.num_qubits):
        z = {q: 1.0 - 2.0 * ((basis >> ansatz.input_layout[q]) & 1) for q in range(n)}
        phase = sum(one[q] * z[q] for q in range(n)) + sum(
            w * z[i] * z[j] for (i, j), w in two.items()
        )
        bit2, bit3 = (basis >> 2) & 1, (basis >> 3) & 1
        out = basis & ~((1 << 2) | (1 << 3))
        out |= bit3 << 2
        out |= bit2 << 3
        want[out, basis] = np.exp(-0.5j * phase)
    assert np.allclose(got, want)


def test_pickle_round_trip_preserves_the_contract():
    import pickle

    from qiskit.quantum_info import Operator

    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    schedule = Schedule(
        initial_layout={i: i for i in range(n)},
        rounds=[Round(pairs=[(0, 1), (1, 2), (2, 3)])],
    )
    ansatz = ansatz_mod.build(schedule, problem, hardware)

    revived = pickle.loads(pickle.dumps(ansatz))
    assert revived.couplings == ansatz.couplings
    assert np.array_equal(revived.mask, ansatz.mask)
    assert revived.input_layout == ansatz.input_layout

    one = {q: 0.1 * (q + 1) for q in range(n)}
    two = dict.fromkeys(ansatz.couplings, 0.2)
    assert np.allclose(
        Operator(ansatz.bind(one, two)).data, Operator(revived.bind(one, two)).data
    )


# -- how the terms in a round are ordered ------------------------------------


def test_edge_coloring_is_proper_and_hits_delta_on_bipartite_graphs():
    from swapnet.ansatz import edge_coloring

    cases = [
        ([], 0, None),
        ([(i, i + 1) for i in range(8)], 2, 2),  # path
        ([(0, 1), (0, 2), (0, 3)], 3, 3),  # star
        ([(i, 3 + j) for i in range(3) for j in range(3)], 3, 3),  # K3,3
        ([(0, 1), (1, 2), (2, 3), (3, 0)], 2, 2),  # even cycle
    ]
    for pairs, delta, expected in cases:
        colour = edge_coloring(pairs)
        if not pairs:
            assert colour == {}
            continue
        adjacency: dict[int, set[int]] = {}
        for u, v in pairs:
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)
        assert max(len(w) for w in adjacency.values()) == delta
        # proper: no qubit sees one colour twice
        for node, neighbours in adjacency.items():
            seen = [colour[frozenset((node, w))] for w in neighbours]
            assert len(seen) == len(set(seen)), f"colour repeats at {node}"
        # bipartite, so Delta classes exist and must be attained
        assert max(colour.values()) + 1 == expected


def test_edge_coloring_stays_within_vizing_on_an_odd_cycle():
    """Not bipartite, so Delta classes need not exist; Delta + 1 always does."""
    from swapnet.ansatz import edge_coloring

    triangle = [(0, 1), (1, 2), (2, 0)]
    colour = edge_coloring(triangle)
    assert max(colour.values()) + 1 <= 3  # Delta = 2, so Vizing allows 3
    assert len({colour[frozenset(e)] for e in triangle}) == 3


def test_edge_coloring_ignores_a_repeated_edge():
    from swapnet.ansatz import edge_coloring

    assert len(edge_coloring([(0, 1), (1, 0), (1, 2)])) == 2


def test_reordering_the_terms_leaves_the_unitary_alone():
    """The terms are diagonal, so grouping them by colour cannot change what is computed.

    Compared against the same terms emitted in schedule order, built here by hand: the
    builder has no option to produce that, and the point is to check the reordering rather
    than to offer it.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import Operator

    n = 4
    problem = small_problem(n)
    hardware = line_hardware(n)
    schedule = Schedule(
        initial_layout={i: i for i in range(n)},
        rounds=[Round(pairs=[(0, 1), (1, 2), (2, 3)])],
    )
    built = ansatz_mod.build(schedule, problem, hardware)

    one = {q: 0.3 * (q + 1) for q in range(n)}
    two = dict.fromkeys(built.couplings, 0.4)

    as_scheduled = QuantumCircuit(QuantumRegister(hardware.num_qubits, "q"))
    for qubit in range(n):
        as_scheduled.rz(one[qubit], schedule.initial_layout[qubit])
    for a, b in schedule.rounds[0].pairs:
        key = (a, b) if a < b else (b, a)
        as_scheduled.rzz(two[key], schedule.initial_layout[a], schedule.initial_layout[b])

    assert np.allclose(
        Operator(built.bind(one, two)).data, Operator(as_scheduled).data
    ), "grouping the terms by colour changed the unitary"


def test_a_round_costs_no_more_layers_than_its_maximum_degree():
    """On a path the interaction subgraph has degree 2, so two layers is the floor."""
    n = 6
    problem = small_problem(n)
    hardware = line_hardware(n)
    schedule = Schedule(
        initial_layout={i: i for i in range(n)},
        rounds=[Round(pairs=[(i, i + 1) for i in range(n - 1)])],
    )
    built = ansatz_mod.build(schedule, problem, hardware)
    depth = built.circuit.depth(lambda instruction: len(instruction.qubits) == 2)
    assert depth == 2, f"expected the degree-2 floor, got {depth}"


def test_each_round_is_coloured_on_its_own_terms():
    """Every round is coloured separately, on the terms that round executes.

    Rounds cannot share a colouring. A SWAP does not commute with an ``rzz``, so no term
    may cross a swap layer, and after one the pair set has moved -- a class that was
    internally disjoint in one round need not be in the next.

    Which *layout* the colouring runs against is immaterial, and this test does not probe
    it: two terms in a round collide exactly when they share a variable, and a layout is
    injective, so the logical and physical pair sets induce the same partition.

    Randomised over 25 schedules, because one hand-built case proves little: any order
    gives depth >= Delta, so a wrong colouring can hit the floor by luck. Reusing one
    round's colouring for the rest fails ~6% of rounds here; a fixed example missed it.
    """
    import random

    from qiskit import QuantumCircuit

    width = 14
    hardware = line_hardware(width)
    problem = Problem(
        n=width,
        couplings={
            (i, j): 0.1 + 0.01 * (i * width + j)
            for i in range(width)
            for j in range(i + 1, width)
        },
    )

    def random_schedule(rng, num_rounds=4):
        """A valid multi-round line schedule: terms on live links, disjoint swaps."""
        layout = {v: v for v in range(width)}
        used: set[tuple[int, int]] = set()
        rounds = []
        for _ in range(num_rounds):
            occupant = {q: v for v, q in layout.items()}
            available = []
            for q in range(width - 1):
                a, b = occupant[q], occupant[q + 1]
                key = (a, b) if a < b else (b, a)
                if key not in used:
                    available.append(key)
            rng.shuffle(available)
            pairs = [pair for pair in available if rng.random() < 0.8]
            used.update(pairs)

            touched: set[int] = set()
            swaps = []
            for q in rng.sample(range(width - 1), width - 1):
                if q in touched or q + 1 in touched:
                    continue
                if rng.random() < 0.5:
                    swaps.append((occupant[q], occupant[q + 1]))
                    touched.update({q, q + 1})
            rounds.append(Round(pairs=pairs, swaps=swaps))
            for a, b in swaps:
                layout[a], layout[b] = layout[b], layout[a]
        return Schedule(initial_layout={v: v for v in range(width)}, rounds=rounds)

    checked = 0
    for seed in range(25):
        schedule = random_schedule(random.Random(seed))
        swapnet.schedule.validate(schedule, width, hardware)
        built = ansatz_mod.build(schedule, problem, hardware)

        # Split the emitted terms at each swap layer: one group per round.
        groups: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        for instruction in built.circuit.data:
            qubits = tuple(built.circuit.find_bit(q).index for q in instruction.qubits)
            if instruction.operation.name == "rzz":
                current.append(qubits)
            elif instruction.operation.name == "swap" and current:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        for edges in groups:
            adjacency: dict[int, set[int]] = {}
            for u, v in edges:
                adjacency.setdefault(u, set()).add(v)
                adjacency.setdefault(v, set()).add(u)
            floor = max(len(w) for w in adjacency.values())
            # Replay just this round's terms, in the order they were emitted.
            isolated = QuantumCircuit(width)
            for u, v in edges:
                isolated.rzz(0.1, u, v)
            depth = isolated.depth(lambda i: len(i.qubits) == 2)
            assert depth == floor, (
                f"seed {seed}: a round of {len(edges)} terms cost {depth} layers "
                f"where its maximum degree is {floor}; colouring is not per-round"
            )
            checked += 1
    assert checked > 50, f"only {checked} rounds exercised"
