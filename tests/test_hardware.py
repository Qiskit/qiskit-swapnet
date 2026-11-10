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

"""The Hardware artifact: live links, the planning graph, and W_ij."""

from __future__ import annotations

import json

import pytest
from conftest import line_hardware

from swapnet.hardware import Hardware


def test_from_error_map_builds_a_line():
    hardware = line_hardware(5, error=0.01)
    assert hardware.num_qubits == 5
    assert hardware.num_links == 4
    assert hardware.usable_qubits == [0, 1, 2, 3, 4]
    assert hardware.unusable_qubits == []
    assert hardware.is_live(1, 2) and hardware.is_live(2, 1)
    assert not hardware.is_live(0, 2)


def test_links_priced_at_or_above_one_are_dead_not_an_error():
    hardware = Hardware.from_error_map(
        {(0, 1): 0.01, (1, 2): 1.0, (2, 3): 0.02}, num_qubits=4
    )
    assert hardware.dead_links == ((1, 2),)
    assert hardware.num_links == 2
    # The dead link splits the line, so the planning graph is the larger side only.
    assert hardware.unusable_qubits in ([0, 1], [2, 3])
    assert len(hardware.usable_qubits) == 2


def test_a_direct_error_above_one_cannot_survive_via_the_other_direction():
    """Both orientations of one link may be reported; dead in either means dead."""
    hardware = Hardware.from_error_map(
        {(0, 1): 0.01, (1, 0): 1.0, (1, 2): 0.02}, num_qubits=3
    )
    assert (0, 1) in hardware.dead_links
    assert not hardware.is_live(0, 1)


def test_the_constructor_rejects_an_error_that_is_not_a_live_rate():
    with pytest.raises(ValueError, match="not a live error rate"):
        Hardware(name="x", num_qubits=2, errors={(0, 1): 1.5})
    with pytest.raises(ValueError, match="no live links at all"):
        Hardware(name="x", num_qubits=2, errors={})


def test_edge_weights_are_mean_one_over_the_planning_graph():
    hardware = Hardware.from_error_map(
        {(0, 1): 0.001, (1, 2): 0.05, (2, 3): 0.01}, num_qubits=4
    )
    weights = hardware.edge_weights
    assert len(weights) == 3
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)
    # Lower error must be cheaper to traverse: that is the whole point of W_ij.
    assert weights[(0, 1)] < weights[(2, 3)] < weights[(1, 2)]


def test_edge_weights_cover_exactly_the_planning_graph():
    """A weight outside `graph` would be silently unused; a missing one raises."""
    hardware = Hardware.from_error_map(
        {(0, 1): 0.01, (1, 2): 1.0, (2, 3): 0.02}, num_qubits=4
    )
    assert set(hardware.edge_weights) == {
        (min(a, b), max(a, b)) for a, b in hardware.graph.edges()
    }


def test_check_capacity_names_the_dead_links_that_caused_it():
    hardware = Hardware.from_error_map(
        {(0, 1): 0.01, (1, 2): 1.0, (2, 3): 0.02}, num_qubits=4
    )
    hardware.check_capacity(2)
    with pytest.raises(ValueError, match=r"needs 3.*Dead links"):
        hardware.check_capacity(3)


def test_round_trips_through_json():
    hardware = line_hardware(6)
    again = Hardware.from_dict(json.loads(json.dumps(hardware.to_dict())))
    assert again.to_dict() == hardware.to_dict()
    assert again.errors == hardware.errors


# -- against a real target ---------------------------------------------------


def test_from_backend_reads_a_fake_boston_target(fake_hardware):
    hardware = fake_hardware
    assert hardware.name == "fake_boston"
    assert hardware.num_qubits == 156
    # 176 couplings, of which one is dead for cz.
    assert hardware.num_links == 175
    assert hardware.dead_links == ((84, 85),)
    assert hardware.priced_gates == ("cz",)
    assert hardware.unusable_qubits == []  # the dead link isolates nobody
    assert sum(hardware.edge_weights.values()) / hardware.num_links == pytest.approx(1.0)


def test_from_backend_rejects_something_without_a_target():
    with pytest.raises(TypeError, match=r"has no \.target"):
        Hardware.from_backend(object())


def test_from_target_rejects_a_gate_the_target_does_not_have():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeBoston

    target = FakeBoston().target
    with pytest.raises(ValueError, match=r"no two-qubit gate \['rzz'\]"):
        Hardware.from_target(target, gates=["rzz"])


def test_isolated_qubits_are_reported_not_dropped():
    """FakeTorino's dead links cut 4 qubits off entirely. They must be visible."""
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    hardware = Hardware.from_backend(FakeTorino())
    assert hardware.unusable_qubits, "expected some qubits outside the largest component"
    assert (
        len(hardware.usable_qubits) + len(hardware.unusable_qubits) == hardware.num_qubits
    )
    # The planning graph is connected, which is what routing requires.
    import networkx as nx

    assert nx.is_connected(hardware.graph)
