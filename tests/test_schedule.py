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

"""The Schedule artifact: the five rules, truncation, and the neutral producer contract.

Each rule gets a test that *violates* it, because a validator nobody has seen reject
anything is not evidence of anything.
"""

from __future__ import annotations

import json

import pytest
from conftest import line_hardware

from swapnet.schedule import Round, Schedule, ScheduleError, validate


def line_schedule() -> Schedule:
    """A valid 4-variable schedule on the line 0-1-2-3."""
    return Schedule(
        initial_layout={i: i for i in range(4)},
        rounds=[
            Round(pairs=[(0, 1), (1, 2), (2, 3)], swaps=[(0, 1), (2, 3)]),
            Round(pairs=[(0, 3)]),
        ],
    )


def test_a_valid_schedule_validates_and_reports_its_replay():
    hardware = line_hardware(4)
    summary = validate(line_schedule(), 4, hardware)
    assert summary["num_executed_pairs"] == 4
    assert summary["num_swap_layers"] == 1
    assert summary["num_swaps"] == 2
    assert summary["final_layout"] == {0: 1, 1: 0, 2: 3, 3: 2}
    assert summary["executed_pairs"] == [(0, 1), (0, 3), (1, 2), (2, 3)]


# -- rule 1: the layout ------------------------------------------------------


def test_rule1_every_variable_must_be_placed():
    hardware = line_hardware(4)
    schedule = Schedule(initial_layout={0: 0, 1: 1, 2: 2}, rounds=[])
    with pytest.raises(ScheduleError, match="no qubit for variables"):
        validate(schedule, 4, hardware)


def test_rule1_the_layout_must_be_injective():
    hardware = line_hardware(4)
    schedule = Schedule(initial_layout={0: 0, 1: 1, 2: 1, 3: 3}, rounds=[])
    with pytest.raises(ScheduleError, match="same qubit"):
        validate(schedule, 4, hardware)


def test_rule1_the_layout_must_stay_on_the_device():
    hardware = line_hardware(4)
    schedule = Schedule(initial_layout={0: 0, 1: 1, 2: 2, 3: 99}, rounds=[])
    with pytest.raises(ScheduleError, match="outside"):
        validate(schedule, 4, hardware)


# -- rule 2: gates on live links --------------------------------------------


def test_rule2_a_pair_must_sit_on_a_live_link_when_it_executes():
    hardware = line_hardware(4)
    schedule = Schedule(
        initial_layout={i: i for i in range(4)}, rounds=[Round(pairs=[(0, 2)])]
    )  # qubits 0 and 2 are not adjacent
    with pytest.raises(ScheduleError, match="not a live link"):
        validate(schedule, 4, hardware)


def test_rule2_is_checked_at_the_moment_of_execution_not_at_the_start():
    """A pair illegal at the initial layout can become legal after a swap.

    Swapping variables 0 and 1 leaves qubit 1 holding variable 0, so the middle link now
    holds the pair (0, 2) -- which was two hops apart at the initial layout.
    """
    hardware = line_hardware(4)
    ok = Schedule(
        initial_layout={i: i for i in range(4)},
        rounds=[Round(swaps=[(0, 1)]), Round(pairs=[(0, 2)])],
    )
    validate(ok, 4, hardware)

    # The same pair without the swap is two hops apart and must be rejected.
    bad = Schedule(
        initial_layout={i: i for i in range(4)}, rounds=[Round(pairs=[(0, 2)])]
    )
    with pytest.raises(ScheduleError, match="not a live link"):
        validate(bad, 4, hardware)


def test_a_self_pair_is_rejected():
    hardware = line_hardware(4)
    schedule = Schedule(
        initial_layout={i: i for i in range(4)}, rounds=[Round(pairs=[(1, 1)])]
    )
    with pytest.raises(ScheduleError, match="self-pair"):
        validate(schedule, 4, hardware)


def test_a_pair_outside_the_variable_range_is_rejected():
    hardware = line_hardware(6)
    schedule = Schedule(
        initial_layout={i: i for i in range(6)}, rounds=[Round(pairs=[(0, 4)])]
    )
    with pytest.raises(ScheduleError, match="not a pair of problem variables"):
        validate(schedule, 4, hardware)


# -- rule 3: swaps live and disjoint ----------------------------------------


def test_rule3_a_swap_must_sit_on_a_live_link():
    hardware = line_hardware(4)
    schedule = Schedule(
        initial_layout={i: i for i in range(4)}, rounds=[Round(swaps=[(0, 2)])]
    )
    with pytest.raises(ScheduleError, match="not a live link"):
        validate(schedule, 4, hardware)


def test_rule3_one_rounds_swaps_must_be_qubit_disjoint():
    """A round is one parallel layer; sharing a qubit means it is not."""
    hardware = line_hardware(4)
    # Both swaps are on live links: (0,1) uses qubits 0-1, then variable 0 sits on qubit 1
    # so (0,2) uses qubits 1-2. They share qubit 1, so the round is not one layer.
    schedule = Schedule(
        initial_layout={i: i for i in range(4)}, rounds=[Round(swaps=[(0, 1), (0, 2)])]
    )
    with pytest.raises(ScheduleError, match="not parallel"):
        validate(schedule, 4, hardware)


def test_rule3_a_swap_may_name_an_idle_qubit():
    """Routing swaps *through* idle qubits, so their labels are legal."""
    hardware = line_hardware(6)
    schedule = Schedule(
        initial_layout={i: i for i in range(6)},
        rounds=[Round(pairs=[(0, 1)], swaps=[(3, 4)])],
    )
    summary = validate(schedule, 3, hardware)
    assert summary["final_layout"][3] == 4 and summary["final_layout"][4] == 3


def test_rule3_a_swap_naming_an_unplaced_label_is_rejected():
    hardware = line_hardware(4)
    schedule = Schedule(
        initial_layout={i: i for i in range(4)}, rounds=[Round(swaps=[(0, 9)])]
    )
    with pytest.raises(ScheduleError, match="not placed"):
        validate(schedule, 4, hardware)


# -- rule 4: each term once --------------------------------------------------


def test_rule4_a_pair_cannot_execute_twice():
    hardware = line_hardware(4)
    schedule = Schedule(
        initial_layout={i: i for i in range(4)},
        rounds=[Round(pairs=[(0, 1)]), Round(pairs=[(1, 0)])],
    )
    with pytest.raises(ScheduleError, match="executed twice"):
        validate(schedule, 4, hardware)


# -- rule 5: the budget ------------------------------------------------------


def test_rule5_the_swap_layer_count_is_capped_by_k():
    hardware = line_hardware(4)
    # After swapping (0,1), variable 0 is on qubit 1, so (0,2) spans the live middle link.
    schedule = Schedule(
        initial_layout={i: i for i in range(4)},
        rounds=[Round(swaps=[(0, 1)]), Round(swaps=[(0, 2)])],
    )
    validate(schedule, 4, hardware, k=2)
    with pytest.raises(ScheduleError, match="budget is k=1"):
        validate(schedule, 4, hardware, k=1)


def test_a_round_with_no_swaps_costs_no_budget():
    """Executing terms is free; only parallel swap layers are the budget."""
    hardware = line_hardware(4)
    schedule = Schedule(
        initial_layout={i: i for i in range(4)},
        rounds=[Round(pairs=[(0, 1)]), Round(pairs=[(1, 2)]), Round(pairs=[(2, 3)])],
    )
    validate(schedule, 4, hardware, k=0)


# -- truncation --------------------------------------------------------------


def test_truncate_keeps_the_pairs_of_the_round_whose_swaps_it_cuts():
    schedule = Schedule(
        initial_layout={i: i for i in range(4)},
        rounds=[
            Round(pairs=[(0, 1)], swaps=[(0, 1)]),
            Round(pairs=[(1, 2)], swaps=[(1, 2)]),
            Round(pairs=[(2, 3)]),
        ],
    )
    cut = schedule.truncate(1)
    assert cut.num_swap_layers == 1
    assert [r.pairs for r in cut.rounds] == [[(0, 1)], [(1, 2)]]
    assert [r.swaps for r in cut.rounds] == [[(0, 1)], []]  # round 1's pairs survive


def test_truncate_to_zero_keeps_the_first_rounds_pairs_only():
    schedule = Schedule(
        initial_layout={i: i for i in range(4)},
        rounds=[Round(pairs=[(0, 1)], swaps=[(0, 1)]), Round(pairs=[(1, 2)])],
    )
    cut = schedule.truncate(0)
    assert cut.num_swap_layers == 0
    assert cut.executed_pairs() == [(0, 1)]


def test_truncate_beyond_the_plan_is_the_whole_plan():
    schedule = line_schedule()
    assert schedule.truncate(99).content() == schedule.content()


def test_truncate_rejects_a_negative_budget():
    with pytest.raises(ValueError, match="non-negative"):
        line_schedule().truncate(-1)


def test_truncation_is_nested():
    """truncate(a).truncate(b) == truncate(min(a,b)) -- what makes a sweep sound."""
    schedule = Schedule(
        initial_layout={i: i for i in range(4)},
        rounds=[
            Round(pairs=[(0, 1)], swaps=[(0, 1)]),
            Round(pairs=[(1, 2)], swaps=[(1, 2)]),
            Round(pairs=[(2, 3)], swaps=[(2, 3)]),
            Round(pairs=[(0, 3)]),
        ],
    )
    for a in range(4):
        for b in range(4):
            assert (
                schedule.truncate(a).truncate(b).content()
                == schedule.truncate(min(a, b)).content()
            )


# -- plumbing ----------------------------------------------------------------


def test_round_trips_through_json():
    schedule = line_schedule()
    schedule.info["producer"] = "test"
    again = Schedule.from_dict(json.loads(json.dumps(schedule.to_dict())))
    assert again.to_dict() == schedule.to_dict()
    assert again.initial_layout == schedule.initial_layout
    assert again.info["producer"] == "test"


def test_executed_pairs_are_normalised_and_in_order():
    schedule = Schedule(
        initial_layout={i: i for i in range(4)},
        rounds=[Round(pairs=[(1, 0)]), Round(pairs=[(3, 2)])],
    )
    assert schedule.executed_pairs() == [(0, 1), (2, 3)]


def test_a_hand_written_producer_satisfies_the_contract():
    """The point of the artifact: something that is not swapnet can produce one.

    This is a complete linear-swap-network round, written by hand, and it passes the same
    validator the swapnet output does.
    """
    n = 6
    hardware = line_hardware(n)
    # Odd-even transposition. Round 0 takes the even links and swaps across them, which
    # leaves qubits holding (1,0,3,2,5,4). The odd links 1-2 and 3-4 then hold the
    # variable pairs (0,3) and (2,5) -- named by variable, not by qubit.
    rounds = [
        Round(pairs=[(0, 1), (2, 3), (4, 5)], swaps=[(0, 1), (2, 3), (4, 5)]),
        Round(pairs=[(0, 3), (2, 5)], swaps=[(0, 3), (2, 5)]),
    ]
    schedule = Schedule(
        initial_layout={i: i for i in range(n)},
        rounds=rounds,
        info={"producer": "odd-even transposition, by hand"},
    )
    summary = validate(schedule, n, hardware, k=2)
    assert summary["num_executed_pairs"] == 5
    assert summary["num_swap_layers"] == 2
