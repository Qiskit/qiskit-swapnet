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

"""The verifier must reject tampered plans.

``verify_plan`` is the safety net the rest of the suite leans on, so it needs its
own tests: a verifier that accepts everything would make every other test
vacuous. Each case takes a valid plan and breaks exactly one property.
"""

from __future__ import annotations

import dataclasses

import pytest

import swapnet
from swapnet.results import Stage, Swap
from swapnet.verify import PlanError

networkx = pytest.importorskip("networkx")


@pytest.fixture
def instance(fast_params):
    hardware = networkx.grid_2d_graph(4, 4)
    interaction = networkx.random_regular_graph(3, 12, seed=23)
    plan = swapnet.map_and_route(hardware, interaction, params=fast_params, seed=5)
    assert plan.total_swaps > 0, "need a plan with swaps to tamper with"
    return hardware, interaction, plan


def test_a_valid_plan_passes(instance):
    hardware, interaction, plan = instance
    swapnet.verify_plan(hardware, interaction, plan)


def test_a_swap_on_a_non_coupling_is_caught(instance):
    hardware, interaction, plan = instance

    # Rewrite the first swap to act on two qubits that are not adjacent.
    corner, far = (0, 0), (3, 3)
    index = next(i for i, s in enumerate(plan.stages) if s.swaps)
    stage = plan.stages[index]
    broken = dataclasses.replace(
        stage,
        swaps=(
            Swap(physical=(corner, far), logical=stage.swaps[0].logical),
            *stage.swaps[1:],
        ),
    )
    tampered = dataclasses.replace(
        plan, stages=(*plan.stages[:index], broken, *plan.stages[index + 1 :])
    )

    with pytest.raises(PlanError):
        swapnet.verify_plan(hardware, interaction, tampered)


def test_a_duplicated_swap_within_a_layer_is_caught(instance):
    hardware, interaction, plan = instance

    index = next(i for i, s in enumerate(plan.stages) if s.swaps)
    stage = plan.stages[index]
    doubled = dataclasses.replace(stage, swaps=(*stage.swaps, stage.swaps[0]))
    tampered = dataclasses.replace(
        plan, stages=(*plan.stages[:index], doubled, *plan.stages[index + 1 :])
    )

    with pytest.raises(PlanError):
        swapnet.verify_plan(hardware, interaction, tampered)


def test_a_dropped_swap_breaks_the_final_layout(instance):
    hardware, interaction, plan = instance

    index = next(i for i, s in enumerate(plan.stages) if s.swaps)
    stage = plan.stages[index]
    shortened = dataclasses.replace(stage, swaps=stage.swaps[1:])
    tampered = dataclasses.replace(
        plan,
        stages=(*plan.stages[:index], shortened, *plan.stages[index + 1 :]),
    )

    with pytest.raises(PlanError):
        swapnet.verify_plan(hardware, interaction, tampered)


def test_a_missing_gate_is_caught(instance):
    hardware, interaction, plan = instance

    index = next(i for i, s in enumerate(plan.stages) if s.gates)
    stage = plan.stages[index]
    fewer = dataclasses.replace(stage, gates=stage.gates[1:])
    tampered = dataclasses.replace(
        plan, stages=(*plan.stages[:index], fewer, *plan.stages[index + 1 :])
    )

    with pytest.raises(PlanError, match="unaccounted for"):
        swapnet.verify_plan(hardware, interaction, tampered)


def test_a_gate_that_is_not_an_interaction_is_caught(instance):
    hardware, interaction, plan = instance

    extra = Stage(gates=((0, 7),), swaps=())
    tampered = dataclasses.replace(plan, stages=(*plan.stages, extra))

    with pytest.raises(PlanError):
        swapnet.verify_plan(hardware, interaction, tampered)


def test_a_gate_scheduled_before_its_qubits_are_adjacent_is_caught(instance):
    hardware, interaction, plan = instance

    # Move every gate into the first stage. Some of them are only executable
    # after later swap layers, so this must fail.
    all_gates = tuple(plan.scheduled_gates())
    stages = (
        dataclasses.replace(plan.stages[0], gates=all_gates),
        *tuple(dataclasses.replace(s, gates=()) for s in plan.stages[1:]),
    )
    tampered = dataclasses.replace(plan, stages=stages)

    with pytest.raises(PlanError):
        swapnet.verify_plan(hardware, interaction, tampered)


def test_a_wrong_final_layout_is_caught(instance):
    hardware, interaction, plan = instance

    swapped = dict(plan.final_full_layout)
    keys = list(swapped)[:2]
    swapped[keys[0]], swapped[keys[1]] = swapped[keys[1]], swapped[keys[0]]
    tampered = dataclasses.replace(plan, final_full_layout=swapped)

    with pytest.raises(PlanError, match="final layout"):
        swapnet.verify_plan(hardware, interaction, tampered)


def test_an_incomplete_initial_layout_is_caught(instance):
    hardware, interaction, plan = instance

    partial = dict(plan.initial_full_layout)
    partial.pop(next(iter(partial)))
    tampered = dataclasses.replace(plan, initial_full_layout=partial)

    with pytest.raises(PlanError):
        swapnet.verify_plan(hardware, interaction, tampered)


def test_an_edge_both_scheduled_and_unrouted_is_caught(instance):
    hardware, interaction, plan = instance

    # Contradictory: the plan claims to execute this interaction and also that it
    # never became executable. Relaxing require_converged must not excuse it.
    tampered = dataclasses.replace(plan, unrouted=(plan.scheduled_gates()[0],))

    with pytest.raises(PlanError, match="both scheduled and"):
        swapnet.verify_plan(hardware, interaction, tampered, require_converged=False)


def test_require_converged_allows_genuinely_unrouted_edges(instance):
    hardware, interaction, plan = instance

    # Drop one interaction's gate and declare it unrouted instead: a consistent
    # description of a plan that did not finish.
    dropped = plan.scheduled_gates()[0]
    stages = tuple(
        dataclasses.replace(stage, gates=tuple(g for g in stage.gates if g != dropped))
        for stage in plan.stages
    )
    tampered = dataclasses.replace(plan, stages=stages, unrouted=(dropped,))

    swapnet.verify_plan(hardware, interaction, tampered, require_converged=False)
    with pytest.raises(PlanError, match="never made"):
        swapnet.verify_plan(hardware, interaction, tampered)
