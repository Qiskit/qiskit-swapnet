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

"""The angle convention, in one place, because getting it wrong is silent.

THE CONTRACT
------------

An :class:`~swapnet.ansatz.Ansatz` implements

    U(theta) = exp(-i G(theta) / 2)        for   G(theta) = sum_alpha theta_alpha Z^alpha

where ``alpha`` runs over the Pauli-Z terms the circuit realises: ``Z_i`` for the one-body
terms and ``Z_i Z_j`` for the two-body ones. ``G`` is the **generator**, not a
Hamiltonian: the ``/2`` is part of the definition, so reading ``U = exp(-i G)`` is wrong
by a factor 2.

WHY THERE IS NO CONVERSION FACTOR
---------------------------------

Qiskit's rotations are defined with exactly this sign and half-angle::

    RZ(t)  = exp(-i t Z / 2)
    RZZ(t) = exp(-i t Z Z / 2)

so ``exp(-i theta Z_i / 2)`` *is* ``rz(theta)`` and ``exp(-i theta Z_i Z_j / 2)`` *is*
``rzz(theta)``. The gate angle equals ``theta``, and :func:`rz_angle` and
:func:`rzz_angle` are the identity.

They exist anyway, and the builder calls them, for two reasons: there is one authority to
cite when a downstream module asks what the angle means, and if the convention ever moves
again this is the only file that changes. A caller that wants the angle should ask here
rather than assume, precisely because the assumption is currently free.

USING IT FOR QAOA
-----------------

For a cost Hamiltonian ``H_C = sum_ij J_ij Z_i Z_j + sum_i h_i Z_i``, one QAOA cost layer
is ``exp(-i gamma H_C)``. Matching that to ``exp(-i G/2)`` gives ``G = 2 gamma H_C``, so

    theta_alpha = 2 * gamma * coeff_alpha

with ``coeff_alpha`` the problem's own ``J_ij`` or ``h_i``.
:func:`~swapnet.ansatz.angles_from_problem` computes exactly this, so a parameter-setting
module supplies ``gamma`` and never handles the factor itself.

WHAT PINS IT
------------

``tests/test_ansatz.py`` builds a small ansatz, binds it, and compares against
``expm(-i G / 2)`` entry by entry, including the permutation the SWAP network performs.
That comparison computes its expectation *without* using this module, so it is the only
check that can catch a wrong factor -- including a negative control that asserts a
``theta`` scaled by 2 fails. ``ansatz.verify_structure`` cannot: it calls the functions
below to compute what it expects, so it and the builder move together.
"""

from __future__ import annotations

__all__ = ["rz_angle", "rzz_angle", "GENERATOR_HALF", "gamma_to_theta"]

# U = exp(-i G/2): the generator carries a half. Qiskit's rotations carry the same half
# with the same sign, so the two cancel and the gate angle is theta itself.
GENERATOR_HALF = 0.5


def rz_angle(theta):
    """The ``rz`` angle realising ``exp(-i theta Z / 2)``.

    The identity, per the module docstring. Accepts a float or a ``ParameterExpression``
    and returns the same kind.
    """
    return theta


def rzz_angle(theta):
    """The ``rzz`` angle realising ``exp(-i theta Z Z / 2)``. The identity."""
    return theta


def gamma_to_theta(coefficient, gamma):
    """``theta`` realising one QAOA cost layer ``exp(-i gamma H_C)`` for this coefficient.

    ``coefficient`` is the problem's own ``J_ij`` or ``h_i``. See the module docstring:
    ``G = 2 gamma H_C``, hence the factor 2.
    """
    return 2.0 * gamma * coefficient
