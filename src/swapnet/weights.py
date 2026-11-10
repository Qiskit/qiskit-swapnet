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

"""Turning device calibration data into solver weights.

Two independent sets of weights steer the pipeline:

``W_ij`` on the *hardware* graph are traversal costs. Lower is better, so they
are error-like: a coupling with twice the error should cost about twice as much
to route through. :func:`from_fidelities` and :func:`from_error_rates` build them
from the numbers a backend actually reports.

``J_lm`` on the *interaction* graph are coupling strengths. Higher is better --
they say how much a term matters. Under a depth truncation the router maximises
the realised ``sum J_lm^2``, so ``J`` carries the same meaning as a Hamiltonian
coefficient and its sign is irrelevant.

Both must be **normalised to mean 1** before use. The annealing temperatures and
the routing reward are calibrated against unit-cost hop counts, and the routing
model's path-length pruning derives its bound from the cheapest edge. Feeding raw
error rates (order 1e-2) makes every path look free, which silently disables the
pruning and inflates the MILP; feeding raw ``-log(fidelity)`` shifts the QAP cost
scale so far that the Metropolis acceptance saturates. :func:`normalize` is
therefore applied by default at every entry point that takes weights.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "normalize",
    "from_fidelities",
    "from_error_rates",
    "mean_normalized",
]


def normalize(weights: Sequence[float]) -> list[float]:
    """Rescale to mean 1, preserving all ratios.

    Raises ``ValueError`` on a negative or non-finite entry, or when every entry
    is zero (which carries no preference and no scale to normalise against).
    """
    values = [float(w) for w in weights]
    if not values:
        return []

    for w in values:
        if not math.isfinite(w):
            raise ValueError(f"weight {w} is not finite")
        if w < 0.0:
            raise ValueError(
                f"weight {w} is negative; weights are costs or magnitudes, so "
                "they must be non-negative"
            )

    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError(
            "every weight is zero, so there is no scale to normalise against "
            "and no edge is preferred; pass None for an unweighted graph instead"
        )

    scale = len(values) / total
    return [w * scale for w in values]


def mean_normalized(weights: Sequence[float]) -> bool:
    """True when `weights` already has mean 1 to within rounding."""
    values = [float(w) for w in weights]
    if not values:
        return True
    return math.isclose(math.fsum(values) / len(values), 1.0, rel_tol=1e-9, abs_tol=1e-12)


def from_fidelities(
    fidelities: Mapping[object, float] | Iterable[float],
    *,
    floor: float = 1e-6,
) -> list[float] | dict[object, float]:
    """Convert two-qubit gate fidelities into normalised traversal costs.

    The cost of an edge is ``-log(fidelity)``, so that the cost of a path is the
    total log-infidelity along it: minimising summed cost maximises the product
    of fidelities, which is what the routed circuit's success probability
    actually is. The result is normalised to mean 1.

    ``floor`` bounds the fidelity away from zero, since a perfectly broken
    coupling would otherwise cost infinity. A coupling that bad should be left
    out of the graph instead.

    Accepts and returns either a mapping keyed by edge or a bare sequence.
    """
    if isinstance(fidelities, Mapping):
        keys = list(fidelities.keys())
        raw = [_neg_log(fidelities[k], floor) for k in keys]
        return dict(zip(keys, normalize(raw), strict=True))

    return normalize([_neg_log(f, floor) for f in fidelities])


def from_error_rates(
    errors: Mapping[object, float] | Iterable[float],
    *,
    floor: float = 1e-6,
) -> list[float] | dict[object, float]:
    """Convert two-qubit error rates into normalised traversal costs.

    Equivalent to ``from_fidelities(1 - error)``: the cost is
    ``-log(1 - error)``, which is very nearly proportional to the error rate for
    the small rates real devices report, but composes correctly across a path.
    """
    if isinstance(errors, Mapping):
        keys = list(errors.keys())
        raw = [_neg_log(1.0 - errors[k], floor) for k in keys]
        return dict(zip(keys, normalize(raw), strict=True))

    return normalize([_neg_log(1.0 - e, floor) for e in errors])


# A reported fidelity of exactly 1 is a rounding artifact, not a free gate, and a
# zero traversal cost is actively harmful: the routing model derives its
# path-length bound from the cheapest edge, so one zero-cost edge makes no finite
# bound safe and silently disables the pruning for every commodity. Cap the
# fidelity just below 1 so every cost is strictly positive. The cap is far smaller
# than any real calibration uncertainty, so it cannot change which edge is
# preferred.
_MAX_FIDELITY = 1.0 - 1e-12


def _neg_log(fidelity: float, floor: float) -> float:
    value = float(fidelity)
    if not math.isfinite(value):
        raise ValueError(f"fidelity {value} is not finite")
    if value > 1.0:
        raise ValueError(f"fidelity {value} exceeds 1")
    if value <= 0.0:
        raise ValueError(
            f"fidelity {value} is not positive; drop the coupling from the graph "
            "rather than modelling it as unusable"
        )
    return -math.log(min(max(value, floor), _MAX_FIDELITY))
