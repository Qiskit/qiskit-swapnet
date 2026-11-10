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

"""Stage 0: the problem artifact -- what to route, independent of any hardware.

A :class:`Problem` is an Ising instance::

    H(z) = sum_{i<j} J_ij z_i z_j  +  sum_i h_i z_i  +  offset        z_i in {-1, +1}

Nothing here knows about devices, circuits or covariance matrices. The constructors
:meth:`Problem.from_matrix`, :meth:`Problem.from_qubo` and
:meth:`Problem.from_covariance` cover the usual ways a caller arrives at one, and
:meth:`Problem.verify` checks the map numerically for the two that do a substitution.

THE COUPLING-POWER ASYMMETRY, WHICH IS THE WHOLE REASON THIS CLASS EXISTS
------------------------------------------------------------------------

The objective under a depth budget is to maximise ``sum J^2`` over the interactions the
circuit realises. The two swapnet stages both accept a ``couplings`` argument, and they
consume it at **different powers**:

* the mapper's assignment cost is ``sum_lm couplings_lm * distance(l, m)`` -- *linear*
  in ``couplings`` (``src/cpp/qap.cpp:93``);
* the router's per-commodity reward and its ``realized_weight`` accumulate
  ``couplings**2`` -- *quadratic* (``src/cpp/pipeline.cpp:42-45``).

So the same ``sum J^2`` objective needs ``J**2`` handed to the mapper and ``|J|`` handed
to the router. Pass ``J**2`` to the router and it silently optimises ``sum J^4`` and
reports coverage of *that*; the plan still validates, the numbers still look healthy, and
the objective is not the one being scored. That is not hypothetical -- it was a live bug
in a demo notebook.

:meth:`mapper_couplings` and :meth:`router_couplings` are therefore the only supported way
to feed the two stages. A caller that uses them cannot get the power wrong, and
``plan.realized_fraction`` then equals :meth:`covered_fraction` exactly, which
``swapnet.plan_schedule`` asserts.

THE TILT, AND WHY THE ROUTER DOES NOT GET RAW |J|
-------------------------------------------------

``route`` normalises whatever it is given to mean 1, then prices each candidate path
against the reward. On a dense problem the raw ``J^2`` can span five orders of magnitude,
and after normalisation a large share of pairs carry less reward than a single arc costs
-- so the router declines to route them at all and a budget increase buys nothing.
Measured on the F2 100-asset objective matrix: 26% of pairs fall below one arc-cost.

:meth:`router_couplings` therefore returns ``sqrt(1 + tilt * J^2 / max(J^2))``, so the
reward (its square) is ``1 + tilt * J^2 / max(J^2)``: every pair can pay its way, while
the heaviest keep up to ``1 + tilt`` times the priority. Ordering is preserved. Pass
``tilt=None`` for the untilted ``|J|``, which is the honest choice when you want
``realized_weight`` to *be* the ``sum J^2`` coverage rather than a monotone proxy for it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Problem", "DEFAULT_TILT"]

# The router reward compression factor. 4.0 lifts the lightest pairs above one arc-cost on
# the F2 dense instances while leaving the heavy pairs 5x the priority of the lightest.
DEFAULT_TILT = 4.0


@dataclass(frozen=True)
class Problem:
    """An Ising instance: the interactions to realise, and the fields that ride along.

    ``couplings`` is keyed by ``(i, j)`` with ``i < j``, and holds only the pairs that
    exist -- a sparse problem simply has fewer keys. ``fields`` is ``h``, length ``n``, or
    ``None`` when the problem has no one-body part. ``offset`` is the constant that makes
    ``H(z)`` equal the original objective.

    ``label`` and ``metadata`` are provenance: they travel with the artifact so a
    downstream module can say which problem it solved without being handed the source
    data again.
    """

    n: int
    couplings: dict[tuple[int, int], float]
    fields: np.ndarray | None = None
    offset: float = 0.0
    label: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        canonical: dict[tuple[int, int], float] = {}
        for pair, value in self.couplings.items():
            a, b = int(pair[0]), int(pair[1])
            if a == b:
                raise ValueError(
                    f"({a}, {b}) is a self-coupling; the diagonal belongs "
                    "in fields and offset, not in couplings"
                )
            if not (0 <= a < self.n and 0 <= b < self.n):
                raise ValueError(f"coupling ({a}, {b}) is outside 0..{self.n - 1}")
            key = (a, b) if a < b else (b, a)
            if key in canonical:
                raise ValueError(f"coupling {key} given twice (both orientations?)")
            canonical[key] = float(value)
        object.__setattr__(self, "couplings", canonical)

        if self.fields is not None:
            array = np.asarray(self.fields, dtype=float)
            if array.shape != (self.n,):
                raise ValueError(f"fields has shape {array.shape}, expected ({self.n},)")
            object.__setattr__(self, "fields", array)
        object.__setattr__(self, "offset", float(self.offset))

    # -- shape ---------------------------------------------------------------

    @property
    def num_pairs(self) -> int:
        return len(self.couplings)

    @property
    def is_complete(self) -> bool:
        return self.num_pairs == self.n * (self.n - 1) // 2

    @property
    def total_j_squared(self) -> float:
        """``sum J^2`` over every pair. The denominator of :meth:`covered_fraction`."""
        return float(sum(v * v for v in self.couplings.values()))

    def j_squared(self, pairs: Iterable[tuple[int, int]]) -> float:
        """``sum J^2`` over an iterable of unordered pairs. Unknown pairs are an error."""
        total = 0.0
        for a, b in pairs:
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            value = self.couplings[key]
            total += value * value
        return float(total)

    def covered_fraction(self, pairs: Iterable[tuple[int, int]]) -> float:
        """The share of ``sum J^2`` that ``pairs`` carries.

        Invariant to any overall rescaling of ``J``, so this -- not the absolute
        ``sum J^2`` -- is the number to compare across conventions.
        """
        total = self.total_j_squared
        return 0.0 if total <= 0.0 else self.j_squared(pairs) / total

    # -- the two coupling vectors, at the right powers -----------------------

    def mapper_couplings(self) -> dict[tuple[int, int], float]:
        """``J**2``, for :func:`swapnet.map_layout`, whose objective is linear."""
        return {pair: value * value for pair, value in self.couplings.items()}

    def router_couplings(
        self, tilt: float | None = DEFAULT_TILT
    ) -> dict[tuple[int, int], float]:
        """For :func:`swapnet.route`, whose reward is quadratic. See the module docstring.

        ``tilt=None`` gives ``|J|``, so the router's ``realized_weight`` *is* the
        ``sum J^2`` coverage. A positive ``tilt`` gives
        ``sqrt(1 + tilt * J^2 / max(J^2))``, which routes the light pairs at the cost of
        making ``realized_weight`` a monotone proxy rather than the objective itself.
        """
        if tilt is None:
            return {pair: abs(value) for pair, value in self.couplings.items()}
        if tilt <= 0.0:
            raise ValueError(f"tilt must be positive or None, got {tilt}")
        mass = {pair: value * value for pair, value in self.couplings.items()}
        peak = max(mass.values(), default=0.0)
        if peak <= 0.0:
            return dict.fromkeys(mass, 1.0)
        return {
            pair: math.sqrt(1.0 + tilt * value / peak) for pair, value in mass.items()
        }

    # -- views ---------------------------------------------------------------

    def matrix(self) -> np.ndarray:
        """The symmetric ``(n, n)`` coupling matrix, zero diagonal."""
        out = np.zeros((self.n, self.n), dtype=float)
        for (a, b), value in self.couplings.items():
            out[a, b] = out[b, a] = value
        return out

    def pairs(self) -> list[tuple[int, int]]:
        """The coupling keys, sorted. The canonical term order for a mask or a vector."""
        return sorted(self.couplings)

    def interaction_graph(self):
        """The unweighted interaction graph, as ``networkx.Graph``.

        Pass this to :func:`swapnet.map_layout` and :func:`swapnet.route` together with
        :meth:`mapper_couplings` / :meth:`router_couplings`; the weights are supplied
        separately so one graph can carry either.
        """
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(self.n))
        graph.add_edges_from(self.couplings)
        return graph

    def with_couplings(
        self,
        couplings: Mapping[tuple[int, int], float],
        *,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Problem:
        """A copy carrying different couplings. Used by :mod:`swapnet.shifts`."""
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return Problem(
            n=self.n,
            couplings=dict(couplings),
            fields=self.fields,
            offset=self.offset,
            label=self.label if label is None else label,
            metadata=merged,
        )

    # -- diagnostics ---------------------------------------------------------

    def concentration(self) -> dict:
        """How concentrated ``sum J^2`` is, which decides whether truncation can pay.

        A problem whose mass sits in a few hundred pairs can be truncated cheaply; one
        that spreads it evenly cannot, and no router will rescue that.
        """
        mass = np.array([v * v for v in self.couplings.values()], dtype=float)
        if mass.size == 0:
            return {"num_pairs": 0, "total_j_squared": 0.0}
        total = float(mass.sum())
        order = np.sort(mass)[::-1]
        cumulative = np.cumsum(order) / total if total > 0 else np.zeros_like(order)

        out = {
            "num_pairs": int(mass.size),
            "total_j_squared": total,
            "abs_j_min": float(np.sqrt(mass.min())),
            "abs_j_median": float(np.sqrt(np.median(mass))),
            "abs_j_max": float(np.sqrt(mass.max())),
            "num_negative_couplings": int(
                sum(1 for v in self.couplings.values() if v < 0)
            ),
        }
        for fraction in (0.5, 0.9, 0.99):
            index = int(np.searchsorted(cumulative, fraction)) + 1
            out[f"pairs_for_{int(fraction * 100)}pct_of_j2"] = min(index, int(mass.size))
        return out

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-ready. Coupling keys become ``"i,j"``, since JSON has no tuple keys."""
        return {
            "n": self.n,
            "couplings": {f"{a},{b}": v for (a, b), v in sorted(self.couplings.items())},
            "fields": None if self.fields is None else [float(x) for x in self.fields],
            "offset": self.offset,
            "label": self.label,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Problem:
        couplings = {}
        for key, value in payload["couplings"].items():
            a, b = key.split(",")
            couplings[(int(a), int(b))] = float(value)
        fields = payload.get("fields")
        return cls(
            n=int(payload["n"]),
            couplings=couplings,
            fields=None if fields is None else np.asarray(fields, dtype=float),
            offset=float(payload.get("offset", 0.0)),
            label=str(payload.get("label", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_matrix(
        cls,
        J: np.ndarray,
        fields: Sequence[float] | None = None,
        *,
        offset: float = 0.0,
        label: str = "",
        metadata: Mapping[str, Any] | None = None,
        drop_zeros: bool = False,
    ) -> Problem:
        """Build from a symmetric coupling matrix. The diagonal is ignored.

        ``J`` is taken as the Ising ``J`` directly -- no substitution, no factor. Use
        :meth:`from_qubo` or :meth:`from_covariance` when starting from a binary
        objective instead.

        ``drop_zeros=True`` omits exactly-zero couplings, turning a padded dense matrix
        into a sparse problem. It is off by default so that a zero coupling stays a term
        the router may schedule -- silently dropping terms would change the problem.
        """
        matrix = np.asarray(J, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"J must be square, got {matrix.shape}")
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
            raise ValueError(
                "J must be symmetric; symmetrise it first if only "
                "J_ij + J_ji is meaningful"
            )
        n = int(matrix.shape[0])
        rows, cols = np.triu_indices(n, 1)
        couplings = {
            (int(i), int(j)): float(matrix[i, j])
            for i, j in zip(rows, cols, strict=True)
            if not (drop_zeros and matrix[i, j] == 0.0)
        }
        return cls(
            n=n,
            couplings=couplings,
            fields=None if fields is None else np.asarray(fields, dtype=float),
            offset=offset,
            label=label,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_qubo(
        cls,
        Q: np.ndarray,
        *,
        scale: float = 1.0,
        label: str = "qubo",
        metadata: Mapping[str, Any] | None = None,
    ) -> Problem:
        """From a QUBO: ``f(x) = scale * x' Q x`` for ``x_i in {0, 1}``.

        A QUBO carries its linear part on the diagonal, because ``x_i^2 == x_i``. The
        substitution ``x_i = (1 - z_i)/2`` sends the diagonal into ``fields`` and
        ``offset``, never into a coupling.

        For symmetric ``Q`` the coefficient of ``x_i x_j`` is ``2 Q_ij``, so ``J_ij =
        Q_ij`` would double-count; the correct value is ``scale * Q_ij / 2``. An
        asymmetric ``Q`` is symmetrised first, since only ``Q_ij + Q_ji`` is observable.
        Run :meth:`verify` to confirm.
        """
        matrix = np.asarray(Q, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Q must be square, got {matrix.shape}")
        symmetric = 0.5 * (matrix + matrix.T)
        return cls._from_binary_quadratic(
            symmetric,
            None,
            quadratic_scale=scale,
            linear_scale=0.0,
            label=label,
            metadata=metadata,
        )

    @classmethod
    def from_covariance(
        cls,
        sigma: np.ndarray,
        mu: np.ndarray | None = None,
        *,
        sigma_scale: float = 1.0,
        mu_scale: float = -1.0,
        label: str = "covariance",
        metadata: Mapping[str, Any] | None = None,
    ) -> Problem:
        """From a portfolio objective ``f(x) = sigma_scale * x' Sigma x + mu_scale * mu'
        x``.

        With ``x_i = (1 - z_i)/2``::

            J_ij   = sigma_scale * Sigma_ij / 2
            h_i    = -sigma_scale * rowsum_i / 2  -  mu_scale * mu_i / 2
            offset =  sigma_scale * (total + trace) / 4  +  mu_scale * sum(mu) / 2

        ``sigma_scale`` is the factor conventions disagree on, and getting it wrong
        rescales ``sum J^2`` while changing no gate count, no depth and no router
        decision -- every one of those depends on ``J`` through ratios only. It is
        recorded in ``metadata`` for that reason. Run :meth:`verify`.
        """
        matrix = np.asarray(sigma, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Sigma must be square, got {matrix.shape}")
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
            raise ValueError("Sigma must be symmetric")
        return cls._from_binary_quadratic(
            matrix,
            mu,
            quadratic_scale=sigma_scale,
            linear_scale=mu_scale,
            label=label,
            metadata=metadata,
        )

    @classmethod
    def _from_binary_quadratic(
        cls,
        matrix: np.ndarray,
        linear: np.ndarray | None,
        *,
        quadratic_scale: float,
        linear_scale: float,
        label: str,
        metadata: Mapping[str, Any] | None,
    ) -> Problem:
        """The substitution behind :meth:`from_qubo` and :meth:`from_covariance`."""
        n = int(matrix.shape[0])
        rows, cols = np.triu_indices(n, 1)
        values = quadratic_scale * matrix[rows, cols] / 2.0
        couplings = {
            (int(i), int(j)): float(v) for i, j, v in zip(rows, cols, values, strict=True)
        }

        fields = -quadratic_scale * matrix.sum(axis=1) / 2.0
        offset = quadratic_scale * (float(matrix.sum()) + float(np.trace(matrix))) / 4.0

        if linear is not None:
            vector = np.asarray(linear, dtype=float)
            if vector.shape != (n,):
                raise ValueError(
                    f"the linear term has shape {vector.shape}, " f"expected ({n},)"
                )
            fields = fields - linear_scale * vector / 2.0
            offset = offset + linear_scale * float(vector.sum()) / 2.0

        info = {
            "quadratic_scale": float(quadratic_scale),
            "linear_scale": float(linear_scale),
        }
        info.update(metadata or {})
        return cls(
            n=n,
            couplings=couplings,
            fields=fields,
            offset=float(offset),
            label=label,
            metadata=info,
        )

    # -- the numerical check ------------------------------------------------

    def verify(
        self,
        matrix: np.ndarray,
        linear: np.ndarray | None = None,
        *,
        trials: int = 64,
        seed: int = 20260817,
    ) -> float:
        """Check ``H(z) == f(x)`` on random assignments. Returns the largest deviation.

        This is what makes the one-body part trustworthy. A wrong ``h`` or ``offset``
        leaves ``J`` correct, so routing, depth and coverage would all look fine and
        only the energies would be wrong -- silently. Run it once per instance built by
        :meth:`from_qubo` or :meth:`from_covariance`.

        ``matrix`` and ``linear`` are the *original* objective's terms, and the scales
        recorded in ``metadata`` are the ones applied.
        """
        quadratic_scale = float(self.metadata.get("quadratic_scale", 1.0))
        linear_scale = float(self.metadata.get("linear_scale", 0.0))
        rng = np.random.default_rng(seed)
        matrix = np.asarray(matrix, dtype=float)
        pairs = np.array(self.pairs(), dtype=int)
        values = np.array(
            [self.couplings[(int(a), int(b))] for a, b in pairs], dtype=float
        )
        fields = np.zeros(self.n) if self.fields is None else self.fields
        worst = 0.0

        for _ in range(trials):
            x = rng.integers(0, 2, size=self.n).astype(float)
            z = 1.0 - 2.0 * x  # x=0 -> z=+1, x=1 -> z=-1
            target = quadratic_scale * float(x @ matrix @ x)
            if linear is not None:
                target += linear_scale * float(np.asarray(linear, dtype=float) @ x)
            got = (
                float(np.sum(values * z[pairs[:, 0]] * z[pairs[:, 1]]))
                + float(fields @ z)
                + self.offset
            )
            worst = max(worst, abs(got - target) / max(1.0, abs(target)))
        return worst

    def __repr__(self) -> str:
        kind = "complete" if self.is_complete else "sparse"
        return (
            f"<Problem {self.label or 'unlabelled'} n={self.n} "
            f"pairs={self.num_pairs} ({kind}) sumJ2={self.total_j_squared:.6g}>"
        )
