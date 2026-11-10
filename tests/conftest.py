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

"""Shared fixtures.

The default search parameters are the upstream ones (70 restarts, 50 000
annealing iterations) and take far too long for a test suite. Everything here
uses a deliberately small budget: these tests check that the pipeline is
correct and self-consistent, not that it finds good layouts.
"""

from __future__ import annotations

import pytest

import swapnet

# Small enough to keep the whole suite in the low seconds, large enough that the
# annealing and tabu phases both actually run.
FAST_MAP = dict(
    num_restarts=4,
    sa_iterations=2000,
    tabu_iterations=1000,
    cycles_per_restart=2,
)


@pytest.fixture
def fast_params() -> swapnet.MapParams:
    return swapnet.MapParams(**FAST_MAP)


@pytest.fixture(params=["highs", "scipy"])
def backend(request) -> str:
    """Each MILP backend that is importable in this environment."""
    if request.param not in swapnet.available_backends():
        pytest.skip(f"{request.param} is not installed")
    return request.param


def factor_covariance(
    n: int = 100, seed: int = 7, sectors: int = 4, vol_sigma: float = 0.8
):
    """A synthetic equity-like covariance matrix. Same generator the README uses.

    One dominant positive market factor plus disjoint sector factors gives a realistic
    correlation structure; lognormal volatilities then scale it into a *covariance*, which
    is what real data is. The volatility spread is the part that matters for
    :mod:`swapnet.shifts`: it is what defeats a scalar shift and what a rank-1 shift
    absorbs. At ``vol_sigma=0.8`` this reproduces the F2 100-asset regime closely --
    off-diagonal std/mean 1.55 vs 1.54, rank-0 29.5% vs 29.7%, rank-1 99.8% vs 99.8%.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    columns = [rng.uniform(0.55, 0.95, size=n)]  # the market: all positive
    labels = rng.integers(0, sectors, size=n)
    for sector in range(sectors):
        columns.append(np.where(labels == sector, rng.uniform(0.15, 0.35, size=n), 0.0))
    loadings = np.column_stack(columns)

    corr = loadings @ loadings.T
    np.fill_diagonal(corr, np.diag(corr) + rng.uniform(0.1, 0.4, size=n))
    scale = np.sqrt(np.diag(corr))
    corr /= np.outer(scale, scale)

    volatility = np.exp(rng.normal(0.0, vol_sigma, size=n))
    return corr * np.outer(volatility, volatility)


@pytest.fixture
def fake_hardware():
    """``Hardware`` from FakeBoston: 156 qubits, 175 live links, 1 dead."""
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeBoston

    return swapnet.Hardware.from_backend(FakeBoston())


def line_hardware(num_qubits: int, error: float = 0.01):
    """A path graph with uniform errors. Enough to build and check an ansatz."""
    return swapnet.Hardware.from_error_map(
        {(q, q + 1): error for q in range(num_qubits - 1)},
        num_qubits=num_qubits,
        name="line",
    )


def grid(rows: int, cols: int):
    """A rows x cols grid as an integer-labelled edge list."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            here = r * cols + c
            if c + 1 < cols:
                edges.append((here, here + 1))
            if r + 1 < rows:
                edges.append((here, here + cols))
    return edges, rows * cols
