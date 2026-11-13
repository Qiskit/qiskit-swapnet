<!-- SHIELDS -->
<div align="left">

  [![Release](https://img.shields.io/pypi/v/swapnet.svg?label=Release)](https://github.com/Qiskit/qiskit-swapnet/releases)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/pypi/pyversions/swapnet?label=Python&logo=python)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D2.0%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
  [![License](https://img.shields.io/github/license/Qiskit/qiskit-swapnet?label=License)](LICENSE.txt)
  [![Downloads](https://img.shields.io/pypi/dm/swapnet.svg?label=Downloads)](https://pypi.org/project/swapnet/)
  [![Tests](https://github.com/Qiskit/qiskit-swapnet/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit/qiskit-swapnet/actions/workflows/test_latest_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/Qiskit/qiskit-swapnet/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-swapnet?branch=main)
</div>

> [!NOTE]
> This repository is under active development and the code here should not be considered stable.

# Qiskit swapnet

Qiskit swapnet is a package for laying out and routing a single QAOA cost layer onto a real device within a budget of `k` parallel SWAP layers. It enables the creation of practical-depth QAOA layers by solving an approximated problem rather than the full original one. Specifically, it contains:

1. Problem definitions from a covariance matrix, a QUBO, or a general coupling matrix, together with the coupling weights that the layout and routing stages expect.
2. Analytic rank-0 and rank-1 shifts, which implement the dominant part of the couplings in a single layer of RZ rotations, with no two-qubit gates and no routing.
3. Qubit layout by quadratic assignment, and SWAP routing by prize-collecting vertex-disjoint multi-commodity flow.
4. Hardware descriptions built from a backend, with per-link traversal costs derived from device error rates.
5. Construction of the resulting cost-layer circuit from an algorithm-neutral `Schedule`, so swapnet can be replaced by another routing algorithm without changing anything downstream.

----------------------------------------------------------------------------------------------------

## Documentation

All documentation will be available on IBM Quantum Platform soon.

----------------------------------------------------------------------------------------------------

## Installation

`swapnet` includes a native C++ extension, so building it requires a C++ compiler and CMake. The package can be either intalled via `pip`:

```bash
pip install 'qiskit-swapnet'
```

Or installed from source:

```bash
git clone git@github.com:Qiskit/qiskit-swapnet.git
cd qiskit-swapnet
pip install -e '.[dev]'
```

The core is Qiskit-free. To install only the core, without Qiskit and the fake backends, use `pip install -e '.[highs,graphs]'`.

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

## Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-swapnet).

The developer guide is located at [CONTRIBUTING.md](CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

----------------------------------------------------------------------------------------------------

## Citing this package

If you use this package in your research, use the [CITATION.bib](CITATION.bib) file in this project’s repository to cite the appropriate reference(s).

----------------------------------------------------------------------------------------------------

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

----------------------------------------------------------------------------------------------------

## Deprecation Policy

We follow [semantic versioning](https://semver.org/). We may occasionally make breaking changes in
order to improve the user experience. When possible, we will keep old interfaces and mark them as
deprecated, as long as they can co-exist with the new ones. Each substantial improvement, breaking
change, or deprecation will be documented in the [release notes](docs/release-notes.rst).
