# Developer guide

Development of the `swapnet` package takes place [on GitHub](https://github.com/Qiskit/qiskit-swapnet).
The [Contributing to Qiskit](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md) guide may serve as a
useful starting point, as this package builds on [Qiskit].

This package is written in [Python] and [C++], and the two are built together with [scikit-build-core] and CMake.
The tests, linters and formatters, and how to run them, are described at [`tests/README.md`](tests/README.md).

Project configuration, including information about dependencies, is stored in [`pyproject.toml`](pyproject.toml).

We use [Sphinx] for documentation and [reno] for release notes.
We use [Google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html),
except we omit the type of each argument, as type information is redundant with Python
[type hints](https://docs.python.org/3/library/typing.html).

Every source file carries the project's copyright header, which is checked by [ruff].

[Qiskit]: https://www.ibm.com/quantum/qiskit
[Python]: https://www.python.org/
[C++]: https://isocpp.org/
[scikit-build-core]: https://scikit-build-core.readthedocs.io/
[Sphinx]: https://www.sphinx-doc.org/
[reno]: https://docs.openstack.org/reno/
[ruff]: https://github.com/astral-sh/ruff
