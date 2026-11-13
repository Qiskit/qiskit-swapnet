# Tests

This repository's tests are run with [pytest]. Install the development dependencies, which include pytest, the linters and the formatters, with:

```sh
pip install -e '.[dev]'
```

Most tests need nothing beyond that. The tests marked `slow` build a full-size device target, and can be deselected.

## Test environment

To run all tests:

```sh
pytest
```

To skip the full-size device case:

```sh
pytest -m "not slow"
```

Pytest is configured in [`pyproject.toml`](../pyproject.toml). In particular, `DeprecationWarning`s are turned into errors.

## Lint and format

The following commands check the code against the project's coding standards, as configured in [`pyproject.toml`](../pyproject.toml):

```sh
ruff check src tests        # lint, including the copyright header on every source file
black --check src tests     # formatting, line length 90, target py311
pymarkdown scan README.md   # markdown lint, config in .pymarkdown.json
```

`pylint src/swapnet` is not wired into a config and is not expected to come back empty -- it is a sanity sweep, not a gate. Its remaining reports are the lazy `import` statements (deliberate: the core stays Qiskit-free), the availability probes in `milp.py`, and members of the compiled `_core` and `rustworkx` extensions that it cannot introspect.

## Documentation environment

The documentation is built with [Sphinx]. Install the `docs` extra, and make sure [pandoc](https://pandoc.org/) is installed, since it is needed to render the notebooks. Pandoc is not available via pip, so it must be installed through some other means, for example `brew install pandoc` on macOS.

```sh
pip install -e '.[docs]'
sphinx-build -W -T --keep-going -b html docs docs/_build/html
```

If the build succeeds, it can be viewed by navigating to `docs/_build/html/index.html` in a web browser.

## Key tests

Two checks are worth knowing about specifically:

- **`tests/test_ansatz.py` pins the convention** by comparing a bound circuit against `expm(-i G/2)` entry by entry, including the SWAP relabelling, computing its expectation *without* `swapnet.conventions` -- plus a negative control asserting that a `theta` scaled by 2 fails. A convention test that only ever confirms agreement proves nothing.
- **`tests/test_pipeline.py` pins the coupling powers** by asserting that with `tilt=None` the router's `realized_fraction` equals `problem.covered_fraction` exactly -- plus a negative control feeding `J**2` to the router and asserting the equality *breaks*.

[pytest]: https://docs.pytest.org/
[Sphinx]: https://www.sphinx-doc.org/
