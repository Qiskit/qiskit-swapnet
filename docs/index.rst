#######
swapnet
#######

swapnet is a package for qubit layout and SWAP routing of a QAOA cost layer. It takes a problem's couplings and a backend, and returns a cost layer that the device can run within a budget of parallel SWAP layers. Specifically, it contains:

1. Problem definitions from a covariance matrix, a QUBO, or a general coupling matrix.
2. Qubit layout by quadratic assignment, and SWAP routing by prize-collecting vertex-disjoint multi-commodity flow.
3. Hardware descriptions built from a backend, with optional error-aware edge weights.
4. Construction of the resulting QAOA cost-layer circuit from an algorithm-neutral schedule.

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/qiskit-swapnet>`_.

By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-swapnet/issues/new/choose>`_ for tracking requests and bugs.

License
-------

`Apache License 2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_. We may occasionally make breaking changes in order to
improve the user experience. When possible, we will keep old interfaces and mark them as deprecated, as long
as they can co-exist with the new ones. Each substantial improvement, breaking change, or deprecation will be
documented in the :doc:`release notes <release-notes>`.

.. toctree::
   :hidden:

   Documentation home <self>
   Installation instructions <install>
   Guides <guides/index>
   GitHub <https://github.com/Qiskit/qiskit-swapnet>

.. toctree::
   :hidden:
   :caption: API reference

   Python API reference <apidocs/index>
   Release notes <release-notes>
