Installation instructions
=========================

Prerequisites
^^^^^^^^^^^^^

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate

``qiskit-swapnet`` includes a native C++ extension, so building it requires a C++ compiler and ``CMake``.


There are two primary ways to install the packages:

- :ref:`Option 1`
- :ref:`Option 2`

.. _Option 1:

Option 1: Install from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most straightforward way to install the ``qiskit-swapnet`` package is via ``PyPI``.

.. code:: sh

    pip install 'qiskit-swapnet'


.. _Option 2:

Option 2: Install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to develop in the repository or run the notebooks locally, you should install from source.
First, clone the ``qiskit-swapnet`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-swapnet.git

Next, upgrade pip and enter the repository.

.. code:: sh

    pip install --upgrade pip
    cd qiskit-swapnet

The next step is to install ``qiskit-swapnet`` to the virtual environment. If you plan on developing in the repository,
install the ``dev`` dependencies, which also include Qiskit and the fake backends.

Adjust the options below to suit your needs.

.. code:: sh

    pip install -e '.[dev]'

To install only the core, without Qiskit, use:

.. code:: sh

    pip install -e '.[highs,graphs]'

The core is Qiskit-free. Building a ``Hardware`` from a backend and building an actual circuit import Qiskit lazily,
so the ``qiskit`` extra is only required if you need those.
