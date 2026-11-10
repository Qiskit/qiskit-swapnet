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

``swapnet`` includes a native C++ extension, so building it requires a C++ compiler and ``CMake``.

Install from source
^^^^^^^^^^^^^^^^^^^

First, clone the ``swapnet`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-swapnet.git

Next, upgrade pip and enter the repository.

.. code:: sh

    pip install --upgrade pip
    cd qiskit-swapnet

The next step is to install ``swapnet`` to the virtual environment. If you plan on developing in the repository,
install the ``dev`` dependencies, which also include Qiskit and the fake backends.

Adjust the options below to suit your needs.

.. code:: sh

    pip install -e '.[dev]'

To install only the core, without Qiskit, use:

.. code:: sh

    pip install -e '.[highs,graphs]'

The core is Qiskit-free. Building a ``Hardware`` from a backend and building an actual circuit import Qiskit lazily,
so the ``qiskit`` extra is only required if you need those.
