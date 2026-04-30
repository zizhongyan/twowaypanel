Installation
============

This page explains how to install **twowaypanel** and its main dependencies.

Prerequisites
-------------

- Python 3.8+ is recommended.
- ``twowaypanel`` relies on standard scientific Python libraries, including **NumPy**, **SciPy**, **pandas**, and **PyTorch**. 

.. note::
    PyTorch is required, because ``twowaypanel`` uses automatic differentiation (autograd) functionality to support numerical optimization routines. You can install PyTorch by following the official instructions at: https://pytorch.org/. **GPU/CUDA is not required.** For most users, the **CPU-only** version of PyTorch is sufficient. ``twowaypanel`` includes an internal copy of the optimization routines based on ``pytorch-minimize`` by Reuben Feinman.


Installation options
--------------------

1. Install from PyPI (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the latest released version from PyPI:

.. code-block:: bash

   pip install twowaypanel

Quick import check:

.. code-block:: bash

   python -c "import twowaypanel; print('twowaypanel imported successfully')"


2. Install the development version from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/zizhongyan/twowaypanel.git
   cd twowaypanel
   pip install -e .

The ``-e`` option performs an *editable install*, meaning local code changes take effect immediately without
reinstalling.

3. Install from a local copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you downloaded the source code (e.g., as a ZIP file) and unzipped it locally, install from the repository root:

.. code-block:: bash

   cd /path/to/twowaypanel
   pip install -e .

This option is convenient for offline use or replication bundles shared as archived source code.

4. Jupyter / notebook users
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Jupyter, it is important to ensure the package is installed into the same Python environment used by the
current notebook kernel. The recommended approach is to use ``%pip`` inside the notebook.

From the repository root directory, run:

.. code-block:: python

   %cd /path/to/twowaypanel
   %pip install -e .

Then verify:

.. code-block:: python

   import twowaypanel



**Reference** for ``pytorch-minimize``

- Feinman, Reuben. (2021). *Pytorch-minimize: a library for numerical optimization with autograd*.  
  Available at https://github.com/rfeinman/pytorch-minimize



