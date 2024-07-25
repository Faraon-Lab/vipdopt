Usage
=====

Requirements
------------

- Python 3.10
- Ansys Lumerical FDTD
- An installation of MPI, for running simulations in parallel
- Qt6
- Dependencies listed in `build_requirements.txt`

If creating a conda environment, there is not official PySide6 package yet. After activating your environment, you will need to install PySide6 with pip.


Using ``vipdopt``
-----------------

To run an optimization, Vipdopt requires a project_directory containing two files:

- A configuration file containing optimization parameters
- A `sim.json` file containing the base simulation file to run with Lumerical

More details regarding configuration files are located in the :doc:`config` section.

To run an optimization from a pre-existing project directory, run:

.. code-block:: bash

    python vipdopt optimize path/to/project/directory

To see more options, run

.. code-block:: bash

    python vipdopt --help