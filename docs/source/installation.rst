Installation
============

``vipdopt`` can be installed using ``pip`` or from `source <https://github.com/Faraon-Lab/vidpopt>`_.

Requirements
------------

``vipdopt`` has a number of prerequisites for installation:


* Python 3.10
* Ansys Lumerical FDTD 2021 edition or later
* An installation of MPI, for running simulations in parallel
* Qt6

There are also a number of dependencies on other Python libraries, all of which
will be installed automatically with ``vipdopt``:

.. code-block:: txt
    scipy==1.12.*
    numpy==1.26.*
    numpy-stl==3.1.*
    matplotlib==3.8.*
    scikit-image==0.22.*
    overrides==7.7.*
    PyYAML==6.*
    types-PyYAML==6.*
    python-magic==0.4.*
    Jinja2==3.1.*
    PySide6==6.3.0
    PySide6-stubs==6.4.2
    gdstk==0.9.50

Installing with pip
-------------------


To install using pip run

.. code-block:: bash

    pip install vipdopt


We discuss installation from the source code below.

Installation from Source
------------------------

To install using the source code, first clone the `GitHub repository <https://github.com/Faraon-Lab/vidpopt>`_
using git. This can be done with the following command:

.. code-block:: bash

    git clone https://github.com/Faraon-Lab/vidpopt.git
    cd vipdopt

Then you want to do the following:

1. Create a development environment (virtual environemnt or conda environment)
2. Install all the dependencies (build, dev, and test requirements)
3. Build `vipdopt` (this will look the same regardless of if you're using vevn or conda)

If you're using pip this will look like:

.. tabs::

    .. tab:: Virtual Env

        Create and activate a new virtual environment named ``venv``.

        .. code-block:: bash

            python -m venv venv
            source venv/bin/activate

        Then install the Python dependencies from PyPi:

        .. code-block:: bash

            python -m pip install -r all_requirements.txt
    
    .. tab:: Anaconda

        Create and activate a new conda environment named ``vipdopt-dev`` with the
        requirements installed with:

        .. code-block:: bash

            conda env create -f environment.yaml
            conda activate vipdopt-dev

To build ``vipdopt`` in an activated environment, run:

.. code-block:: bash

    python setup.py install


Lumerical Installation Path
___________________________

By default, running ``setup.py`` will search for Lumerical in the default directory.
This is ``C:\Program Files\Lumerical`` for Windows or ``/opt/lumerical/`` on Linux.

If you would like to use an installation of Lumerical that is *not* in the default
installation diectory, you can specify this by editing ``setup.cfg``.

The property ``lumapi_path`` is the path to the Lumerical Python API. It might be

.. code-block::

    C:\Program Files\Lumerical\v221\api\python\lumapi.py

You can also change your preferred version with the ``lumerical_version`` option.
This option supports glob patterns. If unset, it will use the latest compatible version.

