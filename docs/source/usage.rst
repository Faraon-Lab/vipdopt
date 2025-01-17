Usage
=====

Requirements
------------

- Python 3.10
- Ansys Lumerical FDTD 2021 edition or later
- An installation of MPI, for running simulations in parallel
- Qt6

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

Outputs
_______

All created files and logs are packaged and output to the selected project folder by default. The internal folder structure is as follows:

- ``data``:  Data generated in the optimization
    - ``opt_info``: captures pertinent information about each iteration and epoch of the optimization. Save data is also collected here for the event that an optimization needs to be restarted.
- ``.tmp``: Lumerical ``.fsp`` simulation files and log files that are used to calculate the adjoint and forward E-fields for calculation of the optimization gradient.

The folder ``opt_info`` collects information about optical figures of merit, the latest 3-D permittivity values of the designed device, the adjoint gradient, and contains auto-generated plots of the evolution of these values throughout the optimization.  
(Please note that values such as transmission are not normalized rigorously for these auto-generated plots, as the inclusion of the proper monitors in Lumerical for rigor would slow down each optimization iteration greatly.)

A log file is also created in the main project folder, named with the specified name using the ``--log`` option (defaults to ``dev.log``). 

GUI
___

The GUI [#]_ can be started by running

.. code-block:: bash

    python vipdopt gui

.. [#] The GUI is deprecated as of ``vipdopt`` version 2.0. Some functionality may still be possible but it is untested and unstable.

This will launch a dashboard that can monitor the progress of a optimization.

.. image:: ../dashboard.png
    :alt: A Screenshot of an example optimization loaded in the GUI

From the dashboard you can also open a dialog for editing [#]_ the optimization parameters

.. [#] Editing and saving an optimization though the GUI is not yet fully supported. However you can still use this window to view the various parameters.

.. image:: ../gui_demo.gif
    :alt: A .GIF demo using the GUI

While starting and stopping the optimization through the GUI is not completely supported yet, there are some experimental features pertaining to those functions.

Clicking the "Start Optimization" button will create a slurm script that can be submitted to run the optimization. To alter the output file format, edit the `submission script`_.
The parameters of running the job are included in this file according to the `SLURM documentation`_.  More information can be found there.


.. _SLURM documentation: https://slurm.schedmd.com/sbatch.html
.. _submission script: ../../vipdopt/submit.sh.

