# `vipdopt`: Volumetric Inverse Photonic Design Optimizer

Vipdopt is a package for streamlining the inverse deisgn of nanophotonic devices. This package largely serves as a Python-based wrapper for Lumerical. Vipdopt also provides an interactable GUI for creating optimizations and monitoring their progress.

This code performs an inverse-design optimization based on the adjoint optimization technique [^2] that designs a freeform multilayer optical Bayer filter and focusing lens as described in [^1]. The Bayer filter is multiwavelength and focuses and sorts different spectral bands into different prescribed locations on the focal plane. Otherwise known as a color router, this code is set by default to create a 10-layer filter for the visible spectral range, with lateral and vertical dimensions of 2.04 microns. Different options are available in the config to, for example, enable polarization sorting or adjust the spectral bands as necessary.

[^1]: https://doi.org/10.1364/OPTICA.384228
[^2]: https://doi.org/10.1364/OE.21.021693


## Tutorials, Examples, and Documentation

See the [documentation](http://vidpopt.readthedocs.io/)

## Requirements

- Python 3.10
- Ansys Lumerical FDTD 2021 edition or later
- An installation of MPI, for running simulations in parallel
- Qt6

If creating a conda environment, there is not official PySide6 package yet. After activating your environment, you will need to install PySide6 with pip.

## Usage (Command Line)

To run an optimization, Vipdopt requires a project_directory containing two files:
- A configuration file containing optimization parameters
- A `sim.json` file containing the base simulation file to run with Lumerical

More details regarding configuration files are located [here](vipdopt/configuration/README.md).

To run an optimization from a pre-existing project directory, run:
```sh
python vipdopt optimize path/to/project/directory
```

To see more options, run
```sh
python vipdopt --help
```

### Outputs
All created files and logs are packaged and output to the selected project folde by default. The internal folder structure is as follows:

- `data`:  Data generated in the optimization
    - `opt_info`: captures pertinent information about each iteration and epoch of the optimization. Save data is also collected here for the event that an optimization needs to be restarted.
- `.tmp`: Lumerical `.fsp` simulation files and log files that are used to calculate the adjoint and forward E-fields for calculation of the optimization gradient.

The folder `opt_info` collects information about optical figures of merit, the latest 3-D permittivity values of the designed device, the adjoint gradient, and contains auto-generated plots of the evolution of these values throughout the optimization.  
(Please note that values such as transmission are not normalized rigorously for these auto-generated plots, as the inclusion of the proper monitors in Lumerical for rigor would slow down each optimization iteration greatly.)

A log file is also created in the main project folder, named with the specified name using the `--log` option (defaults to `dev.log`). 

### GUI

The GUI can be started by running
```
python vipdopt gui
```

This will launch a dashboard that can monitor the progress of a optimization.

![A Screenshot of an example optimization loaded in the GUI](/docs/dashboard.png)

From the dashboard you can also open a dialog for editing [^3] the optimization parameters

[^3]: Editing and saving an optimization though the GUI is not yet fully supported. However you can still use this window to view the various parameters.

![A .GIF demo using the GUI](/docs/gui_demo.gif)

While starting and stopping the optimization through the GUI is not completely supported yet, there are some experimental features pertaining to those functions.

Clicking the "Start Optimization" button will create a slurm script that can be submitted to run the optimization. To alter the output file format, edit [`vipdopt/submit.sh`](/vipdopt/submit.sh). The parameters of running the job are included in this file according to the [SLURM Documentation](https://slurm.schedmd.com/sbatch.html). More information can be found there.
