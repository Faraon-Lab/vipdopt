# `vipdopt` Configuration

Vipdopt requires two files within a project in order to run an optimization: the [configuration file](#configuration-file) and the [simulation file](#simulation-file).

Currently the following file formats are supported:
- YAML
- JSON

# Configuration File

The configuration file contains basic information needed to setup the optimization.

If multiple configuration files are located in your project directory, you can specify which to use in the optimization:
```sh
python vipdopt optimize /path/to/project --config config_name.ext
```
If the specified config file is not found, the program will attempt to automatically locate a file using the glob pattern `project/dir/**/*config*.{yaml,yml,json}`

Below is a list of all required parameters in the configuration file

## Required Parameters
### `optimizer` and `optimizer_settings`
`optimizer`: The optimizer to use
`optimizer_settings` is a dictionary specifying the settings to instantiate the optimizer with.

Currently supported optimizers are:
- `AdamOptimizer`, which implements the adam algorithm[^1]
    - `step_size` (float): The step size to use; defaults to 0.01
    - `betas` (float, float): The betas to use; defaults to (0.9, 0.999)
    - `eps` (float): The epsilon value to use; defaults to 1e-8
- `GradientDescentOptimizer`
    - `step_size` (float): The step size to use; defaults to 0.01


### Lumerical Environment Variables
- `mpi_exe`: The path to the installation of MPI to use
- `solver_exe`: The path to the FDTD solver to run the sumlations with
- `nprocs`: The number of processeros to use for each simulation

When running the simulations, the following script will be called by Lumerical:
```sh
#!/bin/sh
MPI_EXE -n NPROCS SOLVER_EXE {PROJECT_FILE_PATH}
exit 0
```

### `figures_of_merit`
To describe the figures of merit (FoMs) to use in your optimization, a dictionary titled `figures_of_merit` is required containing a sub-dictionaries for each FoM, using the name of the FoM as the key. The overall FoM will be computed as the weighted sum of all the individual FoMs. The FoM sub-dictionaries contain the following values:
- `type`: The type of FoM to use. Currently includes `BayerFilterFoM` and `UniformFoM`
- `fom_monitors`: A list of monitors in the simulation to connect to for computing the FoM
    - Each monitor has the format `[source_name, monitor_name]`, with the appropriate values from the [simulation](#simulation-file)
- `grad_monitors`: A list of monitors in the simulation to connect to for computing the gradient
- `polarization`: The polarization to use
- `freq`: A list of all frequencies in the optimization
- `opt_ids` (optional): A List of the indices of the frequencies to use in the calculations; defaults to all frequencies
- `weight` (optional): The weight to appply to this FoM; defaults to 1.0

### Miscellaneous Parameters
- `num_bands`: The number of frequency bands
- `num_design_frequency_points`: 
- `lambda_values_um`: The wavelengths of light to use in the optimization
- `simulator_dimension`: Whether the simulator is `2D` or `3D`



## Optional Parameters
- `current_epoch`: The current epoch to start the optimization from; defaults to 0
- `current_iteration`: The current iteration within an epoch to start from; defaults to 0
- `max_epochs`: The maximum number of epochs to run before ending the optimization; defaults to 1.
- `iter_per_epoch`: The number of iterations within an epoch; defaults to 100.

You're also welcome to include your own parameters.

The following are some parameters in [`config_example.yaml`](/vipdopt/configuration/config_example.yml) that are often adjusted, and their default values:
```yaml
mesh_spacing_um: 0.017 
geometry_spacing_lateral_um: 0.085
device_scale_um: 0.051    # Controls vertical dimensions e.g. focal length, device vertical layer voxels, and FDTD gap sizes.
num_vertical_layers: 10
vertical_layer_height_um: 0.204
device_size_lateral_um: 2.04
sidewall_thickness_um: 0.24
sidewall_material: 'air'
lambda_min_um: 0.375
lambda_max_um: 0.725
f_number: 2.2
source_angle_theta_vacuum_deg: 0
source_angle_phi_deg: 0
num_epochs: 10
num_iterations_per_epoch: 30
desired_peaks_per_band_um: [ 0.48, 0.52, 0.59 ]
```  

### `device`
While technically the `device` parameter is optional, not including one will result in using our default device.

If you'd like to provide your own device specification, you have two options:
1. Provide a path to a `.npy` file saved with `vipdopt.optimization.device.Device.save`. This is mainly for loading existing projects.
2. Provide a dictionary with the following values:
    - `size` (int, int): The size (in voxels) of a device layer
    - `permittivity_constraints` (float, float): The min/max permittivity
    - `coords` (float, float, float): The coordinates in 3D space of the device
    - `name` (str): The name of the device
    - `init_density` (float): The initial density of the device, when not randomized
    - `randomize` (bool): Whether to randomize the initial device
    - `init_seed` (int): The random seed to use when initializing the device
    - `symmetric` (bool): Whther the initial device should be symmetric
    - `filters`: A List of filter objects to use with format
        ```json
        {
            "type": TYPE,
            "parameters": {
                ...
            }
        }
        ```
        - The types of filters currently supported are
            - `Sigmoid`, which has an `eta` and `beta` parameter defininfg a sigmoid function
            - `Scale`, which has `variable_bounds`, the min/max of scaled variables

# Simulation File

The simulation file is a JSON file containing all of the objects to create in a Lumerical project. This serves as the "base simulation" from which the optimization gets it's data.

The simulation consists of two sub-dictionaries: `info` and `objects`.

## `info`
This contains general information about a simulation. The values are
- `name`: The name of the simulation
- `path`: The path to save this simulation to.
- `simulator_name`: What simulator to use. Currently only supports `LumericalFDTD`
- `coordinates`: A dictionary of ...

## `objects`
The objects dictionary contains all of the simulation objects to create in the FDTD software.

Each object has the format:
```json
NAME: {
    "name": NAME,
    "obj_type": TYPE,
    "properties": {
        ...
    }
}
```
`properties` contains all of the properties one would set in Lumerical, e.g. `x span`, `z min`, `override global monitor settings`, etc.

The available types of objects are
- `fdtd`
- `rect`
- Source objects:
    - `gaussian`
    - `dipole`
    - `tfsf`
- `power`
- Device objects:
    - `profile`
    - `index`
    - `import`
    - `mesh`


Source objects (gaussian, dipole, and tfsf) have an additional field `attatched_monitor`, which is the index of the `power` monitor measuring data from this source.

It is possible to have multiple devices in your optimization. Therefore, all device objects (profile, index, import, and mesh) have an additional field `dev_id` which is the index of the device these values correspond to.

# Generating Configuration Files with Jinja2
Maintaining all of these different properties inside the configuration and simulation files can be tedious. For that reason, we provide an easy way to generate these files using Jinja2.

For an intro to using Jinja2 templating, see [this guide](https://ttl255.com/jinja2-tutorial-part-1-introduction-and-variable-substitution/)

We've provided a set of templates to use in the [`jinja_templates` directory](/jinja_templates/).

Once you have your template file created, and a data file (in YAMl or JSON format) to substitute values in, you can run
```
python -m vipdopt.configuration.template template_file_name.ext /path/to/data/file output_file_name.ext
```

Note that by default, this script will search for your template in `/jinja_templates/`. To overwrite this, use the `-s` command. For more details use `--help`


[^1]: https://arxiv.org/abs/1412.6980