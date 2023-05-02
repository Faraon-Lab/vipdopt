# SONY Volumetric Inverse Photonic (VIP) Design Optimizer
Copyright © 2023, California Institute of Technology. All rights reserved.

Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:

* Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
* Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  


# Project Description
This code performs an inverse-design optimization based on the adjoint optimization technique [2] that designs a freeform multilayer optical Bayer filter and focusing lens as described in [1]. The Bayer filter is multiwavelength and focuses and sorts different spectral bands into different prescribed locations on the focal plane. Otherwise known as a color router, this code is set by default to create a 10-layer filter for the visible spectral range, with lateral and vertical dimensions of 2.04 microns. Different options are available in the config to, for example, enable polarization sorting or adjust the spectral bands as necessary.

References: [[1]](https://doi.org/10.1364/OPTICA.384228), [[2]](https://doi.org/10.1364/OE.21.021693)  


## Folder Structure
This section will give basic descriptions of the software workflow and what is stored where.
References and adapts [this](https://theaisummer.com/best-practices-deep-learning-code/) article.
Further [best practices](https://neptune.ai/blog/how-to-organize-deep-learning-projects-best-practices).

- `configs`: in configs we define every single thing that can be configurable and can be changed in the future. Good examples are optimization hyperparameters, folder paths, the optimization architecture, metrics, flags.
- `evaluation`: is a collection of code that aims to evaluate the performance and accuracy of our model.
- `trials`: contains past trials with all the save data necessary to reconstruct (permittivity data, config etc.)
- `utils`: utilities functions that are used in more than one places and everything that don’t fall in on the above come here.

![This image](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/DL-project-directory.png) also gives a good overview of what forms folder structure for optimization code can take.  


# Tutorial

## Configuration

The code is set up such that many of the hyperparameters of the optimization (both physical and computational) are easily changed from two config files, both located in the `/configs/` folder. The first is called `test_config_sony.yaml` and contains the simplest level of parameters, i.e. parameters that do not depend on anything else. The second is called `SonyBayerFilterParameters.py` and performs processing of these parameters to derive other values, e.g. Device Height = Num. Vertical Layers $\times$ Layer Height.

The following are some parameters in `test_config_sony.yaml` that are often adjusted, and their default values:
```
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


## Running the Code

### Environment
First, we point the optimization code to the current Lumerical installation. The following lines must be edited:
```
/configs/test_config_sony.yaml:
    lumapi_filepath_local: "<LUMERICAL_FOLDER>\\api\\python\\lumapi.py"
    lumapi_filepath_hpc: "<LUMERICAL_FOLDER>/api/python/lumapi.py"

/utils/run_proc.sh:
    <LUMERICAL_FOLDER>/mpich2/nemesis/bin/mpiexec -verbose -n 8 -host $1 <LUMERICAL_FOLDER>/bin//fdtd-engine-mpich2nem -t 1 $2 
    > dev/null 2> /dev/null
```
Parts of the code distinguish between running on a local machine and running on a HPC cluster, so we just have to fill in the appropriate addresses as needed.  


### Initiating the Optimization
The current code is set up to interface with the SLURM job scheduler on a HPC cluster. The batch script `slurm_vis10lyr.sh` should be called with the command:
```
sbatch slurm_vis10lyr.sh
```
and the parameters of running the job are included in this file according to the [SLURM Documentation](https://slurm.schedmd.com/sbatch.html). More information can be found there.

That file calls `SonyBayerFilterOptimization.py` with the `filename` argument `/configs/test_config_sony.yaml`. If different configuration files must be used, the filename argument should be adjusted accordingly. This file contains the bulk of the optimization code.

### Outputs
All created files and logs are packaged and output to the folder `/_trials_<PROJECT_FOLDER>` by default, where `<PROJECT_FOLDER>` indicates the folder to which this code is installed. The internal folder structure is as follows:

- `lumproc_<PROJECT_FOLDER>`: Packages selected output files for easy evaluation.
- `opt_info`: captures pertinent information about each iteration and epoch of the optimization. Save data is also collected here for the event that an optimization needs to be restarted.
- `saved_scripts`: Saves out a snapshot of important script files like `SonyBayerFilterOptimization.py` so that results can be reconstructed easily.
- `Assorted .fsp files`: Lumerical simulation files that are used to calculate the adjoint and forward E-fields for calculation of the optimization gradient.

The folder `opt_info` collects information about optical figures of merit, the latest 3-D permittivity values of the designed device, the adjoint gradient, and contains auto-generated plots of the evolution of these values throughout the optimization.  
(Please note that values such as transmission are not normalized rigorously for these auto-generated plots, as the inclusion of the proper monitors in Lumerical for rigor would slow down each optimization iteration greatly.)

A log file is also created in the main project folder, named with the timestamp (obtained from `datetime.now().timestamp()`) and, if the code was run on a HPC cluster, the job ID for easy identification.


### Restarting
Occasionally the optimization may glitch out partway due to a simulation error in Lumerical, or simply a timeout on the cluster workload manager. In this case, it is straightforward to restart the optimization from the last working iteration:

1. Make sure that the `opt_info` folder contains the most updated savefiles and data. The pertinent files are, in order of importance (but all must be, at the very least, present):
- `cur_design_variable.npy`
- `figure_of_merit.npy`
- `<foo>_by_focal_pol_wavelength.npy` (multiple files)
- `average_design_change_evolution.npy`
- `max_design_change_evolution.npy`
- `step_size_evolution.npy`

2. Check what the most recent epoch and iteration were. This will be contained in the latest logfile toward the end. Then, adjust the following parameters in `/configs/test_config_sony.yaml`, for example:
```
restart_epoch: 5            # Both 0 if not restarting
restart_iter: 14
```

3. Run `slurm_vis10lyr.sh` again.