## PSEUDOCODE

import configs      # Use a Namespace to keep track of all config variables
import eval         
import utils

# Are we running it locally or on SLURM or on AWS or?
workload_manager = configs.workload_manager
# Are we running using Lumerical or ceviche or fdtd-z or SPINS or?
simulator = configs.simulator
# Depending on which one we choose here - different converter files should be run, 
# e.g. lum_utils.py for Lumerical and slurm_handler.py for SLURM


#* Parallel computing
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

comm = MPI.COMM_WORLD
resource_size = MPI.COMM_WORLD.Get_size()
processor_rank = MPI.COMM_WORLD.Get_rank()


#* The master job has rank 0 and contains the main process:
if processor_rank == 0:

	
	#* Step 0: Set up simulation conditions and environment.
	# i.e. current sources, boundary conditions, supporting structures, surrounding regions.
	# Also any other necessary editing of the Lumerical environment and objects.
	
	simulation_environment = configs.simulation_environment		# Pointer to a JSON/YAML file describing each simulation object
	simulations = simulator.import(simulation_environment)
	# simulations is a list of Simulations objects. 
	# Each Simulation object should contain a dictionary from which the corresponding simulation file can be regenerated
	# Multiple simulations may be created here due to the need for large-area simulation segmentation, or genetic optimizations
	
	
	#* Step 1: Design Region(s) + Constraints
	# e.g. feature sizes, bridging/voids, permittivity constraints, corresponding E-field monitor regions
	
	# Create instances of devices to store permittivity values
	devices = []
	for num_region, region in enumerate(region_list):
		devices[num_region] = Device.Device(
			voxel_array_size,
			feature_dimensions,
			permittivity_constraints,
			region_coordinates,
			other_arguments
		)
		
		for simulation in simulations:
			# check coordinates of device region. Does it fit in this simulation?
			# If yes, construct all or part of region and mesh in simulation
			simulation.create_device_region(region)
			simulation.create_device_region_mesh(region)

			
			
	#! Must keep track of which devices correspond to which regions and vice versa.
	#? What is the best method to keep track of maps? Try: https://stackoverflow.com/a/21894086
	device_to_region_map = bidict(devices)
	# For each region, create a field_shape variable to store the E- and H-fields for adjoint calculation.

	#! Shape Check! Check the mesh for the E-field monitors and adjust each field_shape variable for each device to match.
	# Instead of a lookup table approach, probably should just directly pull from the simulator - we can always interpolate anyway.
	
	
	#* Step 2: Set up figure of merit. Determine monitors from which E-fields, Poynting fields, and transmission data will be drawn.
	# Handle all processing of weighting and functions that go into the figure of merit
	
	def overall_figure_of_merit( indiv_foms, weights ):
		return 3

	weights = configs.weights
	indiv_foms = configs.indiv_foms
	
	fom_monitors = []
	for indiv_fom in indiv_foms:
		
		# Adjoint monitors and what data to obtain should be included in indiv_fom.
		fom_monitor = create_monitor_dict(indiv_fom)
	
		for simulation in simulations:
			# check coordinates of monitor region. Does it fit in this simulation?
			# If yes, construct all or part of region and mesh in simulation
			simulation.create_monitors(fom_monitor)
		
	fom_to_monitor_map = bidict( indiv_foms, fom_monitors)
	# During the optimization we would disable and enable monitors depending on which adjoint simulation
	
	#*
	#* AT THIS POINT THE OPTIMIZATION LOOP BEGINS
	#*

	#
	#* Run the optimization
	#

	start_epoch = restart_epoch
	for epoch in range(start_epoch, num_epochs):
		
		# Each epoch the filters get stronger and so the permittivity must be passed through the new filters
		opt_device.update_filters(epoch)
		opt_device.update_permittivity()
  
		start_iter = 0

		if epoch == start_epoch:
			start_iter = restart_iter
			opt_device.current_iteration = start_epoch * num_iterations_per_epoch + start_iter

		for iteration in range(start_iter, num_iterations_per_epoch):
			logging.info("Working on epoch " + str(epoch) + " and iteration " + str(iteration))

			job_names = {}

			#
			# Step 3: Import the current epoch's permittivity value to the device. We create a different optimization job for:
			# - each of the polarizations for the forward source waves
			# - each of the polarizations for each of the adjoint sources
			# Since here, each adjoint source is corresponding to a focal location for a target color band, we have 
			# <num_wavelength_bands> x <num_polarizations> adjoint sources.
			# We then enqueue each job and run them all in parallel.
			# NOTE: Polymer slab / substrate, if included, must change index when the cube index changes.
			#

			if start_from_step == 0 or start_from_step == 3:
				if start_from_step == 3:
					load_backup_vars(shelf_fn)
			
				logging.info("Beginning Step 3: Forward and Adjoint Optimizations")


				# import cur_index into device regions
				for device in devices:
					check regions
					# Get current density of bayer filter after being passed through current extant filters.
					cur_density = opt_device.get_permittivity()		# Despite the name, here cur_density will have values between 0 and 1
					convert device to permittivity and then index
					calculate material % and binarization level, store away
					cut cur_index by region
					upsample and then importnk2

				job_names = {}
				# Construct forward and adjoint sources.
				# Disable and enable monitors depending on which adjoint simulation
				# Adjoint simulations are NOT needed if you have autograd

				for each job:
					job_names[ ( 'pdaf_fwd', xy_idx, dispersive_range_idx ) ] = add_job( job_name, jobs_queue )
					# There's got to be a better way to label these job_names

				# Run all jobs
				run_jobs( jobs_queue )



			#
			# Step 4: Compute the figure of merit
   			#
	
			if start_from_step == 0 or start_from_step == 4:
				if start_from_step == 4:
					load_backup_vars(shelf_fn)
			
				logging.info("Beginning Step 4: Computing Figure of Merit")

				for device in devices:

					def get_figure_of_merit(indiv_fom):
						load every forward simulation file
						grab E-fields from monitors according to fom_to_monitor_data[]
						calculate figure of merit according to weights and FoM instructions
						return figures_of_merit_numbers
		
					figures_of_merit_number = []
					for indiv_fom in indiv_foms:
						figures_of_merit_number.append( get_figure_of_merit(indiv_fom) )
					list_overall_figure_of_merit = overall_figure_of_merit( figures_of_merit_number, weights )
					# each device needs to remember its FoM too
	
					# Save out variables that need saving for restart and debug purposes
					np.save(os.path.abspath(OPTIMIZATION_INFO_FOLDER + "/figure_of_merit.npy"), np.sum(fom_evolution[optimization_fom], (2,3,4)))
					# ... etc
		
					# Plot key information such as Figure of Merit evolution for easy visualization and checking in the middle of optimizations
					plotter.plot_fom_trace(figure_of_merit_evolution, OPTIMIZATION_PLOTS_FOLDER)
					plotter.plot_quadrant_transmission_trace(fom_evolution['transmission'], OPTIMIZATION_PLOTS_FOLDER)
					plotter.plot_individual_quadrant_transmission(quadrant_transmission_data, lambda_values_um, OPTIMIZATION_PLOTS_FOLDER,
															epoch, iteration=30)	# continuously produces only one plot per epoch to save space
					plotter.plot_overall_transmission_trace(fom_evolution['transmission'], OPTIMIZATION_PLOTS_FOLDER)
					plotter.visualize_device(cur_index, OPTIMIZATION_PLOTS_FOLDER)
					# plotter.plot_moments(adam_moments, OPTIMIZATION_PLOTS_FOLDER)
					# plotter.plot_step_size(adam_moments, OPTIMIZATION_PLOTS_FOLDER)
		
					#* Here is where we would start plotting the loss landscape. Probably should be accessed by a separate class...
					# Or we could move it to the device step part
					loss_landscape_mapper = LossLandscapeMapper.LossLandscapeMapper(simulations, devices)


			#
			# Step 5: Run all the adjoint optimizations for both x- and y-polarized adjoint sources and use the results to compute the
			# gradients for x- and y-polarized forward sources.
			#

			if start_from_step == 0 or start_from_step == 5:
				if start_from_step == 5:
					load_backup_vars(shelf_fn)

				logging.info("Beginning Step 5: Adjoint Optimization Jobs")

				def overall_adjoint_gradient( indiv_adjoints, weights ):
					return 3

				def get_adjoint(indiv_adjoint):
					load every adjoint simulation file
					grab E-fields from design region monitor for each adjoint simulation
					calculate figure of merit according to weights and FoM instructions
					return adjoint gradient

				for device in devices:
					# Initialize arrays to store the gradients of each polarization, each the shape of the voxel mesh
					# This array therefore has shape (polarizations, mesh_lateral, mesh_lateral, mesh_vertical)
					xy_polarized_gradients = [ np.zeros(field_shape, dtype=np.complex128), np.zeros(field_shape, dtype=np.complex128) ]

					
					#* Process the various figures of merit for performance weighting.
					# Copy everything so as not to disturb the original data
					process_fom_evolution = copy.deepcopy(fom_evolution)
	 

					adjoint_indiv_number = []
					for indiv_fom in indiv_foms:
						adjoint_indiv_number.append( get_adjoint(indiv_adjoint) )
					list_overall_adjoint_gradient = overall_adjoint_gradient( adjoint_indiv_number, weights )
	 
					# Get the full device gradient by summing the x,y polarization components 
					device_gradient = 2 * ( xy_polarized_gradients[0] + xy_polarized_gradients[1] )
					# Project / interpolate the device_gradient, the values of which we have at each (mesh) voxel point, and obtain it at each (geometry) voxel point
					device_gradient_interpolated = interpolate.interpn( ( gradient_region_x, gradient_region_y, gradient_region_z ), device_gradient, bayer_filter_region, method='linear' )
		
	
					# Each device needs to remember its gradient!
	
	
	
				logging.info("Completed Step 5: Retrieved Adjoint Optimization results and calculated gradient.")
				gc.collect()
				backup_all_vars()
	



			#
			# Step 6: Step the design variable.
			#

			if start_from_step == 0 or start_from_step == 6:
				if start_from_step == 6:
					load_backup_vars(shelf_fn)
			
				logging.info("Beginning Step 6: Stepping Design Variable.")
				# TODO: Rewrite this part to use optimizers / NLOpt. -----------------------------------
	
				if enforce_xy_gradient_symmetry:
					device_gradient_interpolated = 0.5 * ( device_gradient_interpolated + np.swapaxes( device_gradient_interpolated, 0, 1 ) )
				device_gradient = device_gradient_interpolated.copy()
	
				# Backpropagate the design_gradient through the various filters to pick up (per the chain rule) gradients from each filter
				# so the final product can be applied to the (0-1) scaled permittivity
				design_gradient = opt_device.backpropagate( device_gradient )

				cur_design_variable = opt_device.get_design_variable()
				# Get current values of design variable before perturbation, for later comparison.
				last_design_variable = cur_design_variable.copy()
	
 
				def scale_step_size(epoch_start_design_change_min, epoch_start_design_change_max, 
                        			epoch_end_design_change_min, epoch_end_design_change_max):
					'''Begin scaling of step size so that the design change stays within epoch_design_change limits in config.'''
					return 3

				# Feed proposed design gradient into bayer_filter, run it through filters and update permittivity
				# negative of the device gradient because the step() function has a - sign instead of a +
				# NOT the design gradient; we do not want to backpropagate twice!

				if optimizer_algorithm.lower() in ['adam']:
					adam_moments = bayer_filter.step_adam(epoch*num_iterations_per_epoch+iteration, -device_gradient,
													adam_moments, step_size, adam_betas)
				else:
					bayer_filter.step(-device_gradient, step_size)
				cur_design_variable = bayer_filter.get_design_variable()
				# Get permittivity after passing through all the filters of the device
				cur_design = bayer_filter.get_permittivity()

				#* Save out overall savefile, save out all save information as separate files as well.

				logging.info("Completed Step 4: Saved out cur_design_variable.npy and other files.")
				backup_all_vars()


#* If not the master, then this is one of the workers.
else:

	#* Run a loop waiting for something to do or the notification that the optimization is done
	while True:

		# This is a blocking receive call, expecting information to be sent from the master
		data = comm.recv( source=0 )

		if data[ "instruction" ] == "terminate":
			break
		elif data[ "instruction" ] == "simulate":
			processor_simulate( data )
		else:
			logging.info( f'Unknown call to master {processor_rank} of {data["instruction"]}.\nExiting...' )
			sys.exit( 1 )



shutil.copy2(utility.loggingArgs['filename'], SAVED_SCRIPTS_FOLDER + "/" + utility.loggingArgs['filename'])
logging.info("Reached end of file. Program completed.")
