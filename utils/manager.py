import time
import queue
import subprocess
import logging
import os
import numpy as np

def create_tempfiles_fwd_parallel(indiv_foms, simulator, simulation):
	# Each FoM has a list of corresponding forward sources. Collect all of the unique ones in a list.
	fwd_src_arr = []
	for f in indiv_foms:
		if any(f.enabled):									# Only add if at least one of the frequencies for this FoM is enabled
			if f.fwd_srcs not in fwd_src_arr:				# Only add it if it's unique
				fwd_src_arr.append(f.fwd_srcs)
			f.tempfile_fwd_name = fwd_src_arr.index(f.fwd_srcs)	# Find where the fwd_srcs entry is in the extant list
																	# NOTE: temporarily stored as index.
 
	# Each entry in this array corresponds to a single simulation which must have all of the forward sources in that entry enabled.
	# Here we create each simulation, and add it to the job queue.
	tempfile_fwd_name_arr = [None]*len(fwd_src_arr)		# Filenames matched to simulations which contain each of the forward source combinations.
	for i, src_list in enumerate(fwd_src_arr):
		
		# Each entry is a unique combination of sources
		for src in src_list:
			src.src_dict['enabled'] = 1			# Enable that source
			# src.src_dict = cfg.lm.fdtd_update_object( simulator.fdtd_hook, src.src_dict, create_object=False )
			src.src_dict = simulator.fdtd_update_object( src.src_dict, create_object=False )

		# Make a temporary file
		tempfile_fwd_name = f'temp_fwd_{i}'
		simulator.save(simulation['SIM_INFO']['path'], tempfile_fwd_name)
		tempfile_fwd_name_arr[i] = tempfile_fwd_name

		# Reset the simulation
		simulator.disable_all_sources(simulation)

	return fwd_src_arr, tempfile_fwd_name_arr

def create_tempfiles_adj_parallel(indiv_foms, simulator, simulation):
	# Make array of active and inactive adjoint sources / dipoles.
	# Each FoM has a list of corresponding forward sources. Collect all of the unique ones in a list.
	active_adj_src_arr = []
	inactive_adj_src_arr = []		# todo: is inactive_adj_src_arr really needed?
	for f in indiv_foms:
		if any(f.enabled) or any(f.enabled_restricted):		# Only add if at least one of the frequencies for this FoM is enabled
			if f.adj_srcs not in active_adj_src_arr:		# Only add it if it's unique
				active_adj_src_arr.append(f.adj_srcs)
			f.tempfile_adj_name = active_adj_src_arr.index(f.adj_srcs)	# Find where the adj_srcs entry is in the extant list
																	# NOTE: temporarily stored as index.
		else:
			inactive_adj_src_arr.append(f.adj_srcs)
		
	# Each entry in this array corresponds to a single simulation which must have all of the adjoint sources in that entry enabled.
	# Here we create each simulation, and add it to the job queue.
	tempfile_adj_name_arr = [None]*len(active_adj_src_arr)		# Filenames matched to simulations which contain each of the adjoint source combinations.
	for i,src_list in enumerate(active_adj_src_arr):

		# Each entry is a unique combination of sources
		for src in src_list:
			src.src_dict['enabled'] = 1			# Enable that source
			# src.src_dict = cfg.lm.fdtd_update_object( simulator.fdtd_hook, src.src_dict, create_object=False )
			src.src_dict = simulator.fdtd_update_object( src.src_dict, create_object=False )

		# Make a temporary file
		tempfile_adj_name = f'temp_adj_{i}'
		simulator.save(simulation['SIM_INFO']['path'], tempfile_adj_name)
		tempfile_adj_name_arr[i] = tempfile_adj_name

		# Reset the simulation
		simulator.disable_all_sources(simulation)

	return active_adj_src_arr, inactive_adj_src_arr, tempfile_adj_name_arr

#
#* Set up queue for parallel jobs
#

def run_jobs(job_names, folder, workload_manager, simulator):
	'''Takes an array of filenames and creates a queue, runs all simulations in batches until everything is done.
	Args: folder where files are stored, workload_manager
	'''
	
	# Running locally and on Lumerical:
	if workload_manager is None and 'Lumerical' in simulator.__class__.__name__:
		for job_name in job_names:
			simulator.fdtd_hook.addjob(job_name+'.fsp')
		simulator.fdtd_hook.runjobs()

	else:
		jobs_queue = queue.Queue()
		for job_name in job_names:
			jobs_queue.put( os.path.join(os.path.abspath(folder), job_name+'.fsp') )
	
		small_queue = queue.Queue()
		while not jobs_queue.empty():
			for node_idx in range( 0, workload_manager.num_nodes_available ):
				# this scheduler schedules jobs in blocks of num_nodes_available, waits for them ALL to finish, and then schedules the next block
				# TODO: write a scheduler that enqueues a new job as soon as one finishes
				# not urgent because for the most part, all jobs take roughly the same time

				if jobs_queue.qsize() > 0:
					small_queue.put( jobs_queue.get() )
		
			run_jobs_inner( small_queue, workload_manager, simulator )

def run_jobs_inner( queue_in, workload_manager, simulator ):
	job_idx = 0
	processes = []

	while not queue_in.empty():
		get_job_path = queue_in.get()
		logging.info(f'Node {workload_manager.cluster_hostnames[job_idx]} is processing file at {get_job_path}.')

		process = subprocess.Popen(
			[
				#! Make sure this points to the right place
				# os.path.abspath(python_src_directory + "/utils/run_proc.sh"),		# TODO: There's some permissions issue??
				'/home/gdrobert/Develompent/adjoint_lumerical/inverse_design/run_proc.sh',
				workload_manager.cluster_hostnames[ job_idx ],
				get_job_path
			]
		)
		processes.append( process )
		job_idx += 1


	job_queue_time = time.time()									# Record the current time
	completed_jobs = [ 0 for i in range( 0, len( processes ) ) ]	# Create a list of Booleans corresponding to each process
	
	while np.sum( completed_jobs ) < len( processes ):		# While there are still jobs uncompleted:
		for job_idx in range( 0, len( processes ) ):
			if completed_jobs[ job_idx ] == 0:	
				poll_result = processes[ job_idx ].poll()	# Check through each uncompleted job if it's completed?
				if not( poll_result is None ):
					completed_jobs[ job_idx ] = 1			# That job is completed
					logging.info(f'Node {workload_manager.cluster_hostnames[job_idx]} has completed process.')



		if time.time()-job_queue_time > workload_manager.max_wait_time:
			logging.CRITICAL(r"Warning! The following jobs are taking more than an hour and might be stalling:")
			logging.CRITICAL(completed_jobs)

		time.sleep( 5 )