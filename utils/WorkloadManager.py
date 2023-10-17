import os
import re
from utils import utility

class SLURM:
    '''Contains functions to interact with the SLURM workload manager used by many HPC systems.'''

    def __init__(self):
        self.job_id = os.getenv('SLURM_JOB_ID')
        self.job_nodelist = os.getenv('SLURM_JOB_NODELIST')
        self.cluster_hostnames = self.get_slurm_node_list()
        self.num_nodes_available = int(os.getenv('SLURM_JOB_NUM_NODES'))	# should equal --nodes in batch script. Could also get this from len(cluster_hostnames)
        self.num_cpus_per_node = int(os.getenv('SLURM_NTASKS_PER_NODE'))	# should equal 8 or --ntasks-per-node in batch script. Could also get this SLURM_CPUS_ON_NODE or SLURM_JOB_CPUS_PER_NODE?
        # logging.info(f'There are {len(cluster_hostnames)} nodes available.')
        logging.info(f'There are {self.num_nodes_available} nodes available, with {self.num_cpus_per_node} CPUs per node. Cluster hostnames are: {self.cluster_hostnames}')
    
    def get_slurm_node_list( self, slurm_job_env_variable=None ):
        if slurm_job_env_variable is None:
            slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
        if slurm_job_env_variable is None:
            raise ValueError('Environment variable does not exist.')

        solo_node_pattern = r'hpc-\d\d-[\w]+'
        cluster_node_pattern = r'hpc-\d\d-\[.*?\]'
        solo_nodes = re.findall(solo_node_pattern, slurm_job_env_variable)
        cluster_nodes = re.findall(cluster_node_pattern, slurm_job_env_variable)
        inner_bracket_pattern = r'\[(.*?)\]'

        output_arr = solo_nodes
        for cluster_node in cluster_nodes:
            prefix = cluster_node.split('[')[0]
            inside_brackets = re.findall(inner_bracket_pattern, cluster_node)[0]
            # Split at commas and iterate through results
            for group in inside_brackets.split(','):
                # Split at hyphen. Get first and last number. Create string in range
                # from first to last.
                node_clump_split = group.split('-')
                starting_number = int(node_clump_split[0])
                try:
                    ending_number = int(node_clump_split[1])
                except IndexError:
                    ending_number = starting_number
                for i in range(starting_number, ending_number+1):
                    # Can use print("{:02d}".format(1)) to turn a 1 into a '01'
                    # string. 111 -> 111 still, in case nodes hit triple-digits.
                    output_arr.append(prefix + "{:02d}".format(i))
        return output_arr
