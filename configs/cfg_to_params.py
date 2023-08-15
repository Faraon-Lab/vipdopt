# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import pickle
import shutil
import sys
import yaml
import torch

#* Establish folder structure & location
python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
python_src_directory = os.path.dirname(python_src_directory)		# Go up one folder


#* Input Arguments (passed in bash script)
if len( sys.argv ) < 3:							# Print instructions on what input arguments are needed.
	print( "Usage: python " + str( sys.argv[ 0 ] ) + " { yaml file name }" )
	sys.exit( 1 )
 
yaml_filename = sys.argv[ 1 ]

#* Load YAML file
with open( yaml_filename, 'rb' ) as yaml_file:
	parameters = yaml.safe_load( yaml_file )


#* Create project folder to store all files
projects_directory_location = 'ares' + os.path.basename(python_src_directory)
# projects_directory_location = sys.argv[ 2 ]
if not os.path.isdir(projects_directory_location):
	os.mkdir(projects_directory_location)

#* Output / Save Paths
parameters['DATA_FOLDER'] = projects_directory_location
parameters['MODEL_PATH'] = os.path.join(projects_directory_location, 'model.pth')
parameters['OPTIMIZER_PATH'] = os.path.join(projects_directory_location, 'optimizer.pth')


#* Save out the various files that exist right before the optimization runs for debugging purposes.
# If these files have changed significantly, the optimization should be re-run to compare to anything new.

if not os.path.isdir( projects_directory_location + "/saved_scripts/" ):
	os.mkdir( projects_directory_location + "/saved_scripts/" )

shutil.copy2( python_src_directory + "/main.py", projects_directory_location + "/saved_scripts/main.py" )
shutil.copy2( os.path.join(python_src_directory, yaml_filename), 
             projects_directory_location + "/saved_scripts/" + os.path.basename(yaml_filename) )
shutil.copy2( python_src_directory + "/configs/cfg_to_params.py", projects_directory_location + "/saved_scripts/cfg_to_params.py" )
#TODO: et cetera... might have to save out various scripts from each folder



#* Derived Parameters

#TODO: Move from main.py to here


#
#* Debug Options
#

running_on_local_machine = False
slurm_job_env_variable = os.getenv('SLURM_JOB_NODELIST')
if slurm_job_env_variable is None:
    running_on_local_machine = True
    
if running_on_local_machine:	
    lumapi_filepath =  r"C:\Program Files\Lumerical\v212\api\python\lumapi.py"
else:	
    #! do not include spaces in filepaths passed to linux
    lumapi_filepath = r"/central/home/ifoo/lumerical/2021a_r22/api/python/lumapi.py"


start_from_step = 0 	    # 0 if running entire file in one shot
shelf_fn = 'save_state'


print('Config imported and processed.')