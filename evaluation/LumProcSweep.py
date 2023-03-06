import os
import sys
import shelve

# Add filepath to default path
sys.path.append(os.getcwd())
#! Gets all parameters from config file - explicitly calls it so that the variables pass into this module's globals()
from configs.LumProcParameters import *

               
print('Reached end of file. Code completed.')

