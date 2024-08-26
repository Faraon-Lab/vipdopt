import os
import sys
from pathlib import Path
import numpy as np
from collections import OrderedDict
import dictdiffer

# Import main package and add program folder to PATH.
sys.path.append( os.getcwd() )  # 20240219 Ian: Only added this so I could debug some things from my local VSCode
import vipdopt
from vipdopt.configuration import Config
from vipdopt.eval import ParameterHandler

TEMPLATE_DIR = "runs\sb2s3_6lyr"
TEMPLATE_FILE = "config_bilge_3d.yml"
OUTPUT_FILE = "config_bilge_3d_amended.yml"
template_path = (Path(TEMPLATE_DIR) / TEMPLATE_FILE).absolute()
output_path = (Path(TEMPLATE_DIR) / OUTPUT_FILE).absolute()

#Params formatted for convenient human editing
params = [
    {
        "var_name": "max_device_index",
        "var_values":[1.6, 1.8, 2.0, 2.2],
        "base_idx": -1,
        "formatStr":  "%.3f",
        "iterating": True,
        "short_form": "n",
        "long_form": "Sb2S3 Index"
    },
    {
        "var_name": "vertical_layer_height_um",
        "var_values":[0.18,0.24,0.30,0.36],
        "base_idx": 0,
        "formatStr":  "%.3f",
        "iterating": True,
        "short_form": "lyrhgt",
        "long_form": "Layer Height"
    },
    {
        "var_name": "r_weight",
        "var_values":[1.0,1.2,1.5],
        "base_idx": 0,
        "formatStr":  "%.3f",
        "iterating": True,
        "short_form": "rwgt",
        "long_form": "Red Weight"
    },
        {
        "var_name": "g_weights",
        "var_values":[1.0,0.8,0.5],
        "base_idx": 0,
        "formatStr":  "%.3f",
        "iterating": True,
        "short_form": "gwgt",
        "long_form": "Green Weights"
    },
    {
        "var_name": "mesh_spacing_um",
        "var_values":[0.099,0.051],
        "base_idx": -1,
        "formatStr":  "%.3f",
        "iterating": False,
        "short_form": "mesh",
        "long_form": "Mesh Spacing"
    },
    {
        "var_name": "geometry_spacing_lateral_um",
        "var_values":[0.165,0.085,0.099],
        "base_idx": -1,
        "formatStr":  "%.3f",
        "iterating": False,
        "short_form": "vxl",
        "long_form": "Lateral Voxel Spacing"
    },
    {
        "var_name": "background_index",
        "var_values":[1,1.5],
        "base_idx": 0,
        "formatStr":  "%.1f",
        "iterating": False,
        "short_form": "bkgind",
        "long_form": "Background Index"
    },
    {
        "var_name": "device_size_lateral_um",
        "var_values":[3.96,2.04],
        "base_idx": 0,
        "formatStr":  "%.2f",
        "iterating": False,
        "short_form": "devsz",
        "long_form": "Lateral Device Size"
    },
    # {
    #     "var_name": "boundary_conditions",
    #     "var_values":['periodic','pml'],
    #     "base_idx": -1,
    #     "formatStr":  "%s",
    #     "iterating": True,
    #     "short_form": "bcs",
    #     "long_form": "Boundary Conditions"
    # },
    {
        "var_name": "num_vertical_layers",
        "var_values":[3,6,8],
        # "var_values":[8,12,20],
        "base_idx": 1,
        "formatStr":  "%d",
        "iterating": False,
        "short_form": "lyr",
        "long_form": "Vertical Layers"
    },
    {
        "var_name": "border_constant_width",
        "var_values":[False, 2],
        "base_idx": 0,
        "formatStr": "%d",
        "iterating": False,
        "short_form": "bdr",
        "long_form": "Border Width"
    },
    {
        "var_name": "enforce_xy_gradient_symmetry",
        "var_values":[False, True],
        "base_idx": 0,
        "formatStr": "%d",
        "iterating": False,
        "short_form": "symm",
        "long_form": "Device Symmetry"
    }
]
# TODO: How to handle nested config changes???
# Probably need to use recursive dictionary functions in utils
# Write in paths as addresses in the params dictionary above
# And apply them to a copy of the config.
# TODO: Change it to handle linked parameter values, i.e. a x b = constant

ph = ParameterHandler.ParameterHandler()
ph.load_parameters(params)

c = Config.from_file(template_path)
change_list = []
change_list_str = []
for n, idx in enumerate(ph.all_coordinates()):
    # d = c.copy()
    # d.data, changes = ph.apply_changes(d.data, 
    #                               ph.current_sweep_parameter_values(idx))
    # ph.cast_to_int_if_int(d.data)
    # change_str = ph.create_parameter_identifier_string(idx)
    # output_path = template_path.parent / f'run{n}_{change_str}' / template_path.name
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # d.save(output_path)
    
    ch = list(dictdiffer.diff(ph.baseline_parameters, ph.current_sweep_parameter_values(idx)))
    ch = [x for x in ch if x[0] in ['change', 'changed']]
    if True: # len(ch) <= 1:
        d = c.copy()
        d.data, changes = ph.apply_changes(d.data, 
                                    ph.current_sweep_parameter_values(idx))
        ph.cast_to_int_if_int(d.data)
        change_str = ph.create_parameter_identifier_string(idx)
        
        if len(ch)==0:      # Baseline 
            output_path = template_path.parent / f'RUN{n}_{change_str}' / template_path.name
        else:
            p_vals = [params[x]['var_values'] for x in range(4)]
            d_vals = [d.data[params[x]['var_name']] for x in range(4)]
            
            if (p_vals[0].index(d_vals[0]) == p_vals[1].index(d_vals[1])) and (p_vals[2].index(d_vals[2]) == p_vals[3].index(d_vals[3])):
                output_path = template_path.parent / f'run{n}_{change_str}' / template_path.name
                
        output_path.parent.mkdir(parents=True, exist_ok=True)
        d.save(output_path, sort_keys=False)        #! NOTE: DOES NOT OVERWRITE
        print(idx)
    
        change_list.append(changes)
        change_list_str.append(change_str)

# c = Config.from_file(template_path)
# c.data, changes = ph.apply_changes(c.data, ph.current_sweep_parameter_values((0,1)))
# ph.cast_to_int_if_int(c.data)
# c.save(output_path, sort_keys=False)

# Compare differences just to check
d = Config.from_file(template_path)
# e = Config.from_file(output_path)
# e = Config.from_file(template_path.with_name('config_bilge_3d_lyr_6_bdr_0.yml'))
# f = Config.from_file(template_path.parent / 'config_bilge_3d_actualsb2s3butnotworking.yml')

# import dictdiffer
# testlist = list(dictdiffer.diff(d.data, e.data))

print(3)