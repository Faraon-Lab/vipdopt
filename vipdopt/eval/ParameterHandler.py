import numpy as np
from collections import OrderedDict
from typing import Any, List
import numpy.typing as npt
import vipdopt
import dictdiffer

PARAM_TEMPLATE = [ {
                    "var_name": "num_vertical_layers",
                    "var_values":[3,6,8],
                    "base_idx": 2,
                    "formatStr":  "%d",
                    "iterating": True,
                    "short_form": "lyr",
                    "long_form": "Vertical Layers"
                } ]

class ParameterHandler():
    def __init__(self):
        self.sweep_parameters = OrderedDict()
        self.const_parameters = OrderedDict()
        self.baseline_parameters = OrderedDict()
        self.sweep_parameter_values = None
    
    def load_parameters(self, parameters: list[dict[str, Any]] = PARAM_TEMPLATE):
        '''The params dict contains both parameters that are being swept and parameters that are not.
        Sorts accordingly.'''
        for val in parameters:
            # Identify parameters that are actually being swept
            if val['iterating']:
                self.sweep_parameters[val['var_name']] = val
            else:
                self.const_parameters[val['var_name']] = val
        self.generate_baseline_parameters()
        self.generate_sweep_parameter_values()
        
    def sweep_parameters_shape(self) -> List:
        '''If parameter A has a values, parameter B has b values, etc. the output is [a,b,...]'''
        return [len(val['var_values']) for val in self.sweep_parameters.values()]
    
    def generate_baseline_parameters(self) -> OrderedDict:
        '''Uses the base_idx property of the parameters to output a dictionary of parameter values that are treated as baseline.'''
        baseline_parameter_values = OrderedDict()
        for val in self.sweep_parameters.values():
            baseline_parameter_values[val['var_name']] = val['var_values'][val['base_idx']]
        for val in self.const_parameters.values():
            baseline_parameter_values[val['var_name']] = val['var_values'][val['base_idx']]
        self.baseline_parameters = baseline_parameter_values
        return self.baseline_parameters
    
    def generate_sweep_parameter_values(self):
        '''Set up a multidimensional array that contains all sweep parameter values.'''
        
        # Create N+1-dim array where the last dimension is of length N.
        sp_shape = self.sweep_parameters_shape()
        sweep_parameter_value_array = np.zeros(sp_shape + [len(sp_shape)])
        # sweep_parameter_value_array = [np.zeros(sp_shape)]*len(sp_shape)
        # At each index it holds the corresponding parameter values that are swept for each parameter's index
        for idx, _ in np.ndenumerate(np.zeros(sp_shape)):
            for t_idx, p_idx in enumerate(idx):
                sweep_parameter_value_array[idx][t_idx] = (list(self.sweep_parameters.values())[t_idx])['var_values'][p_idx]
                # sweep_parameter_value_array[t_idx][idx] = (list(self.sweep_parameters.values())[t_idx])['var_values'][p_idx]
                
        self.sweep_parameter_values = sweep_parameter_value_array
        return self.sweep_parameter_values
    
    def sweep_parameter_values_1d(self):
        '''Output: Reshapes sweep parameter values into a list of length (num. parameters)'''
        return self.sweep_parameter_values.reshape(-1, self.sweep_parameter_values.shape[-1])
    
    def all_coordinates(self):
        '''Outputs all coordinates of the n-D array for easy access.'''
        coord_list = []
        for idx, _ in np.ndenumerate(np.zeros(self.sweep_parameters_shape())):
            coord_list.append(idx)
        return coord_list
    
    def create_parameter_identifier_string(self, idx: tuple[int, ...]) -> str:
        ''' Input: tuple coordinate in the N-D array. Cross-matches against the sweep_parameters dictionary.
        NOTE: The dictionary cannot have been modified in order or sorted in any way!!!
        Output: unique identifier string for naming the corresponding job'''
        
        output_string = ''
        
        for t_idx, p_idx in enumerate(idx):
            try:
                variable = list(self.sweep_parameters.values())[t_idx]
                variable_name = variable['short_form']
                if isinstance(p_idx, slice):
                    output_string += variable_name + '_swp_'
                else:
                    variable_value = variable['var_values'][p_idx]
                    variable_format = variable['formatStr']
                    output_string += variable_name + '_' + variable_format%(variable_value) + '_'
            except Exception as err:
                vipdopt.logging.warning(f'Could not find info at index {p_idx} of variable {t_idx}.')
        
        return output_string[:-1]       # Strip the last "_"
    
    def current_parameter_values(self, idx: tuple[int, ...]) -> str:
        '''Edits a copy of the baseline parameters to include the current 
        values of the sweep parameters based on input index.'''
        
        current_parameter_values = self.generate_baseline_parameters().copy()
        
        # NOTE: Much easier to read version of what's going on, but takes up more compute time and memory.
        # for t_idx, p_idx in enumerate(idx):
        #     # We create a job corresponding to the coordinate in the parameter value space.

        #     variable = list(self.sweep_parameters.values())[t_idx]	# Access the sweep parameter being changed in this job
        #     variable_name = variable['var_name']				    # Get its name
        #     variable_value = variable['var_values'][p_idx]		    # Get the value it will be changed to in this job
            
        #     current_parameter_values[variable_name] = variable_value
        
        current_parameter_values.update(self.current_sweep_parameter_values(idx))
        
        return current_parameter_values
    
    def current_sweep_parameter_values(self, idx: tuple[int, ...]) -> str:
        '''Given a coordinate in parameter space, returns a dictionary of which parameters are changed and to what values.'''
        return OrderedDict(zip(list(self.sweep_parameters.keys()),
                                    self.sweep_parameter_values[idx]
                            ))
    
    @staticmethod
    def apply_changes(old_dict:dict, new_dict:dict) -> dict:
        '''Checks each key-value pair in the new dictionary and sees if it has changed from the old dictionary.
        Subsequently applies those changes. ORDER MATTERS!
        Output: Patched dictionary and changes.'''
        # https://dictdiffer.readthedocs.io/en/latest/
        result = dictdiffer.diff(old_dict, new_dict)        # Returns a generator object
        a = list(result)                                    # This voids whatever is in result.
        b = [x for x in a if x[0] in ['change', 'changed']] # Only keep the changes, not the additions or removals.
        result = iter(b) #(y for y in b)                    # Convert back to iterator. https://stackoverflow.com/a/27443047
        
        patched = dictdiffer.patch(result, old_dict)
        return patched, b

    @staticmethod
    def cast_to_int_if_int(e:list|npt.NDArray|dict[Any,List|npt.NDArray]):
        '''Checks all values. If they are floats with decimal .0, casts to int type.
        Support for multiple kinds of input values and also dictionary values.'''
        def is_int(a:int|float|list|npt.NDArray):
            a = np.array(a)
            return a-a.astype(int)==0
        
        if isinstance(e, dict):
            for k,v in e.items():
                if k=='mesh_spacing_um':
                    print(is_int(v))
                if isinstance(v, int) or isinstance(v, float):
                    e[k] = int(v) if is_int(v) else float(v)
                elif isinstance(v, list) or isinstance(v, (np.ndarray, np.generic) ):
                    e[k] = ParameterHandler.cast_to_int_if_int(v)
                else:
                    continue
        else:
            for k,v in enumerate(e):
                try:
                    e[k] = int(v) if is_int(v) else v
                except Exception as err:
                    pass
                
        return e