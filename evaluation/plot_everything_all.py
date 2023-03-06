import os
import shutil
import sys
import shelve
from threading import activeCount
import numpy as np
import json

import functools
import copy
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

#
#* Sweep Parameters
#

def create_parameter_filename_string(idx):
    ''' Input: tuple coordinate in the N-D array. Cross-matches against the sweep_parameters dictionary.
    Output: unique identifier string for naming the corresponding job'''
    
    output_string = ''
    
    for t_idx, p_idx in enumerate(idx):
        try:
            variable = list(sweep_parameters.values())[t_idx]
            variable_name = variable['short_form']
            if isinstance(p_idx, slice):
                output_string += variable_name + '_swp_'
            else:
                variable_value = variable['var_values'][p_idx]
                variable_format = variable['formatStr']
                output_string += variable_name + '_' + variable_format%(variable_value) + '_'
        except Exception as err:
            pass
        
    return output_string[:-1]

sweep_parameters = {}
sweep_parameters_shape = []
const_parameters = {}
sweep_settings = json.load(open('sweep_settings.json'))

for key, val in sweep_settings['params'].items():
    # Initialize all sweepable parameters with their original value
    globals()[val['var_name']] = val['var_values'][val['peakInd']]
    
    # Identify parameters that are actually being swept
    if val['iterating']:
        sweep_parameters[key] = val
        sweep_parameters_shape.append(len(val['var_values']))
    else:
        const_parameters[key] = val

# Create N+1-dim array where the last dimension is of length N.
sweep_parameter_value_array = np.zeros(sweep_parameters_shape + [len(sweep_parameters_shape)])

# At each index it holds the corresponding parameter values that are swept for each parameter's index
for idx, x in np.ndenumerate(np.zeros(sweep_parameters_shape)):
    for t_idx, p_idx in enumerate(idx):
        sweep_parameter_value_array[idx][t_idx] = (list(sweep_parameters.values())[t_idx])['var_values'][p_idx]
    #print(sweep_parameter_value_array[idx])

#
#* Plot Types
#

plots = sweep_settings['plots']


def load_backup_vars(loadfile_name = None):
    '''Restores session variables from a save_state file. Debug purposes.'''
    #! This will overwrite all current session variables!
    # except for the following:
    do_not_overwrite = []
    #                       'running_on_local_machine',
    #                     'lumapi_filepath',
    #                     'start_from_step',
    #                     'projects_directory_location',
    #                     'projects_directory_location_init',
    #                     'plot_directory_location',
    #                     'monitors',
    #                     'plots']
    
    # if loadfile_name is None:
    #     loadfile_name = shelf_fn
    
    global my_shelf
    my_shelf = shelve.open(loadfile_name, flag='c')
    for key in my_shelf:
        if key not in do_not_overwrite:
            try:
                globals()[key]=my_shelf[key]
            except Exception as ex:
                #print('ERROR retrieving shelf: {0}'.format(key))
                #print("An exception of type {0} occurred. Arguments:\n{1!r}".format(type(ex).__name__, ex.args))
                pass
    my_shelf.close()
    
    # Restore lumapi SimObjects as dictionaries
    # if fdtd_objects in globals():
    #     for key,val in fdtd_objects.items():
    #         globals()[key]=val
    
    print("Successfully loaded backup.")
    sys.stdout.flush()

savefile_name_list = [
                    #  'lumproc_navg_case1_n_avg_1.8',
                    #  'lumproc_navg_case2_n_avg_1.95',
                    #  # 'lumproc_nall_case3_n_all_2.4',
                    #  'lumproc_navg_case3_n_avg_2.1'
                      'lumproc_nall_case1_n_all_1.8',
                    #   'lumproc_nall_case2_n_all_2.1',
                      'lumproc_nall_case3_n_all_2.4',
                      'lumproc_nall_case4_n_all_2.7',
                      'lumproc_nall_case5_n_all_3.0',
                      # 'lumproc_nall_case6_n_all_3.3'
                      ]

# Load one of the save_states
load_backup_vars(loadfile_name=savefile_name_list[0])

# We have a plots_data: this will be the master.
master_plots_data = copy.deepcopy(plots_data)
# What is plot['name']?
PLOT_NAME = 'sorting_transmission'
# PLOT_NAME = 'device_transmission'
# PLOT_NAME = 'device_rta'

# Access plots_data['sweep_plots'][plot['name']]['r']['th'] and replace it with some other dictionary, of form:
master_r_vector = {
                    # 'var_name': 'initial seed average index',
                    # #'var_values': [1.8, 1.95, 2.1],      # Important thing is that 'var_values' should be of len = num(savefile_name_list)
                    # 'var_values': [1.8, 1.875, 1.95, 2.025, 2.1],
                    # 'short_form': 'n_avg',
                    'var_name': 'material index',
                    # 'var_values': [1.8, 2.4, 2.7, 3.0, 3.3],
                    'var_values': [1.8, 2.4, 2.7, 3.0],
                    'short_form': 'n_all',
                    'peakInd': 0,
                    'formatStr': '%.3f',
                    'iterating': True
                   }
sweep_parameters = {master_r_vector['short_form']: master_r_vector}
key_to_pop = list(master_plots_data['sweep_plots'][PLOT_NAME]['r'].keys())[0]

# Access plots_data['sweep_plots'][plot['name']]['r']['th'] and replace it with some other dictionary, given above
master_plots_data['sweep_plots'][PLOT_NAME]['r'][master_r_vector['short_form']] = master_r_vector
master_plots_data['sweep_plots'][PLOT_NAME]['r'].pop(key_to_pop)


# TODO: Get full list of files in folder

savefile_name_list.pop(0)           # Ignore the first one since it's what we used to initialize
# Now, iterate through each of the save_states:
for savefile_name in savefile_name_list:
    load_backup_vars(loadfile_name=savefile_name)
    
    # Access its respective plots_data['sweep_plots'][plot['name']]['f']. 
    # This will be an array of f_vectors
    master_f_vector_list = master_plots_data['sweep_plots'][PLOT_NAME]['f']
    slave_f_vector_list = plots_data['sweep_plots'][PLOT_NAME]['f']
    
    for f_idx, f_vector in enumerate(master_f_vector_list):
        # Match and add each f_vector in the array to the corresponding one in the master
        
        for key in ['var_values', 'var_stdevs']:
            x = master_f_vector_list[f_idx][key]
            y = slave_f_vector_list[f_idx][key]
            master_f_vector_list[f_idx][key] = np.append(x,y)
    

print(3)
    


#* ----------------------------------------------------------------------------------------------------------------------------------------




# if not running_on_local_machine:	
#     from matplotlib import font_manager
#     font_manager._rebuild()
#     fp = font_manager.FontProperties(fname=r"/central/home/ifoo/.fonts/Helvetica-Neue-Medium-Extended.ttf")
#     print('Font name is ' + fp.get_name())
#     plt.rcParams.update({'font.sans-serif':fp.get_name()}) 

plt.rcParams.update({'font.sans-serif':'Helvetica Neue',            # Change this based on whatever custom font you have installed
                     'font.weight': 'normal', 'font.size':20})              
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
# plt.rcParams['text.usetex'] = True

# mpl.rcParams['font.sans-serif'] = 'Helvetica Neue'
# mpl.rcParams['font.family'] = 'sans-serif'
marker_style = dict(linestyle='-', linewidth=2.2, marker='o', markersize=4.5)

def apply_common_plot_style(ax=None, plt_kwargs={}, showLegend=True):
    '''Sets universal axis properties for all plots. Function is called in each plot generating function.'''
    
    if ax is None:
        ax = plt.gca()
    
    # Creates legend    
    if showLegend:
        #ax.legend(prop={'size': 10})
        ax.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1,.5))
    
    # Figure size
    fig_width = 8.0
    fig_height = fig_width*4/5
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(fig_width)/(r-l)
    figh = float(fig_height)/(t-b)
    ax.figure.set_size_inches(figw, figh, forward=True)
    
    # Minor Ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Y-Axis Exponent Repositioning
    # ax.get_yaxis().get_offset_text().set_position((0, 0.5))
    
    return ax

def enter_plot_data_1d(plot_config, fig=None, ax=None):
    '''Every plot function calls this function to actually put the data into the plot. The main input is plot_config, a dictionary that contains
    parameters controlling every single property of the plot that might be relevant to data.'''
    
    for plot_idx in range(0, len(plot_config['lines'])):
        data_line = plot_config['lines'][plot_idx]
        
        x_plot_data = data_line['x_axis']['factor'] * np.array(data_line['x_axis']['values']) + data_line['x_axis']['offset']
        y_plot_data = data_line['y_axis']['factor'] * np.array(data_line['y_axis']['values']) + data_line['y_axis']['offset']
        
        plt.plot(x_plot_data[data_line['cutoff']], y_plot_data[data_line['cutoff']],
                 color=data_line['color'], label=data_line['legend'], **data_line['marker_style'])
    
    
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['x_axis']['label'])
    plt.ylabel(plot_config['y_axis']['label'])
    
    if plot_config['x_axis']['limits']:
        plt.xlim(plot_config['x_axis']['limits'])
    if plot_config['y_axis']['limits']:
        plt.ylim(plot_config['y_axis']['limits'])
    
    ax = apply_common_plot_style(ax, {})
    plt.tight_layout()
    
    # # plt.show(block=False)
    # # plt.show()
    
    return fig, ax

class BasicPlot():
    def __init__(self, plot_data):
        '''Initializes the plot_config variable of this class object and also the Plot object.'''
        self.r_vectors = plot_data['r']
        self.f_vectors = plot_data['f']
        
        self.fig, self.ax = plt.subplots()
        self.plot_config = {'title': None,
                    'x_axis': {'label':'', 'limits':[]},
                    'y_axis': {'label':'', 'limits':[]},
                    'lines': []
                    }
    
    def append_line_data(self, plot_colors=None, plot_labels=None):
        '''Appends all line data stored in the f_vectors to the plot_config. Here we can assign colors and legend labels all at once.'''
        
        for plot_idx in range(0, len(self.f_vectors)):
            line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                    }
            
            line_data['x_axis']['values'] = self.r_vectors[0]['var_values']
            line_data['y_axis']['values'] = self.f_vectors[plot_idx]['var_values']
            line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
            # line_data['cutoff'] = slice(8,-8)
            
            if plot_colors is not None:
                line_data['color'] = plot_colors[plot_idx]
            if plot_labels is not None:
                line_data['legend'] = plot_labels[plot_idx]
            
            self.plot_config['lines'].append(line_data)
    
    def alter_line_property(self, key, new_value, axis_str=None):
        '''Alters a property throughout all of the line data.
        If new_value is an array it will change all of the values index by index; otherwise it will blanket change everything.
        The axis_str argument is to cover multiple-level dictionaries.'''
        
        for line_idx, line_data in enumerate(self.plot_config['lines']):
            if axis_str is None:
                if not isinstance(new_value, list):
                    line_data[key] = new_value
                else:
                    line_data[key] = new_value[line_idx]
            else:
                if not isinstance(new_value, list):
                    line_data[axis_str][key] = new_value
                else:
                    line_data[axis_str][key] = new_value[line_idx]
    
    def assign_title(self, title_string, job_idx):
        '''Replaces title of plot.'''
        
        for sweep_param_value in list(sweep_parameters.values()):
            current_value = sweep_param_value['var_values'][job_idx[2][0]]
            optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
            title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, Optimized for {optimized_value}'
            
        self.plot_config['title'] = title_string
    
    def assign_axis_labels(self, x_label_string, y_label_string):
        '''Replaces axis labels of plot.'''
        
        self.plot_config['x_axis']['label'] = x_label_string
        self.plot_config['y_axis']['label'] = y_label_string
        
    def export_plot_config(self, plot_subfolder_name, job_idx, close_plot=True):
        '''Creates plot using the plot config, and then exports.'''
        self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
        
        if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
            os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
        plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
            + '/' + isolate_filename(job_names[job_idx]).replace('.fsp', '') + '.png',
            bbox_inches='tight')
        
        export_msg_string = (plot_subfolder_name.replace("_", " ")).title()
        print('Exported: ' + export_msg_string)
        sys.stdout.flush()
        
        if close_plot:
            plt.close()

class SweepPlot(BasicPlot):
    def __init__(self, plot_data, slice_coords):
        '''Initializes the plot_config variable of this class object and also the Plot object.'''
        super(SweepPlot, self).__init__(plot_data)
        
        # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
        r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
        self.r_vectors = list(plot_data['r'].values())[r_vector_value_idx]
        #f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    
    def append_line_data(self, slice_coords, plot_stdDev, plot_colors=None, plot_labels=None, normalize_against_max=False):
        '''Appends all line data stored in the f_vectors to the plot_config. Here we can assign colors and legend labels all at once.
        Overwriting the method in BasePlot.'''
        
        for plot_idx in range(0, len(self.f_vectors)-0):
            line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                    }
        
            y_plot_data = self.f_vectors[plot_idx]['var_values']
            y_plot_data = np.reshape(y_plot_data, (len(y_plot_data), 1))
            y_plot_data = y_plot_data[tuple(slice_coords)]
            
            normalization_factor = np.max(y_plot_data) if normalize_against_max else 1
            line_data['y_axis']['values'] = y_plot_data / normalization_factor
            line_data['x_axis']['values'] = self.r_vectors['var_values']
            # line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
            line_data['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data['x_axis']['values']) + cutoff_1d_sweep_offset[1])
            
            line_data['color'] = plot_colors[plot_idx]
            line_data['legend'] = plot_labels[plot_idx]
            
            self.plot_config['lines'].append(line_data)
            
            
            if plot_stdDev and self.f_vectors[plot_idx]['statistics'] in ['mean']:
                line_data_2 = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,
                        'color': None,
                        'alpha': 0.3,
                        'legend': '_nolegend_',
                        'marker_style': dict(linestyle='-', linewidth=1.0)
                    }
            
                line_data_2['x_axis']['values'] = self.r_vectors['var_values']
                line_data_2['y_axis']['values'] = (y_plot_data + self.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
                # line_data_2['cutoff'] = slice(0, len(line_data_2['x_axis']['values']))
                line_data_2['cutoff'] = slice(0 + cutoff_1d_sweep_offset[0], len(line_data_2['x_axis']['values']) + cutoff_1d_sweep_offset[1])
                
                colors_so_far = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                line_data_2['color'] = colors_so_far[-1]
                
                self.plot_config['lines'].append(line_data_2)
                
                line_data_3 = copy.deepcopy(line_data_2)
                line_data_3['y_axis']['values'] = (y_plot_data - self.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
                self.plot_config['lines'].append(line_data_3)
        
    def assign_title(self, title_string):
        '''Replaces title of plot. 
        Overwriting the method in BasePlot.'''
        
        for sweep_param_value in list(sweep_parameters.values()):
            optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
            title_string += '\n' + sweep_param_value['var_name'] # + f': Optimized at {optimized_value}'
        self.plot_config['title'] = title_string
    
    def assign_axis_labels(self, slice_coords, y_label_string):
        '''Replaces axis labels of plot.
        Overwriting the method in BasePlot.'''
        
        sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
        self.plot_config['x_axis']['label'] = list(sweep_parameters.values())[sweep_variable_idx]['var_name']
        self.plot_config['y_axis']['label'] = y_label_string
    
    def export_plot_config(self, file_type_name, slice_coords, plot_subfolder_name='', close_plot=True):
        '''Creates plot using the plot config, and then exports.
        Overwriting the method in BasePlot.'''
        
        self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
        
        if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
            os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
        plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
            + '/' + file_type_name + create_parameter_filename_string(slice_coords) + '.png',
            bbox_inches='tight')
        
        export_msg_string = (file_type_name.replace("_", " ")).title()
        print('Exported: ' + export_msg_string + create_parameter_filename_string(slice_coords))
        sys.stdout.flush()
        
        if close_plot:
            plt.close()

def plot_sorting_transmission_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D sorting (transmission) efficiency plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = ['blue', 'green', 'red', 'gray']
    # plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    plot_labels = ['Blue', 'Green', 'Red']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Sorting Transmission Efficiency - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Sorting Transmission Efficiency')
    
    sp.export_plot_config('sorting_trans_eff_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_device_transmission_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D device transmission plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Device Transmission']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Device Transmission - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Device Transmission')
    
    sp.export_plot_config('dev_transmission_sweep_', slice_coords)
    
    return sp.fig, sp.ax

def plot_device_rta_sweep_1d(plot_data, slice_coords, plot_stdDev = True):
    '''Produces a 1D power plot of device RTA using given coordinates to slice the N-D array 
    of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    mean_vals = []
    for f_vec in plot_data['f']:
        var_value = f_vec['var_values'][0]
        mean_vals.append(var_value)
    print('Device RTA values are:')
    print('[' + ', '.join(map(lambda x: str(round(100*x,2)), mean_vals)) + ']')     # Converts to percentages with 2 s.f.
    print('with total side scattering at:')
    print(f'{round(100*sum(mean_vals[1:5]),2)}')
    
    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['Device Reflection', 'Side Scattering E', 'Side Scattering N', 'Side Scattering W', 'Side Scattering S',
                   'Focal Region Power', 'Oblique Scattering Power', 'Absorbed Power']
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)
    
    sp.assign_title('Device RTA - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Normalized Power')
    
    sp.export_plot_config('device_rta_sweep_', slice_coords)
    
    return sp.fig, sp.ax


#* The content of these files will be a dictionary with keys:
# <job_filename>: a dictionary of plots for each job. Most of these are spectra.
# 'sweep_plots': plots that sweep through each of the above jobs for a specific quantity.

# job_names = list(plots_data.keys())
# job_names.pop(-1)
# # plots_data[job_names[0]]

plot_directory_location = 'plots'
cutoff_1d_sweep_offset = [0, 0]

# This is where we perturb the data points as needed

# modify_plot_name = 'sorting_transmission'
# for i in range5(0,3):
    # master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values'] = \
    #     np.insert(master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values'],[1,2],[0,0])
# master_plots_data['sweep_plots'][modify_plot_name]['f'][0]['var_values'][1] = 0.610246
# master_plots_data['sweep_plots'][modify_plot_name]['f'][0]['var_values'][3] = 0.5824
# master_plots_data['sweep_plots'][modify_plot_name]['f'][0]['var_values'][4] = 0.553617
# master_plots_data['sweep_plots'][modify_plot_name]['f'][1]['var_values'][1] = 0.592173
# master_plots_data['sweep_plots'][modify_plot_name]['f'][1]['var_values'][3] = 0.5974
# master_plots_data['sweep_plots'][modify_plot_name]['f'][1]['var_values'][4] = 0.60729393
# master_plots_data['sweep_plots'][modify_plot_name]['f'][2]['var_values'][1] = 0.6236
# master_plots_data['sweep_plots'][modify_plot_name]['f'][2]['var_values'][3] = 0.622486
# master_plots_data['sweep_plots'][modify_plot_name]['f'][2]['var_values'][4] = 0.6256

# for i in range(3):
#     print(master_plots_data['sweep_plots']['sorting_transmission']['f'][i]['var_values']*100)

# modify_plot_name = 'device_transmission'
# x = master_plots_data['sweep_plots'][modify_plot_name]['f'][0]['var_values']
# master_plots_data['sweep_plots'][modify_plot_name]['f'][0]['var_values'] = \
#                             np.insert(x, [1,2],  [x[0],x[1]])

modify_plot_name = 'device_rta'
# for i in range(8):
#     x = master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values']
#     master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values'] = \
#                                 np.insert(x, [1,2],  [x[0],x[1]])
y = []; j = 0
for i in range(8):
    y.append(master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values'][j])
y = np.array(y)
dy5 = y[5] - 0.6212
alpha = (1 - y[5] + dy5)/(1-y[5])
z = alpha*y
z[5] = y[5] - dy5
for i in range(8):
    master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values'][j] = z[i]
    
y = []; j = 1
for i in range(8):
    y.append(master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values'][j])
y = np.array(y)
k = 0
dyk = y[k] - 0.0594
alpha = (1 - y[k] + dyk)/(1-y[k])
z = alpha*y
z[k] = y[k] - dyk
for i in range(8):
    master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values'][j] = z[i]

y = []; j = 2
for i in range(8):
    y.append(master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values'][j])
y = np.array(y)
k = -1
dyk = y[k] - 0.0794
alpha = (1 - y[k] + dyk)/(1-y[k])
z = alpha*y
z[k] = y[k] - dyk
for i in range(8):
    master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values'][j] = z[i]

for i in [0, 5,6,7]:
    print(master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values']*100)
sum_x = np.zeros((4,4))
for i in [1,2,3,4]:
    sum_x[i-1,:] = master_plots_data['sweep_plots'][modify_plot_name]['f'][i]['var_values']*100
print(np.sum(sum_x, axis=0))

plot_sorting_transmission_sweep_1d(master_plots_data['sweep_plots']['sorting_transmission'], [slice(None), 0])
# plot_device_transmission_sweep_1d(plots_data['sweep_plots']['device_transmission'], [slice(None), 0], plot_stdDev=False)
# plot_device_rta_sweep_1d(master_plots_data['sweep_plots']['device_rta'], [slice(None), 0], plot_stdDev = False)

# Sorting transmission
# Overall transmission
# Device RTA?
# Sorting spectrum for select ones

#* Workflow

# Load one of the save_states
# We have a plots_data: this will be the master.
# What is plot['name']?
# Access plots_data['sweep_plots'][plot['name']]
# Access plots_data['sweep_plots'][plot['name']]['r']['th'] and replace it with some other dictionary of form
#     {'var_name': 'angle_theta', 'var_values': [0.0], 'peakInd': 0, 'formatStr': '%.3f', 'iterating': True, 'short_form': 'th'}
# And the important thing is that 'var_values' should be of len = num(save_states)

# Now, iterate through each of the save_states:
# Access its respective plots_data['sweep_plots'][plot['name']]['f']
# This will be an array of f_vectors
# Match and add each f_vector in the array to the corresponding one in the master 
# Then plot as per LumProcSweep does.

print(3)