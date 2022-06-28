import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os

from SonyBayerFilterParameters import *

#* Reinstall the right font on the cluster if necessary
if not running_on_local_machine:	
    from matplotlib import font_manager
    font_manager._rebuild()
    fp = font_manager.FontProperties(fname=r"/central/home/ifoo/.fonts/Helvetica-Neue-Medium-Extended.ttf")
    print('Font name is ' + fp.get_name())
    plt.rcParams.update({'font.sans-serif':fp.get_name()}) 

plt.rcParams.update({'font.sans-serif':'Helvetica Neue',            # Change this based on whatever custom font you have installed
                     'font.weight': 'normal', 'font.size':20})              
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'

# mpl.rcParams['font.sans-serif'] = 'Helvetica Neue'
# mpl.rcParams['font.family'] = 'sans-serif'
marker_style = dict(linestyle='-', linewidth=2.2, marker='o', markersize=4.5)

#* Plotting Functions

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
        
        plt.plot(x_plot_data[data_line['cutoff']], np.squeeze(y_plot_data[data_line['cutoff']]),
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

class SpectrumPlot(BasicPlot):
    def __init__(self, plot_data):
        '''Initializes the plot_config variable of this class object and also the Plot object.'''
        super(SpectrumPlot, self).__init__(plot_data)
    
    def append_line_data(self, plot_colors=None, plot_labels=None):
        super(SpectrumPlot, self).append_line_data(plot_colors, plot_labels)
        self.alter_line_property('factor', 1e6, axis_str='x_axis')
    
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
            title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
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

# figure_of_merit_evolution = np.load(
#              projects_directory_location + "/figure_of_merit.npy")
# figure_of_merit_by_wl_evolution = np.load(
#              projects_directory_location + "/figure_of_merit_by_wl.npy")
transmission_by_wl_evolution = np.load(
             projects_directory_location + "/transmission_by_wl.npy")
intensity_fom_by_wavelength_evolution = np.load(
             projects_directory_location + "/intensity_fom_by_wavelength.npy")
mode_overlap_fom_by_wavelength_evolution = np.load(
             projects_directory_location + "/mode_overlap_fom_by_wavelength.npy")

transmission_by_focal_spot_and_wavelength_evolution = np.load(
             projects_directory_location + "/transmission_by_focal_spot_by_wl.npy")
intensity_fom_by_focal_spot_and_wavelength_evolution = np.load(
             projects_directory_location + "/intensity_fom_by_focal_spot_by_wl.npy")
mode_overlap_fom_by_focal_spot_and_wavelength_evolution = np.load(
             projects_directory_location + "/mode_overlap_fom_by_focal_spot_by_wl.npy")

#* First Test: Compare mode overlap FoM with transmission and intensity FoM: x-axis is wavelength

def NormalizeData(data):
    
    def normalize(values):
        # Normalizes a 1D vector to between 0 and 1
        return (values - np.min(values)) / (np.max(values) - np.min(values))
    
    # Either the entire section, or for each quadrant
    if np.shape(data)[2] == 4:
        norm_data = np.zeros(np.shape(data))            # Initialize
        for section_idx in range(np.shape(data)[2]):
            data_part = data[:,:,section_idx,:]
            norm_data[:,:,section_idx,:] = (data_part - np.min(data_part)) / (np.max(data_part) - np.min(data_part))
            if section_idx in [1,3]:
                norm_data[:,:,section_idx,:] *= 0.5
        return norm_data
    else:
        return normalize(data)
    

transmission_by_wl_evolution_norm = NormalizeData(transmission_by_wl_evolution)
mode_overlap_fom_by_wavelength_evolution_norm = NormalizeData(mode_overlap_fom_by_wavelength_evolution)
intensity_fom_by_wavelength_evolution_norm = NormalizeData(intensity_fom_by_wavelength_evolution)

transmission_by_fswl_evolution_norm = NormalizeData(transmission_by_focal_spot_and_wavelength_evolution)
intensity_fom_by_fswl_evolution_norm = NormalizeData(intensity_fom_by_focal_spot_and_wavelength_evolution)
mode_overlap_fom_by_fswl_evolution_norm = NormalizeData(mode_overlap_fom_by_focal_spot_and_wavelength_evolution)
# transmission_by_fswl_evolution_norm = transmission_by_focal_spot_and_wavelength_evolution
# mode_overlap_fom_by_fswl_evolution_norm = mode_overlap_fom_by_focal_spot_and_wavelength_evolution

def moving_average(x, w):
    return np.convolve(np.squeeze(x), np.ones(w), 'valid') / w
mode_overlap_fom_by_wavelength_evolution_norm_movingaverage = np.array(mode_overlap_fom_by_wavelength_evolution_norm[:,:,0])
mode_overlap_fom_by_wavelength_evolution_norm_movingaverage = np.append(mode_overlap_fom_by_wavelength_evolution_norm_movingaverage, moving_average(mode_overlap_fom_by_wavelength_evolution_norm, 3))
mode_overlap_fom_by_wavelength_evolution_norm_movingaverage = np.append(mode_overlap_fom_by_wavelength_evolution_norm_movingaverage, mode_overlap_fom_by_wavelength_evolution_norm[:,:,-1])

# def moving_average(x, w):
#     x2 = x
#     w = 3
#     for idx in range(1,len(x2)-1):
#         x2[idx] = np.average([x[idx-1], x[idx], x[idx+1]])
# mode_overlap_fom_by_wavelength_evolution_norm_movingaverage = moving_average(mode_overlap_fom_by_wavelength_evolution_norm, 3)
# transmission_by_wl_evolution_norm = np.squeeze(transmission_by_wl_evolution_norm)
# intensity_fom_by_wavelength_evolution_norm = np.squeeze(intensity_fom_by_wavelength_evolution_norm)


# TODO: Two separate graphs, one with all the quadrants added together, and one with individual quadrant plots.
r_vectors = [{'var_values': lambda_values_um}]
f_vectors = [#{'var_values': list(transmission_by_wl_evolution_norm[:] * 1)},
                {'var_values': list(transmission_by_fswl_evolution_norm[:,:,0,:] * 1)},
                {'var_values': list((transmission_by_fswl_evolution_norm[:,:,1,:] + transmission_by_fswl_evolution_norm[:,:,3,:]) * 1)},
                {'var_values': list(transmission_by_fswl_evolution_norm[:,:,2,:] * 1)},
                # {'var_values': list(transmission_by_fswl_evolution_norm[:,:,3,:] * 1)},
            #{'var_values': list(intensity_fom_by_wl_evolution_norm[:] * 1)},
                {'var_values': list(intensity_fom_by_fswl_evolution_norm[:,:,0,:] * 1)},
                {'var_values': list((intensity_fom_by_fswl_evolution_norm[:,:,1,:] + intensity_fom_by_fswl_evolution_norm[:,:,3,:]) * 1)},
                {'var_values': list(intensity_fom_by_fswl_evolution_norm[:,:,2,:] * 1)},
             #{'var_values': list(mode_overlap_fom_by_wavelength_evolution_norm[:] * 1)},
                # {'var_values': list(mode_overlap_fom_by_fswl_evolution_norm[:,:,0,:] * 1)},
                # {'var_values': list((mode_overlap_fom_by_fswl_evolution_norm[:,:,1,:] + mode_overlap_fom_by_fswl_evolution_norm[:,:,3,:]) * 1)},
                # {'var_values': list(mode_overlap_fom_by_fswl_evolution_norm[:,:,2,:] * 1)}#,
             #{'var_values': list(intensity_fom_by_wavelength_evolution_norm[:] * 1)}
             ]

fig, ax = plt.subplots()
plot_config = {'title': None,
                'x_axis': {'label':'', 'limits':[]},
                'y_axis': {'label':'', 'limits':[]},
                'lines': []
            }

plot_colors = [#'black', 
               'blue','red','green',
               #'blue', 
               'brown','orange','purple'#,
               #'black'
               ]
plot_labels = [#'Transmission', 
               'Transmission_0', 'Transmission_1+3', 'Transmission_2', 
               #'Transmission', 
               'Intensity FoM_0', 'Intensity FoM_1+3', 'Intensity FoM_2'
               #'Mode Overlap FoM',
            #    'Mode Overlap FoM _ 0','Mode Overlap FoM _ 1+3','Mode Overlap FoM _ 2'#,
               ]

for plot_idx in range(0, len(f_vectors)):
    line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                    'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                    'cutoff': None,
                    'color': None,
                    'alpha': 1.0,
                    'legend': None,
                    'marker_style': marker_style
            }
    
    
    line_data['x_axis']['values'] = r_vectors[0]['var_values']
    line_data['x_axis']['factor'] = 1 # 1e6
    line_data['y_axis']['values'] = f_vectors[plot_idx]['var_values']
    line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
    
    line_data['color'] = plot_colors[plot_idx]
    line_data['legend'] = plot_labels[plot_idx]
    
    plot_config['lines'].append(line_data)
    

title_string = 'Correlation of Intensity FoM to Transmission'
# title_string = 'Correlation of Mode Overlap FoM to Transmission'
plot_config['title'] = title_string

plot_config['x_axis']['label'] = 'Wavelength (um)'
plot_config['y_axis']['label'] = ''



fig, ax = enter_plot_data_1d(plot_config, fig, ax)

plot_directory_location = r'C:\Users\Ian\Dropbox\Caltech\Faraon Group\Simulations\Mode Overlap FoM\fom_dev\fom_correlation_test\plots'
plot_subfolder_name = 'fom_correlation_transmission'
if not os.path.isdir(plot_directory_location + '/' + plot_subfolder_name):
    os.makedirs(plot_directory_location + '/' + plot_subfolder_name)
plt.savefig(plot_directory_location + '/' + plot_subfolder_name\
    + '/' + 'correlation_fom_with_transmission' + '.png',
    bbox_inches='tight')
print('Exported: FoM Correlation with Transmission')

# TODO: Individual wavelength bin correlations.
# print('Correlation of Mode Overlap FoM with Transmission is:' +\
#     str(np.corrcoef(np.squeeze(mode_overlap_fom_by_wavelength_evolution_norm), np.squeeze(transmission_by_wl_evolution_norm))))

print('Correlation of Mode Overlap FoM with Transmission, Wavelength Bin 0, is:' +\
    str(np.corrcoef(np.squeeze(mode_overlap_fom_by_fswl_evolution_norm[:,:,0,:]), np.squeeze(transmission_by_fswl_evolution_norm[:,:,0,:]))))
print('Correlation of Mode Overlap FoM with Transmission, Wavelength Bin 1+3, is:' +\
    str(np.corrcoef(np.squeeze(mode_overlap_fom_by_fswl_evolution_norm[:,:,1,:] + mode_overlap_fom_by_fswl_evolution_norm[:,:,3,:]), 
                    np.squeeze(transmission_by_fswl_evolution_norm[:,:,1,:] + transmission_by_fswl_evolution_norm[:,:,3,:]))))
print('Correlation of Mode Overlap FoM with Transmission, Wavelength Bin 2, is:' +\
    str(np.corrcoef(np.squeeze(mode_overlap_fom_by_fswl_evolution_norm[:,:,2,:]), np.squeeze(transmission_by_fswl_evolution_norm[:,:,2,:]))))

print('Correlation of Intensity FoM with Transmission, Wavelength Bin 0, is:' +\
    str(np.corrcoef(np.squeeze(intensity_fom_by_fswl_evolution_norm[:,:,0,:]), np.squeeze(transmission_by_fswl_evolution_norm[:,:,0,:]))))
print('Correlation of Intensity FoM with Transmission, Wavelength Bin 1+3, is:' +\
    str(np.corrcoef(np.squeeze(intensity_fom_by_fswl_evolution_norm[:,:,1,:] + intensity_fom_by_fswl_evolution_norm[:,:,3,:]), 
                    np.squeeze(transmission_by_fswl_evolution_norm[:,:,1,:] + transmission_by_fswl_evolution_norm[:,:,3,:]))))
print('Correlation of Intensity FoM with Transmission, Wavelength Bin 2, is:' +\
    str(np.corrcoef(np.squeeze(intensity_fom_by_fswl_evolution_norm[:,:,2,:]), np.squeeze(transmission_by_fswl_evolution_norm[:,:,2,:]))))

print('Correlation of Intensity FoM with Transmission is:' +\
    str(np.corrcoef(np.squeeze(intensity_fom_by_wavelength_evolution_norm), np.squeeze(transmission_by_wl_evolution_norm))))
    

plt.close()


# Choose (x0,y0) = (0,0)
# Take a 1D slice through the cube vertically


print('Reached end of code. Operation completed.')
    