# Copyright © 2023, California Institute of Technology. All rights reserved.
#
# Use in source and binary forms for nonexclusive, nonsublicenseable, commercial purposes with or without modification, is permitted provided that the following conditions are met:
# - Use of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# - Use in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the software.
# - Neither the name of the California Institute of Technology (Caltech) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import copy
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, PercentFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Custom Classes and Imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
# import configs.yaml_to_params as cfg
# # Gets all parameters from config file - store all those variables within the namespace. Editing cfg edits it for all modules accessing it
# # See https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
# # from utils import Device
# # from utils import LumericalUtils
# # from utils import FoM
# from utils import utility
import vipdopt


#* Template
# The structure of the dictionaries we are passing into these functions are as follows:
# https://i.imgur.com/J1OctoM.png
# https://tree.nathanfriend.io/?s=(%27opNons!(%27fancy!true~fullPath!fqse~trailingSlash!true~rootDot!fqse)~U(%27U%27Os_dataCjob1CQCjob%202CQCsweep_OsC*OMJrJ*LMFmateriq%20indexE0B1B2XLKn_qlWpeakInd80-formatStrK%25.3fWiteraNng%22%3A*TrueJfY0FReflecNonE449B556B6G5B02B04Hmean%22Y1FAbsorpNonE549B656B3G12B09B01Hpeak%22JQJNtleZ%22device_rta_sweep%22%27)~version!%271%27)*%20%20-J**%22KZB%2C%200.C%5Cn*EWz_vqueVF-zMKG17Xz_stdevV0HXstaNsNcsKJC**K8%22Lshort_formM_nameNtiOplotQ*...Usource!Vs8%5B0.W%22-X%5D-YJ*line_Z%3A%20qalzvar%01zqZYXWVUQONMLKJHGFECB8-*
# Broadly, the plot_data has keys 'r', 'f', and 'title'.
# The values of 'r' and 'f' are a dictionary and array of dictionaries respectively, that contain plot metadata as well as values.
# Examples are given below:

TEMPLATE_R_VECTOR = {
                    'var_name': 'Circle Radius',
                    'var_values': [1.8, 2.4, 2.7, 3.0],
                    'short_form': 'c_rad',
                    'peakInd': 0,
                    'formatStr': '%.3f',
                    'iterating': True
                   }
sweep_parameters = [TEMPLATE_R_VECTOR]

TEMPLATE_F_VECTOR = {'var_name': 'Circle Diameter', 
                    'var_values': [10.179, 18.096, 22.902, 28.274], 
                    'var_stdevs': [0.1, 0.1, 0.1, 0.1], 
                    'statistics': 'mean'
                    }

TEMPLATE_PLOT_DATA = {'r':copy.deepcopy(sweep_parameters),
                      'f': [copy.deepcopy(TEMPLATE_F_VECTOR)],
                      'title': 'template'
                    }


#* Plot Style Params
# if not running_on_local_machine:	
#     from matplotlib import font_manager
#     font_manager._rebuild()
#     fp = font_manager.FontProperties(fname=r"/central/home/ifoo/.fonts/Helvetica-Neue-Medium-Extended.ttf")
#     print('Font name is ' + fp.get_name())
#     plt.rcParams.update({'font.sans-serif':fp.get_name()}) 

# plt.rcParams.update({'font.sans-serif':'Helvetica Neue',            # Change this based on whatever custom font you have installed
#                      'font.weight': 'normal', 'font.size':20})
plt.rcParams.update({'font.weight': 'normal', 'font.size':20})              
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
# plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
# plt.rcParams['text.usetex'] = True

# mpl.rcParams['font.sans-serif'] = 'Helvetica Neue'
# mpl.rcParams['font.family'] = 'sans-serif'

marker_style = dict(linestyle='-', linewidth=2.2, marker='o', markersize=4.5)
vline_style = {'color': 'gray', 'linestyle': '--', 'linewidth': 1}



#* Define a class called BasicPlot with basic plotting methods and data storage options.

def adjust_figure_size(ax, fig_width, fig_height=None):
    if fig_height is None:
        fig_height = fig_width*4/5

    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(fig_width)/(r-l)
    figh = float(fig_height)/(t-b)
    ax.figure.set_size_inches(figw, figh, forward=True)

    return ax

def apply_common_plot_style(ax=None, plt_kwargs={}, show_legend=True):
    '''Sets universal axis properties for all plots. Function is called in each plot generating function.'''
    
    if ax is None:
        ax = plt.gca()
    
    # Creates legend    
    if show_legend:
        #ax.legend(prop={'size': 10})
        ax.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1.1,.5))
    
    # Figure size
    ax = adjust_figure_size(ax, 8.0)
    
    # Minor Ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Y-Axis Exponent Repositioning
    ax.get_yaxis().get_offset_text().set_position((-0.1, 0.5))
    
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
        self.r_vectors = plot_data['r'][0]
        self.f_vectors = plot_data['f']
        
        self.fig, self.ax = plt.subplots()
        self.plot_config = {'title': plot_data['title'],
                            'x_axis': {'label':'', 'limits':[]},
                            'y_axis': {'label':'', 'limits':[]},
                            'lines': []
                            }
    
    def append_line_data(self, plot_colors=None, plot_labels=None, plot_alphas=None):
        '''Appends all line data stored in the f_vectors to the plot_config. Here we can assign colors and legend labels all at once.'''
        
        for plot_idx in range(0, len(self.f_vectors)):
            line_data = {'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                        'cutoff': None,		# A slice of two numbers i.e. slice(1,8) that defines the indices at which the data is truncated
                        'color': None,
                        'alpha': 1.0,
                        'legend': None,
                        'marker_style': marker_style
                        }
            
            line_data['x_axis']['values'] = self.r_vectors['var_values']
            line_data['y_axis']['values'] = self.f_vectors[plot_idx]['var_values']
            line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
            # line_data['cutoff'] = slice(8,-8)
            
            if plot_colors is not None:
                line_data['color'] = plot_colors[plot_idx]
            if plot_labels is not None:
                line_data['legend'] = plot_labels[plot_idx]
            else: line_data['legend'] = self.f_vectors[plot_idx]['var_name']
            if plot_alphas is not None:
                line_data['alpha'] = plot_alphas[plot_idx]
            
            self.plot_config['lines'].append(line_data)
    
    def alter_line_property(self, key, new_value, nested_keys=[]):
        '''Alters a property throughout all of the line data.
        If new_value is an array it will change all of the values index by index; otherwise it will blanket change everything.
        The nested_keys argument is to cover multiple-level dictionaries.'''
        
        for line_idx, line_data in enumerate(self.plot_config['lines']):
            if not isinstance(new_value, list):
                utility.set_by_path(line_data, nested_keys+[key], new_value)
            else: 
                utility.set_by_path(line_data, nested_keys+[key], new_value[line_idx])
    
    def assign_title(self, title_string=None):
        '''Replaces title of plot.'''
        if title_string is not None:
            self.plot_config['title'] = title_string
    
    def assign_axis_labels(self, x_label_string=None, y_label_string=None):
        '''Replaces axis labels of plot.'''
        if x_label_string is None:
            x_label_string = self.r_vectors['var_name']
        else:
            self.plot_config['x_axis']['label'] = x_label_string

        if y_label_string is None:
            y_label_string = self.f_vectors[0]['var_name']
        else:
            self.plot_config['y_axis']['label'] = y_label_string

        self.plot_config['x_axis']['label'] = x_label_string
        self.plot_config['y_axis']['label'] = y_label_string
        
    def export_plot_config(self, plot_directory_location, plot_subfolder_name, filename, close_plot=True):
        '''Creates plot using the plot config, and then exports.'''
        self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
  
        #! NOTE: This is where you do any additional adjustment of the plot before saving out
        # plot adjustment code

        SAVE_LOCATION = os.path.abspath(os.path.join(plot_directory_location, plot_subfolder_name))
        if not os.path.isdir(SAVE_LOCATION):
            os.makedirs(SAVE_LOCATION)

        plt.savefig(SAVE_LOCATION + f'/{filename}.png', bbox_inches='tight')
        
        # export_msg_string = (SAVE_LOCATION + f'/{filename}'.replace("_", " ")).title()
        export_msg_string = filename.replace("_", "").title()
        vipdopt.logger.info('Exported: ' + export_msg_string)
        
        if close_plot:
            plt.close()

class SpectrumPlot(BasicPlot):
    def __init__(self, plot_data, sweep_parameters_):
        '''Initializes the plot_config variable of this class object and also the Plot object. Here, accounts for sweep parameters.'''
        super(SpectrumPlot, self).__init__(plot_data)
        self.sweep_parameters = sweep_parameters_

    def assign_title(self, title_string, job_idx):
        '''Replaces title of plot. Overwrites function in BasicPlot and accounts for sweep parameters.'''
        
        for sweep_param_value in list(self.sweep_parameters.values()):
            current_value = sweep_param_value['var_values'][job_idx[2][0]]
            optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
            title_string += '\n' + sweep_param_value['var_name'] + f': Current Value {current_value}, \nOptimized for {optimized_value}'
            
        self.plot_config['title'] = title_string	

    def append_line_data(self, plot_colors=None, plot_labels=None):
        '''Overwrites function in BasicPlot by applying a 1e6 factor to the x-axis (which is supposed to be wavelength in um.)'''
        super(SpectrumPlot, self).append_line_data(plot_colors, plot_labels)
        self.alter_line_property('factor', 1e6, ['x_axis'])

    def export_plot_config(self, plot_directory_location, plot_subfolder_name, filename, close_plot=True):
        '''Creates plot using the plot config, and then exports.'''
        self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
  
        #! NOTE: This is where you do any additional adjustment of the plot before saving out
        # plot adjustment code

        SAVE_LOCATION = os.path.abspath(os.path.join(plot_directory_location, plot_subfolder_name))
        if not os.path.isdir(SAVE_LOCATION):
            os.makedirs(SAVE_LOCATION)
        filename = utility.isolate_filename(filename).replace('.fsp', '')

        plt.savefig(SAVE_LOCATION + f'/{filename}.png', bbox_inches='tight')
        
        export_msg_string = (SAVE_LOCATION + f'/{filename}'.replace("_", " ")).title()
        vipdopt.logger.info('Exported: ' + export_msg_string)
        
        if close_plot:
            plt.close()

class SweepPlot(BasicPlot):
    def __init__(self, plot_data, slice_coords):
        '''Initializes the plot_config variable of this class object and also the Plot object.'''
        super(SweepPlot, self).__init__(plot_data)
        
        # Whichever entry is a slice means that variable in r_vectors is the x-axis of this plot
        r_vector_value_idx = [index for (index , item) in enumerate(slice_coords) if isinstance(item, slice)][0]
        self.r_vectors = list(plot_data['r'][0].values())[r_vector_value_idx]
        #f_vectors = plot_data['f']              # This is a list of N-D arrays, containing the function values for each parameter combination
    
    
    def append_line_data(self, slice_coords, plot_stdDev, plot_colors=None, plot_labels=None, 
                          normalize_against_max=False):
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
            line_data['cutoff'] = slice(0 + lmp.cutoff_1d_sweep_offset[0], len(line_data['x_axis']['values']) + lmp.cutoff_1d_sweep_offset[1])
            
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
                line_data_2['cutoff'] = slice(0 + lmp.cutoff_1d_sweep_offset[0], len(line_data_2['x_axis']['values']) + lmp.cutoff_1d_sweep_offset[1])
                
                colors_so_far = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                line_data_2['color'] = colors_so_far[-1]
                
                self.plot_config['lines'].append(line_data_2)
                
                line_data_3 = copy.deepcopy(line_data_2)
                line_data_3['y_axis']['values'] = (y_plot_data - self.f_vectors[plot_idx]['var_stdevs']) / normalization_factor
                self.plot_config['lines'].append(line_data_3)
        
    def assign_title(self, title_string):
        '''Replaces title of plot. 
        Overwriting the method in BasePlot.'''
        
        for sweep_param_value in list(lmp.sweep_parameters.values()):
            optimized_value = sweep_param_value['var_values'][sweep_param_value['peakInd']]
            title_string += '\n' + sweep_param_value['var_name'] + f': Optimized at {optimized_value}'
        self.plot_config['title'] = title_string
    
    def assign_axis_labels(self, slice_coords, y_label_string):
        '''Replaces axis labels of plot.
        Overwriting the method in BasePlot.'''
        
        sweep_variable_idx = [index for (index, item) in enumerate(slice_coords) if type(item) is slice][0]
        self.plot_config['x_axis']['label'] = list(lmp.sweep_parameters.values())[sweep_variable_idx]['var_name']
        self.plot_config['y_axis']['label'] = y_label_string
   
    def export_plot_config(self, plot_directory_location, plot_subfolder_name, filename, 
                            slice_coords, close_plot=True):
        '''Creates plot using the plot config, and then exports.
        Overwriting the method in BasePlot.'''
        self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)
  
        #! NOTE: This is where you do any additional adjustment of the plot before saving out
        # plot adjustment code

        SAVE_LOCATION = os.path.abspath(os.path.join(plot_directory_location, plot_subfolder_name))
        if not os.path.isdir(SAVE_LOCATION):
            os.makedirs(SAVE_LOCATION)

        param_filename_str = lmp.create_parameter_filename_string(slice_coords)
        plt.savefig(SAVE_LOCATION + f'/{filename}{param_filename_str}.png', bbox_inches='tight')
        
        # export_msg_string = (SAVE_LOCATION + f'/{filename}'.replace("_", " ")).title()
        export_msg_string = filename.replace("_", "").title()
        vipdopt.logger.info('Exported: ' + export_msg_string + param_filename_str)
        
        if close_plot:
            plt.close()

#* Functions that call one of the Plot classes

def plot_basic_1d(plot_data,  plot_directory_location, plot_subfolder_name, filename, 
                  title=None, xlabel_txt=None, ylabel_txt=None):
    # Calls an instance of BasicPlot, initializes with the data.
    bp = BasicPlot(plot_data)
    bp.append_line_data()		# Sets up plot config with line data.
    bp.assign_title(title_string=title)			# Assign title
    bp.assign_axis_labels(x_label_string=xlabel_txt, y_label_string=ylabel_txt)		 # Assign x and y axis labels
    bp.export_plot_config(plot_directory_location, plot_subfolder_name, filename)	 # Do actual plotting and then save out the figure.

#* Evaluation during Optimization

def plot_fom_trace(f, plot_directory_location, epoch_list=None):
    '''Plot FOM trace during evolution of optimization.'''

    if epoch_list is None:
        epoch_list = np.linspace(0, len(f), 10)
    upperRange = np.max(f) # np.ceil(np.max(f))

    iterations = copy.deepcopy(TEMPLATE_R_VECTOR)
    iterations.update({'var_name': 'Iterations', 'var_values': range(f.size), 'short_form': 'iter'})
    fom = copy.deepcopy(TEMPLATE_F_VECTOR)
    fom.update({'var_name': 'FoM', 'var_values': f})
    plot_data = {'r':[iterations],
                'f': [fom],
                'title': 'Figure of Merit - Trace'}
 
    bp = BasicPlot(plot_data)
    bp.append_line_data(plot_colors=['orange'])
    bp.assign_title()
    bp.assign_axis_labels()
    fig, ax = bp.fig, bp.ax
    for i in epoch_list:
        plt.vlines(i, 0,upperRange, **vline_style)
    bp.fig, bp.ax = fig, ax
    bp.export_plot_config(plot_directory_location,'','fom_trace')
 
    return fig
 
def plot_quadrant_transmission_trace(f, plot_directory_location, epoch_list=None,filename='quad_trans_trace'):
    '''Plot evolution trace of quadrant transmission'''
    if epoch_list is None:
        epoch_list = np.linspace(0, len(f), 10)
    upperRange = np.ceil(np.max(f))

    num_adjoint_src = f.shape[1]
    trace = np.zeros((num_adjoint_src, f.shape[0]))

    counter = 0
    for iteration in range(f.shape[0]):
        for adj_src in range(num_adjoint_src):
            trace[adj_src, counter] = np.max(f[iteration, adj_src, :])
        counter += 1

    iterations = copy.deepcopy(TEMPLATE_R_VECTOR)
    iterations.update({'var_name': 'Iterations', 'var_values': range(trace.shape[1]), 'short_form': 'iter'})
    quad_trace = num_adjoint_src * [copy.deepcopy(TEMPLATE_F_VECTOR)]
    for adj_src in range(num_adjoint_src):
        quad_trace[adj_src] = copy.deepcopy(quad_trace)[adj_src]
        quad_trace[adj_src].update({'var_name': f'Q{adj_src}', 'var_values': trace[adj_src]})
    vipdopt.logger.info('Quadrant Transmissions Trace is:')
    plot_data = {'r':[iterations],
                'f': quad_trace,
                'title': 'Quadrant Transmissions - Trace'}

    # colors = [ 'b', 'g', 'r', 'm' ]
    # for quad_idx in range( 0, 4 ):
    # 	plt.plot( quad_trace[quad_idx]['var_values'], color=colors[ quad_idx ] )
    # plt.show()
 
    bp = BasicPlot(plot_data)
    bp.append_line_data(plot_colors=['blue', 'green', 'red', 'xkcd:fuchsia'])
    bp.assign_title()
    bp.assign_axis_labels(y_label_string='Quad Trans.')
    fig, ax = bp.fig, bp.ax
    for i in epoch_list:
        plt.vlines(i, 0,upperRange, **vline_style)
    bp.fig, bp.ax = fig, ax
    bp.export_plot_config(plot_directory_location,'', filename)

    return fig

def plot_individual_quadrant_transmission(f, r, plot_directory_location, iteration):
    '''Plot the most updated quadrant transmission.'''

    f = f[iteration]
    upperRange = np.max(f)

    num_adjoint_src = 4
    lambda_vector = copy.deepcopy(TEMPLATE_R_VECTOR)
    lambda_vector.update({'var_name': 'Wavelength', 'var_values': r, 'short_form': 'wl'})

    quad_trace = num_adjoint_src * [copy.deepcopy(TEMPLATE_F_VECTOR)]
    for adj_src in range(num_adjoint_src):
        quad_trace[adj_src] = copy.deepcopy(quad_trace)[adj_src]
        quad_trace[adj_src].update({'var_name': f'Q{adj_src}', 'var_values': f[adj_src,:]})
    plot_data = {'r':[lambda_vector],
                'f': quad_trace,
                'title': 'Quadrant Transmissions - Trace'}
    
    # plot_subfolder_name = 'quad_trans'
    # plot_directory_location = plot_directory_location / plot_subfolder_name
    # if not os.path.isdir(plot_directory_location):
    #     os.makedirs(plot_directory_location)

    bp = BasicPlot(plot_data)
    fig, ax = bp.fig, bp.ax
    bp.plot_config['y_axis']['limits'] = [0.0, 1.0]
    bp.append_line_data(plot_colors=['blue', 'green', 'red', 'xkcd:fuchsia'])
    bp.assign_title()
    bp.assign_axis_labels(y_label_string='Quad Trans.')
    # fig, ax = bp.fig, bp.ax
    # for i in range(0,numEpochs+1):
    # 	plt.vlines(i*numIter, 0,upperRange, **vline_style)
    # bp.fig, bp.ax = fig, ax
    bp.export_plot_config(plot_directory_location,'quad_trans', f'trans_i{iteration}')
    
    return fig

def visualize_device(cur_data, plot_directory_location, num_visualize_layers=1, iteration=''):
    '''Visualizes each (voxel) layer of cur_data. This data can be either density, permittivity, or index,
    and should be processed as such before passing to this function.
    Uses the binarization sigmoid corresponding to the epoch passed as input argument.'''
    
    output_plot = {}
    r_vectors = []      # Variables
    f_vectors = []      # Functions
    lambda_vectors = [] # Wavelengths
    
    r_vectors.append({'var_name': 'x-axis',
                      'var_values': range(cur_data.shape[0])
                      })
    r_vectors.append({'var_name': 'y-axis',
                      'var_values': range(cur_data.shape[1])
                      })
    
    f_vectors.append({'var_name': 'Device Data',
                      'var_values': cur_data
                        })
    
    output_plot['r'] = r_vectors
    output_plot['f'] = f_vectors
    output_plot['title'] = 'Device Visualization'
    
    # plot_layers = np.linspace(0, cur_data.shape[2]-1, num_visualize_layers).astype(int)
    plot_layers = np.linspace(0, 1, num_visualize_layers).astype(int)
    # actual_layers = np.linspace(0, gp.cv.num_vertical_layers, num_visualize_layers).astype(int)
    actual_layers = plot_layers		# todo: replace with above
    for layer_idx, layer in enumerate(plot_layers):
        fig, ax = plt.subplots()

        Y_grid, X_grid = np.meshgrid(np.squeeze(r_vectors[0]['var_values']),
                                    np.squeeze(r_vectors[1]['var_values']))
        # todo: Something is wrong here. The top face (y=2.04) gets printed as the left face in the device plot

        c = ax.pcolormesh(X_grid, Y_grid,
                            np.transpose(np.real(f_vectors[0]['var_values'][:,:,layer])),
                            # f_vectors[0]['var_values'][:,:,layer],
                            # f_vectors[0]['var_values'][:,:,layer][:-1, :-1],	# compensate for error when shading='flat'
                            cmap='jet', shading='auto')      # cmap='RdYlBu_r' is also good
        plt.gca().set_aspect('equal')

        title_string = f'Device Layer {actual_layers[layer_idx]}'
        plt.title(title_string)
        # plt.xlabel('x (um)')
        # plt.ylabel('y (um)')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        fig.colorbar(c, cax=cax)

        fig_width = 8.0
        fig_height = fig_width*1.0
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(fig_width)/(r-l)
        figh = float(fig_height)/(t-b)
        ax.figure.set_size_inches(figw, figh, forward=True)
        plt.tight_layout()

        plot_subfolder_name = 'device_layers'
        if not os.path.isdir(plot_directory_location / plot_subfolder_name):
            os.makedirs(plot_directory_location / plot_subfolder_name)
            
        plt.savefig(plot_directory_location / plot_subfolder_name / f'L{layer}_i{iteration}.png',
            bbox_inches='tight')
        print(f'Exported: Device Layer {actual_layers[layer_idx]}')

        plt.close()

    return fig, ax
        


def plot_overall_transmission_trace(f, plot_directory_location, epoch_list=None):
    '''Plot evolution trace of overall transmission'''

    if epoch_list is None:
        epoch_list = np.linspace(0, len(f), 10)
    upperRange = np.ceil(np.max(f))

    num_adjoint_src = f.shape[1]
    trace = np.zeros((f.shape[0]))

    for iteration in range(f.shape[0]):
        trace[iteration] = np.max( np.sum(f[iteration], 0) )

    iterations = copy.deepcopy(TEMPLATE_R_VECTOR)
    iterations.update({'var_name': 'Iterations', 'var_values': range(len(trace)), 'short_form': 'iter'})
    trans_trace = copy.deepcopy(TEMPLATE_F_VECTOR)
    trans_trace.update({'var_name': 'Transmission', 'var_values': trace})
    plot_data = {'r':[iterations],
                'f': [trans_trace],
                'title': 'Overall Peak Transmission - Trace'}
 
    bp = BasicPlot(plot_data)
    bp.append_line_data(plot_colors=['gray'])
    bp.assign_title()
    bp.assign_axis_labels()
    fig, ax = bp.fig, bp.ax
    for i in epoch_list:
        plt.vlines(i, 0,upperRange, **vline_style)
    bp.fig, bp.ax = fig, ax
    bp.export_plot_config(plot_directory_location,'','overall_trans_trace')

    return fig

def plot_Enorm_focal_2d(f, r, wl, plot_directory_location, iteration, wl_idxs = [0,-1]):
    '''Plot the most updated quadrant transmission.'''

    upperRange = np.max(f)

    num_adjoint_src = 1 #f.shape[0]
    f = f[np.newaxis, ...]
    r_vector = copy.deepcopy(TEMPLATE_R_VECTOR)
    r_vector.update({'var_name': 'x', 'var_values': r, 'short_form': 'x'})

    for wl_idx in wl_idxs:
        quad_trace = num_adjoint_src * [copy.deepcopy(TEMPLATE_F_VECTOR)]
        for adj_src in range(num_adjoint_src):
            quad_trace[adj_src] = copy.deepcopy(quad_trace)[adj_src]
            quad_trace[adj_src].update({'var_name': f'F{adj_src}', 'var_values': f[adj_src,:,wl_idx]})
        plot_data = {'r':[r_vector],
                    'f': quad_trace,
                    'title': 'E-field Plot'}

        bp = BasicPlot(plot_data)
        # bp.plot_config['y_axis']['limits'] = [0.0, 1.0]
        bp.append_line_data(plot_colors=['blue', 'green', 'red', 'xkcd:fuchsia'])
        bp.assign_title()
        bp.assign_axis_labels(y_label_string='Intensity')
        # fig, ax = bp.fig, bp.ax
        # for i in range(0,numEpochs+1):
        # 	plt.vlines(i*numIter, 0,upperRange, **vline_style)
        # bp.fig, bp.ax = fig, ax
        bp.export_plot_config(plot_directory_location,'Efield_plots', f'Enorm_wl{ int(1e9*wl[wl_idx])}nm_i{iteration}')
        
def plot_Enorm_focal_3d(f, x, y, wl, plot_directory_location, iteration, wl_idxs = [0,-1]):
    '''Plot the most updated quadrant transmission.'''

    '''For the given job index, plots the E-norm f.p. image at specific input spectra, corresponding to that job.
    Note: Includes focal scatter region. '''

    plot_subfolder_name = 'Enorm_fp_image_spectra'
    plot_directory_location = plot_directory_location / plot_subfolder_name
    if not os.path.isdir(plot_directory_location):
        os.makedirs(plot_directory_location)
    
    r_vectors = 2 * [copy.deepcopy(TEMPLATE_R_VECTOR)]
    r_vectors[0].update({'var_name': 'x', 'var_values': x, 'short_form': 'x'})
    r_vectors[1].update({'var_name': 'y', 'var_values': y, 'short_form': 'y'})
    f_vectors = [copy.deepcopy(TEMPLATE_F_VECTOR)]
    f_vectors[0].update({'var_name': 'Enorm', 'var_values': f})
    
    for wl_idx in wl_idxs:
        plot_wl = float(wl[wl_idx])
        
        fig, ax = plt.subplots()
        
        # Find the index of plot_wl in the wl_vector
        # wl_index = min(range(len(wl_vector)), key=lambda i: abs(wl_vector[i]-plot_wl))

        # If the monitor area is rectangular, truncate to a square
        max_spatial_idx = np.min(np.shape(f_vectors[0]['var_values'])[0:2])
        r_vectors[0]['var_values'] = r_vectors[0]['var_values'][0:max_spatial_idx]
        r_vectors[1]['var_values'] = r_vectors[1]['var_values'][0:max_spatial_idx]

        Y_grid, X_grid = np.meshgrid(np.squeeze(r_vectors[0]['var_values'])*1e6, np.squeeze(r_vectors[1]['var_values'])*1e6)
        c = ax.pcolormesh(X_grid, Y_grid, f_vectors[0]['var_values'][0:max_spatial_idx,0:max_spatial_idx,wl_idx],
                            cmap='jet', shading='auto')      # cmap='RdYlBu_r' is also good
        plt.gca().set_aspect('equal')
        
        wl_str = f'{plot_wl*1e9:.0f}nm' if plot_wl < 1e-6 else f'{plot_wl*1e6:.3f}um'
        title_string = r'$E_{norm}$' + ' at Focal Plane: $\lambda = $ ' + f'{wl_str}'
        plt.title(title_string)
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        fig.colorbar(c, cax=cax)

        fig_width = 8.0
        fig_height = fig_width*1.0
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(fig_width)/(r-l)
        figh = float(fig_height)/(t-b)
        ax.figure.set_size_inches(figw, figh, forward=True)
        plt.tight_layout()

        #plt.show(block=False)
        #plt.show()
            
        plt.savefig(plot_directory_location / f'Enorm_{wl_str}_i{iteration}.png',
            bbox_inches='tight')
        vipdopt.logger.info('Exported: Enorm Focal Plane Image at wavelength ' + wl_str)

        plt.close()
        
    return fig, ax

#* Evaluation after Optimization

# -- Spectrum Plot Functions (Per Job)

def plot_sorting_eff_spectrum(plot_data, job_idx, sweep_parameters, job_names, plot_folder, include_overall=False):
    '''For the given job index, plots the sorting efficiency (transmission) spectrum corresponding to that job.'''
    
    # plot_colors = ['blue', 'green', 'red', 'pink']
    # plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    plot_colors = ['blue', 'green', 'red']	
    plot_labels = ['Blue', 'Green', 'Red']
    
    if include_overall:
        plot_colors.append('gray')
        plot_labels.append('Trans.')
    else:
        plot_data['f'].pop(-1)		# remove overall transmission from f-vectors

    sp = SpectrumPlot(plot_data, sweep_parameters)
    
    sp.append_line_data(plot_colors, plot_labels)
    
    sp.assign_title('Spectrum in Each Quadrant', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Sorting Efficiency')
    
    sp.export_plot_config(plot_folder, 'sorting_efficiency_spectra', utility.isolate_filename(job_names[job_idx]).replace('.fsp', ''))
    return sp.fig, sp.ax

def plot_sorting_transmission_spectrum(plot_data, job_idx, sweep_parameters, job_names, plot_folder, include_overall=False, normalize_against='input_power'):
    '''For the given job index, plots the sorting efficiency (transmission) spectrum corresponding to that job.'''
    
    # plot_colors = ['blue', 'green', 'red', 'pink']
    # plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    plot_colors = ['blue', 'green', 'red']	
    plot_labels = ['Blue', 'Green', 'Red']

    if include_overall:
        plot_colors.append('gray')
        plot_labels.append('Trans.')
    else:
        plot_data['f'].pop(-1)		# remove overall transmission from f-vectors

    sp = SpectrumPlot(plot_data, sweep_parameters)
    
    sp.append_line_data(plot_colors, plot_labels)
    
    sp.assign_title('Spectrum in Each Quadrant', job_idx)
    if normalize_against == 'unity':
        sp.assign_axis_labels('Wavelength (um)', 'Sorting Transmission Power')
    else:	sp.assign_axis_labels('Wavelength (um)', 'Sorting Transmission Efficiency')
    
    sp.export_plot_config(plot_folder, 'sorting_trans_efficiency_spectra', utility.isolate_filename(job_names[job_idx]).replace('.fsp', ''))
    return sp.fig, sp.ax

def plot_crosstalk_adj_quadrants_spectrum(plot_data, job_idx, plot_colors, plot_labels, sweep_parameters, job_names, plot_folder):
    '''For the given job index, plots the crosstalk of adjacent quadrants according to that job, and the quadrant indices given to generate_crosstalk_adj_quadrants_spectrum().'''
    
    sp = SpectrumPlot(plot_data, sweep_parameters)
    
    sp.append_line_data(plot_colors, plot_labels)
    
    sp.assign_title('Spectrum in Each Quadrant', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Power')
    
    sp.export_plot_config(plot_folder, 'crosstalk_adj_quadrants_spectra', utility.isolate_filename(job_names[job_idx]).replace('.fsp', ''))
    return sp.fig, sp.ax

def plot_Enorm_focal_plane_image_spectrum(plot_data, job_idx, sweep_parameters, job_names, plot_folder,
                                          plot_wavelengths = None, ignore_opp_polarization = True):
    '''For the given job index, plots the E-norm f.p. image at specific input spectra, corresponding to that job.
    Note: Includes focal scatter region. '''
    
    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    lambda_vectors = plot_data['lambda']
    wl_vector = np.squeeze(lambda_vectors[0]['var_values']).tolist()
 
    if plot_wavelengths is None:
        ## We split the wavelength range into equal bands, and grab their midpoints.
        # plot_wavelengths = [lambda_min_um, lambda_values_um[int(len(lambda_values_um/2))]lambda_max_um]
        spectral_band_midpoints = np.multiply(gp.cv.num_points_per_band, np.arange(0.5, gp.cv.num_bands + 0.5)).astype(int)         # This fails if num_points_per_band and num_bands are different than what was optimized
        plot_wavelengths = np.array(wl_vector)[spectral_band_midpoints].tolist()
    
    if ignore_opp_polarization:
        plot_wavelengths = plot_wavelengths[:-1]
    
    for plot_wl in plot_wavelengths:
        plot_wl = float(plot_wl)
            
        fig, ax = plt.subplots()
        
        # Find the index of plot_wl in the wl_vector
        wl_index = min(range(len(wl_vector)), key=lambda i: abs(wl_vector[i]-plot_wl))

        # If the monitor area is rectangular, truncate to a square
        max_spatial_idx = np.min(np.shape(f_vectors[0]['var_values'])[0:2])
        r_vectors[0]['var_values'] = r_vectors[0]['var_values'][0:max_spatial_idx]
        r_vectors[1]['var_values'] = r_vectors[1]['var_values'][0:max_spatial_idx]

        Y_grid, X_grid = np.meshgrid(np.squeeze(r_vectors[0]['var_values'])*1e6, np.squeeze(r_vectors[1]['var_values'])*1e6)
        c = ax.pcolormesh(X_grid, Y_grid, f_vectors[0]['var_values'][0:max_spatial_idx,0:max_spatial_idx,wl_index],
                            cmap='jet', shading='auto')      # cmap='RdYlBu_r' is also good
        plt.gca().set_aspect('equal')
        
        wl_str = f'{plot_wl*1e9:.0f} nm' if plot_wl < 1e-6 else f'{plot_wl*1e6:.3f} um'
        title_string = r'$E_{norm}$' + ' at Focal Plane: $\lambda = $ ' + f'{wl_str}'
        plt.title(title_string)
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        fig.colorbar(c, cax=cax)
        
        fig_width = 8.0
        fig_height = fig_width*1.0
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(fig_width)/(r-l)
        figh = float(fig_height)/(t-b)
        ax.figure.set_size_inches(figw, figh, forward=True)
        plt.tight_layout()
        
        #plt.show(block=False)
        #plt.show()
        
        plot_subfolder_name = 'Enorm_fp_image_spectra'
        if not os.path.isdir(plot_folder + '/' + plot_subfolder_name):
            os.makedirs(plot_folder + '/' + plot_subfolder_name)
            
        plt.savefig(plot_folder + '/' + plot_subfolder_name\
            + '/' + utility.isolate_filename(job_names[job_idx]).replace('.fsp', '')\
            + '_' + f'{wl_str}' + '.png',
            bbox_inches='tight')
        vipdopt.logger.info('Exported: Enorm Focal Plane Image at wavelength ' + wl_str)
    
        plt.close()
    
    return fig, ax

def plot_device_cross_section_spectrum(plot_data, job_idx, sweep_parameters, job_names, plot_folder,
                                       plot_wavelengths = None, ignore_opp_polarization = True):
    '''For the given job index, plots the device cross section image plots at specific input spectra, corresponding to that job.'''
    import gc

    r_vectors = plot_data['r']
    f_vectors = plot_data['f']
    lambda_vectors = plot_data['lambda']
    wl_vector = np.squeeze(lambda_vectors[0]['var_values']).tolist()
    
    if plot_wavelengths is None:
        ## We split the wavelength range into equal bands, and grab their midpoints.
        # plot_wavelengths = [lambda_min_um, lambda_values_um[int(len(lambda_values_um/2))]lambda_max_um]
        spectral_band_midpoints = np.multiply(gp.cv.num_points_per_band, np.arange(0.5, gp.cv.num_bands + 0.5)).astype(int)         # This fails if num_points_per_band and num_bands are different than what was optimized
        plot_wavelengths = np.array(wl_vector)[spectral_band_midpoints].tolist()
    
    if ignore_opp_polarization:
        plot_wavelengths = plot_wavelengths[:-1]
    

    for device_cross_idx in range(0,6):
        
        #* Create E-norm images
        for plot_wl in plot_wavelengths:
            plot_wl = float(plot_wl)
            
            fig, ax = plt.subplots()
        
            is_x_slice = False
            y_grid = r_vectors[device_cross_idx * 3 + 2]['var_values']                   # Plot the z-axis as the vertical
            x_grid = r_vectors[device_cross_idx * 3]['var_values']                       # Check if it's the x-axis or y-axis that is the slice; plot the non-slice as the horizontal of the image plot
            slice_val = r_vectors[device_cross_idx * 3 + 1]['var_values']
            slice_str = f'$y = '
            if isinstance(x_grid, (int, float)):
                is_x_slice = True
                x_grid = r_vectors[device_cross_idx * 3 + 1]['var_values']
                slice_val = r_vectors[device_cross_idx * 3]['var_values']
                slice_str = f'$x = '
            slice_str +=  f'{slice_val:.2e}$; '
            
            # Find the index of plot_wl in the wl_vector
            wl_index = min(range(len(wl_vector)), key=lambda i: abs(wl_vector[i]-plot_wl))

            Y_grid, X_grid = np.meshgrid(np.squeeze(y_grid)*1e6, np.squeeze(x_grid)*1e6)
            c = ax.pcolormesh(X_grid, Y_grid, 
                              f_vectors[device_cross_idx * 2]['var_values'][:,:,wl_index],
                            cmap='jet', shading='auto')      # cmap='RdYlBu_r' is also good
            plt.axhline(0,color='black', linestyle='-',linewidth=2.2)
            plt.gca().set_aspect('auto')
            
            wl_str = f'{plot_wl*1e9:.0f} nm' if plot_wl < 1e-6 else f'{plot_wl*1e6:.3f} um'
            title_string = r'$E_{norm}$' + ', Device Cross-Section:\n' + slice_str + '$\lambda = $ ' + f'{wl_str}'
            plt.title(title_string)
            plt.xlabel('y (um)' if is_x_slice else 'x (um)')
            plt.ylabel('z (um)')
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.25)
            fig.colorbar(c, cax=cax)
            
            fig_width = 8.0
            fig_height = fig_width / np.shape(X_grid)[0] * np.shape(X_grid)[1]
            l = ax.figure.subplotpars.left
            r = ax.figure.subplotpars.right
            t = ax.figure.subplotpars.top
            b = ax.figure.subplotpars.bottom
            figw = float(fig_width)/(r-l)
            figh = float(fig_height)/(t-b)
            ax.figure.set_size_inches(figw, figh, forward=True)
            plt.tight_layout()
            
            #plt.show(block=False)
            #plt.show()
            
            plot_subfolder_name = 'device_cross_section_spectra'
            if not os.path.isdir(plot_folder + '/' + plot_subfolder_name):
                os.makedirs(plot_folder + '/' + plot_subfolder_name)
                
            plt.savefig(plot_folder + '/' + plot_subfolder_name\
                + '/' + utility.isolate_filename(job_names[job_idx]).replace('.fsp', '')\
                + '_Enorm_' + f'{device_cross_idx}' + '_' + f'{wl_str}' + '.png',
                bbox_inches='tight')
            vipdopt.logger.info(f'Exported: Device Cross Section Enorm Image {device_cross_idx} at wavelength ' + wl_str)

            plt.close()

        #* Create real_E_x images
        display_vector_labels = ['x','y','z']
        display_vector_component = 0            # 0 for x, 1 for y, 2 for z
        
        for display_vector_component in range(0,3):
            for plot_wl in plot_wavelengths:
                plot_wl = float(plot_wl)
                
                fig, ax = plt.subplots()
                
                is_x_slice = False
                y_grid = r_vectors[device_cross_idx * 3 + 2]['var_values']                   # Plot the z-axis as the vertical
                x_grid = r_vectors[device_cross_idx * 3]['var_values']                       # Check if it's the x-axis or y-axis that is the slice; plot the non-slice as the horizontal of the image plot
                slice_val = r_vectors[device_cross_idx * 3 + 1]['var_values']
                slice_str = f'$y = '
                if isinstance(x_grid, (int, float)):
                    is_x_slice = True
                    x_grid = r_vectors[device_cross_idx * 3 + 1]['var_values']
                    slice_val = r_vectors[device_cross_idx * 3]['var_values']
                    slice_str = f'$x = '
                slice_str +=  f'{slice_val:.2e}$; '
                    
                # Find the index of plot_wl in the wl_vector
                wl_index = min(range(len(wl_vector)), key=lambda i: abs(wl_vector[i]-plot_wl))

                Y_grid, X_grid = np.meshgrid(np.squeeze(y_grid)*1e6, np.squeeze(x_grid)*1e6)
                c = ax.pcolormesh(X_grid, Y_grid, 
                                  np.real(f_vectors[device_cross_idx * 2 + 1]['var_values'][display_vector_component][:,:,wl_index]),
                                cmap='jet', shading='auto')      # cmap='RdYlBu_r' is also good
                plt.axhline(0,color='black', linestyle='-',linewidth=2.2)
                plt.gca().set_aspect('auto')
                
                wl_str = f'{plot_wl*1e9:.0f} nm' if plot_wl < 1e-6 else f'{plot_wl*1e6:.3f} um'
                title_string = r'$Re(E_x)$' + ', Device Cross-Section:\n' + slice_str + '$\lambda = $ ' + f'{wl_str}'
                plt.title(title_string)
                plt.xlabel('y (um)' if is_x_slice else 'x (um)')
                plt.ylabel('z (um)')
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.25)
                fig.colorbar(c, cax=cax)
                
                fig_width = 8.0
                fig_height = fig_width / np.shape(X_grid)[0] * np.shape(X_grid)[1]
                l = ax.figure.subplotpars.left
                r = ax.figure.subplotpars.right
                t = ax.figure.subplotpars.top
                b = ax.figure.subplotpars.bottom
                figw = float(fig_width)/(r-l)
                figh = float(fig_height)/(t-b)
                ax.figure.set_size_inches(figw, figh, forward=True)
                plt.tight_layout()
                
                #plt.show(block=False)
                #plt.show()
                
                plot_subfolder_name = 'device_cross_section_spectra'
                if not os.path.isdir(plot_folder + '/' + plot_subfolder_name):
                    os.makedirs(plot_folder + '/' + plot_subfolder_name)
                    
                plt.savefig(plot_folder + '/' + plot_subfolder_name\
                    + '/' + utility.isolate_filename(job_names[job_idx]).replace('.fsp', '')\
                    + '_realE' + f'{display_vector_labels[display_vector_component]}' + '_' + f'{device_cross_idx}' + '_' + f'{wl_str}' + '.png',
                    bbox_inches='tight')
                vipdopt.logger.info(f'Exported: Device Cross Section E{display_vector_labels[display_vector_component]}_real Image {device_cross_idx} at wavelength ' + wl_str)
        
        plt.close()
    
    gc.collect()
    
    return fig, ax

def plot_crosstalk_power_spectrum(plot_data, job_idx, sweep_parameters, job_names, plot_folder):
    '''For the given job index, plots the crosstalk power corresponding to that job.'''

    plot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_labels = ['NW', 'N', 'NE', 'W', 'Center', 'E', 'SW', 'S', 'SE']

    sp = SpectrumPlot(plot_data, sweep_parameters)

    sp.append_line_data(plot_colors, plot_labels)
    
    sp.assign_title('Device Crosstalk Power', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Normalized Power')
    
    sp.export_plot_config(plot_folder, 'crosstalk_power_spectra', utility.isolate_filename(job_names[job_idx]).replace('.fsp', ''))
    return sp.fig, sp.ax

def plot_exit_power_distribution_spectrum(plot_data, job_idx, sweep_parameters, job_names, plot_folder):
    '''For the given job index, plots the power spectrum for focal region power and oblique scattering power, corresponding to that job.
    The two are normalized against their sum.'''
    
    sp = SpectrumPlot(plot_data, sweep_parameters)

    sp.append_line_data(None, None)

    sp.assign_title('Power Distribution at Focal Plane', job_idx)
    sp.assign_axis_labels('Wavelength (um)', 'Normalized Power')
    
    sp.export_plot_config(plot_folder,  'exit_power_distribution_spectra', utility.isolate_filename(job_names[job_idx]).replace('.fsp', ''))
    return sp.fig, sp.ax

# -- Sweep Plot Functions (Overall)

def plot_sorting_transmission_sweep_1d(plot_data, slice_coords, plot_folder,
                                       include_overall=True, plot_stdDev = True, cutoff_1d_sweep_offset = [0,0]):
    '''Produces a 1D sorting (transmission) efficiency plot using given coordinates to slice the N-D array of data contained in plot_data.
    X-axis i.e. sweep variable is given by the index of the entry of type 'slice' in the slice_coords.'''
    
    sp = SweepPlot(plot_data, slice_coords)
    
    # plot_colors = ['blue', 'green', 'red', 'gray']
    # plot_labels = ['Blue', 'Green (x-pol.)', 'Red', 'Green (y-pol.)']
    plot_colors = ['blue', 'green', 'red']	
    plot_labels = ['Blue', 'Green', 'Red']

    if include_overall:
        plot_colors.append('gray')
        plot_labels.append('Trans.')
    else:
        plot_data['f'].pop(-1)		# remove overall transmission from f-vectors
    sp.append_line_data(slice_coords, plot_stdDev, plot_colors, plot_labels)

    sp.assign_title('Sorting Transmission Efficiency - Parameter Range:')
    sp.assign_axis_labels(slice_coords, 'Sorting Transmission Efficiency')

    sp.export_plot_config(plot_folder, '', 'sorting_trans_eff_sweep_', slice_coords)
    
    return sp.fig, sp.ax

 
if __name__ == "__main__":
    plot_directory_location = 'plots'
    cutoff_1d_sweep_offset = [0, 0]

    #* Call plotting function and export 
    plot_basic_1d(TEMPLATE_PLOT_DATA, 'evaluation/plots', 'test', 'test1',
                           title='Variation of Circle Diameter with Radius')
 
# print(3)