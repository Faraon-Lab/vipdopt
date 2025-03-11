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
from pathlib import Path
import copy

import matplotlib
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.ticker import (  # type: ignore
    AutoMinorLocator,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
matplotlib.use('TkAgg')

# Custom Classes and Imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
# # Gets all parameters from config file - store all those variables within the namespace. Editing cfg edits it for all modules accessing it
# # See https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
import vipdopt
from vipdopt import utils

# * Template
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
    'iterating': True,
}
sweep_parameters = [TEMPLATE_R_VECTOR]

TEMPLATE_F_VECTOR = {
    'var_name': 'Circle Diameter',
    'var_values': [10.179, 18.096, 22.902, 28.274],
    'var_stdevs': [0.1, 0.1, 0.1, 0.1],
    'statistics': 'mean',
}

TEMPLATE_PLOT_DATA = {
    'r': copy.deepcopy(sweep_parameters),
    'f': [copy.deepcopy(TEMPLATE_F_VECTOR)],
    'title': 'template',
}

# * Plot Style Params
# if not running_on_local_machine:
#     from matplotlib import font_manager
#     font_manager._rebuild()
#     fp = font_manager.FontProperties(fname=r"/central/home/ifoo/.fonts/Helvetica-Neue-Medium-Extended.ttf")
#     print('Font name is ' + fp.get_name())
#     plt.rcParams.update({'font.sans-serif':fp.get_name()})

# plt.rcParams.update({'font.sans-serif':'Helvetica Neue',            # Change this based on whatever custom font you have installed
#                      'font.weight': 'normal', 'font.size':20})
plt.rcParams.update({'font.weight': 'normal', 'font.size': 20})
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
# plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
# plt.rcParams['text.usetex'] = True

# mpl.rcParams['font.sans-serif'] = 'Helvetica Neue'
# mpl.rcParams['font.family'] = 'sans-serif'

marker_style = {'linestyle': '-', 'linewidth': 2.2, 'marker': 'o', 'markersize': 4.5}
vline_style = {'color': 'gray', 'linestyle': '--', 'linewidth': 1}


def adjust_figure_size(ax, fig_width, fig_height=None):
    if fig_height is None:
        fig_height = fig_width * 4 / 5

    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(fig_width) / (r - l)
    figh = float(fig_height) / (t - b)
    ax.figure.set_size_inches(figw, figh, forward=True)

    return ax

def apply_common_plot_style(ax=None, plt_kwargs=None, show_legend=True):
    """Sets universal axis properties for all plots. Function is called in each plot generating function."""
    if plt_kwargs is None:
        plt_kwargs = {}
    if ax is None:
        ax = plt.gca()

    # Creates legend
    if show_legend:
        # ax.legend(prop={'size': 10})
        ax.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1.1, 0.5))

    # Figure size
    ax = adjust_figure_size(ax, 8.0)

    # Minor Ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Y-Axis Exponent Repositioning
    ax.get_yaxis().get_offset_text().set_position((-0.1, 0.5))

    return ax

def enter_plot_data_1d(plot_config, fig=None, ax=None):
    """Every plot function calls this function to actually put the data into the plot. The main input is plot_config, a dictionary that contains
    parameters controlling every single property of the plot that might be relevant to data.
    """

    for plot_idx in range(len(plot_config['lines'])):
        data_line = plot_config['lines'][plot_idx]

        x_plot_data = (
            data_line['x_axis']['factor'] * np.array(data_line['x_axis']['values'])
            + data_line['x_axis']['offset']
        )
        y_plot_data = (
            data_line['y_axis']['factor'] * np.array(data_line['y_axis']['values'])
            + data_line['y_axis']['offset']
        )

        plt.plot(
            x_plot_data[data_line['cutoff']],
            y_plot_data[data_line['cutoff']],
            color=data_line['color'],
            label=data_line['legend'],
            **data_line['marker_style'],
        )

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

# * Define a class called BasicPlot with basic plotting methods and data storage options.
class BasicPlot:
    def __init__(self, plot_data):
        """Initializes the plot_config variable of this class object and also the Plot object."""
        self.r_vectors = plot_data['r'][0]
        self.f_vectors = plot_data['f']

        self.fig, self.ax = plt.subplots()
        self.plot_config = {
            'title': plot_data['title'],
            'x_axis': {'label': '', 'limits': []},
            'y_axis': {'label': '', 'limits': []},
            'lines': [],
        }

    def append_line_data(self, plot_colors=None, plot_labels=None, plot_alphas=None,
                         *args, **kwargs):
        """Appends all line data stored in the f_vectors to the plot_config. Here we can assign colors and legend labels all at once."""

        for plot_idx in range(len(self.f_vectors)):
            line_data = {
                'x_axis': {'values': None, 'factor': 1, 'offset': 0},
                'y_axis': {'values': None, 'factor': 1, 'offset': 0},
                'cutoff': None,  # A slice of two numbers i.e. slice(1,8) that defines the indices at which the data is truncated
                'color': None,
                'alpha': 1.0,
                'legend': None,
                'marker_style': marker_style,
            }

            line_data['x_axis']['values'] = self.r_vectors['var_values']
            line_data['y_axis']['values'] = self.f_vectors[plot_idx]['var_values']
            line_data['cutoff'] = slice(0, len(line_data['x_axis']['values']))
            # line_data['cutoff'] = slice(8,-8)

            if plot_colors is not None:
                line_data['color'] = plot_colors[plot_idx]
            if plot_labels is not None:
                line_data['legend'] = plot_labels[plot_idx]
            else:
                line_data['legend'] = self.f_vectors[plot_idx]['var_name']
            if plot_alphas is not None:
                line_data['alpha'] = plot_alphas[plot_idx]

            self.plot_config['lines'].append(line_data)

    def alter_line_property(self, key, new_value, nested_keys=None):
        """Alters a property throughout all of the line data.
        If new_value is an array it will change all of the values index by index; otherwise it will blanket change everything.
        The nested_keys argument is to cover multiple-level dictionaries.
        """

        if nested_keys is None:
            nested_keys = []
        for line_idx, line_data in enumerate(self.plot_config['lines']):
            if not isinstance(new_value, list):
                utils.set_by_path(line_data, [*nested_keys, key], new_value)
            else:
                utils.set_by_path(line_data, [*nested_keys, key], new_value[line_idx])

    def assign_title(self, title_string=None, *args, **kwargs):
        """Replaces title of plot."""
        if title_string is not None:
            self.plot_config['title'] = title_string

    def assign_axis_labels(self, x_label_string=None, y_label_string=None, *args, **kwargs):
        """Replaces axis labels of plot."""
        if x_label_string is None:
            x_label_string = self.r_vectors['var_name']
        if y_label_string is None:
            y_label_string = self.f_vectors[0]['var_name']

        self.plot_config['x_axis']['label'] = x_label_string
        self.plot_config['y_axis']['label'] = y_label_string

    def export_plot_config(self, plot_directory_location, plot_subfolder_name,
                           filename, close_plot=True,
                        ):
        """Creates plot using the plot config, and then exports."""

        self.fig, self.ax = enter_plot_data_1d(self.plot_config, self.fig, self.ax)

        # ! NOTE: This is where you do any additional adjustment of the plot before saving out
        # plot adjustment code

        SAVE_LOCATION = Path(plot_directory_location) / plot_subfolder_name
        SAVE_LOCATION.mkdir(parents=True, exist_ok=True)

        plt.savefig(SAVE_LOCATION / f'{filename}.png', bbox_inches='tight')

        export_msg_string = filename.replace('_', '').title()
        vipdopt.logger.info('Exported: ' + export_msg_string)

        if close_plot:
            plt.close()


def plot_basic_1d(  plot_data,
                    plot_directory_location,
                    plot_subfolder_name,
                    filename,
                    title=None,
                    xlabel_txt=None,
                    ylabel_txt=None,
                    *args, **kwargs,
                ):

    # Calls an instance of BasicPlot, initializes with the data.
    bp = BasicPlot(plot_data)
    bp.append_line_data(*args, **kwargs)  # Sets up plot config with line data.
    bp.assign_title(title_string=title)  # Assign title
    bp.assign_axis_labels(
        x_label_string=xlabel_txt, y_label_string=ylabel_txt
    )  # Assign x and y axis labels
    bp.export_plot_config(
        plot_directory_location, plot_subfolder_name, filename
    )  # Do actual plotting and then save out the figure.


def plot_history_traces(f_vectors,      # Assume all have the same x-axis / r-vector.
                       plot_directory_location, filename,
                       plot_subfolder = '',
                       epoch_list=None,
                       title_str='Optimization History',
                       f_labels=None,
                       statistics_func = np.max,
                       *args, **kwargs):

    """Plot trace of some FOM during evolution of optimization."""

    f_vectors = np.array(f_vectors)     # Make sure shape 0 = length of x-axis.
    if f_labels is None:
        f_labels = [None]*f_vectors.shape[1]

    if epoch_list is None:
        epoch_list = np.linspace(0, f_vectors.shape[0], 10)
    upperRange = np.max(f_vectors)  # np.ceil(np.max(f))

    iterations = copy.deepcopy(TEMPLATE_R_VECTOR)
    iterations.update({
        'var_name': 'Iterations',
        'var_values': range(f_vectors.shape[0]),
        'short_form': 'iter',
    })

    fom_traces = f_vectors.shape[1] * [copy.deepcopy(TEMPLATE_F_VECTOR)]
    for adj_src in range(f_vectors.shape[1]):
        fom_traces[adj_src] = copy.deepcopy(fom_traces)[adj_src]
        fom_traces[adj_src].update({
            'var_name': f_labels[adj_src],
            'var_values': f_vectors[:, adj_src],
        })

    plot_data = {
        'r': [iterations],
        'f': fom_traces,
        'title': title_str,
    }

    bp = BasicPlot(plot_data)
    bp.append_line_data(*args, **kwargs)
    bp.assign_title()
    bp.assign_axis_labels(*args, **kwargs)
    fig, ax = bp.fig, bp.ax
    for i in epoch_list:
        plt.vlines(i, 0, upperRange, **kwargs.get('vline_style', vline_style))
    bp.fig, bp.ax = fig, ax
    bp.export_plot_config(plot_directory_location, plot_subfolder, filename)

    return fig


def plot_fom_trace(f, plot_directory_location, epoch_list=None ,filename='fom_trace'):
    """Plot FOM trace during evolution of optimization."""

    return plot_history_traces(f.reshape(-1,1),
                                plot_directory_location, filename,
                                epoch_list=epoch_list,
                                title_str='Figure of Merit - Trace',
                                plot_colors=['orange'],
                                )

def plot_bayer_quadrant_transmission_trace(
    f,      # axis 0: iterations, axis 1: num_adjoint_sources, axis 2: wavelength
    plot_directory_location, epoch_list=None, filename='quad_trans_trace',
    statistics_func = np.max, # np.mean, etc.
    # display_greens='sum',# 'separate', 'both' <---
    line_labels=['Q0', 'Q1', 'Q2', 'Q3'],
):
    """Plot evolution trace of quadrant transmission."""

    trace = statistics_func(np.array(f).reshape(f.shape[0],f.shape[1],-1), axis=2)
    vipdopt.logger.info('Quadrant Transmissions Trace is:')

    return plot_history_traces(trace, plot_directory_location, filename,
                    epoch_list=epoch_list,
                    title_str = 'Quadrant Transmissions - Trace',
                    plot_colors=['blue', 'green', 'red', 'xkcd:fuchsia'],
                    y_label_str='Quad Trans.',
                    f_labels=line_labels,
                )


def plot_Enorm_2d(r, f_vectors, wl,
                    plot_directory_location, filename,
                    plot_subfolder = 'Efield_plots',
                    wl_idxs=None,
                ):
    """Plot the most updated E-field over the monitor."""

    if wl_idxs is None:
        wl_idxs = [0, -1]

    r_vector = copy.deepcopy(TEMPLATE_R_VECTOR)
    r_vector.update({'var_name': 'x', 'var_values': r, 'short_form': 'x'})

    figs = []
    axs = []
    for wl_idx in wl_idxs:
        Enorm = copy.deepcopy(TEMPLATE_F_VECTOR)
        Enorm.update({'var_name': f'E_norm',
                        'var_values': f_vectors[:, wl_idx],
                    })
        plot_data = {'r': [r_vector], 'f': [Enorm], 'title': f'E-field at Focal Plane, {int(1e3 * wl[wl_idx]):.3f}nm'}

        bp = BasicPlot(plot_data)
        bp.append_line_data(plot_colors=['blue', 'green', 'red', 'xkcd:fuchsia'])
        bp.assign_title()
        bp.assign_axis_labels(y_label_string='Intensity')
        wl_str = f'{wl[wl_idx]:.3f}um' if 1e3*wl[wl_idx] >= 1000 else f'{int(1e3*wl[wl_idx]):d}nm'
        bp.export_plot_config(
            plot_directory_location,
            plot_subfolder_name=plot_subfolder,
            filename=filename+f'_wl{wl_str}',
        )
        figs.append(bp.fig)
        axs.append(bp.ax)

    return figs


def plot_spectrum(wl, f_vectors,
                plot_directory_location, filename,
                plot_subfolder = '',
                band_vals=None,
                title_str='Spectrum',
                line_labels=None,
                y_limits=None, #[0.0, 1.0],
                statistics_func=np.max,
                *args, **kwargs):

    """Plot spectrum of some FOM at some point of optimization."""

    f_vectors = np.array(f_vectors)     # Make sure shape 0 = length of x-axis.
    if line_labels is None:
        line_labels = [None]*f_vectors.shape[1]

    if band_vals is None:
        band_vals = []
    upperRange = np.max(f_vectors)  # np.ceil(np.max(f))

    lambda_vector = copy.deepcopy(TEMPLATE_R_VECTOR)
    lambda_vector.update({
        'var_name': 'Wavelength',
        'var_values': wl,
        'short_form': 'wl',
    })

    stat = []
    fom_spectra = f_vectors.shape[0] * [copy.deepcopy(TEMPLATE_F_VECTOR)]
    for adj_src in range(f_vectors.shape[0]):
        fom_spectra[adj_src] = copy.deepcopy(fom_spectra)[adj_src]
        fom_spectra[adj_src].update({
            'var_name': line_labels[adj_src],
            'var_values': f_vectors[adj_src, :],
        })
        stat.append(statistics_func(f_vectors[adj_src, :]))

    plot_data = {
        'r': [lambda_vector],
        'f': fom_spectra,
        'title': title_str,
    }

    bp = BasicPlot(plot_data)
    if y_limits is not None:
        bp.plot_config['y_axis']['limits'] = y_limits
    bp.append_line_data(*args, **kwargs)
    bp.assign_title()
    bp.assign_axis_labels(*args, **kwargs)
    fig, ax = bp.fig, bp.ax
    for i in band_vals:
        plt.vlines(i, 0, upperRange, **kwargs.get('vline_style', vline_style))
    bp.fig, bp.ax = fig, ax
    bp.export_plot_config(plot_directory_location, plot_subfolder, filename)

    return fig, stat


def plot_bayer_quadrant_transmission_spectra(
    wl,
    f_vectors,      # axis 0: quadrant, axis 1-3: xyz, axis 4: wavelength
    plot_directory_location, filename='trans_spec',
    band_vals=None,
    statistics_func = np.max, # np.mean, etc.
    # display_greens='sum', # 'separate', 'both'
    line_labels=['Q0', 'Q1', 'Q2', 'Q3'],
    plot_colors=['blue', 'green', 'red', 'xkcd:fuchsia'],
    ):

    # Keep the first and last axes - 0: quadrant, -1: wavelength
    f_vectors = f_vectors.reshape(f_vectors.shape[0], f_vectors.shape[-1])

    fig, stat = plot_spectrum(wl, f_vectors,
                              plot_directory_location, filename, plot_subfolder='quad_trans',
                              band_vals=band_vals, title_str='Quadrant Transmission Spectra',
                              y_label_string='Quad Trans.',
                              line_labels=line_labels,
                            #   y_limits=[0.0,1.0],
                              plot_colors=plot_colors,
                              statistics_func=statistics_func,
                            )

    return fig, stat

def plot_data_2d(r_vectors, f_vector,
                plot_directory_location, filename,
                plot_subfolder = '',
                title_str='Spectrum',
                normalize=False,
                *args, **kwargs):

    if normalize:
        def normalize(x,minx=0,maxx=1):
            return minx + (maxx-minx) * (x-np.min(x))/(np.max(x)-np.min(x))
        f_vector['var_values'] = normalize(f_vector['var_values'])


    fig, ax = plt.subplots()

    X_grid, Y_grid = np.meshgrid(
        np.squeeze(r_vectors[0]['var_values']),
        np.squeeze(r_vectors[1]['var_values']),
        indexing='ij'
    )

    c = ax.pcolor(
        X_grid, Y_grid, f_vector['var_values'],
        cmap='jet',     # 'RdYlBu_r' is also good
        shading='auto',
    )
    plt.gca().set_aspect('equal')
    plt.title(title_str)
    plt.xlabel(r_vectors[0]['var_name'])
    plt.ylabel(r_vectors[1]['var_name'])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.25)
    fig.colorbar(c, cax=cax)

    fig_width = 8.0
    fig_height = fig_width * 1.0
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(fig_width) / (r - l)
    figh = float(fig_height) / (t - b)
    ax.figure.set_size_inches(figw, figh, forward=True)
    plt.tight_layout()

    SAVE_LOCATION = Path(plot_directory_location) / plot_subfolder
    SAVE_LOCATION.mkdir(parents=True, exist_ok=True)
    plt.savefig( SAVE_LOCATION / f'{filename}.png', bbox_inches='tight')
    plt.close()

    return fig, ax

def visualize_device(r1,r2, cur_data,
                     plot_directory_location,
                     filename='',
                     plot_subfolder='device_layers',
                     num_visualize_layers=1,
                ):
    """Visualizes each (voxel) layer of cur_data. This data can be either density, permittivity, or index,
    and should be processed as such before passing to this function.
    Uses the binarization sigmoid corresponding to the epoch passed as input argument.
    """

    r_vectors = []  # Variables
    f_vectors = []  # Functions

    r_vectors.append({'var_name': 'x-axis', 'var_values': r1})
    r_vectors.append({'var_name': 'y-axis', 'var_values': r2})
    f_vectors.append({'var_name': 'Device Data', 'var_values': cur_data})

    # plot_layers = np.linspace(0, cur_data.shape[2]-1, num_visualize_layers).astype(int)
    plot_layers = np.linspace(0, 1, num_visualize_layers).astype(int)
    # actual_layers = np.linspace(0, gp.cv.num_vertical_layers, num_visualize_layers).astype(int)
    actual_layers = plot_layers  # todo: replace with above
    for layer_idx, layer in enumerate(plot_layers):
        
        device_data = {'var_name': 'Device', 
                       'var_values': np.real(f_vectors[0]['var_values'][:, :, layer]),
                                    # f_vectors[0]['var_values'][:,:,layer],
                                    # f_vectors[0]['var_values'][:,:,layer][:-1, :-1],	# compensate for error when shading='flat', 
                    }
        
        fig, ax = plot_data_2d(r_vectors, 
                                device_data,
                                plot_directory_location,
                                filename=f'L{layer}_{filename}',
                                plot_subfolder='device_layers',
                                title_str=f'Device Layer {actual_layers[layer_idx]}',
                            )
        print(f'Exported: Device Layer {actual_layers[layer_idx]}')

    return fig, ax

def close_all():
    plt.close('all')