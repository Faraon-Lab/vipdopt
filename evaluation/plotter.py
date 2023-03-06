import os
import sys
import copy
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
							   AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.getcwd())
from utils import utility

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
		self.r_vectors = plot_data['r'][0]
		self.f_vectors = plot_data['f']
		
		self.fig, self.ax = plt.subplots()
		self.plot_config = {'title': plot_data['title'],
							'x_axis': {'label':'', 'limits':[]},
							'y_axis': {'label':'', 'limits':[]},
							'lines': []
							}
	
	def append_line_data(self, plot_colors=None, plot_labels=None):
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
		
		export_msg_string = (SAVE_LOCATION + f'/{filename}'.replace("_", " ")).title()
		logging.info('Exported: ' + export_msg_string)
		
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

def plot_fom_trace(f, plot_directory_location):
	# Plot FOM trace
	trace = []
	numEpochs = len(f); numIter = len(f[0])
	upperRange = np.ceil(np.max(f))

	for epoch_fom in f:
		trace = np.concatenate((trace, epoch_fom), axis=0)

	iterations = copy.deepcopy(TEMPLATE_R_VECTOR)
	iterations.update({'var_name': 'Iterations', 'var_values': range(f.size), 'short_form': 'iter'})
	fom = copy.deepcopy(TEMPLATE_F_VECTOR)
	fom.update({'var_name': 'FoM', 'var_values': trace})
	plot_data = {'r':[iterations],
				'f': [fom],
				'title': 'Figure of Merit - Trace'}
 
	bp = BasicPlot(plot_data)
	bp.append_line_data(plot_colors=['orange'])
	bp.assign_title()
	bp.assign_axis_labels()
	fig, ax = bp.fig, bp.ax
	for i in range(0,numEpochs+1):
		plt.vlines(i*numIter, 0,upperRange, **vline_style)
	bp.fig, bp.ax = fig, ax
	bp.export_plot_config(plot_directory_location,'','fom_trace')
 

 
if __name__ == "__main__":
	plot_directory_location = 'plots'
	cutoff_1d_sweep_offset = [0, 0]

	#* This is where we perturb the data points as needed
	# TODO: modify data code

	#* Call plotting function and export 
	plot_basic_1d(TEMPLATE_PLOT_DATA, 'evaluation/plots', 'test', 'test1',
			  			 title='Variation of Circle Diameter with Radius')
 
print(3)