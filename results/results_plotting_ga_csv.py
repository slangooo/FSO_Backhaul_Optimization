#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

import numpy as np
import pandas as pd
import os
import sys
import math
import pickle
import matplotlib.pyplot as plt
from matplotlib import markers, lines, colors as mcolors
import matplotlib as mpl

results_folder = os.path.join(os.getcwd(), "ga\\")

_markers = list(markers.MarkerStyle.markers.keys())
_markers.pop(0)
_markers.pop(0)
if None in _markers: _markers.remove(None)
if 'None' in _markers: _markers.remove('None')
if '' in _markers: _markers.remove('')
if ' ' in _markers: _markers.remove(' ')
_linestyles = list(lines.lineStyles.keys())
if 'None' in _linestyles: _linestyles.remove('None')
if '' in _linestyles: _linestyles.remove('')
if ' ' in _linestyles: _linestyles.remove(' ')
_linestyles.append((0, (3, 5, 1, 5, 1, 5)))
_linestyles.append((0, (5, 10)))
_linestyles.pop(0)
_colors = list(mcolors.BASE_COLORS.keys())
mpl.rc('font', family='Times New Roman')

def procent_formatter(x, pos):
    return '{:.0f} %'.format(x * 100)

def step(values, tics):
    min_value = min(values)
    max_value = max(values)
    step = (max_value - min_value) / tics
    magnitude_index = math.floor(math.log10(abs(step)))

    magnitude_value = 10 ** magnitude_index

    step_normalized = abs(step / magnitude_value)
    if step_normalized <= 1.5:
        tic_normalized = 1
    elif step_normalized <= 3:
        tic_normalized = 2
    elif step_normalized <= 7.5:
        tic_normalized = 5
    else:
        tic_normalized = 10
    
    tic_value = tic_normalized * magnitude_value

    min_value_norm = math.floor(min_value / tic_value) * tic_value
    max_value_norm = math.ceil (max_value / tic_value) * tic_value

    value = min_value_norm
    values_norm = []
    while value <= max_value_norm:
        values_norm.append(value)
        value += tic_value
    
    return values_norm

def min_max_with_indent(values, indent):
    min_value = min(values)
    max_value = max(values)
    range = max_value - min_value
    ind_value = range * indent
    return (min_value - ind_value, max_value + ind_value)

def figure (series, x_values, data_values, legend_loc,
            x_range, y_range,
            x_label, y_label,
            x_label_formatter, y_label_formatter,
            indent, tics):
    fig, axs = plt.subplots()
    for i, serie_label in enumerate(series):
        axs.plot(
            x_values,
            data_values[i],
            color=_colors[i],
            # markersize=4,
            # marker=_markers[i],
            # ls=_linestyles[0],
            lw=1.8,
            label=serie_label)
    
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc=legend_loc, ncol=2)

    axs.set_xlabel(x_label, fontsize=10)
    if x_label_formatter:
        axs.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(x_label_formatter))
    if not x_range:
        x_range = x_values
    xticks = step(x_values, tics)
    min_max_x = min_max_with_indent(xticks, indent)
    axs.set_xlim(min_max_x[0], min_max_x[1])
    axs.set_xticks(xticks)

    axs.set_ylabel(y_label, fontsize=10)
    if y_label_formatter:
        axs.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(y_label_formatter))
    if not y_range:
        y_range = [np.min(data_values), np.max(data_values)]
    yticks = step(y_range, tics)
    min_max_y = min_max_with_indent(yticks, indent)
    axs.set_ylim(min_max_y[0], min_max_y[1])
    axs.set_yticks(yticks)

    axs.grid(True)

    return fig

def ga(file, x_label, y_label, y_label_formatter, y_range, legend_x, legend_y):
    data = pd.read_csv(results_folder + f'{file}.csv', sep=';', decimal=',', header=None)

    fig = figure(
        series            = data.values[1:,0],
        x_values          = data.values[0,1:],
        data_values       = data.values[1:,1:],
        legend_loc        = (legend_x, legend_y),
        x_range           = None,
        x_label           = x_label,
        x_label_formatter = None,
        y_range           = y_range,
        y_label           = y_label,
        y_label_formatter = y_label_formatter,
        indent            = 0.05,
        tics              = 5
    )

    fig.savefig(results_folder + f'{file}.pdf', format='pdf')

if __name__ == '__main__':
    args = sys.argv[1:]
    file = 'test' if not args else args[0]
    left   = 0.15
    right  = 0.60
    top    = 0.68
    bottom = 0.15
    ga('fd', f'Maximum backhaul distance $d_\mathrm{{max}}$ [m]', f'Probability of success' , procent_formatter, [0, 1]     , right, bottom)
    ga('fn', f'Number of DBSs'                                  , f'Probability of success' , procent_formatter, [0, 1]     , left , top   )
    ga('sd', f'Maximum backhaul distance $d_\mathrm{{max}}$ [m]', f'DBSs surplus throughput', None             , [0, 180000], right, top   )
    ga('sn', f'Number of DBSs'                                  , f'DBSs surplus throughput', None             , [0, 180000], left , top   )
