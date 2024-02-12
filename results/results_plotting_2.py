#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import markers, lines, colors as mcolors
import matplotlib as mpl

results_folder = os.path.join(os.getcwd(), "jsac\\")

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

if __name__ == '__main__':
    min_n_degrees = [1, 2, 3, 4]
    max_fso_distance = [1e3, 2e3, 3e3, 4e3, 5e3]
    max_coverage_radius = [500, 1000, 1500, 2000, 2500, 3000]
    res_1 = list(np.load(results_folder + 'results_1_1.npz').values())[0]
    n_drones_fig, n_drones_axs = plt.subplots()
    for _n_deg in range(res_1.shape[0]):
        for _cov_radius in range(3, res_1.shape[2], 2):
            n_drones_axs.plot(max_fso_distance, res_1[_n_deg, :, _cov_radius], color=_colors[_n_deg],
                              ls=_linestyles[_n_deg], marker=_markers[_cov_radius // 2 - 1], lw=1,
            label='$N_{\mathrm{B}}$=' + f'{min_n_degrees[_n_deg]},  ' +
                  '$R_{\mathrm{A}}$=' + f'{int(max_coverage_radius[_cov_radius] / 1e3)} Km')

    # for _fso_d in range(res_1.shape[1]):
    #     for _cov_radius in range(1, res_1.shape[2], 2):
    #         n_drones_axs.plot(min_n_degrees, res_1[:, _fso_d, _cov_radius], color=_colors[_fso_d],
    #                           ls=_linestyles[_fso_d], marker=_markers[_cov_radius//2 ], lw=1)
    #                           # label='$N_{\mathrm{B}}$=' + f'{min_n_degrees[_n_deg]}' +
    #                           #       '$R_{\mathrm{A}}$=' + f'{int(max_coverage_radius[_cov_radius] / 1e3)} Km')

    # Legend
    total_handles = []
    total_labels = []
    handles, labels = n_drones_axs.get_legend_handles_labels()
    n_drones_fig.legend(handles, labels, loc=(0.4, 0.67), ncol=2)
    # n_drones_axs.set_yscale('log')

    # color_legend = [lines.Line2D([0], [0], color=_colors[_n_deg], ls=_linestyles[_n_deg],
    #                              label='$N_{\mathrm{B}}$=' + f'{min_n_degrees[_n_deg]}') for _n_deg in
    #                 range(len(min_n_degrees))]
    # ls_legend = [lines.Line2D([0], [0], marker=_markers[_cov_radius // 2 - 1],
    #                           label='$R_{\mathrm{A}}$=' + f'{int(max_coverage_radius[_cov_radius] / 1e3)} Km') for
    #              _cov_radius in range(3, res_1.shape[2], 2)]
    # leg1 = plt.legend(handles=color_legend, loc='upper right')
    # leg2 = plt.legend(handles=ls_legend, loc='upper right')
    # n_drones_fig.legend(handles=color_legend + ls_legend, loc=(0.58, 0.67), ncol=2)

    # Axes, ticks, labels
    n_drones_axs.set_xlabel(f'Maximum backhaul distance $d_\mathrm{{max}}$ [Km]', fontsize=10)
    n_drones_axs.set_xticks(max_fso_distance, labels=[int(max_fso_distance[i]) for i in range(len(max_fso_distance))])
    n_drones_axs.set_ylabel('Required number of DBSs $M$', fontsize=10)
    n_drones_axs.grid(True)
    n_drones_fig.show()
    n_drones_fig.savefig(results_folder + f'n_drones.eps', format='eps')
