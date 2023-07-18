#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import markers, lines, colors as mcolors
import matplotlib as mpl


results_folder = os.getcwd() + "\\"

run_idx = 2

_markers = list(markers.MarkerStyle.markers.keys())
if None in _markers: _markers.remove(None)
if 'None' in _markers: _markers.remove('None')
if '' in _markers: _markers.remove('')
if ' ' in _markers: _markers.remove(' ')
_linestyles = list(lines.lineStyles.keys())
if 'None' in _linestyles: _linestyles.remove('None')
if '' in _linestyles: _linestyles.remove('')
if ' ' in _linestyles: _linestyles.remove(' ')
_linestyles.append((0, (3, 5, 1, 5, 1, 5)))
_colors = list(mcolors.BASE_COLORS.keys())
mpl.rc('font', family='Times New Roman')

if __name__ == '__main__':
    # Results: n_solutions, score_ga, score_exact, time_ga,
    # time_first, ga_percentage, time_full, em_n_iters
    res = np.load(results_folder + f'results_of_run{run_idx}.npy')

    # Run params: N_drones, ue_rate, max_fso_distance, fso_transmit_power
    with open(results_folder + f"params_run{run_idx}.pkl", 'rb') as f:
        run_params = pickle.load(f)

    # Number of solutions, ga_percentage, time_ga and time_first per number of drones and different ue_rate
    fso_transmit_power_idx = 0
    max_ga_percentage, max_n_solutions = 0, 0
    n_drones_fig, n_drones_axs = plt.subplots(3, len(run_params[1]), sharex=True, sharey='row')
    if len(run_params[1]) <2:
        for idx, _ax in enumerate(n_drones_axs):
            n_drones_axs[idx] = [_ax]

    for ue_rate_idx, ue_rate in enumerate(run_params[1]):
        for max_fso_distance_idx, max_fso_distance in enumerate(run_params[2]):
            # N solutions
            n_drones_axs[0][ue_rate_idx].plot(run_params[0], res[0, :, ue_rate_idx, max_fso_distance_idx,
                                                             fso_transmit_power_idx],
                                              color=_colors[max_fso_distance_idx],
                                              ls=_linestyles[max_fso_distance_idx],
                                              label='$d_{max}$=' + f'{int(max_fso_distance / 1000)}k')
            # Ga percentage
            n_drones_axs[1][ue_rate_idx].plot(run_params[0], res[5, :, ue_rate_idx, max_fso_distance_idx,
                                                             fso_transmit_power_idx],
                                              color=_colors[max_fso_distance_idx],
                                              ls=_linestyles[max_fso_distance_idx])
            # time ga/ time full
            ga_time_percentage = np.array(res[3, :, ue_rate_idx, max_fso_distance_idx, fso_transmit_power_idx]) / np.array(
                res[6, :, ue_rate_idx, max_fso_distance_idx,
                fso_transmit_power_idx])
            ga_time_percentage = np.nan_to_num(ga_time_percentage)
            max_ga_percentage = max(max_ga_percentage, ga_time_percentage.max())
            max_n_solutions = max(max_n_solutions, max(res[0, :, ue_rate_idx, max_fso_distance_idx,fso_transmit_power_idx]))
            n_drones_axs[2][ue_rate_idx].plot(run_params[0], ga_time_percentage, label='Time first',
                                              color=_colors[max_fso_distance_idx],
                                              ls=_linestyles[max_fso_distance_idx])
            # n_drones_axs[2][ue_rate_idx].plot(run_params[0], res[3, :, ue_rate_idx, max_fso_distance_idx,
            #                                   fso_transmit_power_idx], label='Time GA', color=_colors[max_fso_distance_idx],
            #                                   ls=_linestyles[max_fso_distance_idx], marker=_markers[0])
            n_drones_axs[2][ue_rate_idx].set_xlabel(f'N drones \n UE rate = {int(ue_rate / 1e6)} Mbps', fontsize=10)
            n_drones_axs[2][ue_rate_idx].set_xticks(range(run_params[0][0], run_params[0][-1] + 1),
                                                    labels=run_params[0])
    n_drones_fig.tight_layout()
    total_handles = []
    total_labels = []
    handles, labels = n_drones_axs[0][0].get_legend_handles_labels()
    n_drones_fig.legend(handles, labels, loc='upper center', ncol=len(run_params[2]))
    n_drones_fig.subplots_adjust(left=0.12, top=0.9, right=0.95, bottom=0.15, wspace=0.1, hspace=0.1)
    n_drones_axs[0][0].set_ylabel('N solutions', fontsize=10)
    n_drones_axs[0][0].set_yscale('log')
    # n_drones_axs[2][0].set_yscale('log')
    n_drones_axs[0][0].set_ylim([0, max_n_solutions + 1])
    n_drones_axs[2][0].set_ylim([0, 40])
    n_drones_axs[1][0].set_ylabel('GA percentage', fontsize=10)
    n_drones_axs[2][0].set_ylabel('GA time/ exact full time', fontsize=10)
    n_drones_axs[2][0].set_xlim([run_params[0][0], run_params[0][-1]])
    if len(run_params[1]) >= 2:
        n_drones_axs[2][1].set_xlim([run_params[0][0], run_params[0][-1]])
        n_drones_axs[2][2].set_xlim([run_params[0][0], run_params[0][-1]])

    n_drones_axs[1][0].set_ylim([0, 110])
    # n_drones_axs[2][0].set_ylim([0, max_ga_percentage])
    # n_drones_axs[2][0].set_yticks(np.arange(0, max_ga_percentage + max_ga_percentage/5, max_ga_percentage/5, dtype=int))
    n_drones_fig.align_ylabels()
    n_drones_fig.savefig(results_folder + f'n_drones_run_idx{run_idx}.eps', format='eps')
    n_drones_fig.savefig(results_folder + f'n_drones_run_idx{run_idx}.png', format='png')
    n_drones_fig.show()

    em_iter_fig, em_iter_axs = plt.subplots(1, 1)
    em_iter_axs.plot(run_params[0], res[-1, :, 0, 0, 0])
    em_iter_axs.set_xticks(range(run_params[0][0], run_params[0][-1] + 1),
                                                        labels=run_params[0])
    em_iter_axs.set_xlabel(f'N drones', fontsize=10)
    em_iter_axs.set_ylabel('Number of iterations', fontsize=10)
    em_iter_fig.savefig(results_folder + f'em_iter_run_idx{run_idx}.eps', format='eps')
    em_iter_fig.savefig(results_folder + f'em_iter_run_idx{run_idx}.png', format='png')
    em_iter_fig.show()