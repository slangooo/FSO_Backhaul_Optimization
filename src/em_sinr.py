#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

import numpy as np
from src.math_tools import lin2db
from scipy.special import softmax
import matplotlib.pyplot as plt

MAX_ITER = 100
SINR_EM_THRESHOLD = 1


def perform_sinr_em(users, dbs):
    n_dbs, n_ues = len(dbs), len(users)

    ues_locs = np.vstack([users[i].coords.as_2d_array() for i in range(n_ues)])

    dbs_locs = [dbs[i].coords for i in range(n_dbs)]

    for iter in range(MAX_ITER):
        ues_sinrs = [users[i].rf_transceiver.received_sinrs[0] for i in range(n_ues)]
        ues_sinrs_db_arr = lin2db(np.vstack(ues_sinrs))
        ues_max_sinr = [users[i].rf_transceiver.received_sinr for i in range(n_ues)]
        ues_max_sinr_db_arr = lin2db(np.vstack(ues_max_sinr))

        sinrs_mask = ues_sinrs_db_arr >= (ues_max_sinr_db_arr - SINR_EM_THRESHOLD)

        # if not np.all(np.max(np.where(sinrs_mask, ues_sinrs_db_arr, -np.inf), 1) > -np.inf):
        #     bla = 5
        # np.argwhere(np.invert(np.max(np.where(sinrs_mask, ues_sinrs_db_arr, -np.inf), 1) > -np.inf))

        posterior_probs = softmax(np.where(sinrs_mask, ues_sinrs_db_arr, -np.inf), 1)
        cluster_probs = np.sum(posterior_probs, 0)
        # get new locs
        new_locs = (np.sum(np.repeat(posterior_probs, 2, 1) * np.tile(ues_locs, n_dbs), 0) / np.repeat(cluster_probs, 2)).reshape(n_dbs, 2)
        stacked_dbs_locs = np.vstack(dbs_locs)[:, :2]
        max_change = np.abs(stacked_dbs_locs - new_locs).max()
        nan_idxs = np.argwhere(np.isnan(new_locs))
        for _nan_idx in nan_idxs:
            new_locs[_nan_idx[0], _nan_idx[1]] = stacked_dbs_locs[_nan_idx[0], _nan_idx[1]]

        # #TODO:remove below tests
        # if np.isnan(new_locs).any():
        #     print("NAN location!")
        #     raise Exception("Nan in SINR-EM")
        # if len(np.unique(new_locs, axis=0)) < len(new_locs):
        #     raise Exception("Duplicate Locations in SINR-EM!")

        for dbs_idx, station in enumerate(dbs):
            station.coords.update_coords_from_array(np.append(new_locs[dbs_idx], station.coords.z))

        if max_change < 1:
            break

        [_user.rf_transceiver.get_serving_bs_info(recalculate=True) for _user in users]
    return iter
    # dbs_xs, dbs_ys = np.zeros((n_dbs,), dtype=float), np.zeros((n_dbs,), dtype=float)
    # for i in range(n_dbs):
    #     dbs_xs[i] = dbs[i].coords.x
    #     dbs_ys[i] = dbs[i].coords.y
    # plt.scatter(dbs_xs, dbs_ys, edgecolor='red', facecolor='black', alpha=1, marker="s", label="DBSs")
    # plt.show()
