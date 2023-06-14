#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from src.main import Simulator
from src.parameters import MAX_FSO_DISTANCE
import numpy as np
from scipy.cluster import vq, hierarchy


def update_drone_locs(sim_ctrl:Simulator, locs):
    for idx, bs in enumerate(sim_ctrl.bs_rf_list):
        bs.coords.update_coords_from_array(locs[idx])

if __name__ == '__main__':
    plot_flag = False
    sim_ctrl = Simulator()
    sim_ctrl.reset_users_model()
    max_distance = 4000
    ues_locs = np.vstack([_user.coords.as_2d_array() for _user in sim_ctrl.users])
    min_n_drones, max_n_drones = 10, 20

    avg_degree = 0
    #Kmeans
    for n_drones in range(min_n_drones, max_n_drones):
        sim_ctrl.set_drones_number(n_drones)
        locs, _ = vq.kmeans(ues_locs, n_drones)
        update_drone_locs(sim_ctrl, locs)
        sim_ctrl.update_users_rfs()
        sim_ctrl.get_fso_capacities(max_distance)
        new_avg = np.sum(sim_ctrl.fso_links_capacs != 0, 1).mean()
        print("Average degrees per node:", np.sum(sim_ctrl.fso_links_capacs != 0, 1).mean())
        print("Degrees:", np.sum(sim_ctrl.fso_links_capacs != 0, 1))
        if new_avg < avg_degree:
            print("Lower Degree!")
            break
        avg_degree = new_avg

    avg_degree = 0
    # Kmeans2
    for n_drones in range(min_n_drones, max_n_drones):
        sim_ctrl.set_drones_number(n_drones)
        locs, _ = vq.kmeans2(ues_locs, n_drones)
        update_drone_locs(sim_ctrl, locs)
        sim_ctrl.update_users_rfs()
        sim_ctrl.get_fso_capacities(max_distance)
        new_avg = np.sum(sim_ctrl.fso_links_capacs != 0, 1).mean()
        print("Average degrees per node:", np.sum(sim_ctrl.fso_links_capacs != 0, 1).mean())
        print("Degrees:", np.sum(sim_ctrl.fso_links_capacs != 0, 1))
        if new_avg < avg_degree:
            print("Lower Degree!")
            break
        avg_degree = new_avg
