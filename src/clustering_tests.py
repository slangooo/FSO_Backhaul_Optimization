#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from src.main import Simulator
from src.parameters import MAX_FSO_DISTANCE
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plot_flag = False
    sim_ctrl = Simulator()
    sim_ctrl.reset_users_model()
    max_distance = 4000
    sim_ctrl.get_fso_capacities(max_distance)
    sim_ctrl.fso_links_capacs
    capacs_figs = []
    top_figs = []
    min_n_drones, max_n_drones = 10, 20
    n_iterations = 10
    avg_degree_per_n = np.zeros(max_n_drones - min_n_drones)

    for iter in range(1, n_iterations+1):
        avg_degree = 0
        avg_degree_per_n_iter = np.zeros(max_n_drones - min_n_drones)
        for n_drones in range(min_n_drones, max_n_drones):
            sim_ctrl.set_drones_number(n_drones)
            sim_ctrl.perform_sinr_en()
            sim_ctrl.get_fso_capacities(max_distance)
            if plot_flag:
                capacs_figs.append(sim_ctrl.generate_plot_capacs())
                fig, _ = sim_ctrl.generate_plot()
                top_figs.append(fig)
            new_avg = np.sum(sim_ctrl.fso_links_capacs != 0, 1).mean()
            print("Average degrees per node:", np.sum(sim_ctrl.fso_links_capacs != 0, 1).mean())
            print("Degrees:", np.sum(sim_ctrl.fso_links_capacs != 0, 1))
            if new_avg < avg_degree:
                print("Lower Degree!")
                # break
            avg_degree = new_avg
            avg_degree_per_n_iter[n_drones - min_n_drones] = new_avg
        avg_degree_per_n = avg_degree_per_n + (avg_degree_per_n_iter - avg_degree_per_n)/iter
            


