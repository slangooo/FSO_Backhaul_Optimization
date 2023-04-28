#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.
from src.main_controller import SimulationController
from src.parameters import NUM_MBS
from src.mhp.MHP import DroneNet
from src.mhp.MHP import GenAlg
from src.parameters import CALCULATE_EXACT_FSO_NET_SOLUTION, FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT, TX_POWER_FSO_DRONE, \
    NUM_UAVS, MEAN_UES_PER_CLUSTER, MAX_FSO_DISTANCE, REQUIRED_UE_RATE
import numpy as np
import random
from multiprocessing import Pool
import os
from time import time
from itertools import repeat
import pickle

n_drones_total = [8, 9, 10, 11, 12, 13, 14]
ue_rate = [5e6, 7e6, 9e6]  # , 11e6]
max_fso_distance = [3000, 4000, 5000]
fso_transmit_power = [0.2]
results_folder = os.path.join(os.getcwd(), "results\\")
NUM_ITER = 10000


class Simulator(SimulationController):
    def __init__(self):
        super().__init__()

    def set_drones_number(self, n_drones=NUM_UAVS):
        current_n_drones = len(self.base_stations) - NUM_MBS
        self.base_stations = self.base_stations[:n_drones + NUM_MBS]
        if current_n_drones < n_drones:
            self.add_drone_stations(n_drones - current_n_drones)

    def perform_simulation_run(self, n_drones=NUM_UAVS, ue_rate=REQUIRED_UE_RATE,
                               max_fso_distance=MAX_FSO_DISTANCE, fso_transmit_power=TX_POWER_FSO_DRONE, em_n_iters=0):
        # self.set_drones_number(n_drones)
        # self.reset_users_model()
        # em_n_iters = self.perform_sinr_en()
        self.ue_required_rate = ue_rate
        self.get_fso_capacities(max_fso_distance, fso_transmit_power)
        mbs_list = self.base_stations[:NUM_MBS]
        dbs_list = self.base_stations[NUM_MBS:]
        dn = DroneNet.createArea(mbs_list, dbs_list, self.get_required_capacity_per_dbs(),
                                 self.fso_links_capacs)

        exactSolution = dn.lookForChainSolution(first=False, mode='bestNodes')
        if exactSolution.found:
            ga = GenAlg(timeLimit=FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT)
            gaSolution = ga.run(dn)
            n_solutions = len(exactSolution.results)
            score_exact = random.choice(exactSolution.bestResults)[1]
            time_first = exactSolution.firstTime
            time_full = exactSolution.fullTime
            score_ga = gaSolution.score
            time_ga = float(gaSolution.time)
            ga_percentage = 100 * gaSolution.score / exactSolution.bestScore

        else:
            n_solutions = 0
            score_ga = 0
            score_exact = 0
            time_ga = 0
            time_first = 0
            time_full = 0
            ga_percentage = 0

        return n_solutions, score_ga, score_exact, time_ga, time_first, ga_percentage, time_full, em_n_iters
        # results


def perform_simulation_run_main(n_drones, ue_rate=ue_rate, max_fso_distance=max_fso_distance,
                                fso_transmit_power=fso_transmit_power):
    sim = Simulator()
    pool = Pool(5)
    sim.set_drones_number(n_drones)
    sim.reset_users_model()
    em_n_iters = sim.perform_sinr_en()
    res = np.zeros((8, len(ue_rate), len(max_fso_distance), len(fso_transmit_power)))
    # for idx_1, _ue_rate in enumerate(ue_rate):
    # sim.ue_required_rate = _ue_rate
    for idx_2, _fso_transmit_power in enumerate(fso_transmit_power):
        for idx_3, _max_fso_distance in enumerate(max_fso_distance):
        #     res[:, idx_1, idx_3, idx_2] = sim.perform_simulation_run(n_drones, _ue_rate, _max_fso_distance,
        #                                                              _fso_transmit_power, em_n_iters)
            res[:, :, idx_3, idx_2] = np.array(pool.starmap(sim.perform_simulation_run,
                                                            zip(repeat(n_drones), ue_rate, repeat(_max_fso_distance),
                                                                repeat(_fso_transmit_power),
                                                                repeat(em_n_iters)))).transpose()
    pool.close()
    return res


if __name__ == '__main__':
    # sim = Simulator()
    # sim.set_drones_number(n_drones_total[0])
    # sim.reset_users_model()
    # em_n_iters = sim.perform_sinr_en()
    # pool = Pool(5)
    # sim.ue_required_rate = ue_rate[0]
    # sim.perform_simulation_run(n_drones_total[0], ue_rate[0], max_fso_distance,
    #                            fso_transmit_power[0], 20)
    # res1 = pool.starmap(sim.perform_simulation_run, zip(repeat(n_drones_total[0]), repeat(ue_rate[0]), max_fso_distance,
    #                                                     repeat(fso_transmit_power[0]), repeat(20)))

    run_idx = 5
    run_params = [n_drones_total, ue_rate, max_fso_distance, fso_transmit_power]
    with open(results_folder + f"params_run{run_idx}.pkl", 'wb') as f:
        pickle.dump(run_params, f)
    res_iter = np.zeros((8, len(n_drones_total), len(ue_rate), len(max_fso_distance), len(fso_transmit_power)))
    for iter in range(1, NUM_ITER+1):
        print(f"iteration {iter}")
        for n_drone_idx, _n_drones in enumerate(n_drones_total):
            print("N Drones =", _n_drones)
            res_iter[:, n_drone_idx, :, :, :] = perform_simulation_run_main(n_drones=_n_drones)
        if np.isnan(res_iter).any():
            print("FOUND NAN!")
            break
        if iter == 1:
            res = res_iter.copy()
        else:
            res = res + (res_iter - res) / iter
        np.save(results_folder + f'results_of_run{run_idx}', res)
    np.save(results_folder + f'results_of_run{run_idx}', res)

    # res = perform_simulation_run_main(7, ue_rate, max_fso_distance, fso_transmit_power)
    # res = sim.perform_simulation_run()

    # #Initialize simulation controller with UE distributed in clusters,
    # # and UAVs distributed randomly, and Macro base stations located according
    # # to settings in parameters.py (MBS_LOCATIONS)
    # sim_ctrl = SimulationController()
    #
    # #Generate plot and show UEs, DBSs, and MBSs
    # fig, _ = sim_ctrl.generate_plot()
    # fig.show()
    #
    # #Locate DBSs according to EM algorithm which optimizes channel quality
    # sim_ctrl.perform_sinr_en()
    #
    # #If we plot again we can see new DBSs locations
    # fig, _ = sim_ctrl.generate_plot()
    # fig.show()
    #
    # #Calculate FSO link capacities for current locations
    # sim_ctrl.get_fso_capacities()
    #
    # # Plot FSO capacities graph with capacity requirement of each cluster
    # sim_ctrl.generate_plot_capacs()
    #
    # # #Call this to generate new UEs distribution
    # # sim_ctrl.reset_users_model()
    #
    # #The controller has the MBSs and DBSs in self.base_stations
    # #The MBSs are in the slice [:NUM_MBS] and DBSs are in [NUM_MBS:]
    # mbs_list = sim_ctrl.base_stations[:NUM_MBS]
    # dbs_list = sim_ctrl.base_stations[NUM_MBS:]
    #
    # #Each DBS has coords attribute (which can also behave like numpy array)
    # print(dbs_list[0].coords.x, dbs_list[0].coords.y, dbs_list[0].coords.z)
    # np.sum(dbs_list[0].coords)
    #
    # #Alternatively we can get locations of all DBSs as arrays
    # print(sim_ctrl.get_uavs_locs())
    # print(np.sum(sim_ctrl.get_uavs_locs(), 1)) # just to show it behaves like array
    #
    # #We can also get the FSO capacities as 2D array where the indexes are indexes of DBSs.
    # # I.e., element i,j s.t. i=j is redundant. Otherwise the array show the capacity between the ith DBS and jth DBS
    # print(sim_ctrl.fso_links_capacs)
    #
    # #To get required capacity from UEs load per DBS
    # print(sim_ctrl.get_required_capacity_per_dbs())
    #
    # # Construct drone net
    # dn = DroneNet.createArea(mbs_list, dbs_list, sim_ctrl.get_required_capacity_per_dbs(), sim_ctrl.fso_links_capacs)
    # dn.print()
    #
    # #Exact solution, long time
    # if CALCULATE_EXACT_FSO_NET_SOLUTION:
    #     exactSolution = dn.lookForChainSolution(first=False, mode='bestNodes')
    #     exactSolution.print()
    #     if exactSolution.found:
    #         ga = GenAlg(timeLimit=FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT)
    #         gaSolution = ga.run(dn)
    #         gaSolution.print(exactSolution)
    # # else:
    # #     ga = GenAlg(timeLimit=FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT)
    # #     gaSolution = ga.run(dn)
    # #     #gaSolution.print(draw = True, drawFileName = 'FSO_net_ga_solution')
    # #     gaSolution.print()
