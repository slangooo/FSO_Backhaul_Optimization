#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.
import itertools

from src.main_controller import SimulationController
from src.parameters import NUM_MBS
from src.mhp.DroneNet import DroneNet
from src.mhp.GenAlg import GenAlg
from src.run_info import Info
from src.mhp.GenAlg import FitnessMode


from src.parameters import CALCULATE_EXACT_FSO_NET_SOLUTION, CALCULATE_EXACT_FSO_NET_SOLUTION_FIRST_ONLY, \
    FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT, \
    TX_POWER_FSO_DRONE, NUM_UAVS, MEAN_UES_PER_CLUSTER, MAX_FSO_DISTANCE, REQUIRED_UE_RATE, SAVE_MHP_DATA, \
    CLUSTERING_METHOD, MIN_N_DEGREES, \
    X_BOUNDARY, Y_BOUNDARY, RANDOMIZE_MBS_LOCS, COVERAGE_RADIUS, FSO_NET_GENECTIC_ALGORITHM_GENERATION_LIMIT,\
FSO_NET_GENECTIC_ALGORITHM_GENERATE_FIRST_ONLY,GENECTIC_ALGORITHM_POPULATION_SIZE, CALCULATE_EXACT_FSO_NET_SOLUTION_TIME_LIMIT,\
CALCULATE_EXACT_FSO_NET_SOLUTION_INSTANCE_LIMIT
import numpy as np
import random
from multiprocessing import Pool
import os
from time import time, perf_counter_ns
from itertools import repeat
import pickle
import string
from datetime import datetime
from tqdm import tqdm

USE_MULTIPROCESSING = True
results_folder = os.path.join(os.getcwd(), "results\\new\\")

class Simulator(SimulationController):
    def __init__(self):
        super().__init__()

    def randomize_mbs_locs(self):
        for _bs in self.base_stations[:NUM_MBS]:
            _bs.coords.update_coords_from_array([np.random.uniform(low=X_BOUNDARY[0], high=X_BOUNDARY[1]),
                                                 np.random.uniform(low=Y_BOUNDARY[0], high=Y_BOUNDARY[1])])

    def perform_simulation_run(self, test_iteration=1, n_drones=NUM_UAVS, ue_rate=REQUIRED_UE_RATE,
                               max_fso_distance=MAX_FSO_DISTANCE, fso_transmit_power=TX_POWER_FSO_DRONE, em_n_iters=0):
        # self.set_drones_number(n_drones)
        # self.reset_users_model()
        min_n_clusters = 0
        if CLUSTERING_METHOD == 2:
            min_n_clusters = self.localize_drones(max_fso_distance=max_fso_distance, min_n_degrees=MIN_N_DEGREES, )
        self.ue_required_rate = ue_rate
        self.get_fso_capacities(max_fso_distance, fso_transmit_power)
        mbs_list = self.base_stations[:NUM_MBS]
        dbs_list = self.base_stations[NUM_MBS:]
        dn = DroneNet.createArea(mbs_list, dbs_list, self.get_required_capacity_per_dbs(),
                                 self.fso_links_capacs)

        ga = GenAlg(timeLimit=FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT)
        gaSolution = ga.run(dn)
        score_ga = gaSolution.score
        return score_ga, min_n_clusters,
        # results


def perform_simulation_run_main(test_iteration, n_drones, ue_rate=20, max_fso_distance=2000,
                                fso_transmit_power=[0.05]):
    sim = Simulator()
    pool = Pool(5, maxtasksperchild=1)
    if RANDOMIZE_MBS_LOCS:
        sim.randomize_mbs_locs()
    sim.set_drones_number(n_drones)
    sim.reset_users_model()
    if CLUSTERING_METHOD < 2:
        em_n_iters = sim.localize_drones(CLUSTERING_METHOD)  # Here we can pass 1 to force using Kmeans
    else:
        em_n_iters = 0
    print("EM iters:", em_n_iters)
    res = np.zeros((8, len(ue_rate), len(max_fso_distance), len(fso_transmit_power)))
    if USE_MULTIPROCESSING:
        # for idx_1, _ue_rate in enumerate(ue_rate):
        # sim.ue_required_rate = _ue_rate
        for idx_2, _fso_transmit_power in enumerate(fso_transmit_power):
            for idx_3, _max_fso_distance in enumerate(max_fso_distance):
                #     res[:, idx_1, idx_3, idx_2] = sim.perform_simulation_run(n_drones, _ue_rate, _max_fso_distance,
                #                                                              _fso_transmit_power, em_n_iters)
                res[:, :, idx_3, idx_2] = np.array(pool.starmap(sim.perform_simulation_run,
                                                                zip(repeat(test_iteration),
                                                                    repeat(n_drones),
                                                                    ue_rate,
                                                                    repeat(_max_fso_distance),
                                                                    repeat(_fso_transmit_power),
                                                                    repeat(em_n_iters)))).transpose()
        pool.close()
        pool.join()
        pool.terminate()
    else:
        for idx_1, _ue_rate in enumerate(ue_rate):
            for idx_2, _fso_transmit_power in enumerate(fso_transmit_power):
                for idx_3, _max_fso_distance in enumerate(max_fso_distance):
                    res[:, idx_1, idx_3, idx_2] = sim.perform_simulation_run(test_iteration, n_drones, _ue_rate,
                                                                             _max_fso_distance,
                                                                             _fso_transmit_power, em_n_iters)
    return res


def run_clustering_nb(min_n_degrees, max_fso_distance, max_coverage_radius=COVERAGE_RADIUS):
    n_iterations = 10
    sim = Simulator()
    min_n_clusters = 0
    for i in range(1, n_iterations):
        sim.randomize_mbs_locs()
        sim.reset_users_model()
        min_n_clusters += (min_n_clusters + sim.localize_drones(max_fso_distance=max_fso_distance,
                                                                min_n_degrees=min_n_degrees,
                                                                max_coverage_radius=max_coverage_radius) - min_n_clusters) / i
    print("Done!!")
    return min_n_clusters


def sim_1_ndbs_fso_M():
    min_n_degrees = [1, 2, 3, 4]
    max_fso_distance = [1e3, 2e3, 3e3, 4e3, 5e3]
    max_coverage_radius = [500, 1000, 1500, 2000, 2500, 3000]
    inputs = list(itertools.product(min_n_degrees, max_fso_distance, max_coverage_radius))
    with Pool(5, maxtasksperchild=4) as pool:
        results = pool.starmap(run_clustering_nb, tqdm(inputs, total=len(inputs)))
    reshaped_results = np.array(results).reshape(len(min_n_degrees), len(max_fso_distance), len(max_coverage_radius))
    return reshaped_results


sim = Simulator()


def run_iter(max_fso_distance, min_nb_degrees):
    sim.randomize_mbs_locs()
    sim.reset_users_model()
    n_drones = sim.localize_drones(_method=2, max_fso_distance=max_fso_distance, min_n_degrees=min_nb_degrees,
                                   max_coverage_radius=200000)

    sim.set_drones_number(n_drones)

    sim.get_fso_capacities(max_fso_distance)
    mbs_list = sim.base_stations[:NUM_MBS]
    dbs_list = sim.base_stations[NUM_MBS:]
    dn = DroneNet.createArea(mbs_list, dbs_list, sim.get_required_capacity_per_dbs(),
                             sim.fso_links_capacs)
    ga = GenAlg(timeLimit=FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT)

    for i in range(3):
        try:
            gaSolution = ga.run(dn)
            if gaSolution:
                score_ga_hc = gaSolution.score
                break
        except:
            score_ga_hc = None


    sim.localize_drones(_method=1, max_fso_distance=max_fso_distance, min_n_degrees=min_nb_degrees,
                        max_coverage_radius=200000)
    sim.get_fso_capacities(max_fso_distance)
    mbs_list = sim.base_stations[:NUM_MBS]
    dbs_list = sim.base_stations[NUM_MBS:]
    dn = DroneNet.createArea(mbs_list, dbs_list, sim.get_required_capacity_per_dbs(),
                             sim.fso_links_capacs)
    ga = GenAlg(timeLimit=FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT)
    for i in range(3):
        try:
            gaSolution = ga.run(dn)
            if gaSolution:
                score_ga_km = gaSolution.score
                break
        except:
            score_ga_km = None
    return n_drones, score_ga_hc, score_ga_km


def sim_2_ga_vs_kmeans_hc():
    max_fso_distance = [1e3, 2e3, 3e3, 4e3]
    min_n_degrees = [1, 2, 3, 4]
    n_iterations = 100
    inputs = list(itertools.product(min_n_degrees, max_fso_distance))
    repeated_inputs = [(x, y) for (x, y) in inputs for _ in range(n_iterations)]
    # with Pool(5, maxtasksperchild=2) as pool:
    #     results = pool.starmap(run_iter, tqdm(repeated_inputs, total=len(repeated_inputs)))
    res = np.zeros((n_iterations, len(min_n_degrees), len(max_fso_distance), 3))

    for fso_idx, fso_dist in tqdm(enumerate(max_fso_distance)):
        for n_degree_idx, n_deg in tqdm(enumerate(min_n_degrees)):
            for _iter in tqdm(range(n_iterations)):
                res[_iter, n_degree_idx, fso_idx, :] = np.array(run_iter(fso_dist, n_deg))

    return res


if __name__ == '__main__':
    # results_1 = sim_1_ndbs_fso_M()
    # results_2 = sim_2_ga_vs_kmeans_hc()
    GENECTIC_ALGORITHM_FITNESS_MODE = 'NVP'
    sim = Simulator()
    sim.reset_users_model()
    min_n_clusters = sim.localize_drones(max_fso_distance=3000, min_n_degrees=5,
                                         max_coverage_radius=200000, n_dbs=50)
    sim.set_drones_number(min_n_clusters)
    sim.get_fso_capacities()
    mbs_list = sim.base_stations[:NUM_MBS]
    dbs_list = sim.base_stations[NUM_MBS:]
    dn = DroneNet.createArea(mbs_list, dbs_list, sim.get_required_capacity_per_dbs(), sim.fso_links_capacs)
    info = Info(1, clustering_method=2, n_drones=min_n_clusters, ue_rate=REQUIRED_UE_RATE, max_fso_distance=3000, fso_transmit_power=TX_POWER_FSO_DRONE, em_n_iters=0,
                dn=dn, mbs_list=mbs_list, dbs_list=dbs_list, fitness_mode=GENECTIC_ALGORITHM_FITNESS_MODE)
    fitnessMode = FitnessMode(mode=GENECTIC_ALGORITHM_FITNESS_MODE, penalty_value=dn.drone_number * dn.max_bases_bandwidth())
    exactSolution = dn.lookForChainSolution(
        time_limit_s=CALCULATE_EXACT_FSO_NET_SOLUTION_TIME_LIMIT,
        instance_limit=CALCULATE_EXACT_FSO_NET_SOLUTION_INSTANCE_LIMIT,
        stop_on_first=CALCULATE_EXACT_FSO_NET_SOLUTION_FIRST_ONLY)
    ga = GenAlg(
        time_limit_s=FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT,
        num_generations=FSO_NET_GENECTIC_ALGORITHM_GENERATION_LIMIT,
        stop_on_first=FSO_NET_GENECTIC_ALGORITHM_GENERATE_FIRST_ONLY,
        fitness_fitness_mode=fitnessMode.fitness_fitness_mode,
        fitness_penalty_mode=fitnessMode.fitness_penalty_mode,
        fitness_penalty_value=fitnessMode.fitness_penalty_value,
        use_solution_func=fitnessMode.use_solution_func,
        solution_fitness_mode=fitnessMode.solution_fitness_mode,
        solution_penalty_mode=fitnessMode.solution_penalty_mode,
        solution_penalty_value=fitnessMode.solution_penalty_value,
        parent_selection_type='tournament',
        sol_per_pop=GENECTIC_ALGORITHM_POPULATION_SIZE,
        num_parents_mating=int(0.3 * GENECTIC_ALGORITHM_POPULATION_SIZE),
        keep_elitism=int(0.1 * GENECTIC_ALGORITHM_POPULATION_SIZE),
        mutation_probability_norm=0.2,
        saturate_stop=30,
        fitness_draw= None
    )
    gaSolution = ga.run(dn)

    # sim = Simulator()
    # if RANDOMIZE_MBS_LOCS:
    #     sim.randomize_mbs_locs()
    # sim.reset_users_model()
    # min_n_clusters = sim.localize_drones(max_fso_distance=3000, min_n_degrees=2,
    #                                      n_dbs=NUM_UAVS, max_coverage_radius=800)
    # sim.set_drones_number(min_n_clusters)
    # sim.get_fso_capacities()
    # fig, _ = sim.generate_plot(plot_fso=False)
    # fig.show()
