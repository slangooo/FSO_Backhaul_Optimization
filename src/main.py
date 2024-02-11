#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.
from src.main_controller import SimulationController
from src.run_info import Info
from src.parameters import NUM_MBS
from src.mhp.DroneNet import DroneNet
from src.mhp.GenAlg import GenAlg
from src.mhp.GenAlg import FitnessMode
from src.mhp.ExactSolution import ExactSolution
from src.mhp.GASolution import GASolution
from src.parameters import \
    CALCULATE_EXACT_FSO_NET_SOLUTION, CALCULATE_EXACT_FSO_NET_SOLUTION_FIRST_ONLY, \
    CALCULATE_EXACT_FSO_NET_SOLUTION_TIME_LIMIT, CALCULATE_EXACT_FSO_NET_SOLUTION_INSTANCE_LIMIT, \
    FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT, FSO_NET_GENECTIC_ALGORITHM_GENERATION_LIMIT, FSO_NET_GENECTIC_ALGORITHM_GENERATE_FIRST_ONLY,\
    TX_POWER_FSO_DRONE, NUM_UAVS, MEAN_UES_PER_CLUSTER, MAX_FSO_DISTANCE, REQUIRED_UE_RATE, \
    CLUSTERING_METHOD, MIN_N_DEGREES, RANDOMIZE_MBS_LOCS, \
    RUN_GENECTIC_ALGORITHM, GENECTIC_ALGORITHM_FITNESS_MODE, GENECTIC_ALGORITHM_POPULATION_SIZE, \
    SAVE_MHP_DATA, SAVE_FIG, SAVE_INSTANCE, X_BOUNDARY, Y_BOUNDARY, RADIUS_BY_DESIRED_VERTEX_DEGREE
import numpy as np
import math
from multiprocessing import Pool
import os
from itertools import repeat
import pickle

USE_MULTIPROCESSING = False


#n_drones_total = range(20, 51, 5)
n_drones_total = range(20, 40)
#n_drones_total = [10, 20]
#n_drones_total = [15]
#ue_rate = [5e6, 7e6]
ue_rate = [1e7]
#ue_rate = range(25000000, 35000001, 1000000)
#max_fso_distance = [3000, 4000]
max_fso_distance = [3000]
#max_fso_distance = range(3000, 4001, 20)
#max_fso_distance = range(1500, 2501, 250)
fso_transmit_power = [0.2]
#fso_transmit_power = [0.1, 0.2, 0.3]

results_folder = os.path.join(os.getcwd(), "results\\")
NUM_ITER = 3
    
class Simulator(SimulationController):
    def __init__(self):
        super().__init__()

    def randomize_mbs_locs(self):
        for _bs in self.base_stations[:NUM_MBS]:
            _bs.coords.update_coords_from_array([np.random.uniform(low=X_BOUNDARY[0], high=X_BOUNDARY[1]), np.random.uniform(low=Y_BOUNDARY[0], high=Y_BOUNDARY[1])])

    def set_drones_number(self, n_drones=NUM_UAVS):
        current_n_drones = len(self.base_stations) - NUM_MBS
        self.base_stations = self.base_stations[:n_drones + NUM_MBS]
        if current_n_drones < n_drones:
            self.add_drone_stations(n_drones - current_n_drones)
        self.set_ues_base_stations()
        self.update_users_rfs()


    def perform_simulation_run(self, test_iteration=1, n_drones=NUM_UAVS, ue_rate=REQUIRED_UE_RATE,
                               max_fso_distance=MAX_FSO_DISTANCE, fso_transmit_power=TX_POWER_FSO_DRONE, em_n_iters=0):
        # self.set_drones_number(n_drones)
        # self.reset_users_model()
        if CLUSTERING_METHOD == 2:
            min_n_clusters = self.localize_drones(max_fso_distance=max_fso_distance, min_n_degrees=MIN_N_DEGREES, n_dbs=n_drones)
        
        # print('perform_simulation_run')

        self.ue_required_rate = ue_rate
        self.get_fso_capacities(max_fso_distance, fso_transmit_power)
        mbs_list = self.base_stations[:NUM_MBS]
        dbs_list = self.base_stations[NUM_MBS:]
        
        key = '_i' + str(test_iteration) + '_c' + str(CLUSTERING_METHOD, ) + '_d' + str(n_drones) + '_r' + str(max_fso_distance).zfill(4) + '_eu' + str(ue_rate) + '_tp0' + str(10*fso_transmit_power)
        
        if SAVE_FIG:
            fig = self.generate_plot_capacs(True)
            fig.savefig('out/img/area' + key + '.png', bbox_inches='tight')
        
        dn = DroneNet.createArea(mbs_list, dbs_list, self.get_required_capacity_per_dbs(), self.fso_links_capacs)

        if SAVE_INSTANCE:
            dn.save('out/data/input' + key + '.json')

        info = Info(test_iteration, CLUSTERING_METHOD, n_drones, ue_rate, max_fso_distance, fso_transmit_power, em_n_iters, dn, mbs_list, dbs_list, GENECTIC_ALGORITHM_FITNESS_MODE);
        
        if CALCULATE_EXACT_FSO_NET_SOLUTION:
            exactSolution = dn.lookForChainSolution(
                time_limit_s   = CALCULATE_EXACT_FSO_NET_SOLUTION_TIME_LIMIT,
                instance_limit = CALCULATE_EXACT_FSO_NET_SOLUTION_INSTANCE_LIMIT,
                stop_on_first  = CALCULATE_EXACT_FSO_NET_SOLUTION_FIRST_ONLY)
        else:
            exactSolution = ExactSolution(
                stopType             = 'none',
                found                = False,
                firstTime            = 0,
                fullTime             = 0,
                firstCorrectInstance = 0,
                processedInstances   = 0,
                results              = [],
                mode                 = 'NONE')

        info.appendExactSolution(exactSolution)
            
        if RUN_GENECTIC_ALGORITHM :
            fitnessMode = FitnessMode(GENECTIC_ALGORITHM_FITNESS_MODE, dn.drone_number * dn.max_bases_bandwidth())
            ga = GenAlg(
                time_limit_s              = FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT,
                num_generations           = FSO_NET_GENECTIC_ALGORITHM_GENERATION_LIMIT,
                stop_on_first             = FSO_NET_GENECTIC_ALGORITHM_GENERATE_FIRST_ONLY,
                fitness_fitness_mode      = fitnessMode.fitness_fitness_mode,
                fitness_penalty_mode      = fitnessMode.fitness_penalty_mode,
                fitness_penalty_value     = fitnessMode.fitness_penalty_value,
                use_solution_func         = fitnessMode.use_solution_func,
                solution_fitness_mode     = fitnessMode.solution_fitness_mode,
                solution_penalty_mode     = fitnessMode.solution_penalty_mode,
                solution_penalty_value    = fitnessMode.solution_penalty_value,
                parent_selection_type     = 'tournament',
                sol_per_pop               = GENECTIC_ALGORITHM_POPULATION_SIZE,
                num_parents_mating        = int(0.3 * GENECTIC_ALGORITHM_POPULATION_SIZE),
                keep_elitism              = int(0.1 * GENECTIC_ALGORITHM_POPULATION_SIZE),
                mutation_probability_norm = 0.2,
                saturate_stop             = 30,
                fitness_draw              = 'out/img/fitness' + key if SAVE_FIG else None
            )

            # print('run genetic algoritm')
            gaSolution = ga.run(dn)
        else:
            gaSolution = GASolution(
                ga = None,
                generations_runs = 0,
                total_runs = 0,
                positive_runs = 0,
                found = False,
                time = 0,
                score = -1,
                result = None,
                first_time = 0,
                first_score = 0,
                first_result = 0,
                runs = 0)

        info.appendGASolution(gaSolution)

        if SAVE_MHP_DATA:
            print(info.msg())
            info.log()
            info.csv()

        if SAVE_FIG:
            if exactSolution.found:
                dn._processChainSolution(exactSolution.bestResults[0][0])
                dn.draw(True, 'out/img/network' + key + '_e.png')
            if gaSolution.found:
                dn._processChainSolution(gaSolution.result)
                dn.draw(True, 'out/img/network' + key + '_g.png')
            if not exactSolution.found and not gaSolution.found:
                dn.draw(False, 'out/img/network' + key + '_n.png')

        return info.exact_solutions(), info.ga_score(), info.exact_score(), info.ga_time(), info.exact_time_first(), info.score_percentage(), info.exact_time_full(), em_n_iters
        # results


def calculate_radius(k, desired_vertex_degree=RADIUS_BY_DESIRED_VERTEX_DEGREE):
    return round(math.sqrt(((desired_vertex_degree + 1) * X_BOUNDARY[1] * Y_BOUNDARY[1]) / (math.pi * k)))

def perform_simulation_run_main(test_iteration, n_drones, ue_rate=ue_rate, max_fso_distance=max_fso_distance, fso_transmit_power=fso_transmit_power):
    sim = Simulator()
    pool = Pool(5, maxtasksperchild=1)
    if RANDOMIZE_MBS_LOCS:
        sim.randomize_mbs_locs()
    sim.set_drones_number(n_drones)
    sim.reset_users_model()
    if CLUSTERING_METHOD < 2:
        em_n_iters = sim.localize_drones(CLUSTERING_METHOD) #Here we can pass 1 to force using Kmeans
    else:
        em_n_iters = 0
    print("EM iters:", em_n_iters)
    res = np.zeros((8, len(ue_rate), len(max_fso_distance), len(fso_transmit_power)))
    if USE_MULTIPROCESSING:
        # for idx_1, _ue_rate in enumerate(ue_rate):
        # sim.ue_required_rate = _ue_rate
        for idx_2, _fso_transmit_power in enumerate(fso_transmit_power):
            for idx_3, _max_fso_distance in enumerate(max_fso_distance):
                if RADIUS_BY_DESIRED_VERTEX_DEGREE:
                    _max_fso_distance = calculate_radius(n_drones)
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
                    res[:, idx_1, idx_3, idx_2] = sim.perform_simulation_run(test_iteration, n_drones, _ue_rate, _max_fso_distance, _fso_transmit_power, em_n_iters)
    return res


if __name__ == '__main__':
################################################ SIMULATION

    run_idx = 3
    continue_sim = False
    
    Info.csv_header()

    def update_run_params(iter_idx=1):
        run_params = [n_drones_total, ue_rate, max_fso_distance, fso_transmit_power, iter_idx]
        with open(results_folder + f"params_run{run_idx}.pkl", 'wb') as f:
            pickle.dump(run_params, f)

    if continue_sim:
        res = np.load(results_folder + f'results_of_run{run_idx}.npy')
        with open(results_folder + f"params_run{run_idx}.pkl", 'rb') as f:
            run_params = pickle.load(f)
        start_iter = run_params[-1] + 1
        assert (start_iter > 1)
    else:
        start_iter = 1

    res_iter = np.zeros((8, len(n_drones_total), len(ue_rate), len(max_fso_distance), len(fso_transmit_power)))
    for test_iteration in range(start_iter, NUM_ITER + 1):
        print(f"iteration {test_iteration}")
        for n_drone_idx, _n_drones in enumerate(n_drones_total):
            print("N Drones =", _n_drones)
            res_iter[:, n_drone_idx, :, :, :] = perform_simulation_run_main(test_iteration=test_iteration, n_drones=_n_drones)
        if np.isnan(res_iter).any():
            print("FOUND NAN!")
            break
        if test_iteration == 1:
            res = res_iter.copy()
        else:
            res = res + (res_iter - res) / test_iteration
        np.save(results_folder + f'results_of_run{run_idx}', res)
        update_run_params(test_iteration + 1)
    np.save(results_folder + f'results_of_run{run_idx}', res)
