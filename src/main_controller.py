#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.
import warnings

from src.em_sinr import perform_sinr_em
from src.users import User
from src.drone_station import DroneStation
from src.macro_base_station import MacroBaseStation
import numpy as np
from src.parameters import TX_POWER_FSO_MBS, TX_POWER_FSO_DRONE, NUM_UAVS, NUM_MBS, NUM_OF_USERS, \
    USER_MOBILITY_SAVE_NAME, TIME_STEP, DRONE_HEIGHT, SKIP_ENERGY_UPDATE, X_BOUNDARY, Y_BOUNDARY, MBS_LOCATIONS, \
    REQUIRED_UE_RATE, MAX_FSO_DISTANCE, MEAN_UES_PER_CLUSTER, CLUSTERING_METHOD, MIN_N_DEGREES
from src.environment.user_modeling import ThomasClusterProcess

from src.data_structures import Coords3d
from src.apparatus.fso_transceiver import FsoTransceiver
from src.types_constants import LinkType
from src.channel_model.fso_a2a import StatisticalModel
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Manager
from time import sleep
from src.math_tools import lin2db
from scipy.cluster import vq
from src.hierarchical_clustering import perform_dbs_hc, get_centroids


class SimulationController:
    plot_flag = False
    bs_rf_list = []
    users_model = None
    steps_count = 1
    users_per_bs = None
    users = []
    fso_links_capacs = None
    base_stations = None
    ue_required_rate = REQUIRED_UE_RATE

    def __init__(self, initial_uavs_coords=None, n_users=NUM_OF_USERS, mbs_locations=MBS_LOCATIONS):
        self.base_stations = [MacroBaseStation(mbs_id=mbs_id, coords=mbs_coords)
                              for mbs_id, mbs_coords in zip(range(len(mbs_locations)), mbs_locations)]
        self.create_drone_stations()
        self.reset_users_model()
        self.irradiation_manager = None
        self.init_users_rfs()
        self.stations_list = []

    def reset_users_model(self, mean_ues_per_cluster=MEAN_UES_PER_CLUSTER):
        self.users_model = ThomasClusterProcess(mean_ues_per_cluster)
        self.users = [User(self.users_model.users[i])
                      for i in range(self.users_model.n_users)]
        if self.base_stations:
            self.init_users_rfs()

    def get_fso_capacities(self, max_fso_distance=MAX_FSO_DISTANCE, fso_transmit_power=TX_POWER_FSO_DRONE):
        dbs_locs = [dbs.coords for dbs in self.base_stations]
        n_dbs = len(dbs_locs)
        poss_links_capacs = np.zeros((n_dbs, n_dbs))

        for idx_i, dbs_i in enumerate(self.base_stations):
            for idx_j, dbs_j in enumerate(self.base_stations):
                if idx_i == idx_j:
                    continue

                if dbs_i.coords.get_distance_to(dbs_j.coords) > max_fso_distance:
                    continue
                _, capacity = StatisticalModel.get_charge_power_and_capacity(dbs_i.coords, dbs_j.coords,
                                                                             transmit_power=fso_transmit_power,
                                                                             power_split_ratio=1)
                capacity = int(capacity / 1e6)  # In MB
                poss_links_capacs[idx_i, idx_j] = capacity if capacity > 0 else 0
        self.fso_links_capacs = poss_links_capacs
        return poss_links_capacs

    def perform_sinr_em(self):
        iter = perform_sinr_em(self.users, self.base_stations[NUM_MBS:])
        self.update_users_rfs()
        return iter

    def perform_kmeans(self):
        ues_locs = self.get_ues_locs()
        while 1:
            try:
                locs, _ = vq.kmeans2(ues_locs, len(self.bs_rf_list), minit='++', missing='raise')
                break
            except:
                print("Empty Cluster, repeat!")
        for idx, bs in enumerate(self.bs_rf_list):
            bs.coords.update_coords_from_array(locs[idx])
        self.update_users_rfs()
        return 0

    def get_ues_locs(self):
        return np.vstack([_user.coords.as_2d_array() for _user in self.users])

    def perform_hierarchical_clustering(self, max_fso_distance, min_n_degrees, n_clusters):
        linkage_matrix, n_clusters_possible = perform_dbs_hc(
            self.users, [_bs.coords.as_2d_array() for _bs in self.base_stations[:NUM_MBS]], max_fso_distance,
            min_n_degrees)
        if n_clusters < n_clusters_possible:
            warnings.warn(f"Number of clusters {n_clusters} is less than possible {n_clusters_possible}!")
        locs = get_centroids(n_clusters, linkage_matrix, self.get_ues_locs())
        for idx, bs in enumerate(self.bs_rf_list):
            bs.coords.update_coords_from_array(locs[idx])
        return n_clusters_possible

    def localize_drones(self, _method=CLUSTERING_METHOD, max_fso_distance=MAX_FSO_DISTANCE, min_n_degrees=MIN_N_DEGREES,
                        n_dbs=NUM_UAVS):
        if _method == 0:
            return self.perform_sinr_em()
        elif _method == 1:
            return self.perform_kmeans()
        else:
            return self.perform_hierarchical_clustering(max_fso_distance, min_n_degrees, n_dbs)

    def get_required_capacity_per_dbs(self):
        return np.array([_bs.n_associated_users * self.ue_required_rate for _bs in self.bs_rf_list])

    def update_fso_stats(self):
        if SKIP_ENERGY_UPDATE:
            return
        [_drone.update_fso_status() for _drone in self.base_stations[NUM_MBS:]]

    def get_users_stats(self):
        return np.array([_user.rf_transceiver.get_stats() for _user in self.users])

    def update_drones_energy(self):
        if SKIP_ENERGY_UPDATE:
            return
        [_drone.update_energy(self.mobility_model.time_step) for _drone in self.base_stations[NUM_MBS:]]

    def get_users_rf_means(self):
        return np.array([[lin2db(_user.rf_transceiver.mean_sinr) for _user in self.users],
                         [_user.rf_transceiver.mean_capacity for _user in self.users]])

    def get_uavs_total_consumed_energy(self):
        return sum([_drone.battery.get_total_energy_consumption() for _drone in self.base_stations[NUM_MBS:]])

    def set_ues_base_stations(self, exclude_mbs=True):
        self.bs_rf_list = [_bs.rf_transceiver for _bs in self.base_stations[NUM_MBS if exclude_mbs else 0:]]
        all_freqs = [_bs.carrier_frequency for _bs in self.bs_rf_list]
        available_freqs = set(all_freqs)
        self.stations_list = []
        for _freq in available_freqs:
            bs_list = []
            for idx, _freq_bs in enumerate(all_freqs):
                if _freq_bs == _freq:
                    bs_list.append(self.bs_rf_list[idx])
            self.stations_list.append(bs_list)
        for _user in self.users:
            _user.rf_transceiver.set_available_base_stations(self.stations_list)

    def link_stations_with_fso_star(self):
        for i in range(NUM_MBS, len(self.base_stations)):
            self.link_two_stations_with_fso(self.base_stations[0], self.base_stations[i])

    def link_stations_with_fso_sequential(self):
        mbs_tr = FsoTransceiver(coords=self.base_stations[0].coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_MBS,
                                bs_id=self.base_stations[0].id)
        dbs_tr = FsoTransceiver(coords=self.base_stations[1].coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_DRONE,
                                bs_id=self.base_stations[1].id, is_backhaul=True, endpoint=mbs_tr)
        mbs_tr.endpoint = dbs_tr
        self.base_stations[0].fso_transceivers.append(mbs_tr)
        self.base_stations[1].fso_transceivers.append(dbs_tr)

        for i in range(1, len(self.base_stations) - 1):
            self.link_two_stations_with_fso(self.base_stations[i], self.base_stations[i + 1])
        self.update_fso_stats()

    @staticmethod
    def link_two_stations_with_fso(backhaul_bs, next_bs):
        """First is backhaul second is normal"""
        if isinstance(backhaul_bs, MacroBaseStation) and isinstance(next_bs, MacroBaseStation):
            bdbs_tr = FsoTransceiver(coords=backhaul_bs.coords, link_type=LinkType.G2G, t_power=TX_POWER_FSO_MBS,
                                     bs_id=backhaul_bs.id, is_backhaul=False)
            ndbs_tr = FsoTransceiver(coords=next_bs.coords, link_type=LinkType.G2G, t_power=TX_POWER_FSO_MBS,
                                     bs_id=next_bs.id, is_backhaul=True, endpoint=bdbs_tr)
        elif isinstance(backhaul_bs, MacroBaseStation) and isinstance(next_bs, DroneStation):
            bdbs_tr = FsoTransceiver(coords=backhaul_bs.coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_MBS,
                                     bs_id=backhaul_bs.id, is_backhaul=False)
            ndbs_tr = FsoTransceiver(coords=next_bs.coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_DRONE,
                                     bs_id=next_bs.id, is_backhaul=True, endpoint=bdbs_tr)
        elif isinstance(backhaul_bs, DroneStation) and isinstance(next_bs, DroneStation):
            bdbs_tr = FsoTransceiver(coords=backhaul_bs.coords, link_type=LinkType.A2A, t_power=TX_POWER_FSO_DRONE,
                                     bs_id=backhaul_bs.id, is_backhaul=False)
            ndbs_tr = FsoTransceiver(coords=next_bs.coords, link_type=LinkType.A2A, t_power=TX_POWER_FSO_DRONE,
                                     bs_id=next_bs.id, is_backhaul=True, endpoint=bdbs_tr)
        elif isinstance(backhaul_bs, DroneStation) and isinstance(next_bs, MacroBaseStation):
            bdbs_tr = FsoTransceiver(coords=backhaul_bs.coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_DRONE,
                                     bs_id=backhaul_bs.id, is_backhaul=False)
            ndbs_tr = FsoTransceiver(coords=next_bs.coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_MBS,
                                     bs_id=next_bs.id, is_backhaul=True, endpoint=bdbs_tr)

        bdbs_tr.endpoint = ndbs_tr
        backhaul_bs.fso_transceivers.append(bdbs_tr)
        next_bs.fso_transceivers.append(ndbs_tr)

    def create_drone_stations(self, check_obstacles=False, irradiation_manager=None, initial_coords_input=None):
        self.base_stations = self.base_stations[:NUM_MBS]
        if initial_coords_input is None:
            xs = np.linspace(X_BOUNDARY[0], X_BOUNDARY[1], int(np.ceil(np.sqrt(NUM_UAVS))) + 2)[1:-1]
            ys = np.linspace(Y_BOUNDARY[0], Y_BOUNDARY[1], int(np.ceil(NUM_UAVS / len(xs))) + 2)[1:-1]
            coords_xs, coords_ys = np.meshgrid(xs, ys)
            coords_xs, coords_ys = coords_xs.flatten(), coords_ys.flatten()
        for i in range(NUM_UAVS):
            if initial_coords_input is None:
                # initial_coords = np.random.uniform(low=X_BOUNDARY[0], high=X_BOUNDARY[1], size=2)
                initial_coords = np.array([coords_xs[i], coords_ys[i]])
                drone_height = DRONE_HEIGHT
            else:
                initial_coords = initial_coords_input[i]
                drone_height = initial_coords_input[i].z

            initial_coords = Coords3d(initial_coords[0], initial_coords[1], drone_height)
            # initial_coords = Coords3d(2, 215, 25)
            new_station = DroneStation(coords=initial_coords, irradiation_manager=irradiation_manager,
                                       drone_id=i + 1)  # ,
            # carrier_frequency=UAVS_FREQS[i])
            self.base_stations.append(new_station)
        self.base_stations = self.base_stations
        self.set_ues_base_stations()

    def add_drone_stations(self, n_drones, irradiation_manager=None):
        for i in range(n_drones):
            initial_coords = np.random.uniform(low=X_BOUNDARY[0], high=X_BOUNDARY[1], size=2)
            drone_height = DRONE_HEIGHT
            initial_coords = Coords3d(initial_coords[0], initial_coords[1], drone_height)
            # initial_coords = Coords3d(2, 215, 25)
            new_station = DroneStation(coords=initial_coords, irradiation_manager=irradiation_manager,
                                       drone_id=self.base_stations[-1].id + 1)  # ,
            # carrier_frequency=UAVS_FREQS[i])
            self.base_stations.append(new_station)

    def update_users_per_bs(self):
        bs_list = [[] for i in range(len(self.bs_rf_list))]
        if not bs_list:
            return
        for _user in self.users:
            bs_id, sinr, snr, rx_power = _user.rf_transceiver.get_serving_bs_info(recalculate=False)
            bs_list[bs_id - NUM_MBS].append(_user)
        for _bs, _users in zip(self.bs_rf_list, bs_list):
            _bs.n_associated_users = len(_users)
        self.users_per_bs = bs_list

    def get_users_per_bs(self):
        return self.users_per_bs

    def get_uavs_locs(self):
        coords = []
        for idx, _bs in enumerate(self.base_stations[NUM_MBS:]):
            coords.append(_bs.coords)
        return coords

    def init_users_rfs(self):
        [_user.rf_transceiver.sinr_coverage_history.fill(0) for _user in self.users]
        for _user in self.users:
            _user.rf_transceiver.sinr_coverage_score = 0
        self.set_ues_base_stations()
        [_user.rf_transceiver.get_serving_bs_info(recalculate=True) for _user in self.users]
        # self.update_sinr_coverage_scores()
        self.update_users_per_bs()
        # self.init_users_rf_stats()

    def init_users_rf_stats(self):
        [_user.rf_transceiver.init_rf_stats() for _user in self.users]

    def update_users_rfs(self):
        [_user.rf_transceiver.get_serving_bs_info(recalculate=True) for _user in self.users]
        self.update_users_per_bs()

    def get_users_sinrs(self):
        return np.array([_user.rf_transceiver.received_sinr for _user in self.users])

    def generate_plot_capacs(self):
        req_capacs = self.get_required_capacity_per_dbs()
        mpl.rc('font', family='Times New Roman')
        fig, ax = self.generate_plot(False)
        for idx_i, dbs_i in enumerate(self.base_stations):
            if idx_i >= NUM_MBS:
                ax.text(dbs_i.coords.x + 10, dbs_i.coords.y,
                        f'{req_capacs[idx_i - NUM_MBS] / 1e6}', fontsize=12, color='red')
            for idx_j, dbs_j in enumerate(self.base_stations):
                if idx_i == idx_j or self.fso_links_capacs[idx_i, idx_j] < 1:
                    continue
                if dbs_i.coords.get_distance_to(dbs_j.coords) > MAX_FSO_DISTANCE:
                    continue
                ax.plot([dbs_i.coords.x, dbs_j.coords.x], [dbs_i.coords.y, dbs_j.coords.y],
                        color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
                ax.text(abs(dbs_i.coords.x + dbs_j.coords.x) / 2, abs(dbs_i.coords.y + dbs_j.coords.y) / 2,
                        f'{self.fso_links_capacs[idx_i, idx_j]}', fontsize=8)

        return fig

    def generate_plot(self, plot_ues=True):
        mpl.rc('font', family='Times New Roman')
        if plot_ues:
            fig, ax = self.users_model.generate_plot()
        else:
            fig, ax = plt.subplots()

        num_dbs = len(self.base_stations) - NUM_MBS
        dbs_xs, dbs_ys = np.zeros((num_dbs,), dtype=float), np.zeros((num_dbs,), dtype=float)
        mbs_xs, mbs_ys = np.zeros((NUM_MBS,), dtype=float), np.zeros((NUM_MBS,), dtype=float)
        for i in range(NUM_MBS, len(self.base_stations)):
            dbs_idx = i - NUM_MBS
            dbs_xs[dbs_idx] = self.base_stations[i].coords.x
            dbs_ys[dbs_idx] = self.base_stations[i].coords.y
        for mbs_idx in range(NUM_MBS):
            mbs_xs[mbs_idx] = self.base_stations[mbs_idx].coords.x
            mbs_ys[mbs_idx] = self.base_stations[mbs_idx].coords.y
        ax.scatter(dbs_xs, dbs_ys, edgecolor='red', facecolor='black', alpha=1, marker="s", label="DBSs")
        ax.scatter(mbs_xs, mbs_ys, edgecolor='green', facecolor='black', alpha=1, marker="o", label="MBSs")
        return fig, ax


if __name__ == "__main__":
    # np.random.seed(10)
    a = SimulationController()
    # a.reset_users_model()
    # n_iters = a.localize_drones(1)
    # a.update_users_rfs()
    #
    # sinrs_db = lin2db(a.get_users_sinrs())
    # min_sinr, max_sinr = np.ceil(sinrs_db.min()), np.ceil(sinrs_db.max())
    # n, bins, patches = plt.hist(x=sinrs_db, bins=40, color='#0504aa',
    #                             alpha=0.7, rwidth=0.85)
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('SINR')
    # plt.ylabel('Frequency')
    # plt.title('SINRs')
    # plt.text(23, 45, rf'$\mu={sinrs_db.mean()}$')
    # _ = plt.xticks(np.arange(min_sinr, max_sinr, 4, dtype=int))
    # plt.show()

    # ###############KMEANS VS EM##################
    # n_iters = 100
    # kmeans_means, em_means = np.zeros(n_iters), np.zeros(n_iters)
    # for _iter in range(n_iters):
    #     a.reset_users_model()
    #     n_iters = a.localize_drones(0)
    #     print("EM Iters: ", n_iters)
    #     a.update_users_rfs()
    #     sinrs_db = lin2db(a.get_users_sinrs())
    #     em_means[_iter] = sinrs_db.mean()
    #
    #     n_iters = a.localize_drones(1)
    #     a.update_users_rfs()
    #     sinrs_db = lin2db(a.get_users_sinrs())
    #     kmeans_means[_iter] = sinrs_db.mean()
    #     print(em_means.mean(), kmeans_means.mean())
    #######################################################

    # fig, _ =a.generate_plot()
    # fig.show()
    # perform_sinr_em(a.users, a.base_stations[NUM_MBS:])
    # b = a.get_fso_capacities()
    # fig, _ =a.generate_plot()
    # fig.show()
    # a.generate_plot_capacs()
    # # a.set_drones_waypoints(TIME_STEP*20)
    # while 1:
    #     a.simulate_time_step()
    #     sleep(1)
    #     print("======================================================")
    #     for _uav in a.base_stations[NUM_MBS:]:
    #         # print(_uav.battery.energy_level)
    #         # print(_uav.battery.recharge_count)
    #         # print(_uav.fso_transceivers[0].received_charge_power)
    #         # print(_uav.fso_transceivers[0].link_capacity)
    #         _uav.coords.update(Coords3d(500, 500, 200), 5)
    #         # print(_uav.coords)
    #         print("distance:", _uav.coords.get_distance_to(a.base_stations[0].coords))
    #     # for _user in a.users:
    #     # _user.rf_transceiver.get_serving_bs_info(recalculate=True)
    #     # bs_id, sinr, snr, rx_power, capacity = _user.rf_transceiver.get_serving_bs_info(recalculate=True)
    #     # print(lin2db(sinr))
    #     # print(_user.coords.get_distance_to(_user.rf_transceiver.serving_bs.coords))
    #     # print(_user.coords)

    # a.mobility_model.generate_model()
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # a.simulate_time_step()
    # for _user in a.users:
    #     # _user.rf_transceiver.get_serving_bs_info(recalculate=True)
    #     # print(lin2db(_user.rf_transceiver.get_serving_bs_info(recalculate=True)[1]))
    #     # print(_user.coords.get_distance_to(_user.rf_transceiver.serving_bs.coords))
    #     # print(_user.rf_transceiver.serving_bs.coords)
    #     print(_user.coords)
    #     print("======================================================")
