#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from src.parameters import *
import numpy as np
from src.channel_model.rf_a2g import PlosModel
# from src.channel_model.rf_g2g import UmaCellular
from src.data_structures import Coords3d
from src.types_constants import StationType
import types
from scipy.ndimage.interpolation import shift


class RfTransceiver:
    received_sinrs = []
    received_snrs = []
    received_powers = []
    serving_bs = None
    received_sinr = None
    received_snr = None
    received_power = None
    capacity = None
    min_capacity, max_capacity, mean_capacity = None, None, None
    min_sinr, max_sinr, mean_sinr = None, None, None
    sinr_coverage_score = 0


    def __init__(self, coords: Coords3d, t_power=DRONE_TX_POWER_RF, bandwidth=USER_BANDWIDTH, bs_id=None,
                 user_id=None, carrier_frequency=DEFAULT_CARRIER_FREQ_MBS, station_type=StationType.UE):
        self.tx_power = t_power
        self.coords = coords
        self.stations_list = []
        self.noise_power = NOISE_POWER_RF
        self.bandwidth = bandwidth
        self.user_id = user_id
        self.bs_id = bs_id
        self.carrier_frequency = carrier_frequency
        self.get_path_loss = get_path_loss_function(station_type)
        self.available_bandwidth = self.bandwidth if station_type != StationType.UE else 0
        self.n_associated_users = 0
        self.sinr_coverage_history = np.zeros(USER_SINR_COVERAGE_HIST, dtype=bool)
        # association_func = get_serving_bs_info_snr if ASSOCIATION_SCHEME == 'SNR' else get_serving_bs_info_sinr
        # self.get_serving_bs_info = types.MethodType(association_func, self)

    def get_received_powers(self):
        for bs_idx, bs_set in enumerate(self.stations_list):
            for idx, bs in enumerate(bs_set):
                path_loss = bs.get_path_loss(bs.coords, self.coords, bs.carrier_frequency)
                self.received_powers[bs_idx][idx] = (self.bandwidth / bs.bandwidth * bs.tx_power / path_loss)
        # self.received_powers = []
        # for bs_idx, bs_set in enumerate(self.stations_list):
        #     rx_powers = np.empty(len(bs_set))
        #     for idx, bs in enumerate(bs_set):
        #         path_loss = bs.get_path_loss(bs.coords, self.coords, bs.carrier_frequency)
        #         rx_powers[idx] = (self.bandwidth / bs.bandwidth * bs.tx_power / path_loss)
        #     self.received_powers.append(rx_powers)

    def get_received_sinrs(self):
        self.get_received_powers()
        for set_idx, bs_set in enumerate(self.stations_list):
            interferences = np.empty(len(bs_set))
            rx_powers = self.received_powers[set_idx]
            for idx, _ in enumerate(bs_set):
                interferences[idx] = np.sum(rx_powers) - rx_powers[idx]
            self.received_sinrs[set_idx] = rx_powers / (interferences + self.noise_power)
            self.received_snrs[set_idx] = rx_powers / self.noise_power

    def get_received_snrs(self):
        self.get_received_powers()
        self.received_snrs = []
        for set_idx, bs_set in enumerate(self.stations_list):
            self.received_snrs.append(self.received_powers[set_idx] / self.noise_power)

    def set_available_base_stations(self, stations_list):
        # all_freqs = [_bs.carrier_frequency for _bs in stations_list]
        # available_freqs = set(all_freqs)
        # self.stations_list = []
        # for _freq in available_freqs:
        #     bs_list = []
        #     for idx, _freq_bs in enumerate(all_freqs):
        #         if _freq_bs == _freq:
        #             bs_list.append(stations_list[idx])
        #     self.stations_list.append(bs_list)
        self.stations_list = stations_list
        self.received_powers = [np.zeros(len(self.stations_list[i])) for i in range(len(self.stations_list))]
        self.received_sinrs = [np.zeros(len(self.stations_list[i])) for i in range(len(self.stations_list))]
        self.received_snrs = [np.zeros(len(self.stations_list[i])) for i in range(len(self.stations_list))]

    def get_serving_bs_info(self, recalculate=False):
        """Defined at initialization. Select SNR association or SINR."""
        """Return base_station ID, SINR, SNR, Received Power, Capacity"""
        if recalculate:
            self.get_received_sinrs()
        if not self.stations_list:
            return None, None, None, None
        max_sinrs_idxs = [self.received_sinrs[idx].argmax() for idx in range(len(self.stations_list))]
        max_idx = np.argmax([self.received_sinrs[idx][max_sinrs_idxs[idx]] for idx in range(len(self.stations_list))])
        max_sinrs_idx = max_sinrs_idxs[max_idx]
        self.serving_bs = self.stations_list[max_idx][max_sinrs_idx]
        self.received_sinr = self.received_sinrs[max_idx][max_sinrs_idx]
        self.received_snr = self.received_powers[max_idx][max_sinrs_idx] / self.noise_power
        self.received_power = self.received_powers[max_idx][max_sinrs_idx]
        return self.serving_bs.bs_id, self.received_sinr, self.received_snr, self.received_power

    def update_sinr_coverage_score(self, steps_count):
        # self.sinr_coverage_score -= self.sinr_coverage_history[0]
        # self.sinr_coverage_history = shift(self.sinr_coverage_history, -1, cval=self.is_sinr_satisfied())
        # self.sinr_coverage_score += self.sinr_coverage_history[-1]
        res = self.is_sinr_satisfied()
        idx = steps_count%USER_SINR_COVERAGE_HIST
        self.sinr_coverage_score -= self.sinr_coverage_history[idx]
        self.sinr_coverage_history[idx] = res
        self.sinr_coverage_score += res


        # self.sinr_coverage_score -= self.sinr_coverage_history[0]

    def update_rf_stats(self, steps_count):
        bandwidth_share = self.serving_bs.available_bandwidth / self.serving_bs.n_associated_users
        self.bandwidth = bandwidth_share
        self.capacity = int(self.bandwidth * np.log2(1 + self.received_sinr))/(1024)
        self.min_capacity, self.max_capacity = min(self.min_capacity, self.capacity), max(self.max_capacity, self.capacity)
        self.mean_capacity = self.mean_capacity + (self.capacity - self.mean_capacity)/ steps_count
        self.min_sinr, self.max_sinr = min(self.min_sinr, self.received_sinr), max(self.max_sinr,
                                                                                          self.received_sinr)
        self.mean_sinr = self.mean_sinr + (self.received_sinr - self.mean_sinr) /steps_count

    def get_stats(self):
        return self.capacity, self.min_capacity, self.max_capacity,\
               self.received_sinr, self.min_sinr, self.max_sinr

    def is_sinr_satisfied(self):
        return self.received_sinr >= SINR_THRESHOLD

    def is_snr_satisfied(self):
        return self.received_snr >= DEFAULT_SNR_THRESHOLD

    def init_rf_stats(self):
        if not self.stations_list:
            return
        bandwidth_share = self.serving_bs.available_bandwidth / self.serving_bs.n_associated_users
        self.bandwidth = bandwidth_share
        if not self.mean_capacity:
            self.mean_capacity = self.bandwidth * np.log2(1 + self.received_sinr)
        if self.mean_sinr is None:
            self.mean_sinr = self.received_sinr

        self.min_capacity, self.max_capacity  = self.bandwidth * np.log2(1 + self.received_sinr), \
                                                                   self.bandwidth * np.log2(1 + self.received_sinr)

        self.min_sinr, self.max_sinr = self.received_sinr, self.received_sinr


def get_path_loss_function(station_type):
    if station_type == StationType.UE:
        return None
    elif station_type == StationType.UMa:
        pass
    elif station_type == StationType.DBS:
        return PlosModel.get_path_loss
    else:
        return None


if __name__ == "__main__":
    rf1 = RfTransceiver(Coords3d(0, 1, 2))
    rf2 = RfTransceiver(Coords3d(5, 6, 7))
    rf3 = RfTransceiver(rf1.coords)
