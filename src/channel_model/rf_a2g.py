#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

import numpy as np
from src.data_structures import Coords3d
from scipy.stats import rice, rayleigh
from src.math_tools import db2lin, lin2db, newton_raphson
from scipy.constants import speed_of_light
from src.parameters import *
from sympy import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.optimize import brentq

PATHLOSS_CONST = 4 * np.pi / speed_of_light

class decentralizedSenseSend:
    """Based on https://ieeexplore.ieee.org/document/8494742. However, they reference the 3GPP model but they don't seem
    To conform to it. UAV - cellular BS model."""

    @staticmethod
    def get_los_probability(drone_coords: Coords3d, base_station_coords: Coords3d) -> float:
        distance_2d = drone_coords.get_distance_to(base_station_coords, flag_2d=True)
        r_i = distance_2d
        r_c = max(294.05 * np.log10(drone_coords.z) - 432.94, 18)

        if r_i <= r_c:
            return 1
        else:
            p_0 = 233.98 * np.log10(drone_coords.z) - 0.95
            res = r_c / r_i + np.exp((-r_i / p_0) * (1 - r_c / r_i))

            return min(res, 1)

    @staticmethod
    def get_los_pathloss(drone_coords: Coords3d, base_station_coords: Coords3d,
                         carrier_freq=DEFAULT_CARRIER_FREQ_MBS) -> float:
        distance_3d = drone_coords.get_distance_to(base_station_coords)
        fspl = lin2db((4 * np.pi * distance_3d * carrier_freq / speed_of_light) ** 2)
        res = 30.9 + (22.25 - 0.5 * np.log10(drone_coords.z)) * np.log10(distance_3d) + 20 * np.log10(
            carrier_freq / 1e9)
        return max(fspl, res)

    @staticmethod
    def get_nlos_pathloss(los_pathloss, drone_coords: Coords3d, base_station_coords: Coords3d,
                          carrier_freq=DEFAULT_CARRIER_FREQ_MBS) -> float:
        distance_3d = drone_coords.get_distance_to(base_station_coords)
        return 32.4 + (43.2 - 7.6 * np.log10(drone_coords.z)) * np.log10(distance_3d) + 20 * np.log10(
            carrier_freq / 1e9)

    @staticmethod
    def get_los_ss_fading_sample(drone_coords: Coords3d) -> float:
        # Small-scale LOS fading obeys Rice distribution
        k_shape_parameter = db2lin(4.217 * np.log10(drone_coords.z) + 5.787)
        return float(rice.rvs(k_shape_parameter, size=1, scale=1))

    @staticmethod
    def get_nlos_ss_fading_sample():
        # Small-scale NLOS fading obeys Rayleigh distribution
        return float(rayleigh.rvs(size=1))

    @staticmethod
    def get_successful_transmission_probability(drone_coords: Coords3d,
                                                base_station_coords: Coords3d = Coords3d(0, 0, 0),
                                                p_t=DRONE_TX_POWER_RF,
                                                snr_threshold=DEFAULT_SNR_THRESHOLD,
                                                carrier_freq=DEFAULT_CARRIER_FREQ_MBS) -> float:
        los_pl = decentralizedSenseSend.get_los_pathloss(drone_coords, base_station_coords, carrier_freq)
        nlos_pl = decentralizedSenseSend.get_nlos_pathloss(los_pl, drone_coords, base_station_coords, carrier_freq)

        x_los = NOISE_POWER_RF * (10 ** (0.1 * los_pl)) * snr_threshold / p_t

        x_nlos = NOISE_POWER_RF * 10 ** (0.1 * nlos_pl) * snr_threshold / p_t

        pr_los = decentralizedSenseSend.get_los_probability(drone_coords, base_station_coords)

        k_shape_parameter = 20 * np.log10(4.217 * np.log10(drone_coords.z) + 5.787)
        # k_shape_parameter = 10
        Fri = min(rice.cdf(x_los, k_shape_parameter, scale=1), 1)
        Fra = rayleigh.cdf(x_nlos)

        return pr_los * (1 - Fri) + (1 - pr_los) * (1 - Fra)

    @staticmethod
    def get_received_power_sample(p_t, drone_coords: Coords3d,
                                  base_station_coords: Coords3d, carrier_freq=DEFAULT_CARRIER_FREQ_MBS) -> float:

        los_gain = decentralizedSenseSend.get_los_ss_fading_sample(drone_coords) / 10 ** \
                   (0.1 * decentralizedSenseSend.get_los_pathloss(drone_coords, base_station_coords, carrier_freq))

        nlos_gain = decentralizedSenseSend.get_nlos_ss_fading_sample(drone_coords) / 10 ** \
                    (0.1 * decentralizedSenseSend.get_nlos_pathloss(drone_coords, base_station_coords, carrier_freq))

        pr_los = decentralizedSenseSend.get_los_probability(drone_coords, base_station_coords)

        return p_t * (pr_los * los_gain + (1 - pr_los) * nlos_gain)

    @staticmethod
    def get_received_snr_sample(p_t, drone_coords: Coords3d,
                                base_station_coords: Coords3d, carrier_freq=DEFAULT_CARRIER_FREQ_MBS) -> float:

        return decentralizedSenseSend.get_received_power_sample(p_t, drone_coords,
                                                                base_station_coords, carrier_freq) / NOISE_POWER_RF


class PlosModel:
    """The model as defined in https://ieeexplore.ieee.org/document/6863654"""

    @staticmethod
    def get_path_loss(ue_coords: Coords3d, bs_coords: Coords3d = Coords3d(0, 0, 0), frequency=DEFAULT_CARRIER_FREQ_MBS):
        """Return path loss in linear"""
        distance_2d = ue_coords.get_distance_to(bs_coords, flag_2d=True)
        distance_3d = np.sqrt(distance_2d ** 2 + (bs_coords.z - ue_coords.z) ** 2)
        los_probability = PlosModel.get_los_probability(abs(bs_coords.z - ue_coords.z), distance_2d)
        path_loss = 20 * np.log10(
            PATHLOSS_CONST * frequency * distance_3d ) + los_probability * PLOS_AVG_LOS_LOSS + \
                    (1 - los_probability) * PLOS_AVG_NLOS_LOSS
        return db2lin(path_loss)

    @staticmethod
    def get_los_probability(height, distance_2d, a_param=PLOS_A_PARAM, b_param=PLOS_B_PARAM):
        return 1 / (1 + a_param * np.exp(-b_param * (180 / np.pi * np.arctan(height / distance_2d) - a_param)))

    @staticmethod
    def get_a_b_params(env_type='Urban', alpha=None, beta=None, gamma=None):
        """https://ieeexplore.ieee.org/document/6863654"""
        if not alpha and not beta and not gamma:
            if env_type == 'Suburban':
                alpha, beta, gamma = 0.1, 750, 8
            elif env_type == 'Urban':
                alpha, beta, gamma = 0.3, 500, 15
            elif env_type == 'Dense Urban':
                alpha, beta, gamma = 0.5, 300, 20
            elif env_type == 'Highrise Urban':
                alpha, beta, gamma = 0.5, 300, 50
            else:
                raise ValueError('Undefined environment type! Choose one of following:'
                                 ' Suburban, Urban. Dense Urban, Highrise Urban')

        cij_a = [[9.34e-1, 2.3e-1, -2.25e-3, 1.86e-5], [1.97e-2, 2.44e-3, 6.58e-6, 0], [-1.24e-4, -3.34e-6, 0, 0],
                 [2.73e-7, 0, 0, 0]]
        cij_b = [[1.17, -7.56e-2, 1.98e-3, -1.78e-5], [-5.79e-3, 1.81e-4, -1.65e-6, 0], [1.73e-5, -2.02e-7, 0, 0],
                 [-2e-8, 0, 0, 0]]

        def get_fitting_parameter(cij_matrix):
            sum = 0
            for i in range(4):
                for j in range(4):
                    sum += cij_matrix[i][j] * (alpha * beta) ** i * gamma ** j
            return sum

        a, b = get_fitting_parameter(cij_a), get_fitting_parameter(cij_b)
        return a, b

    @staticmethod
    def get_optimal_height_radius(min_snr=DEFAULT_SNR_THRESHOLD, transmission_power=DRONE_TX_POWER_RF,
                           noise_power=NOISE_POWER_RF, carrier_freq=DEFAULT_CARRIER_FREQ_DRONE,
                           avg_loss_los=PLOS_AVG_LOS_LOSS, avg_loss_nlos=PLOS_AVG_NLOS_LOSS, env_a=PLOS_A_PARAM,
                           env_b=PLOS_B_PARAM):
        max_path_loss = lin2db(transmission_power / (min_snr * noise_power))
        # max_path_loss = 100
        A = avg_loss_los - avg_loss_nlos
        B = 20 * np.log10(carrier_freq) + 20 * np.log10(4 * np.pi / speed_of_light) + avg_loss_nlos
        theta_opt = PlosModel.get_optimal_elevation_angle(A, env_a, env_b)

        R = (10 ** ((max_path_loss - A / (
                    1 + env_a * np.exp(-env_b * (theta_opt * 180 / np.pi - env_a))) - B) / 20)) * np.cos(theta_opt)
        h = np.tan(theta_opt) * R
        return h, R

    @staticmethod
    def get_coverage_radius(uav_height=UAVS_HEIGHTS[0], min_snr=DEFAULT_SNR_THRESHOLD, transmission_power=DRONE_TX_POWER_RF,
                           noise_power=NOISE_POWER_RF, carrier_freq=DEFAULT_CARRIER_FREQ_DRONE,
                           avg_loss_los=PLOS_AVG_LOS_LOSS, avg_loss_nlos=PLOS_AVG_NLOS_LOSS, env_a=PLOS_A_PARAM,
                           env_b=PLOS_B_PARAM, ue_bandwidth=USER_BANDWIDTH, drone_bandwidth=DRONE_BANDWIDTH):
        tx_power = ue_bandwidth/ drone_bandwidth * transmission_power
        max_path_loss = lin2db(tx_power / (min_snr * noise_power))
        A = avg_loss_los - avg_loss_nlos
        B = 20 * np.log10(carrier_freq) + 20 * np.log10(4 * np.pi / speed_of_light) + avg_loss_nlos
        # R = symbols('R')
        # f = max_path_loss - A / (1 + env_a * exp(-env_b * (atan(uav_height/R) * 180 / np.pi - env_a))) - 10 * log(uav_height**2 + R**2, 10) - B
        # fderivative = f.diff(R)
        # res = newton_raphson(f, fderivative, R, initial_guess=1)[0]

        func = lambda R: max_path_loss - A / (1 + env_a * np.exp(-env_b * (np.arctan(uav_height/R) * 180 / np.pi - env_a))) - 10 * np.log10(uav_height**2 + R**2) - B
        # res = fsolve(func, 1)[0]
        try:
            res = brentq(func, 1, 1e9)
        except:
            print("Infeasible solution for coverage radius!")
            return 0
        # PlosModel.get_path_loss(Coords3d(0,0,0), Coords3d(res/np.sqrt(2), res/np.sqrt(2), uav_height))


        theta_2 = np.arctan(uav_height/res)
        max_pl = A / (1 + env_a * np.exp(-env_b * (theta_2 * 180 / np.pi - env_a))) + 20 * np.log10(res / np.cos(theta_2)) + B
        # func2 = lambda theta: max_path_loss - A / (1 + env_a * np.exp(-env_b * (theta * 180 / np.pi - env_a))) + 20 * np.log10(uav_height/ (np.arctan(theta)* np.cos(theta))) + B
        # res2 = brentq(func2, 1, 1e9)
        assert (res>0 and abs(max_pl - max_path_loss)<10)
        return res

    @staticmethod
    def get_optimal_elevation_angle(A, env_a, env_b):
        # # env_b, env_a = PlosModel.get_a_b_params()
        #
        # theta = symbols('theta')
        # f = np.pi / (9 * log(10, E)) * tan(theta) + env_a * env_b * A * exp(-env_b * (theta * 180 / np.pi - env_a)) / (
        #         env_a * exp(-env_b * (theta * 180 / np.pi - env_a)) + 1) ** 2
        # fderivative = f.diff(theta)
        # res1 = newton_raphson(f, fderivative, theta, initial_guess=np.pi/4)[0]

        func = lambda theta: np.pi / (9 * np.log(10)) * np.tan(theta) + env_a * env_b * A * np.exp(
            -env_b * (theta * 180 / np.pi - env_a)) / (env_a * np.exp(-env_b * (theta * 180 / np.pi - env_a)) + 1) ** 2
        return fsolve(func, np.pi/4)[0]#, res1

    @staticmethod
    def get_avg_loss(env_type='Urban', frequency=DEFAULT_CARRIER_FREQ_MBS):
        if env_type == 'Suburban':
            if frequency == 700e6:
                return 0, 18
            elif frequency == 2e9:
                return 0.1, 21
            elif frequency == 5.8e9:
                return 0.2, 24
            else:
                raise ValueError('Only 700, 2000, or 5800 MHz are applicable!')
        elif env_type == 'Urban':
            if frequency == 700e6:
                return 0.6, 17
            elif frequency == 2e9:
                return 1, 20
            elif frequency == 5.8e9:
                return 1.2, 23
            else:
                raise ValueError('Only 700, 2000, or 5800 MHz are applicable!')
        elif env_type == 'Dense Urban':
            if frequency == 700e6:
                return 1, 20
            elif frequency == 2e9:
                return 1.6, 23
            elif frequency == 5.8e9:
                return 1.8, 23
            else:
                raise ValueError('Only 700, 2000, or 5800 MHz are applicable!')
        elif env_type == 'Highrise Urban':
            if frequency == 700e6:
                return 1.5, 29
            elif frequency == 2e9:
                return 2.4, 34
            elif frequency == 5.8e9:
                return 2.5, 41
            else:
                raise ValueError('Only 700, 2000, or 5800 MHz are applicable!')
        else:
            raise ValueError('Undefined environment type! Choose one of following:'
                             ' Suburban, Urban. Dense Urban, Highrise Urban')


if __name__ == "__main__":
    # print(PlosModel.get_a_b_params())
    # a, b = PlosModel.get_a_b_params()
    # print(PlosModel.get_optimal_height_radius(env_a=a, env_b=b))
    # print(PlosModel.get_coverage_radius())
    rs = []
    for uav_height in range(20, 100000, 10):
        rs.append(PlosModel.get_coverage_radius(uav_height))
    plt.plot(range(20, 100000, 10), rs)
    plt.show()