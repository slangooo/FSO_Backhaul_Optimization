#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

import numpy as np
from src.data_structures import Coords3d
from scipy.stats import rice, rayleigh
from src.math_tools import db2lin, lin2db
from scipy.constants import speed_of_light
from src.parameters import *
from math import erf


# class fsoMultiCasting:
#     """Based on the model used in https://ieeexplore.ieee.org/document/9405354."""
#
#     @staticmethod
#     def get_atmoshpheric_turbulence_fading(tx_coords: Coords3d, rx_coords: Coords3d, rx_diameter: float = RX_DIAMETER, \
#                                            divergence_angle: float = DIVERGENCE_ANGLE,
#                                            weather_coefficient: float = WEATHER_COEFF):
#         d = tx_coords.get_distance_to(rx_coords)
#         return rx_diameter ** 2 / (divergence_angle ** 2 * (d ** 2)) * 10 ** (-weather_coefficient * d / 10)
#
#     @staticmethod
#     def get_channel_gain(tx_coords: Coords3d, rx_coords: Coords3d,
#                          rx_responsivity: float = RX_RESPONSIVITY, avg_gml: float = AVG_GML,
#                          rx_diameter: float = RX_DIAMETER, divergence_angle: float = DIVERGENCE_ANGLE,
#                          weather_coefficient: float = WEATHER_COEFF):
#         atm_loss = fsoMultiCasting.get_atmoshpheric_turbulence_fading(tx_coords, rx_coords, rx_diameter,
#                                                                       divergence_angle, weather_coefficient)
#
#         return rx_responsivity * avg_gml * atm_loss
#
#     @staticmethod
#     def get_capacity(channel_gain, tx_power=TX_POWER_FSO_MBS, bandwidth=BANDWIDTH_FSO, noise_power=NOISE_POWER_FSO,
#                      empirical_snr_losses=EMPIRICAL_SNR_LOSSES, power_split_ratio=POWER_SPLIT_RATIO):
#         return bandwidth / 2 * np.log2(1 + (tx_power * channel_gain * power_split_ratio) /
#                                        (noise_power * empirical_snr_losses))


class StatisticalModel:
    """As in https://ieeexplore.ieee.org/document/9040587"""

    @staticmethod
    def get_charge_power_and_capacity(tx_coords: Coords3d, rx_coords: Coords3d, transmit_power=TX_POWER_FSO_MBS,
                                      wavelength=WAVELENGTH, responsivity=RX_RESPONSIVITY, lens_radius=RX_DIAMETER / 2,
                                      noise_power=NOISE_VARIANCE_FSO, bandwidth=BANDWIDTH_FSO,
                                      power_split_ratio=POWER_SPLIT_RATIO, beamwaist_radius=BEAMWAIST_RADIUS,
                                      get_gain=False, fixed_bw=None,
                                      fixed_orientation=True):
        """Fixed Beamwidth"""
        distance = tx_coords.get_distance_to(rx_coords)
        atmospheric_loss = StatisticalModel.get_atmospheric_loss(distance)
        atm_turb_induced_fading = 1  # for completeness
        c_n = StatisticalModel.get_refraction_index(tx_coords, rx_coords)
        rho = StatisticalModel.get_coherence_length(c_n, wavelength, distance)
        beam_width = StatisticalModel.get_beamwidth(distance, rho, beamwaist_radius=beamwaist_radius)
        beam_width = fixed_bw if fixed_bw else beam_width  # TODO: REMOVE
        v1 = StatisticalModel.get_v1(beam_width)
        phi, theta = StatisticalModel.get_orientation_angles(tx_coords,
                                                             rx_coords) if not fixed_orientation else (np.pi / 2, np.pi)
        v2 = StatisticalModel.get_v2(v1, phi, theta)
        t1 = StatisticalModel.get_t1(v1)
        t2 = StatisticalModel.get_t2(v2, phi, theta)
        t = (t1 + t2) / 2
        a0 = StatisticalModel.get_max_fraction_a0(v1, v2)
        lambda_1, lambda_2 = StatisticalModel.get_ig_fluctuations_eigen_values(tx_coords, rx_coords, phi, theta,
                                                                               distance,
                                                                               lens_radius)
        gml = StatisticalModel.get_gml(a0, t, beam_width, misalignment=np.sqrt(lambda_1 + lambda_2))
        gain = gml * atmospheric_loss * atm_turb_induced_fading * responsivity
        received_charge_power = (1 - power_split_ratio) * gain * transmit_power
        if get_gain:
            return lin2db(gain), gml, atmospheric_loss, atm_turb_induced_fading, responsivity
        # print("Lens radius, beamwidth, l/b:", lens_radius, beam_width, beam_width / lens_radius)

        # (41)
        c = np.exp(1) / (2 * np.pi) * responsivity ** 2 * atmospheric_loss ** 2 * (
                power_split_ratio * transmit_power) ** 2 / noise_power
        r_max = 0.5 * np.log2(c * a0 ** 2)
        r_delta = 2 / (t * beam_width ** 2 * np.log(2)) * (lambda_1 + lambda_2)
        capacity = (r_max - r_delta) * bandwidth
        # print("gml, misalignment_loss_rate:", gml, r_delta* bandwidth/(1024))
        return received_charge_power, capacity

    @staticmethod
    def get_refraction_index(tx_coords: Coords3d, rx_coords: Coords3d):
        h_d = (tx_coords.z + rx_coords.z) / 2  # TODO: Remember
        c_0_2 = 1.7 * 10 ** (-14)  # Nominal refractive index on the ground SQUARED
        return c_0_2 * np.exp(-h_d / 100)

    @staticmethod
    def get_coherence_length(refraction_index, wavelength, distance):
        return (0.55 * refraction_index * (2 * np.pi / wavelength) ** 2 * distance) ** (-3 / 5)

    @staticmethod
    def get_beamwidth(distance, coherence_length, beamwaist_radius=BEAMWAIST_RADIUS, wavelength=WAVELENGTH):
        # res = []
        # for bw in np.linspace(beamwaist_radius*1e-5, beamwaist_radius*1e+5):
        #     res.append(bw * np.sqrt(1 + (1 + (2 * bw ** 2) / coherence_length ** 2) *
        #                                   (wavelength * distance / (np.pi * bw ** 2))**2))
        return beamwaist_radius * np.sqrt(1 + (1 + (2 * beamwaist_radius ** 2) / coherence_length ** 2) *
                                          (wavelength * distance / (np.pi * beamwaist_radius ** 2)) ** 2)

    @staticmethod
    def get_atmospheric_loss(distance, weather_coefficient=WEATHER_COEFF):
        return 10 ** (-weather_coefficient * distance / 10)

    @staticmethod
    def get_max_fraction_a0(v1, v2):
        a0 = erf(v1) * erf(v2)
        return a0

    @staticmethod
    def get_v1(beam_width, lens_radius=RX_DIAMETER / 2):
        v1 = lens_radius / beam_width * np.sqrt(np.pi / 2)
        return v1

    @staticmethod
    def get_v2(v1, phi, theta):
        v2 = v1 * abs(np.sin(phi) * np.cos(theta))
        return v2

    @staticmethod
    def get_gml(a0, t, beam_width, misalignment=0):
        gml = a0 * np.exp(-2 * misalignment ** 2 / (t * beam_width ** 2))
        return gml

    @staticmethod
    def get_t1(v1):
        t1 = np.sqrt(np.pi) * erf(v1) / (2 * v1 * np.exp(-v1 ** 2))
        return t1

    @staticmethod
    def get_t2(v2, phi, theta):
        t2 = np.sqrt(np.pi) * erf(v2) / (2 * v2 * np.exp(-v2 ** 2) * np.sin(phi) ** 2 * np.cos(theta) ** 2)
        return t2

    @staticmethod
    def get_relative_location(tx_coords: Coords3d, rx_coords: Coords3d):
        return tx_coords.x - rx_coords.x, tx_coords.y - rx_coords.y, tx_coords.z - rx_coords.z

    @staticmethod
    def get_orientation_angles(tx_coords: Coords3d, rx_coords: Coords3d):
        _x, _y, _z = StatisticalModel.get_relative_location(tx_coords, rx_coords)
        phi = np.pi - np.arccos(_z / (np.sqrt(_x ** 2 + _y ** 2 + _z ** 2)))
        if _x > 0:
            theta = np.pi + np.arctan(_y / _x)
        else:
            theta = np.arctan(_y / _x)
        return phi, theta

    @staticmethod
    def get_fluctuations_variances(distance, lens_radius, sigma=0.2):
        # #From ergodic
        # sigma_x, sigma_y, sigma_z = [0.01]*3
        # sigma_phi, sigma_theta = [0.3*10**-3]*2
        # OR from statistical model
        sigma_x = sigma * lens_radius * 0.8
        sigma_y = sigma * lens_radius * 0.27
        sigma_z = sigma * lens_radius * 0.53
        sigma_phi = sigma * lens_radius / distance * 0.44
        sigma_theta = sigma * lens_radius / distance * 0.9
        return sigma_x, sigma_y, sigma_z, sigma_phi, sigma_theta

    @staticmethod
    def get_fluctuations_constants(_x, _y, _z, phi, theta):
        """ (21)"""
        c1 = -np.tan(theta)
        c2 = -_x / (np.cos(theta) ** 2)
        c3 = _x / (np.sin(phi) ** 2 * np.cos(theta))
        c4 = -(_x * (1 / np.tan(phi)) * np.tan(theta)) / np.cos(theta)
        c5 = -1 / (np.cos(theta) * np.tan(phi))
        return c1, c2, c3, c4, c5

    @staticmethod
    def get_ig_fluctuations_eigen_values(tx_coords: Coords3d, rx_coords: Coords3d, phi, theta, distance, lens_radius):
        #TODO: Most constants are zero. Simplify.
        _x, _y, _z = StatisticalModel.get_relative_location(tx_coords, rx_coords)
        sigma_x, sigma_y, sigma_z, sigma_phi, sigma_theta = \
            StatisticalModel.get_fluctuations_variances(distance, lens_radius)

        c1, c2, c3, c4, c5 = StatisticalModel.get_fluctuations_constants(_x, _y, _z, phi, theta)

        _matrix = [[sigma_y ** 2 + (c1 * sigma_x) ** 2 + (c2 * sigma_theta) ** 2,
                    c1 * c5 * sigma_x ** 2 + c2 * c4 * sigma_theta ** 2],
                   [c1 * c5 * sigma_x ** 2 + c2 * c4 * sigma_theta ** 2,
                    sigma_z ** 2 + (c3 * sigma_phi) ** 2 + (c4 * sigma_theta) ** 2 + (c5 * sigma_x) ** 2]]

        return np.linalg.eig(_matrix)[0]


if __name__ == "__main__":
    tx_coords = Coords3d(500, 0, 100)
    rx_coords = Coords3d(0, 0, 100)
    distance = tx_coords.get_distance_to(rx_coords)
    # print(np.rad2deg(StatisticalModel.get_orientation_angles(distance, tx_coords, rx_coords)))
    # print(np.rad2deg(StatisticalModel.get_orientation_angles2(distance, tx_coords, rx_coords)))
    # cg1 = StatisticalModel.get_channel_gain_and_capacity(tx_coords, rx_coords)
    # cg2 = fsoMultiCasting.get_channel_gain(tx_coords, rx_coords)
    # r2 = fsoMultiCasting.get_capacity(cg2)
