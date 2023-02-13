
#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from src.parameters import *
from itertools import count
import numpy as np
# from src.apparatus.uav_battery import UavBattery
from src.apparatus.rf_transciever import RfTransceiver
from src.types_constants import StationType

import time


class DroneStation:
    _ids = count(NUM_MBS)  # before is reserved for MBS
    moving_flag = False

    def __init__(self, drone_id: int = None, coords: Coords3d = Coords3d(0.0, 0.0, 0.0), irradiation_manager=None,
                 carrier_frequency=DEFAULT_CARRIER_FREQ_DRONE):
        """If irradiation_manager, equip with solar panels"""
        self.id = next(self._ids) if drone_id is None else drone_id
        self.coords = coords.copy()
        # self.battery = UavBattery(irradiation_manager=irradiation_manager, coords=self.coords)
        self.battery = None
        self.rf_transceiver = RfTransceiver(coords=self.coords, bandwidth=DRONE_BANDWIDTH, bs_id=self.id,
                                            carrier_frequency=carrier_frequency, station_type=StationType.DBS)
        self.fso_transceivers = []
        self.next_waypoint = None
        self.speeds = Coords3d(0, 0, 0)  # x,y,z
        self.speed = 0

    def update_energy(self, t_step):
        """To be called after moving UAV"""
        tx_powers = np.sum([_fso_tr.tx_power for _fso_tr in self.fso_transceivers]) + self.rf_transceiver.tx_power
        return self.battery.update_energy(t_step, self.speeds.x, self.speeds.y, self.speeds.z, tx_powers,
                                          self.fso_transceivers[0].received_charge_power)

    def update_fso_status(self):
        for fso_tr in self.fso_transceivers:
            fso_tr.calculate_link_capacity_and_received_power()

    def move(self, t_step):
        """If true, UAV reached"""
        if self.next_waypoint is None or self.next_waypoint==self.coords:
            self.next_waypoint = None
            self.moving_flag = False
            return True
        distance = t_step * self.speed
        required_distance = self.coords.get_distance_to(self.next_waypoint)
        if self.coords.update(self.next_waypoint, distance):
            t_travel = required_distance/distance * t_step
            self.update_energy(t_travel)
            self.speed, self.speeds = 0, Coords3d(0, 0, 0)
            self.update_energy(t_step - t_travel)
            self.battery.skip_movement_energy = True #Skips next update only
            self.moving_flag = False
            return True
        self.moving_flag = True
        return False

    def set_waypoint(self, waypoint: Coords3d, travel_duration):
        self.next_waypoint = waypoint
        self.speeds = (self.next_waypoint - self.coords) / travel_duration
        self.speed = np.linalg.norm(self.speeds)


if __name__ == '__main__':
    a = Coords3d(5, 6, 7)
    time1 = time.time()
    for i in range(500000):
        np.linalg.norm(a)
    time2 = time.time()
    for i in range(500000):
        np.sqrt(np.sum(a.np_array() ** 2))
    time3 = time.time()
    print(time2 - time1, time3 - time2)
