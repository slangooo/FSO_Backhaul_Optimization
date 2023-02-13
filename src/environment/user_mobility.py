#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

# from src.environment import graph_tools
from src.environment.user_modeling import UserWalker
import numpy as np
from src.environment import plotting
from multiprocessing import Process, Manager
from time import sleep
from src.parameters import *
from src.environment.user_modeling import UserSpatialModel
from datetime import datetime, time, timedelta
from dateutil import tz
from src.environment.obstacles import Obstacles, get_madrid_buildings

from shutil import rmtree

import os
from tqdm import tqdm

import time as timee


class ObstaclesMobilityModel(UserSpatialModel):
    loaded = False
    seeks = []
    loaded_duration = 0

    def __init__(self, time_step=TIME_STEP, number_of_users=NUM_OF_USERS, graph=None):
        self.obstacles_objects = get_madrid_buildings()
        self.users = []
        self.plotter_func = plotter_func
        self.current_time = 0
        self.plot_flag = False
        self.users_coords = None
        self.plot_sleep = 0
        self.plotter_process = None
        self.time_step = time_step
        self.auxiliary_updates = []
        self.base_stations_coords = None
        self.n_users = number_of_users
        # self.reset_users()
        self.max_time_of_day = None
        self.starting_solar_hour, self.starting_solar_minute = STARTING_HOUR, STARTING_MINUTE
        self.time_of_day = datetime(2022, STARTING_MONTH, STARTING_DAY, STARTING_HOUR,
                                    STARTING_MINUTE,
                                    0, tzinfo=tz.gettz(TIME_ZONE))
        self.set_max_time_of_day(MAX_HOUR_DAY, 59)
        self.seconds_buffer = 0

    def reset_users(self, speed_divisor=1):
        self.current_time = 0
        self.load_model(save_directory_in=self.loaded, duration=self.loaded_duration, max_num_users=self.n_users,
                        speed_divisor=speed_divisor)

    def generate_plot(self, plot_sleep=TIME_SLEEP):
        """If called, an animation of users mobility will be shown"""
        self.plot_flag = True
        self.plot_sleep = plot_sleep
        manager = Manager()
        self.users_coords = manager.list()
        for _user in self.users:
            self.users_coords.append([_user.current_coords.x, _user.current_coords.y])
        obstacles_list = self.obstacles_objects.obstaclesList
        self.plotter_process = Process(target=self.plotter_func,
                                       args=(None, obstacles_list, self.users_coords,
                                             self.base_stations_coords if self.base_stations_coords else None))
        self.plotter_process.start()

    def update_plot_users_coords(self):
        for idx, _user in enumerate(self.users):
            self.users_coords[idx] = [_user.current_coords.x, _user.current_coords.y]

    def update_plot(self):
        self.update_plot_users_coords()
        if self.plot_sleep:
            sleep(self.plot_sleep)

    def update_users_locations(self):
        for _user in self.users:
            # if _user.id == 0:
            #     bla = 5
            if _user.update_location(self.time_step):
                with open(os.path.join(self.loaded, f"user_{_user.id}"), 'rb') as _file:
                    _file.seek(self.seeks[_user.id])
                    _wayps = np.load(_file, allow_pickle=True)
                    waypoints = [Coords3d.from_array(_arr) for _arr in _wayps]
                    next_pause_time = int(np.load(_file, allow_pickle=True))
                    self.seeks[_user.id] = _file.tell()
                _user.set_waypoints(waypoints.copy(), 0)
                _user.last_waypoint = waypoints[-1]
                _user.next_pause_time = next_pause_time

    def generate_model(self, duration=None):
        while duration is None or self.current_time < duration:
            if not self.generate_model_step():
                return

    def generate_model_step(self, time_step=None, static_users=False):
        self.current_time += self.time_step
        if self.current_time > self.loaded_duration > 0:
            return False
        if not static_users:
            self.update_users_locations()
        for _update in self.auxiliary_updates:
            _update()
        if self.plot_flag:
            self.update_plot()
        # return [_user.current_coords for _user in self.users]

    def get_obstacles(self):
        return self.obstacles_objects.obstaclesList

    def add_base_stations_to_plot(self, base_stations_coords):
        self.base_stations_coords = base_stations_coords

    def set_max_time_of_day(self, max_hour, max_minutes):
        if max_hour > self.time_of_day.time().hour:

            self.max_time_of_day = datetime.combine(self.time_of_day.date(), time(hour=max_hour, minute=max_minutes),
                                                    tzinfo=self.time_of_day.tzinfo)
        else:
            self.max_time_of_day = datetime.combine(self.time_of_day.date() + timedelta(days=1),
                                                    time(hour=max_hour, minute=max_minutes),
                                                    tzinfo=self.time_of_day.tzinfo)

    def increment_time(self, minutes_to_add=None):
        self.seconds_buffer += self.time_step
        if minutes_to_add is None and self.seconds_buffer > 60:
            minutes_to_add = int(self.seconds_buffer / 60)
            self.seconds_buffer = self.seconds_buffer % 60

            self.time_of_day = self.time_of_day + timedelta(minutes=minutes_to_add)
            if self.time_of_day > self.max_time_of_day:
                self.seconds_buffer = 0
                self.time_of_day = datetime.combine(self.time_of_day.date() + timedelta(days=1),
                                                    time(hour=self.starting_solar_hour,
                                                         minute=self.starting_solar_minute),
                                                    tzinfo=self.time_of_day.tzinfo)
                self.set_max_time_of_day(MAX_HOUR_DAY, 59)

    def load_model(self, folder_name='mobility_model_saved',
                   save_name="default", duration=60 * 60 * 2, max_num_users=None, save_directory_in=None,
                   speed_divisor=USER_SPEED_DIVISOR):
        save_directory = os.path.join(os.path.abspath(os.path.join(os. getcwd(), os.pardir)),
                                      folder_name, save_name) if not save_directory_in else save_directory_in
        self.seeks = []
        n_users = len(os.listdir(save_directory))
        self.n_users = min(n_users, max_num_users if max_num_users else n_users)
        for uid in range(self.n_users):
            with open(os.path.join(save_directory, f"user_{uid}"), 'rb') as _file:
                _file.seek(0)
                speed = float(np.load(_file, allow_pickle=True)) / speed_divisor
                wyps = np.load(_file, allow_pickle=True)
                next_pause_time = int(np.load(_file, allow_pickle=True))
                self.seeks.append(_file.tell())
                if wyps.size == 0:
                    wyps = [Coords3d.from_array(_arr) for _arr in np.load(_file, allow_pickle=True)]
                    waypoints = [wyps[0]]
                else:
                    waypoints = [Coords3d.from_array(_arr) for _arr in wyps]
            starting_coords = Coords3d.from_array(waypoints[0])
            if len(self.users) > uid:
                self.users[uid].current_coords.set(starting_coords)
                self.users[uid].next_pause_time = next_pause_time
                self.users[uid].set_waypoints(waypoints.copy(), 0)
                self.users[uid].remaining_pause_time = 0
            else:
                _user = UserWalker(uid, initial_coords=starting_coords)
                _user.speed = speed
                _user.next_pause_time = next_pause_time
                _user.set_waypoints(waypoints.copy(), 0)
                self.users.append(_user)

        self.loaded = save_directory
        self.loaded_duration = duration
        # self.users = tuple(self.users)


def plotter_func(path_segments, obstacles_list, users_coords, base_stations_coords):
    plotter = plotting.Plotter()
    background_objects = []
    if (path_segments):
        for segment in path_segments:
            xs, ys = zip(*segment)
            background_objects.append([xs, ys])
        plotter.set_fixed_background(background_objects, color='g', width=0.5, style='dotted')

    background_objects = []
    for _building in obstacles_list:
        xs, ys = zip(*_building.vertices + [_building.vertices[0]])
        background_objects.append([xs, ys])
    plotter.set_fixed_background(background_objects, color='k', width=1, style='dashed')

    plotter.set_users(users_coords)
    if base_stations_coords:
        plotter.set_base_stations(base_stations_coords)
    plotter.start_plotter()


if __name__ == '__main__':
    mobility_model = ObstaclesMobilityModel()
    time1 = timee.time()
    mobility_model.load_model(save_name='extended_4_madrids_500_users', max_num_users=20)
    mobility_model.generate_model_step(TIME_STEP)
    mobility_model.generate_plot()
    # mobility_model.generate_model(duration=60 * 60 * 4)
    # print(timee.time() - time1)
    #
    # for j in range(100):
    #     mobility_model.generate_model_step(TIME_STEP)