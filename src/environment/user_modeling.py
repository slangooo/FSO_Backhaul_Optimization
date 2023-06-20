#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from src.parameters import *
from itertools import count
import numpy as np
from random import randint
import matplotlib.pyplot as plt

class UserWalker:
    _ids = count(0)

    def __init__(self, user_id, starting_vertex=None, initial_coords=None):
        if starting_vertex:
            self.last_waypoint = starting_vertex.id
            self.current_coords = starting_vertex.coords.copy()
            self.current_coords.z = UE_HEIGHT
            self.next_pause_time = 0
        elif initial_coords is not None:
            self.current_coords = Coords3d(initial_coords[0], initial_coords[1], UE_HEIGHT)
        else:
            raise ValueError('No coordinates were provided!')

        self.id = next(self._ids) if user_id is None else user_id
        self.waypoints = []
        self.speed = np.random.uniform(USER_SPEED[0], USER_SPEED[1])
        self.remaining_pause_time = 0

    def set_waypoints(self, waypoints, last_waypoint_id):
        assert (waypoints[0] == self.current_coords)
        waypoints.pop(0)
        self.waypoints = waypoints
        self.last_waypoint = last_waypoint_id

    def update_location(self, delta_t):
        """Update location towards next waypoint or pause. Returns True when pause
         is done or last waypoint reached"""
        if not self.remaining_pause_time:
            if self.waypoints:
                if self.current_coords.update(self.waypoints[0], delta_t * self.speed):
                    self.current_coords.set(self.waypoints[0])
                    self.waypoints.pop(0)
                return False
            else:
                if self.next_pause_time:
                    self.remaining_pause_time = self.next_pause_time
                    self.next_pause_time = 0
                else:
                    self.remaining_pause_time = randint(PAUSE_INTERVAL[0], PAUSE_INTERVAL[1])
        else:
            self.remaining_pause_time -= delta_t
            if self.remaining_pause_time <= 0:
                self.remaining_pause_time = 0

        if not self.remaining_pause_time:
            return True
        return False


class UserSpatialModel:
    users = []

    def __init__(self, ):
        pass


class ThomasClusterProcess(UserSpatialModel):
    n_clusters = 0
    n_ues_per_cluster = None
    clusters_centers = None
    n_users = 0

    def __init__(self, mean_ues_per_cluster=MEAN_UES_PER_CLUSTER):
        self.rng = np.random.default_rng()
        # self.rng = np.random.default_rng(seed=10)
        self.generate_distribution(mean_ues_per_cluster)

    def populate_num_of_clusters(self):
        self.n_clusters = self.rng.poisson(N_CLUSTERS)

    def populate_num_of_ues_per_cluster(self, mean_ues_per_cluster):
        self.n_ues_per_cluster = self.rng.poisson(mean_ues_per_cluster, (self.n_clusters))
        self.n_users = self.n_ues_per_cluster.sum()

    def generate_distribution(self, mean_ues_per_cluster=MEAN_UES_PER_CLUSTER):
        self.populate_num_of_clusters()
        self.populate_num_of_ues_per_cluster(mean_ues_per_cluster)
        clusters_xs = (X_BOUNDARY[1] - X_BOUNDARY[0]) * np.random.uniform(0, 1, self.n_clusters)
        clusters_ys = (Y_BOUNDARY[1] - Y_BOUNDARY[0]) * np.random.uniform(0, 1, self.n_clusters)
        self.clusters_centers = np.stack((clusters_xs, clusters_ys), 1)

        childs_xs = np.random.normal(0, SIGMA_UE_PER_CLUSTER, self.n_users)
        childs_ys = np.random.normal(0, SIGMA_UE_PER_CLUSTER, self.n_users)

        self.childs_xs = np.repeat(clusters_xs, self.n_ues_per_cluster)
        self.childs_ys = np.repeat(clusters_ys, self.n_ues_per_cluster)

        self.childs_xs += childs_xs
        self.childs_ys += childs_ys

        for uid in range(self.n_users):
            self.users.append(UserWalker(uid, initial_coords=Coords3d(self.childs_xs[uid], self.childs_ys[uid], UE_HEIGHT)))

    def generate_plot(self):
        fig, ax = plt.subplots()
        ax.scatter(self.childs_xs, self.childs_ys, edgecolor='b', facecolor='none', alpha=0.5, marker=".", label="UEs")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(X_BOUNDARY[0] - 6*SIGMA_UE_PER_CLUSTER,
                 X_BOUNDARY[1] + 6*SIGMA_UE_PER_CLUSTER)
        ax.set_ylim(Y_BOUNDARY[0] - 6*SIGMA_UE_PER_CLUSTER,
                 Y_BOUNDARY[1] + 6*SIGMA_UE_PER_CLUSTER)
        return fig, ax


if __name__ == '__main__':
    print(N_CLUSTERS)
    tc = ThomasClusterProcess()
    tc.generate_distribution()
    fig, ax = tc.generate_plot()
    fig.show()
    print(tc.n_users)
    print(tc.n_ues_per_cluster)
    print(tc.clusters_centers)
