#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from src.data_structures import Obstacle
from src.math_tools import line_point_angle, get_mid_azimuth
from src.parameters import UAVS_HEIGHTS, SUN_SEARCH_STEP, SUN_SEARCH_COUNT, MBS_LOCATIONS, \
    EXTEND_TIMES_FOUR

obstacles_madrid_list = [[9, 423, 129, 543, 52.5],
                         [9, 285, 129, 405, 49],
                         [9, 147, 129, 267, 42],
                         [9, 9, 129, 129, 45.5],
                         [147, 423, 267, 543, 31.5],
                         [147, 147, 267, 267, 52.5],
                         [147, 9, 267, 129, 28],
                         [297, 423, 327, 543, 31.5],
                         [297, 285, 327, 405, 45.5],
                         [297, 147, 327, 267, 38.5],
                         [297, 9, 327, 129, 42],
                         [348, 423, 378, 543, 45.5],
                         [348, 285, 378, 405, 49],
                         [348, 147, 378, 267, 38.5],
                         [348, 9, 378, 129, 42]]


class Obstacles(object):
    obstaclesList = []

    def __init__(self, obstacles_data_list, vertices_format='axes'):
        for obstacle_id, obstacle in enumerate(obstacles_data_list):
            vertices = []
            if vertices_format == 'coordinates':
                for idx in range(0, (len(obstacle) - 1), 2):
                    vertices.append((obstacle[idx], obstacle[idx + 1]))
            elif vertices_format == 'axes':
                vertices = [(obstacle[0], obstacle[1]), (obstacle[0], obstacle[3]),
                            (obstacle[2], obstacle[3]), (obstacle[2], obstacle[1])]
            self.obstaclesList.append(Obstacle(obstacle_id, obstacle[-1], vertices))

    def check_overlap(self, other_coords):
        for _obstacle in self.obstaclesList:
            if _obstacle.is_overlapping(other_coords.x, other_coords.y, other_coords.z):
                return True
        return False

    def get_total_vertices(self):
        total_vertices = []
        for obstacle in self.obstaclesList:
            total_vertices = total_vertices + obstacle.vertices
        return total_vertices

    def get_total_edges(self):
        edges = []
        for obstacle in self.obstaclesList:
            obstacle_poly = obstacle.vertices + [obstacle.vertices[0]]
            for idx in range(len(obstacle.vertices)):
                edges.append([obstacle_poly[idx], obstacle_poly[idx + 1]])
        return edges

    def print_obstacles(self):
        for obstacle in self.obstaclesList:
            print(obstacle.id, ": ", obstacle.vertices, obstacle.height)

    def plot_obstacles(self, show_flag=False, fill_color=None):
        if not fill_color:
            for obstacle in self.obstaclesList:
                xs, ys = zip(*obstacle.vertices + [obstacle.vertices[0]])
                plt.plot(xs, ys, c='dimgray')
        else:
            rects = []
            for obstacle in self.obstaclesList:
                xs, ys = zip(*obstacle.vertices + [obstacle.vertices[0]])
                xs = set(xs)
                ys = set(ys)
                corner = min(xs), min(ys)
                height = max(ys) - min(ys)
                width = max(xs) - min(xs)
                rects.append(Rectangle(corner, width, height, color=fill_color))
            return rects

        if show_flag:
            print("SHOWING")
            plt.show()


def get_madrid_buildings():
    extension = []
    if EXTEND_TIMES_FOUR:
        x_shift = 400  # Total dimensions for Madrid Grid is 387 m (east-west) and 552 m (south north).  The
        #   building height is uniformly distributed between 8 and 15 floors with 3.5 m per floor
        y_shift = 570
        for _bldg in obstacles_madrid_list:
            extension.append([_bldg[0] + x_shift, _bldg[1], _bldg[2] + x_shift, _bldg[3], _bldg[4]])
        for _bldg in obstacles_madrid_list:
            extension.append([_bldg[0], _bldg[1] + y_shift, _bldg[2], _bldg[3] + y_shift, _bldg[4]])
        for _bldg in obstacles_madrid_list:
            extension.append([_bldg[0] + x_shift, _bldg[1] + y_shift, _bldg[2] + x_shift, _bldg[3] + y_shift, _bldg[4]])
        new_obstacles = obstacles_madrid_list + extension
    return Obstacles(new_obstacles)


if __name__ == '__main__':
    _rects = get_madrid_buildings().plot_obstacles(False, fill_color='gray')
    plt.plot(MBS_LOCATION.x, MBS_LOCATION.y, c='blue', marker='s', label='MBS', linestyle='none')
    ax = plt.gca()
    for _rect in _rects:
        ax.add_patch(_rect)
    # xs = np.array([0, 15.0, 200.0])
    # ys = np.array([0.0, 390.0, 370.0])
    # plt.plot(xs, ys, c='red')
    # plt.plot([], [], c='red', label='FSO link')
    # plt.plot(250.0, 380.0, c='green', marker='o', label='Hotspot center', linestyle='none')
    # # plt.plot(200.0, 370.0, c='green', marker='o', label='Hotspot center', linestyle='none')

    xs = [82.5, 165, 82.5, 165, 345.5, 263, 345.5, 263]
    ys = [113.5, 227, 479.5, 366, 113.5, 227, 479.5, 366]

    for x, y in zip(xs, ys):
        plt.plot(x, y, c='green', marker='o', linestyle='none')
        # circle = plt.Circle((x, y), 300, color='r')
        # ax = plt.gca()
        # ax.add_patch(circle)

    # plt.legend(loc="upper left")
    plt.show()
    # plt.savefig('plots/madrid_modified_plus_stations.eps', format='eps')
