#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from dataclasses import dataclass
import numpy as np
import timeit
from scipy.spatial import distance
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from src.math_tools import rotate, line_point_angle, point_inside_prlgm


@dataclass
class Coords3d:
    __slots__ = ["x", "y", "z"]
    x: float
    y: float
    z: float

    def set(self, other_coords):
        self.x = other_coords.x
        self.y = other_coords.y
        self.z = other_coords.z

    def __str__(self):
        return f'{{x: {str(self.x)}, y:{str(self.y)}, z:{str(self.z)}}}'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __add__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x + other, self.y + other, self.z + other)
        elif isinstance(other, np.ndarray):
            if len(other) == 2:
                return Coords3d(self.x + other[0], self.y + other[1], self.z)
            elif len(other) == 3:
                return Coords3d(self.x + other[0], self.y + other[1], self.z + other[2])
        else:
            raise ValueError("Undefined operation for given operand!")

    def __sub__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x - other, self.y - other, self.z - other)
        elif isinstance(other, np.ndarray):
            if len(other) == 2:
                return Coords3d(self.x - other[0], self.y - other[1], self.z)
            elif len(other) == 3:
                return Coords3d(self.x - other[0], self.y - other[1], self.z - other[2])
        else:
            raise ValueError("Undefined operation for given operand!")

    def __truediv__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x / other, self.y / other, self.z / other)
        else:
            raise ValueError("Undefined operation for given operand!")

    def __rtruediv__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x / other, self.y / other, self.z / other)
        else:
            raise ValueError("Undefined operation for given operand!")

    def __mul__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError("Undefined operation for given operand!")

    def __rmul__(self, other):
        if isinstance(other, Coords3d):
            return Coords3d(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, int) or isinstance(other, float):
            return Coords3d(self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError("Undefined operation for given operand!")

    def get_distance_to(self, other_coords, flag_2d=False):

        if isinstance(other_coords, Coords3d):
            return np.sqrt(((self.x - other_coords.x) ** 2 + (self.y - other_coords.y) ** 2
                            + ((self.z - other_coords.z) ** 2 if not flag_2d else 0)))
        elif isinstance(other_coords, tuple) or isinstance(other_coords, list) or isinstance(other_coords, np.ndarray):
            squared_sum = (other_coords[0] - self.x) ** 2 + (other_coords[1] - self.y) ** 2
            if len(other_coords) > 2 and not flag_2d:
                squared_sum += (other_coords[2] - self.z) ** 2
            return np.sqrt(squared_sum)
        else:
            raise ValueError('Unidentified input format!')

    def copy(self):
        return Coords3d(self.x, self.y, self.z)

    def np_array(self):
        return np.asarray((self.x, self.y, self.z))

    def as_2d_array(self):
        return np.asarray((self.x, self.y))

    def __array__(self, dtype=None):
        if dtype is None:
            return np.asarray((self.x, self.y, self.z))
        elif dtype == Coords3d:
            # arr = np.empty(1, dtype=Coords3d)
            # arr[0] = self
            return self

    def __len__(self):
        return 3

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        else:
            raise ValueError("Out of bounds!")

    @staticmethod
    def from_array(array):
        return Coords3d(array[0], array[1], array[2])

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def norm(self):
        return np.sqrt(self.x**2 + self.y ** 2 + self.z ** 2)

    def update(self, destination, distance):
        """Update coordinates towards the destination (x,y) with given distance.
            Return True if arrived to destination """
        if self == destination:
            return True

        # coord_array = self.np_array()
        # direction = - coord_array + destination.np_array()
        direction = destination - self
        # from time import time
        # time1= time()
        # for i in range(100000):
        # remaining_distance = np.linalg.norm(direction)
        remaining_distance = direction.norm()
        # assert(remaining_distance == np.linalg.norm(direction))
        # time2 = time()
        # print(time2 - time1)
        direction = direction / remaining_distance
        if remaining_distance <= distance:
            self.set(destination)
            return True
        else:
            self.set(self + direction * distance)
            # self.set(coord_array)
            return False

    def update_coords_from_array(self, np_array):
        self.x = np_array[0]
        self.y = np_array[1]
        self.z = np_array[2]


def to_coords_3d(_array):
    return Coords3d(_array[0], _array[1], _array[2] if len(_array) > 2 else 0)


@dataclass
class Obstacle:
    __slots__ = ["id", "height", "vertices"]
    id: int
    height: float
    vertices: list

    def get_adjacent_vertices(self, _vertex):
        idx = self.vertices.index(_vertex)
        return [self.vertices[(idx + 1) % len(self.vertices)],
                self.vertices[(idx - 1 + len(self.vertices)) % len(self.vertices)]]

    def is_overlapping(self, x, y, ref_height=0):
        if ref_height > self.height:
            return False
        point = Point(x, y)
        polygon = Polygon(self.vertices)
        return polygon.contains(point)

    def is_intersecting(self, src_coords, dest_coords):
        # TODO: not tested
        def on_segment(p, q, r):
            if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
                    (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
                return True
            return False

        def orientation(p, q, r):
            # to find the orientation of an ordered triplet (p,q,r)
            # function returns the following values:
            # 0 : Collinear points
            # 1 : Clockwise points
            # 2 : Counterclockwise
            val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
            if val > 0:
                # Clockwise orientation
                return 1
            elif val < 0:
                # Counterclockwise orientation
                return 2
            else:
                # Collinear orientation
                return 0

        def do_intersect(p1, q1, p2, q2):
            # Find the 4 orientations required for
            # the general and special cases
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)
            # General case
            if (o1 != o2) and (o3 != o4):
                return True
            # Special Cases
            # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
            if (o1 == 0) and on_segment(p1, p2, q1):
                return True
            # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
            if (o2 == 0) and on_segment(p1, q2, q1):
                return True
            # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
            if (o3 == 0) and on_segment(p2, p1, q2):
                return True
            # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
            if (o4 == 0) and on_segment(p2, q1, q2):
                return True
            # If none of the cases
            return False

        for p1, p2 in zip(self.vertices, rotate(self.vertices, 1)):
            if do_intersect(Coords3d(p1[0], p1[1], 0), Coords3d(p2[0], p2[1], 0), src_coords, dest_coords):
                return True
        return False

    def is_blocking(self, src_x, src_y, azimuth_to_dest, elevation_to_dest, reference_height=0, max_distance=None):
        if elevation_to_dest < 0:
            return True
        ##########
        shadow_length = (self.height - reference_height) / np.tan(np.deg2rad(elevation_to_dest))
        p0p = line_point_angle(shadow_length, self.vertices[0], angle_x=azimuth_to_dest + 180, angle_y=None,
                               rounding=False)
        p1p = line_point_angle(shadow_length, self.vertices[1], angle_x=azimuth_to_dest + 180, angle_y=None,
                               rounding=False)
        p2p = line_point_angle(shadow_length, self.vertices[2], angle_x=azimuth_to_dest + 180, angle_y=None,
                               rounding=False)
        p3p = line_point_angle(shadow_length, self.vertices[3], angle_x=azimuth_to_dest + 180, angle_y=None,
                               rounding=False)
        projections = [p0p, p1p, p2p, p3p]
        for p1, p2, p1p, p2p in zip(self.vertices, rotate(self.vertices, 1), projections, rotate(projections, 1)):
            if point_inside_prlgm(src_x, src_y, [p1, p1p, p2p, p2]):
                return True

        return False


if __name__ == '__main__':
    a = Coords3d(1, 2, 3)
    b = Coords3d(1.5, 2.5, 3.5)
    an = np.array([1, 2, 3])
    bn = np.array([1.5, 2.5, 3])
    print(np.linalg.norm(an - bn), a.get_distance_to(b, flag_2d=True), distance.euclidean(an, bn))
    setup = '''
from src.data_structures import Coords3d
a = Coords3d(1, 2, 3)
b = Coords3d(1.5, 2.5, 3.5)
'''
    print(min(timeit.Timer('a.get_distance_to(b)', setup=setup).repeat(7, 1000)))
    setup = '''
import numpy as np
from scipy.spatial import distance
an = np.array([1, 2, 3])
bn = np.array([1.5, 2.5, 3.5])
'''
    print(min(timeit.Timer('np.linalg.norm(an-bn)', setup=setup).repeat(7, 1000)))
    print(min(timeit.Timer('distance.euclidean(an, bn)', setup=setup).repeat(7, 1000)))
