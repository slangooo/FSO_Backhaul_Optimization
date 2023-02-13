#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

import random
import numpy as np
from numpy import degrees, arctan2


def decision(probability):
    return random.random() < probability


def distance_to_line(p, p1, p2):
    return np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)


def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))  # edit
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return tuple(ans)


def lin2db(linear_input):
    return 10 * np.log10(linear_input)


def db2lin(db_input):
    return 10 ** (db_input / 10)


def numpy_object_array(_object, dtype=None):
    """1D for now only"""
    obj_len = len(_object)
    object_type = type(_object[0]) if obj_len > 1 else type(_object)
    arr = np.empty(obj_len, dtype=object_type)
    if obj_len < 2:
        arr[0] = _object
        return arr
    else:
        for i in range(obj_len):
            arr[i] = _object[i]
        return arr


def rotate(l, n):
    return l[n:] + l[:n]


def line_point_angle(length, point, angle_x=None, angle_y=None, rounding=True):
    """
    Get endpoint from starting point, distance, and azimuth

    Parameters
    ----------
    angle_y : angle in degrees from y axis
    angle_x : angle in degrees from x axis
    """
    dest = [0, 0]
    if angle_x is not None:
        if rounding:
            dest[0] = point[0] + length * np.round(np.cos(angle_x / 180 * np.pi), decimals=5)
            dest[1] = point[1] + length * np.round(np.sin(angle_x / 180 * np.pi), decimals=5)
        else:
            dest[0] = point[0] + length * np.cos(angle_x / 180 * np.pi)
            dest[1] = point[1] + length * np.sin(angle_x / 180 * np.pi)
        return dest

    elif angle_y is not None:
        dest[0] = point[0] + length * np.round(np.sin(angle_x / 180 * np.pi), decimals=5)
        dest[1] = point[1] + length * np.round(np.cos(angle_x / 180 * np.pi), decimals=5)
        return dest
    else:
        raise ValueError('No angle was provided!')


def point_inside_prlgm(x, y, poly):
    inside = False
    xb = poly[0][0] - poly[1][0]
    yb = poly[0][1] - poly[1][1]
    xc = poly[2][0] - poly[1][0]
    yc = poly[2][1] - poly[1][1]
    xp = x - poly[1][0]
    yp = y - poly[1][1]
    d = xb * yc - yb * xc
    if (d != 0):
        oned = 1.0 / d
        bb = (xp * yc - xc * yp) * oned
        cc = (xb * yp - xp * yb) * oned
        inside = (bb >= 0) & (cc >= 0) & (bb <= 1) & (cc <= 1)
    return inside


def get_mid_azimuth(p_origin, p1, p2):
    azim1 = get_azimuth(p_origin[0], p_origin[1], p1[0], p1[1])
    azim2 = get_azimuth(p_origin[0], p_origin[1], p2[0], p2[1])
    azim11 = angle_in_range(azim1 + 180, 360)
    azim22 = angle_in_range(azim2 + 180, 360)
    if angle_in_range(azim2 - azim1, 360) >= 180:
        return angle_in_range(angle_in_range(azim2 - azim1, 360) / 2 + azim1, 360), azim22, azim11
    elif angle_in_range(azim1 - azim2, 360) >= 180:
        return angle_in_range(angle_in_range(azim1 - azim2, 360) / 2 + azim2, 360), azim11, azim22
    else:
        raise ValueError("Ensure that polygon is convex!")


def get_azimuth(center_x, center_y, x, y):
    angle = degrees(arctan2(y - center_y, x - center_x))
    bearing = (angle + 360) % 360
    return bearing


def angle_in_range(angle, range):
    """Limit angle to given range."""
    angle = angle % range
    return (angle + range) % range


def wh_to_joules(wh):
    return wh * 3600


def joules_to_wh(joules):
    return joules / 3600


def newton_raphson(f, fderivative, variable, initial_guess=1, max_error=1e-15, max_iter=1000):
    xn = initial_guess
    error = 10
    iter = 0
    while error > max_error and iter < max_iter:
        xn = xn - float(f.evalf(subs={variable: xn})) / float(fderivative.evalf(subs={variable: xn}))
        error = float(f.evalf(subs={variable: xn}))
        iter+=1

    return xn, float(f.evalf(subs={variable: xn}))