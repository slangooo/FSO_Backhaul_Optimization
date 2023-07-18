#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

# Copyright (C) Damian Eads, 2007-2008. New BSD License.

# hierarchy.py (derived from cluster.py, http://scipy-cluster.googlecode.com)
#
# Author: Damian Eads
# Date:   September 22, 2007
#
# Copyright (c) 2007, 2008, Damian Eads
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   - Redistributions of source code must retain the above
#     copyright notice, this list of conditions and the
#     following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer
#     in the documentation and/or other materials provided with the
#     distribution.
#   - Neither the name of the author nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from src.parameters import MAX_FSO_DISTANCE, MEAN_UES_PER_CLUSTER
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from dataclasses import dataclass, field
from scipy.cluster import hierarchy
import heapq
import warnings
from scipy.cluster.hierarchy import fcluster, cut_tree, dendrogram, to_tree
from src.users import User
from src.environment.user_modeling import ThomasClusterProcess


warnings.filterwarnings("error", category=RuntimeWarning)


def centroid_update(d_xi, d_yi, d_xy, nx, ny):
    # res = (d_xi * nx + d_yi * ny) / (nx + ny) - nx * ny / (nx + ny) ** 2 * d_xy
    res = np.sqrt((((nx * d_xi * d_xi) + (ny * d_yi * d_yi)) -
                   (nx * ny * d_xy * d_xy) / (nx + ny)) /
                  (nx + ny))
    return res


@dataclass(order=True)
class Pair:
    key: int = field(compare=False, init=True)
    value: int
    removed = False


def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return int(n * i - (i * (i + 1) / 2) + (j - i - 1))
    elif i > j:
        return int(n * j - (j * (j + 1) / 2) + (i - j - 1))


def find_min_dist(n, condensed_distance, cardinalities, x):
    current_min = np.inf
    y = -1
    for i in range(x + 1, n):
        if cardinalities[i] == 0:
            continue

        dist = condensed_distance[condensed_index(n, x, i)]
        if dist < current_min:
            current_min = dist
            y = i

    return Pair(y, current_min)


def perform_dbs_hc(users, mbs_locs, distance_threshold=MAX_FSO_DISTANCE, min_n_degrees=5):
    n_ues = len(users)
    n_mbs = len(mbs_locs)
    ues_mbs_locs = np.vstack([users[i].coords.as_2d_array() for i in range(n_ues)] + mbs_locs)
    ues_locs = np.vstack([users[i].coords.as_2d_array() for i in range(n_ues)])
    flat_p_0 = pdist(ues_locs).round(2)  # D

    flat_ue_mbs_dist = pdist(ues_mbs_locs).round(2)
    dist_flags = flat_ue_mbs_dist <= distance_threshold
    dist_flags_square = squareform(dist_flags)
    degrees_vec = np.sum(dist_flags_square, 1)
    old_degrees_vec = np.empty_like(degrees_vec)
    old_degrees_vec[:] = degrees_vec
    unsatisfied_idxs = np.argwhere(
        degrees_vec <= min_n_degrees + 1)  # +1 to account for possibility of having two clusters removed from neighborhood (x and y)
    clusters_lock = np.reshape(~dist_flags_square[unsatisfied_idxs].sum(0).astype(bool), ues_mbs_locs.shape[0])
    clusters_lock[unsatisfied_idxs] = False

    mbs_dists = np.vstack((np.array([euclidean(_ue_loc, mbs_locs[0]) for _ue_loc in ues_locs]),
                           np.array([euclidean(_ue_loc, mbs_locs[1]) for _ue_loc in ues_locs]))).T

    linkage_matrix = np.empty((n_ues - 1, 4), dtype=float)
    cardinalities = np.ones(n_ues, dtype=int)
    cluster_id = np.arange(n_ues, dtype=int)

    neighbor = np.empty(n_ues - 1, dtype=int)
    min_dist = np.empty(n_ues - 1)

    min_dist_heap = []
    min_dist_heap_2 = []
    heap_dict = {}

    def remove_heap_entry(key, dict_in):
        entry = dict_in.pop(key)
        entry.removed = True

    def replace_entry(pair, dict_in, heap_in):
        remove_heap_entry(pair.key, dict_in)
        entry = Pair(pair.key, pair.value)
        dict_in[pair.key] = entry
        heapq.heappush(heap_in, entry)

    for x in range(n_ues - 1):
        pair = find_min_dist(n_ues, flat_p_0, cardinalities, x)
        neighbor[x] = pair.key
        min_dist[x] = pair.value
        new_pair = Pair(x, pair.value)
        heap_dict[x] = new_pair
        if clusters_lock[x] and clusters_lock[pair.key]:
            heapq.heappush(min_dist_heap, new_pair)
        else:
            heapq.heappush(min_dist_heap_2, new_pair)
    detectd_first_breach = False
    for k in range(n_ues - 1):
        for i in range(n_ues - 1):
            heap_empty = False
            while 1:
                if len(min_dist_heap) == 0:
                    heap_empty = True
                    break
                pair = min_dist_heap[0]
                if pair.removed:
                    heapq.heappop(min_dist_heap)
                    continue
                else:
                    break
            if heap_empty:
                if not detectd_first_breach:
                    detectd_first_breach = n_ues - k + 1
                    print("min n_clusters_possible:", detectd_first_breach)
                pair = min_dist_heap_2[0]
                while pair.removed:
                    heapq.heappop(min_dist_heap_2)
                    pair = min_dist_heap_2[0]

            if (not clusters_lock[pair.key] or not clusters_lock[neighbor[pair.key]]) and not heap_empty and not detectd_first_breach:
                replace_entry(pair, heap_dict, min_dist_heap_2)
                heapq.heappop(min_dist_heap)
                continue

            x, dist = pair.key, pair.value
            y = neighbor[x]

            if dist == flat_p_0[condensed_index(n_ues, x, y)]:
                if not heap_empty:
                    heapq.heappop(min_dist_heap)
                else:
                    heapq.heappop(min_dist_heap_2)
                break

            pair = find_min_dist(n_ues, flat_p_0, cardinalities, x)
            y, dist = pair.key, pair.value
            neighbor[x] = y
            min_dist[x] = dist
            new_pair = Pair(x, pair.value)
            if not heap_empty:
                remove_heap_entry(x, heap_dict)
                heapq.heapreplace(min_dist_heap, new_pair)
                heap_dict[x] = new_pair
            else:
                heapq.heappop(min_dist_heap_2)
                replace_entry(new_pair, heap_dict, min_dist_heap)

        id_x = cluster_id[x]
        id_y = cluster_id[y]
        nx = cardinalities[x]
        ny = cardinalities[y]
        if id_x > id_y:
            id_x, id_y = id_y, id_x

        linkage_matrix[k, 0] = id_x
        linkage_matrix[k, 1] = id_y
        linkage_matrix[k, 2] = dist
        linkage_matrix[k, 3] = nx + ny

        cardinalities[x] = 0  # Cluster x will be dropped.

        old_mbs_dists = mbs_dists[y].copy()
        mbs_dists[y] = centroid_update(mbs_dists[x], mbs_dists[y], dist, nx, ny)
        degrees_vec[x] = min_n_degrees + 10
        old_degrees_vec[:] = degrees_vec
        degrees_vec[y] += (
                (mbs_dists[y] < distance_threshold).astype(int) - (old_mbs_dists < distance_threshold).astype(
            int)).sum()
        dist_flags_square[y, -n_mbs:] = dist_flags_square[-n_mbs:, y] = mbs_dists[y] <= distance_threshold
        dist_flags_square[:, x] = dist_flags_square[x, :] = False
        if dist <= distance_threshold:
            degrees_vec[y] -= 1

        cardinalities[y] = nx + ny  # Cluster y will be replaced with the new cluster.
        cluster_id[y] = n_ues + k

        # Update the distance matrix.
        for z in range(n_ues):
            nz = cardinalities[z]
            if nz == 0 or z == y:
                continue
            old_distance = flat_p_0[condensed_index(n_ues, z, y)]
            dist_zx = flat_p_0[condensed_index(n_ues, z, x)]
            new_distance = centroid_update(dist_zx
                                           , flat_p_0[condensed_index(n_ues, z, y)],
                                           dist, nx, ny)
            flat_p_0[condensed_index(n_ues, z, y)] = new_distance
            if dist_zx <= distance_threshold:
                degrees_vec[z] -= 1
            if old_distance <= distance_threshold < new_distance:
                degrees_vec[z] -= 1
                degrees_vec[y] -= 1
                dist_flags_square[z, y] = dist_flags_square[y, z] = False
            elif old_distance > distance_threshold >= new_distance:
                degrees_vec[z] += 1
                degrees_vec[y] += 1
                dist_flags_square[z, y] = dist_flags_square[y, z] = True

        for z in range(x):
            if cardinalities[z] > 0 and neighbor[z] == x:
                neighbor[z] = y
        # Update lower bounds of distance.
        for z in range(y):
            if cardinalities[z] == 0:
                continue
            dist = flat_p_0[condensed_index(n_ues, z, y)]
            if dist < min_dist[z]:
                min_dist[z] = dist
                replace_entry(Pair(z, dist), heap_dict, min_dist_heap)
                neighbor[z] = y

        if y < n_ues - 1:
            pair = find_min_dist(n_ues, flat_p_0, cardinalities, y)
            z, dist = pair.key, pair.value
            if z != -1:
                min_dist[y] = dist
                replace_entry(Pair(y, dist), heap_dict, min_dist_heap)
                neighbor[y] = z

        degree_switch_idxs = np.argwhere(np.logical_xor(old_degrees_vec > min_n_degrees, degrees_vec > min_n_degrees))

        # if np.any(degrees_vec < min_n_degrees):
        #     print("Detected", n_ues - k)
        #     if np.any(np.argwhere(degrees_vec < min_n_degrees) != y):
        #         assert detectd_first_breach

        # new_assert = (squareform(flat_p_0) <= distance_threshold)
        # new_assert[np.argwhere(cardinalities <= 0), :] = False
        # new_assert[:, np.argwhere(cardinalities <= 0)] = False
        # new_assert = new_assert.sum(0)
        # assert (np.all(
        #     (degrees_vec[np.argwhere(cardinalities > 0)] - 1 <= new_assert[np.argwhere(cardinalities > 0)])) and \
        #         np.all((degrees_vec[np.argwhere(cardinalities > 0)] >= new_assert[np.argwhere(cardinalities > 0)])))
        if degree_switch_idxs.size != 0 and not detectd_first_breach:
            unsatisfied_idxs = np.argwhere(degrees_vec <= min_n_degrees + 1)
            clusters_lock = np.reshape(~dist_flags_square[unsatisfied_idxs].sum(0).astype(bool), ues_mbs_locs.shape[0])
            clusters_lock[unsatisfied_idxs] = False
            for i in range(len(min_dist_heap_2) - 1, -1, -1):
                _item = min_dist_heap_2[i]
                if _item.removed:
                    min_dist_heap_2.pop(i)
                    continue
                elif clusters_lock[_item.key] and clusters_lock[neighbor[_item.key]]:
                    heapq.heappush(min_dist_heap, _item)
                    min_dist_heap_2.pop(i)
                else:
                    continue

    return linkage_matrix, detectd_first_breach


def extract_levels(row_clusters, labels):
    clusters = {}
    for row in range(row_clusters.shape[0]):
        cluster_n = row + len(labels)
        # which clusters / labels are present in this row
        glob1, glob2 = int(row_clusters[row, 0]), int(row_clusters[row, 1])

        # if this is a cluster, pull the cluster
        this_clust = []
        for glob in [glob1, glob2]:
            if glob > (len(labels) - 1):
                this_clust += clusters[glob]
            # if it isn't, add the label to this cluster
            else:
                this_clust.append(glob)

        clusters[cluster_n] = this_clust
    return clusters


def flatten_cluster_from_tree(cluster_node):
    leaves = []
    if cluster_node.count > 1:
        leaves += flatten_cluster_from_tree(cluster_node.get_left())
        leaves += flatten_cluster_from_tree(cluster_node.get_right())
    else:
        return [cluster_node.id]
    return leaves


def get_clustering(linkage_matrix, n_clusters):
    clusters = []
    _tree = to_tree(linkage_matrix, True)
    idx = -1
    idx_1, idx_2 = int(linkage_matrix[idx, 0]), int(linkage_matrix[idx, 1])
    clusters.append(idx_1)
    clusters.append(idx_2)
    while len(clusters) < n_clusters:
        idx -= 1
        for cl_idx, _cluster_idx in enumerate(clusters):
            if _tree[1][_cluster_idx].count == 1:
                continue
            if _tree[1][_cluster_idx].get_left().id in linkage_matrix[idx][0:2]:
                break
        clusters.pop(cl_idx)
        idx_1, idx_2 = int(linkage_matrix[idx, 0]), int(linkage_matrix[idx, 1])
        clusters.append(idx_1)
        clusters.append(idx_2)

    return [flatten_cluster_from_tree(_tree[1][_cluster_idx]) for _cluster_idx in clusters]


def get_centroids(n_clusters, linkage_matrix, ues_locs):
    N = n_clusters
    labels = np.empty(ues_locs.shape[0], dtype=int)
    res = get_clustering(linkage_matrix, N)
    for cluster_idx, children in enumerate(res):
        labels[children] = cluster_idx
    aa = labels
    n_clusters = len(np.unique(aa))
    centroids = np.zeros((n_clusters, 2))
    for i in range(0, n_clusters):  # Starts from 1
        centroids[i, :] = ues_locs[np.argwhere(aa == i)].mean(0)
    return centroids


if __name__ == '__main__':
    users_model = ThomasClusterProcess(MEAN_UES_PER_CLUSTER)
    users = [User(users_model.users[i])
                  for i in range(users_model.n_users)]
    mbs_locs = [(10, 10), (10000, 10000)]
    min_n_degrees = 1  # Number of required degrees (i.e., number of adjacent clusters within FSO distance).
    linkage_matrix, n_clusters_possible = perform_dbs_hc(users, mbs_locs, min_n_degrees=min_n_degrees)
    ues_locs = np.vstack([users[i].coords.as_2d_array() for i in range(users_model.n_users)])

    # Cluster so that max dist between nodes is N
    # max_dist = 50
    # aa = fcluster(linkage_matrix, max_dist, 'distance')

    # Cluster into N clusters
    # aa = fcluster(linkage_matrix, N, 'maxclust') - 1

    dists = pdist(np.append(get_centroids(n_clusters_possible, linkage_matrix, ues_locs), mbs_locs, 0))
    dist_flags = squareform(dists) <= MAX_FSO_DISTANCE
    np.fill_diagonal(dist_flags, False)
    # Make sure all have enough degrees
    print(np.all(dist_flags.sum(0)[:-2] >= min_n_degrees))
