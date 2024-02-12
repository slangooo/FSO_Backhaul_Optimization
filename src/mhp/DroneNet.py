'''
@author: Piotr Wawrzyniak
'''

import numpy
import random
import math
import networkx
import matplotlib
import json
from time import perf_counter_ns

from src.mhp.pygad import GA
from src.mhp.ExactSolution import ExactSolution

class DroneNet:
    
    CYCLE_PENALTY = 100
    
    def __init__(self, drone_number, base_number):
        self.drone_number = drone_number
        self.base_number = base_number
        
        self.bandwidth = numpy.array([0] * drone_number)
        self.net = numpy.zeros((self.node_number(), self.node_number()), dtype=int)
        
        self.pos = []
        for i in range(self.node_number()):
            self.pos.append(numpy.zeros(2, dtype=int))
            
        self.labels = []
        self.longlabels = []
        for i in range(drone_number):
            self.labels    .append(      str(i+1))
            self.longlabels.append("(" + str(i+1) + ")")
        base_label = "A"
        base_label_ord = ord(base_label[0])
        for i in range(base_number):
            self.labels    .append(      chr(base_label_ord+i))
            self.longlabels.append("(" + chr(base_label_ord+i) + ")")
            
    def to_json(self):
        return {
            'drone_number': self.drone_number,
            'base_number': self.base_number,
            'bandwidth': self.bandwidth.tolist(),
            'net': self.net.tolist(),
            'pos': self.pos,
            'labels': self.labels,
            'longlabels': self.longlabels
        }

    @classmethod
    def from_json(cls, data):
        drone_net = cls(data['drone_number'], data['base_number'])
        drone_net.bandwidth = numpy.array(data['bandwidth'])
        drone_net.net = numpy.array(data['net'])
        drone_net.pos = data['pos']
        drone_net.labels = data['labels']
        drone_net.longlabels = data['longlabels']
        return drone_net
    
    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_json(), f)
    
    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            return DroneNet.from_json(json.load(f))
        
    def node_number(self):
        return self.drone_number + self.base_number
    
    def edge_number(self):
        edge_number = 0
        for i in range(self.node_number()):
            for j in range(i):
                if self.net[i][j] > 0:
                    edge_number += 1
        return edge_number
    
    def calc_bases_bandwidth(self):
        bases_bandwidth_max   = 0
        bases_bandwidth_total = 0
        for base in range(self.drone_number, self.drone_number + self.base_number):
            max_base_bandwidth = 0
            for drone in range(self.drone_number):
                max_base_bandwidth = max(max_base_bandwidth, self.net[base][drone])
            bases_bandwidth_max = max(bases_bandwidth_max, max_base_bandwidth)
            bases_bandwidth_total += max_base_bandwidth
        return (bases_bandwidth_max, bases_bandwidth_total)

    def max_bases_bandwidth(self):
        return self.calc_bases_bandwidth()[0]

    # Square area with four single base stations at corners
    # size                         : square area dimension
    # base_number                  : number of base stations FSO links
    # drone_number                 : number of drones in area
    # max_bandwidth                : max of random bandwidth delivered by drone
    # net_bandwidth_percentage     : max of random bandwidth between drones (and also to four base stations),
    #                                  % of drone_number * max_bandwidth/2 (=avg_bandwidth) / base_number
    # max_range_percentage         : max range for connection between drones
    #                                  % of size
    # min_net_bandwidth_percentage : percentage for minimal net bandwidth between nodes at maximum range
    #                                  % of max_bandwidth
    # topology                     : default 'corners', other options 'center'
    @staticmethod
    def createRandomAreaRelative(
            size, base_number, drone_number,
            max_bandwidth,
            net_bandwidth_percentage,
            max_range_percentage,
            min_net_bandwidth_percentage = 25,
            topology = 'corners'):
        net_bandwidth = int(drone_number * max_bandwidth * net_bandwidth_percentage / (2 * base_number * 100))
        max_range = int(size * max_range_percentage / 100)
        
        return DroneNet.createRandomArea(
            size, base_number, drone_number, max_bandwidth,
            net_bandwidth, max_range,
            min_net_bandwidth_percentage,
            topology)
        
    @staticmethod
    def randomBandwidth(net_bandwidth, min_net_bandwidth_percentage, distance, max_range):
        if distance > max_range:
            return 0
        else:
            min_net_bandwidth = int(net_bandwidth * min_net_bandwidth_percentage / 100)
            rst_net_bandwidth = net_bandwidth - min_net_bandwidth
            distance_bandwidth = int(rst_net_bandwidth * (max_range - distance) / max_range)
            random_bandwidth = min_net_bandwidth + random.randint(0, distance_bandwidth)
            return random_bandwidth
    
    # Square area with four single base stations at corners
    # size                         : square area dimension
    # base_number                  : number of base stations FSO links
    # drone_number                 : number of drones in area
    # min_bandwidth                : min of random bandwidth delivered by drone
    # max_bandwidth                : max of random bandwidth delivered by drone
    # net_bandwidth                : max of random bandwidth between drones (and also to base stations)
    # max_range                    : max range for connection between drones
    # min_net_bandwidth_percentage : percentage for minimal net bandwidth between nodes at maximum range
    # topology                     : default 'corners', other options 'center'
    @staticmethod
    def createRandomArea(
            size, base_number, drone_number,
            min_bandwidth, max_bandwidth,
            net_bandwidth,
            max_range,
            min_net_bandwidth_percentage = 25,
            topology = 'corners'):
        dn = DroneNet(drone_number, base_number)
        
        for i in range(drone_number):
            dn.bandwidth [i] = random.randint(min_bandwidth, max_bandwidth)
            dn.longlabels[i] = str(dn.bandwidth[i]) + " (" + str(i+1) + ")"
        
        for i in range(drone_number):
            repeat = True
            while repeat:
                repeat = False
                
                dn.pos[i] = [random.randint(1, size-1), random.randint(1, size-1)]
                
                for j in range(0,i):
                    if (dn.pos[i][0] == dn.pos[j][0]) and (dn.pos[i][1] == dn.pos[j][1]):
                        repeat = True
                        break
        
        if topology == 'corners':
            corners = [[   0,    0],
                       [size, size],
                       [size,    0],
                       [   0, size]]
            for i in range(base_number):
                dn.pos[drone_number + i] = corners[i % 4]
        elif topology == 'center':
            for i in range(base_number):
                dn.pos[drone_number + i] = [size/2, size/2]
        else:
            for i in range(base_number):
                dn.pos[drone_number + i] = [random.randint(0, size), random.randint(0, size)]
        
        for i in range(drone_number + base_number):
            for j in (range(i) if i < drone_number else range(drone_number)):
                distance = math.dist(dn.pos[i], dn.pos[j])
                net = DroneNet.randomBandwidth(net_bandwidth, min_net_bandwidth_percentage, distance, max_range)
                dn.net[i][j] = net
                dn.net[j][i] = net
                
        return dn
    
    # mbs_list                  : Main base stations list
    # dbs_list                  : Drone base station list
    # required_capacity_per_dbs : Required capacity per drone base station
    # fso_links_capacities      : FSO links capacities
    @staticmethod
    def createArea(mbs_list, dbs_list, required_capacity_per_dbs, fso_links_capacities):
        drone_number = len(dbs_list)
        base_number  = len(mbs_list)
        dn = DroneNet(drone_number, base_number)
        
        for i in range(base_number):
            i_mod = drone_number + i
            dn.labels[i_mod] = "B " + str(mbs_list[i].id)
            dn.longlabels[i_mod] = "Base[" + str(mbs_list[i].id) + "]"
            dn.pos[i_mod] = [mbs_list[i].coords.x, mbs_list[i].coords.y]
        
        for i in range(drone_number):
            dn.labels[i] = "D " + str(dbs_list[i].id)
            dn.bandwidth[i] = required_capacity_per_dbs[i] / 1e6
            dn.longlabels[i] = "D[" + str(dbs_list[i].id) + "] +" + str(dn.bandwidth[i])
            dn.pos[i] = [dbs_list[i].coords.x, dbs_list[i].coords.y]
        
        for i in range(base_number + drone_number):
            for j in range(i):
                i_mod = i + drone_number if i < base_number else i - base_number
                j_mod = j + drone_number if j < base_number else j - base_number
                cap = fso_links_capacities[i,j]
                dn.net[i_mod][j_mod] = cap
                dn.net[j_mod][i_mod] = cap
                
        return dn
    
    @staticmethod
    def createAndAdd(o, size, total_bandwidth, net_bandwidth, min_net_bandwidth_percentage, max_range):
        drone_number = o.drone_number + 1
        base_number  = o.base_number
        n = DroneNet(drone_number, base_number)
        
        drone_bandwidth = int(total_bandwidth / drone_number)
        
        for i in range(o.base_number):
            i_mod_n = n.drone_number + i
            i_mod_o = o.drone_number + i
            n.labels    [i_mod_n] = o.labels    [i_mod_o]
            n.longlabels[i_mod_n] = o.longlabels[i_mod_o]
            n.pos       [i_mod_n] = o.pos       [i_mod_o]
        
        for i in range(drone_number):
            n.labels    [i] = o.labels    [i]
            n.bandwidth [i] = drone_bandwidth
            n.longlabels[i] = str(drone_bandwidth) + " (" + str(i+1) + ")"
            n.pos       [i] = o.pos       [i]
        
        n.bandwidth[o.drone_number] = drone_bandwidth
        n.pos      [o.drone_number] = [random.randint(1, size-1), random.randint(1, size-1)]
        
        for i in range(o.drone_number):
            for j in range(o.drone_number):
                n.net[i][j] = o.net[i][j]
            for j in range(o.base_number):
                n.net[                 i][o.drone_number+1+j] = o.net[               i][o.drone_number+j]
                n.net[o.drone_number+1+j][                 i] = o.net[o.drone_number+j][               i]
        
        for i in range(o.base_number):
            for j in range(o.base_number):
                n.net[o.drone_number+1+i][o.drone_number+1+j] = 0
        
        for i in range(drone_number-1):
            distance = math.dist(n.pos[i], n.pos[drone_number-1])
            net = DroneNet.randomBandwidth(net_bandwidth, min_net_bandwidth_percentage, distance, max_range)
            n.net[             i][drone_number-1] = net
            n.net[drone_number-1][             i] = net
        
        n.net[drone_number-1][drone_number-1] = 0
        
        for i in range(drone_number, n.node_number()):
            distance = math.dist(n.pos[i], n.pos[drone_number-1])
            net = DroneNet.randomBandwidth(net_bandwidth, min_net_bandwidth_percentage, distance, max_range)
            n.net[             i][drone_number-1] = net
            n.net[drone_number-1][             i] = net
            
        return n
    
    def lookForRandomChainSolution(self, stop_on_first = True, time_limit_s = 0, instance_limit = 0, fitness_mode = 'nodeSurplus', penalty_mode = 'penalty', penalty_value = 0):
        if time_limit_s == 0 and instance_limit == 0:
            time_limit_s = 1
        time_limit = time_limit_s * (10**9)
        start_time = perf_counter_ns()
        end_time = start_time + time_limit
        stop_type = 'time'
        results = []
        processed = 0
        while True:
            if time_limit and perf_counter_ns() > end_time:
                stop_type = 'time'
                break;
            if instance_limit and processed > instance_limit:
                stop_type = 'iter'
                break;
            solution = self.randomSolution(allow_missing_edges = False)
            if solution:
                score = self.fitnessChain(solution, 0, fitness_mode = fitness_mode, penalty_mode = penalty_mode, penalty_value = penalty_value)
                if score >= 0:
                    results.append((solution.copy(), score, perf_counter_ns() - start_time, processed))
                    if stop_on_first:
                        stop_type = 'first'
                        break
            processed += 1

        return ExactSolution(
            stopType             = stop_type,
            found                = len(results) > 0,
            firstTime            = results[0][2] if len(results) > 0 else -1,
            fullTime             = perf_counter_ns() - start_time,
            firstCorrectInstance = results[0][3] if len(results) > 0 else -1,
            processedInstances   = processed,
            results              = results,
            mode                 = 'RANDOM')
    
    def lookForChainSolution(self, stop_on_first = True, mode = 'nodeSurplus', time_limit_s = 0, instance_limit = 0):
        time_limit = time_limit_s * (10**9)
        start_time = perf_counter_ns()
        end_time = start_time + time_limit
        stop_type = 'end'
        results = []
        solutionNodes = [-1] * (self.node_number() - 1)
        solutionFlow  = [ 0] * (self.node_number() - 1)
        instance_count = 0
        for node in range(self.drone_number + 1):
            if instance_limit and instance_count > instance_limit:
                stop_type = 'iter'
                break
            if time_limit and perf_counter_ns() > end_time:
                stop_type = 'time'
                break
            solutionNodes[0] = node
            stop_type, force_stop, instance_count = self.recursiveLookForChainSolution(
                    1, solutionNodes, solutionFlow, results,
                    self.drone_number + 1 if node == self.drone_number else self.drone_number,
                    stop_on_first, start_time, time_limit, instance_count, instance_limit, mode)
            if force_stop:
                break
        
        stop_time = perf_counter_ns()
        
        return ExactSolution(
            stopType             = stop_type,
            found                = len(results) > 0,
            firstTime            = results[0][2] if len(results) > 0 else -1,
            fullTime             = stop_time - start_time,
            firstCorrectInstance = results[0][3] if len(results) > 0 else -1,
            processedInstances   = instance_count,
            results              = results,
            mode                 = 'EXACT')
    
    def recursiveLookForChainSolution(
            self, index, solutionNodes, solutionFlow, results,
            nextBase,
            stop_on_first, start_time, time_limit_s, instance_count, instance_limit,
            mode = 'nodeSurplus'):
        time_limit = time_limit_s * (10**9)
        start_time = perf_counter_ns()
        end_time = start_time + time_limit

        if index < (self.node_number() - 1):
            # if next base is the last one, process only drones
            for node in range(self.drone_number + 1 if nextBase < self.node_number()-1 else self.drone_number):
                instance_count += 1
                if instance_limit and instance_count > instance_limit:
                    return 'iter', True, instance_count
                if time_limit and perf_counter_ns() > end_time:
                    return 'time', True, instance_count
                if node == self.drone_number:
                    node = nextBase
                    nextBase += 1
                if node not in solutionNodes:
                    nextBandwidth = solutionFlow[index - 1] + self.bandwidth[solutionNodes[index - 1]] if solutionNodes[index-1] < self.drone_number else 0
                    if self.net[solutionNodes[index-1]][node] >= nextBandwidth:
                        solutionNodes[index] = node
                        solutionFlow[index] = nextBandwidth
                        stop_type, force_stop, instance_count = self.recursiveLookForChainSolution(
                                index+1, solutionNodes,
                                solutionFlow, results, nextBase,
                                stop_on_first, start_time, time_limit, instance_count, instance_limit,
                                mode)
                        if force_stop:
                            return stop_type, True, instance_count
            solutionNodes[index] = -1
        else:
            node = self.node_number() - 1
            nextBandwidth = solutionFlow[index-1] + self.bandwidth[solutionNodes[index-1]] if solutionNodes[index-1] < self.drone_number else 0
            if self.net[solutionNodes[index-1]][node] >= nextBandwidth:
                score = self.fitnessChain(
                    solutionNodes,
                    0,
                    mode)
                results.append((solutionNodes.copy(), score, perf_counter_ns() - start_time, instance_count))
                if stop_on_first:
                    return 'first', True, instance_count
                if instance_limit and len(results) > instance_limit:
                    return 'iter', True, instance_count
            instance_count += 1
            if instance_limit and instance_count > instance_limit:
                return 'iter', True, instance_count

        return 'end', False, instance_count
    
    def solutionNodeToSolutionChain(self, solutionNodes):
        solutionDic = [-1] * (self.node_number())
        for nodeIndex, node in enumerate(solutionNodes):
            solutionDic[node] = nodeIndex
            
        solutionChain = [-1] * (self.node_number() - 1)
        base = self.node_number() - 1
        node = base
        for index in range(self.node_number() - 2, -1):
            if node in solutionDic:
                node = solutionDic[node]
            else:
                base -= 1
                node = base
            solutionChain[index] = node
        
        return solutionChain
        
    def fitnessNode(self, solution, solution_idx):  # @UnusedVariable
        result = 0
        a = self.net.copy()
        for drone in range(self.drone_number):
            u = [False] * self.drone_number
            u[drone] = True
            prevDrone = drone
            nextDrone = solution[drone]
            while True:
                if nextDrone < self.drone_number:
                    if u[nextDrone]:
                        result -= self.CYCLE_PENALTY
                        break
                    u[nextDrone] = True
                a[nextDrone][prevDrone] -= self.bandwidth[drone]
                a[prevDrone][nextDrone] -= self.bandwidth[drone]
                if nextDrone >= self.drone_number:
                    break
                prevDrone = nextDrone
                nextDrone = solution[nextDrone]
                
        for row in a:
            for cell in row:
                if cell < 0:
                    result += cell
                    
        return 1 if result == 0 else int(result/2)
    
    # fitness_mode :
    #   any         - 0 for all correct, -1 if incorrect
    #   edgeSurplus - sum of surplus on edges
    #   nodeSurplus - sum of surplus on nodes
    # penalty_mode :
    #   same    - same as fitness
    #   penalty - sum of all deficits in edge flows
    # penalty_value : value to decrease fitness value if result is incorrect
    def fitnessChain(self, solution, solution_idx, fitness_mode = 'any', penalty_mode = 'penalty', penalty_value = 0):  # @UnusedVariable
        resultPenalty = 0
        resultAwardEdges = 0
        resultAwardNodes = 0
        sumFlow = 0
        
        if penalty_mode == 'same': penalty_mode = fitness_mode

        solutionFlow = []
        for index in range (1, self.node_number() + 1):
            if index < self.node_number():
                prev = solution[index-1]
                node = solution[index] if index < len(solution) else index
            else:
                prev = self.drone_number
                node = self.drone_number

            if prev < self.drone_number:
                sumFlow += self.bandwidth[prev]
                edgeFlowSurplus = self.net[prev][node] - sumFlow
                solutionFlow.insert(0, edgeFlowSurplus)
                resultAwardEdges += edgeFlowSurplus
                if edgeFlowSurplus < 0:
                    resultPenalty += edgeFlowSurplus
            else:
                if fitness_mode == 'nodeSurplus':
                    solutionFlowLength = len(solutionFlow)
                    if solutionFlowLength > 0:
                        minFlow = solutionFlow[0]
                        for chainIndex in range(1, solutionFlowLength):
                            if solutionFlow[chainIndex] > minFlow:
                                solutionFlow[chainIndex] = minFlow
                            else:
                                minFlow = solutionFlow[chainIndex]
                        solutionFlowSum = sum(solutionFlow)
                        resultAwardNodes += solutionFlowSum
                        
                sumFlow = 0
                solutionFlow = []
                
        if resultPenalty < 0:
            if penalty_mode == 'edgeSurplus':
                result = resultAwardEdges
            elif penalty_mode == 'nodeSurplus':
                result = resultAwardNodes
            elif penalty_mode == 'penalty':
                result = resultPenalty
            else: # any
                result = -1
            return result - penalty_value
        else:
            if fitness_mode == 'edgeSurplus':
                return resultAwardEdges
            elif fitness_mode == 'nodeSurplus':
                return resultAwardNodes
            else: # any
                return 0
            
    def fillChainsFlows(self):
        self.chainsFlows = {}
        for baseIndex in range(self.drone_number, self.node_number()):
            solutionBandwidth = [0] * len(self.chainsOrder[baseIndex])
            for solutionBaseIndex, solutionBaseNode in enumerate(self.chainsOrder[baseIndex]):
                for solutionBaseSubIndex in range(solutionBaseIndex + 1):
                    solutionBandwidth[solutionBaseSubIndex] += self.bandwidth[solutionBaseNode]
                    
            self.chainsFlows[baseIndex] = solutionBandwidth
    
    def randomSolutions(self, size, mode = 'random', allow_missing_edges = True):
        solutions = []
        for sol_i in range(size):
            if mode == 'random':
                solution = self.randomSolution()
            else:
                solution = self.pathSolution(allow_missing_edges = allow_missing_edges)
            if solution:
                solutions.append(solution)
        
        return solutions

    def randomSolution(self):
        solution = list(range(self.drone_number))
        numpy.random.shuffle(solution)
        
        indices = numpy.random.choice(self.drone_number, self.base_number - 1, replace=False)
        indices.sort()
        for i, idx in enumerate(indices):
            solution.insert(idx + i, self.drone_number + i)
        
        return solution
    
    def pathSolution(self, allow_missing_edges = True):
        # Generate random solution by edges
        net_adjacency_list = []
        num_vertices = len(self.net)

        for i in range(num_vertices):
            neighbors = []
            for j in range(num_vertices):
                if self.net[i][j] > 0:
                    neighbors.append(j)
            net_adjacency_list.append(neighbors)

        nodes = list(range(self.drone_number))

        bases = []
        for base in range(self.base_number):
            base_node = self.drone_number + base
            bases.append([base_node])
            for net_adjacency in net_adjacency_list:
                if base_node in net_adjacency:
                    net_adjacency.remove(base_node)
        
        while nodes:
            out_of_edges_add_need = True
            for baseList in bases:
                last = baseList[-1]
                if net_adjacency_list[last]:
                    random_next_node = random.choice(net_adjacency_list[last])
                    baseList.append(random_next_node)
                    for net_adjacency in net_adjacency_list:
                        if random_next_node in net_adjacency:
                            net_adjacency.remove(random_next_node)
                    nodes.remove(random_next_node)
                    out_of_edges_add_need = False
            if out_of_edges_add_need:
                if allow_missing_edges:
                    baseList = random.choice(bases)
                    random_next_node = random.choice(nodes)
                    baseList.append(random_next_node)
                    for net_adjacency in net_adjacency_list:
                        if random_next_node in net_adjacency:
                            net_adjacency.remove(random_next_node)
                    nodes.remove(random_next_node)
                else:
                    return None
        
        solution = []
        for baseList in bases:
            while baseList:
                solution.append(baseList.pop())
        solution.pop()

        return solution

    # [5 20 0 8 7 A    18 3 13 B    1 2 17 C    19 9 15 6 10 11] (D)
    # A :  5 20  0  8  7
    # B : 18  3 13
    # C :  1  2 17
    # D : 19  9 15  6 10 11
    
    # chainFix: best random
    def singleChainCrossover(self, parents, offspring_size, ga : GA, chainFix = 'best'):
        
        if ga.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=ga.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)
            
        parentCount = len(parents)
        
        for k in range(offspring_size[0]):
            parentA_idx = random.randrange(parentCount)
            parentB_idx = random.randrange(parentCount)
            while parentA_idx == parentB_idx:
                parentB_idx = random.randrange(parentCount)
            
            selectedParents = (parents[parentA_idx], parents[parentB_idx])
            chains  = ({}, {})
            for p in (0,1):
                parent = selectedParents[p]
                chain  = chains[p]
                base = len(parent)
                chain[base] = []
                for node in reversed(parent):
                    if node < self.drone_number:
                        chain[base].append(node)
                    else:
                        base = node
                        chain[base] = []
                        
            status = [0] * self.drone_number
            masterBase = random.randrange(self.drone_number, self.node_number())
            childChains = {}
            duplicated = 0
            for base in range(self.drone_number, self.node_number()):
                childChains[base] = chains[0 if base == masterBase else 1][base]
                for node in childChains[base]:
                    if status[node] == 0:
                        status[node] = base
                    else:
                        status[node] = -status[node] if base == masterBase else -base
                        duplicated += 1
                        
            notUsed = []
            for node in range(self.drone_number):
                if status[node] == 0:
                    notUsed.append(node)
            
            bases = [*range(self.drone_number, self.node_number())]
            bases.remove(masterBase)
            
            if chainFix == 'best':
                toRemove = []
                for base in bases:
                    chain = childChains[base]
                    for nodeIndex, node in enumerate(chain):
                        if status[node] < 0:
                            toRemove.append((base, node))
                for itemToRemove in toRemove:
                    chain = childChains[itemToRemove[0]]
                    chain.remove(itemToRemove[1])
                    
                while len(notUsed) > 0:
                    maxDiff = -1
                    maxAddN = -1
                    maxBase = -1
                    maxIndx = -1
                    for nodeToAdd in notUsed:
                        for base in bases:
                            chain = childChains[base]
                            prev = -1
                            for nodeIndex, node in enumerate(chain):
                                currFlow = 0 if prev == -1 else self.net[prev][node]
                                
                                prevFlow = -1 if prev == -1 else self.net[prev][nodeToAdd]
                                nextFlow = self.net[nodeToAdd][node]
                                
                                addNFlow = nextFlow if prevFlow == -1 else min(prevFlow, nextFlow)
                                
                                flowDiff = addNFlow - currFlow
                                if flowDiff > maxDiff:
                                    maxAddN = nodeToAdd
                                    maxBase = base
                                    maxIndx = nodeIndex
                                    maxDiff = flowDiff
                                
                                prev = node
                                
                            currFlow = 0 if prev == -1 else self.net[prev][base]
                            
                            prevFlow = -1 if prev == -1 else self.net[prev][base]
                            nextFlow = self.net[nodeToAdd][base]
                            
                            addNFlow = nextFlow if prevFlow == -1 else min(prevFlow, nextFlow)
                            
                            flowDiff = addNFlow - currFlow
                            if flowDiff > maxDiff:
                                maxAddN = nodeToAdd
                                maxBase = base
                                maxIndx = -1
                                maxDiff = flowDiff
                    notUsed.remove(maxAddN)
                    chain = childChains[maxBase]
                    if maxIndx == -1:
                        chain.append(maxAddN)
                    else:
                        chain.insert(maxIndx, maxAddN)
                        
            else:
                if len(notUsed) < duplicated:
                    for i in range(len(notUsed), duplicated):
                        notUsed.append(-1)
                random.shuffle(notUsed)
                for base in childChains:
                    chain = childChains[base]
                    if base != masterBase:
                        newChain = []
                        for nodeIndex, node in enumerate(chain):
                            if status[node] < 0:
                                node = notUsed.pop()
                            if node >= 0:
                                newChain.append(node)
                        childChains[base] = newChain
                if len(notUsed) > 0:
                    bases = [*range(self.drone_number, self.node_number())]
                    bases.remove(masterBase)
                    for node in notUsed:
                        base = random.choice(bases)
                        chain = childChains[base]
                        index = random.randint(0, len(chain))
                        if index < len(chain):
                            chain.insert(index, node)
                        else:
                            chain.append(node)
                            
            i = 0
            for base in range(self.drone_number, self.node_number()):
                for node in childChains[base]:
                    offspring[k][i] = node
                    i += 1
                if base < (self.node_number() - 1):
                    offspring[k][i] = base
                    i += 1
                    
        return offspring
    
    def _processNodeSolution(self, solution):
        solutionDic = {}
        for nodeIndex, node in enumerate(solution):
            solutionDic[node] = nodeIndex
            
        self.chainsOrder = {}
        for baseIndex in range(self.drone_number, self.node_number()):
            solutionBase = []
            solutionNode = baseIndex
            
            while solutionNode in solutionDic:
                solutionNode = solutionDic[solutionNode]
                solutionBase.append(solutionNode)
                
            self.chainsOrder[baseIndex] = solutionBase
            
        self.fillChainsFlows()
        
    def _printSolution(self, draw = False, drawFileName = None, fixedSize = False):
        for baseIndex in range(self.drone_number, self.node_number()):
            print(self.longlabels[baseIndex], end="")
            
            solutionBasePrevNode = baseIndex
            for solutionBaseIndex, solutionBaseNode in enumerate(self.chainsOrder[baseIndex]):
                print(" <--" + str(self.chainsFlows[baseIndex][solutionBaseIndex])
                      + "/" + str(self.net[solutionBasePrevNode][solutionBaseNode])
                      + "-- " + str(self.longlabels[solutionBaseNode]), end="")
                solutionBasePrevNode = solutionBaseNode
            print()

        if draw:
            self.draw(showSolution = True, fileName = drawFileName, fixedSize = fixedSize)
    
    def printNodeSolution(self, solution, draw = False, drawFileName = None, fixedSize = False):
        for nodeIndex in solution:
            print(self.labels[nodeIndex], end=" ")
        print()
        self._processNodeSolution(solution)
        self._printSolution(draw = draw, drawFileName = drawFileName, fixedSize = fixedSize)
        
    def _processChainSolution(self, solution):
        self.chainsOrder = {}
        baseOrder = []
        for node in solution:
            if node < self.drone_number:
                baseOrder.insert(0, node)
            else:
                self.chainsOrder[node] = baseOrder
                baseOrder = []
                
        self.chainsOrder[self.node_number() - 1] = baseOrder
        
        self.fillChainsFlows()
        
    def printChainSolution(self, solution, draw = False, drawFileName = None, fixedSize = False):
        self._processChainSolution(solution)
        self._printSolution(draw = draw, drawFileName = drawFileName, fixedSize = fixedSize)
    
    def printInfo(self, mode = 'full'):
        net_bandwidth = []
        for i in range(0, self.node_number()):
            for j in range(i+1, self.node_number()):
                if self.net[i][j] > 0:
                    net_bandwidth.append(self.net[i][j])
        node_distance = []
        node_range    = []
        for i in range(0, self.drone_number):
            for j in range(i+1, self.node_number()):
                distance = round(math.dist(self.pos[i], self.pos[j]))
                node_distance.append(distance)
                if self.net[i][j] > 0:
                    node_range.append(distance)
        if mode == 'full':
            print('= INPUT ==================================')
            print('number of base links       = ', self.base_number)
            print('drone number               = ', self.drone_number)
            print('node bandwidth min/avg/max = ', numpy.min(self.bandwidth), '/', round(numpy.average(self.bandwidth)), '/', numpy.max(self.bandwidth))
            if len(net_bandwidth):
                print('net  bandwidth min/avg/max = ', numpy.min(net_bandwidth ), '/', round(numpy.average(net_bandwidth )), '/', numpy.max(net_bandwidth ))
            else:
                print('no net connections')
            print('net  distance  min/avg/max = ', numpy.min(node_distance ), '/', round(numpy.average(node_distance )), '/', numpy.max(node_distance ))
            if len(node_range):
                print('net  range     min/avg/max = ', numpy.min(node_range    ), '/', round(numpy.average(node_range    )), '/', numpy.max(node_range    ))
            else:
                print('no net connections')
            print('------------------------------------------')
            print('node bandwidth', self.bandwidth)
            print('node position' , self.pos)
            print(self.net)
            
            print('==========================================')
        elif mode == 'csv':
            print(                    self.base_number   , end=';')
            print(                    self.drone_number  , end=';')
            print(      numpy.min    (self.bandwidth   ) , end=';')
            print(round(numpy.average(self.bandwidth   )), end=';')
            print(      numpy.max    (self.bandwidth   ) , end=';')
            print(      numpy.min    (net_bandwidth    ) , end=';')
            print(round(numpy.average(net_bandwidth    )), end=';')
            print(      numpy.max    (net_bandwidth    ) , end=';')
            print(      numpy.min    (node_distance    ) , end=';')
            print(round(numpy.average(node_distance    )), end=';')
            print(      numpy.max    (node_distance    ) , end=';')
            print(      numpy.min    (node_range       ) , end=';')
            print(round(numpy.average(node_range       )), end=';')
            print(      numpy.max    (node_range       ) , end=';')
            
    def drawSolution(self, solution, fileName = None, fixedSize = False):
        if solution.found:
            self._processChainSolution(solution.bestResults[0][0])
        self.draw(showSolution = solution.found, fileName = fileName, fixedSize = fixedSize)

    def draw(self, showSolution = False, fileName = None, fixedSize = False):
        #G = networkx.from_numpy_matrix(self.net)
        G = networkx.Graph(self.net)
        
        edgelist = list(G.edges())
        
        edge_labels = {}
        edge_width = list()
        edge_color = list()
        edge_to_index = {}
        
        maxEdgeValue = 1
        for edgeIndex, edge in enumerate(edgelist):
            maxEdgeValue = max(maxEdgeValue, self.net[edge[1]][edge[0]])
            
        for edgeIndex, edge in enumerate(edgelist):
            edge_to_index[(edge[0], edge[1])] = edgeIndex
            if not showSolution:
                edge_labels[(edge[0], edge[1])] = self.net[edge[1]][edge[0]]
            edge_width.append(1 if showSolution else 10*self.net[edge[1]][edge[0]]/maxEdgeValue)
            edge_color.append("#696969") #DimGray
            
        node_size  = numpy.empty(self.node_number())
        node_color = numpy.array(["#000000"] * self.node_number())
        dark_color = numpy.array(["#000000"] * self.node_number())
        edgecolors = numpy.array(["#000000"] * self.node_number())
        
        maxNodeValue = 1
        for i in range(self.drone_number):
            maxNodeValue = max(maxNodeValue, self.bandwidth[i])
        
        for i in range(self.drone_number):
            node_size [i] = 2500 if fixedSize else self.bandwidth[i] * 10000 / maxNodeValue
            node_color[i] = "#D2B48C" #Tan
            edgecolors[i] = "#888888"
        for i in range(self.base_number):
            node_size [self.drone_number + i] = 2500
            if i == 0:
                node_color[self.drone_number + i] = "#87CEEB" #SkyBlue
                dark_color[self.drone_number + i] = "#67AECB"
            elif i == 1:
                node_color[self.drone_number + i] = "#FF7F50" #Coral
                dark_color[self.drone_number + i] = "#DF5F30"
            elif i == 2:
                node_color[self.drone_number + i] = "#90EE90" #LightGreen
                dark_color[self.drone_number + i] = "#70CE70"
            elif i == 3:
                node_color[self.drone_number + i] = "#DDA0DD" #Plum
                dark_color[self.drone_number + i] = "#BD80BD"
            else:
                node_color[self.drone_number + i] = "#DAA520" #GoldenRod
                dark_color[self.drone_number + i] = "#BA8500"
                
        if showSolution:
            for baseIndex in range(self.drone_number, self.node_number()):
                baseColor = node_color[baseIndex]
                darkColor = dark_color[baseIndex]
                prevNodeIndex = baseIndex
                for chainIndex, nodeIndex in enumerate(self.chainsOrder[baseIndex]):
                    key = (prevNodeIndex, nodeIndex) if prevNodeIndex < nodeIndex else (nodeIndex, prevNodeIndex)
                    edgeIndex = edge_to_index[key]
                    
                    node_color[nodeIndex] = baseColor
                    edgecolors[nodeIndex] = darkColor
                    edge_color[edgeIndex] = baseColor
                    edge_width[edgeIndex] = 10 * self.net[prevNodeIndex][nodeIndex] / maxEdgeValue
                    
                    edge_labels[key] = str(self.chainsFlows[baseIndex][chainIndex]) + "/" + str(self.net[prevNodeIndex][nodeIndex])
                    
                    prevNodeIndex = nodeIndex
                    
        max_pos = 0
        for pos in self.pos:
            max_pos = max(max_pos, pos[0], pos[1])
        
        norm_pos = {}
        for i in range(self.node_number()):
            norm_pos[i] = numpy.zeros(2, dtype=int)
            norm_pos[i][0] = int(100 * self.pos[i][0] / max_pos)
            norm_pos[i][1] = int(100 * self.pos[i][1] / max_pos)
            
        fig, ax = matplotlib.pyplot.subplots(figsize=(20,20))  # @UnusedVariable
        networkx.draw_networkx_nodes(
            G,
            norm_pos,
            ax         = ax,
            node_size  = node_size,
            node_color = node_color,
            edgecolors = edgecolors)
        networkx.draw_networkx_labels(
            G,
            norm_pos,
            ax     = ax,
            labels = {i: self.longlabels[i] for i in range(len(self.longlabels))})
        networkx.draw_networkx_edges(
            G,
            norm_pos,
            ax         = ax,
            edgelist   = edgelist,
            edge_color = edge_color,
            width      = edge_width)
        networkx.draw_networkx_edge_labels(
            G,
            norm_pos,
            ax=ax,
            edge_labels=edge_labels)
        
        if fileName is not None:
            matplotlib.pyplot.savefig(fileName, dpi=300, bbox_inches='tight')
        
        matplotlib.pyplot.close()
