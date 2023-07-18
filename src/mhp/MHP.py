'''
@author: Piotr Wawrzyniak
'''

import numpy
import random
import math
from time import perf_counter_ns
import networkx
import matplotlib

from src.mhp.pygad import GA

class DroneNet:
    
    CYCLE_PENALTY = 100
    
    def __init__(self, drone_number, base_number):
        self.drone_number = drone_number
        self.base_number = base_number
        
        self.bandwidth = numpy.array([0] * drone_number)
        self.net = numpy.zeros((self.node_number(), self.node_number()), dtype=int)
        
        self.pos = {}
        for i in range(self.node_number()):
            self.pos[i] = numpy.zeros(2, dtype=int)
        
        self.labels = {}
        self.longlabels = {}
        for i in range(drone_number):
            self.labels[i] = str(i + 1)
            self.longlabels[i] = "(" + str(i + 1) + ")"
        base_label = "A"
        base_label_ord = ord(base_label[0])
        for i in range(base_number):
            self.labels[drone_number + i] = chr(base_label_ord + i)
            self.longlabels[drone_number + i] = "(" + chr(base_label_ord + i) + ")"
            
    def node_number(self):
        return self.drone_number + self.base_number
    
    def edge_number(self):
        edge_number = 0
        for i in range(self.node_number()):
            for j in range(i):
                if self.net[i][j] > 0:
                    edge_number += 1
        return edge_number

    # Square area with four single base stations at corners
    # size                     : square area dimension
    # base_number              : number of base stations FSO links
    # drone_number             : number of drones in area
    # max_bandwidth            : max of random bandwidth delivered by drone
    # net_bandwidth_percentage : max of random bandwidth between drones (and also to four base stations),
    #                            % of : drone_number * max_bandwidth/2 (=avg_bandwidth) / base_number
    # max_range_percentage     : max range for connection between drones
    #                            % of : size
    # topology                 : default 'corners', other options 'center'
    @staticmethod
    def createRandomAreaRelative(size, base_number, drone_number, max_bandwidth, net_bandwidth_percentage, max_range_percentage, topology = 'corners'):
        net_bandwidth = int(drone_number * max_bandwidth * net_bandwidth_percentage / (2 * base_number * 100))
        max_range = int(size * max_range_percentage / 100)
        
        return DroneNet.createRandomArea(size, base_number, drone_number, max_bandwidth, net_bandwidth, max_range, topology)
        
    # Square area with four single base stations at corners
    # size          : square area dimension
    # base_number   : number of base stations FSO links
    # drone_number  : number of drones in area
    # max_bandwidth : max of random bandwidth delivered by drone
    # net_bandwidth : max of random bandwidth between drones (and also to four base stations)
    # max_range     : max range for connection between drones
    # topology      : default 'corners', other options 'center'
    @staticmethod
    def createRandomArea(size, base_number, drone_number, max_bandwidth, net_bandwidth, max_range, topology = 'corners'):
        dn = DroneNet(drone_number, base_number)
        
        for i in range(drone_number):
            dn.bandwidth [i] = random.randint(1, max_bandwidth)
            dn.longlabels[i] = str(dn.bandwidth[i]) + " (" + str(i+1) + ")"
        
        for i in range(drone_number):
            repeat = True
            while repeat:
                repeat = False
                
                dn.pos[i] = [random.randint(1, size-1), random.randint(1, size-1)]
                
                for j in range(0,i):
                    if (dn.pos[i][0] == dn.pos[j][0]) & (dn.pos[i][1] == dn.pos[j][1]):
                        repeat = True
                        break
        
        if (topology == 'corners'):
            corners = [[   0,    0],
                       [   0, size],
                       [size,    0],
                       [size, size]]
            for i in range(base_number):
                dn.pos[drone_number + i] = corners[i % 4]
        elif (topology == 'center'):
            for i in range(base_number):
                dn.pos[drone_number + i] = [size/2, size/2]
        else:
            for i in range(base_number):
                dn.pos[drone_number + i] = [random.randint(0, size), random.randint(0, size)]

        min_net_bandwidth_factor = 0.25

        min_net_bandwidth = int(min_net_bandwidth_factor * net_bandwidth)
        rst_net_bandwidth = net_bandwidth - min_net_bandwidth

        for i in range(drone_number):
            for j in range(i):
                distance = math.dist(dn.pos[i], dn.pos[j])
                net = 0 if distance > max_range else 2 * random.randint(min_net_bandwidth, min_net_bandwidth
                                                     + int(rst_net_bandwidth * (max_range - distance) / max_range))
                dn.net[i][j] = net
                dn.net[j][i] = net

        for i in range(drone_number, drone_number + base_number):
            for j in range(drone_number):
                distance = math.dist(dn.pos[i], dn.pos[j])
                net = 0 if distance > max_range else 2 * random.randint(min_net_bandwidth, min_net_bandwidth
                                                     + int(rst_net_bandwidth * (max_range - distance) / max_range))
                dn.net[i][j] = net
                dn.net[j][i] = net

        return dn

    # mbs_list : Main base stations list
    # dbs_list : Drone base station list
    # required_capacity_per_dbs : required capacity per drone base station
    # fso_links_capacs : FSO links capacities
    @staticmethod
    def createArea(mbs_list, dbs_list, required_capacity_per_dbs, fso_links_capacs):
        drone_number = len(dbs_list)
        base_number = len(mbs_list)
        dn = DroneNet(drone_number, base_number)

        for i in range(base_number):
            imod = drone_number + i
            dn.labels[imod] = "B " + str(mbs_list[i].id)
            dn.longlabels[imod] = "Base[" + str(mbs_list[i].id) + "]"
            dn.pos[imod] = [mbs_list[i].coords.x, mbs_list[i].coords.y]

        for i in range(drone_number):
            dn.labels[i] = "D " + str(dbs_list[i].id)
            dn.bandwidth[i] = required_capacity_per_dbs[i] / 1e6
            dn.longlabels[i] = "D[" + str(dbs_list[i].id) + "] +" + str(dn.bandwidth[i])
            dn.pos[i] = [dbs_list[i].coords.x, dbs_list[i].coords.y]

        for i in range(base_number + drone_number):
            for j in range(i):
                imod = i + drone_number if i < base_number else i - base_number
                jmod = j + drone_number if j < base_number else j - base_number
                cap = fso_links_capacs[i, j]
                dn.net[imod][jmod] = cap
                dn.net[jmod][imod] = cap

        return dn

    def lookForChainSolution(self, first=True, mode='any'):
        results = []
        start = perf_counter_ns()
        firstStop = 0
        solutionNode = [-1] * (self.node_number() - 1)
        solutionFlow = [0] * (self.node_number() - 1)
        for node in range(self.drone_number + 1):
            solutionNode[0] = node
            resultFound, firstStop = self.recursiveLookForChainSolution(
                    1, solutionNode, solutionFlow, results,
                    self.drone_number + 1 if node == self.drone_number else self.drone_number,
                    first, firstStop, mode)
            if resultFound:
                if first:
                    break

        stop = perf_counter_ns()
        return ExactSolution(self, len(results) > 0, firstStop - start, stop - start, results)

    def recursiveLookForChainSolution(
            self, index, solutionNode, solutionFlow, results, nextBase, first, firstStop, mode='bestNodes'):
        if index < (self.node_number() - 1):
            # if next base is the last one, process only drones
            for node in range(self.drone_number + 1 if nextBase < self.node_number() - 1 else self.drone_number):
                if node == self.drone_number:
                    node = nextBase
                    nextBase += 1
                if node not in solutionNode:
                    nextBandwidth = solutionFlow[index-1] + self.bandwidth[solutionNode[index-1]] if solutionNode[index-1] < self.drone_number else 0
                    if self.net[solutionNode[index - 1]][node] >= nextBandwidth:
                        solutionNode[index] = node
                        solutionFlow[index] = nextBandwidth
                        resultFound, firstStop = self.recursiveLookForChainSolution(
                                index + 1, solutionNode,
                                solutionFlow, results, nextBase,
                                first, firstStop, mode)
                        if resultFound:
                            return True, firstStop
            solutionNode[index] = -1
        else:
            node = self.node_number() - 1
            nextBandwidth = solutionFlow[index-1] + self.bandwidth[solutionNode[index-1]] if solutionNode[index-1] < self.drone_number else 0
            if self.net[solutionNode[index - 1]][node] >= nextBandwidth:
                results.append((solutionNode.copy(), self.fitnessChain(solutionNode, 0, mode)))
                if firstStop == 0:
                    firstStop = perf_counter_ns()
                if first:
                    return True, firstStop

        return False, firstStop

    def processNodeSolution(self, solution):
        soldic = {}
        for nodeIndex, node in enumerate(solution):
            soldic[node] = nodeIndex

        self.chainsOrder = {}
        for baseIndex in range(self.drone_number, self.node_number()):
            solbase = []
            solutionNode = baseIndex

            while solutionNode in soldic:
                solutionNode = soldic[solutionNode]
                solbase.append(solutionNode)

            self.chainsOrder[baseIndex] = solbase

        self.fillChainsFlows()

    def solutionNodeToSolutionChain(self, solutionNode):
        soldic = [-1] * (self.node_number())
        for nodeIndex, node in enumerate(solutionNode):
            soldic[node] = nodeIndex

        solutionChain = [-1] * (self.node_number() - 1)
        base = self.node_number() - 1
        node = base
        for index in range(self.node_number() - 2, -1):
            if node in soldic:
                node = soldic[node]
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
                    if (u[nextDrone]):
                        result -= self.CYCLE_PENALTY
                        break
                    u[nextDrone] = True
                a[nextDrone][prevDrone] -= self.bandwidth[drone]
                a[prevDrone][nextDrone] -= self.bandwidth[drone]
                if (nextDrone >= self.drone_number):
                    break
                prevDrone = nextDrone
                nextDrone = solution[nextDrone]

        for row in a:
            for cell in row:
                if (cell < 0):
                    result += cell

        return 1 if result == 0 else int(result / 2)

    # mode :
    #   any  - For all correct return 1,
    #   bestEdges - For correct return sum of surplus on edges
    #   bestNodes - For correct return sum of surplus on nodes
    def fitnessChain(self, solution, solution_idx, mode='any'):  # @UnusedVariable
        resultPenalty = 0
        resultAwardEdges = 0
        resultAwardNodes = 0
        sumFlow = 0

        solutionFlow = []
        for index in range(1, self.node_number() + 1):
            if (index < self.node_number()):
                prev = solution[index - 1]
                node = solution[index] if index < len(solution) else index
            else:
                prev = self.drone_number;
                node = self.drone_number;
            if prev < self.drone_number:
                sumFlow += self.bandwidth[prev]
                edgeFlowSurplus = self.net[prev][node] - sumFlow
                solutionFlow.insert(0, edgeFlowSurplus)
                resultAwardEdges += edgeFlowSurplus
                if edgeFlowSurplus < 0:
                    resultPenalty += edgeFlowSurplus
            else:
                if mode == 'bestNodes':
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
            return resultPenalty
        else:
            if mode == 'bestEdges':
                return resultAwardEdges
            elif mode == 'bestNodes':
                return resultAwardNodes
            else:
                return 1

    def processChainSolution(self, solution):
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

    def fillChainsFlows(self):
        self.chainsFlows = {}
        for baseIndex in range(self.drone_number, self.node_number()):
            solband = [0] * len(self.chainsOrder[baseIndex])
            for solbaseIndex, solbaseNode in enumerate(self.chainsOrder[baseIndex]):
                for solbaseSubIndex in range(solbaseIndex + 1):
                    solband[solbaseSubIndex] += self.bandwidth[solbaseNode]

            self.chainsFlows[baseIndex] = solband

    # [ 5 20  0  8  7 AAA 18  3 13 BBB  1  2 17 CCC 19  9 15  6 10 11]
    # A :  5 20  0  8  7
    # B : 18  3 13
    # C :  1  2 17
    # D : 19  9 15  6 10 11

    # chainFix: best random
    def singleChainCrossover(self, parents, offspring_size, ga: GA, chainFix='best'):

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
            chains = ({}, {})
            for p in (0, 1):
                parent = selectedParents[p]
                chain = chains[p]
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

            notused = []
            for node in range(self.drone_number):
                if status[node] == 0:
                    notused.append(node)

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

                while len(notused) > 0:
                    maxDiff = -1
                    maxAddN = -1
                    maxBase = -1
                    maxIndx = -1
                    for nodeToAdd in notused:
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
                    notused.remove(maxAddN)
                    chain = childChains[maxBase]
                    if maxIndx == -1:
                        chain.append(maxAddN)
                    else:
                        chain.insert(maxIndx, maxAddN)

            else:
                if len(notused) < duplicated:
                    for i in range(len(notused), duplicated):
                        notused.append(-1)
                random.shuffle(notused)
                for base in childChains:
                    chain = childChains[base]
                    if base != masterBase:
                        newChain = []
                        for nodeIndex, node in enumerate(chain):
                            if status[node] < 0:
                                node = notused.pop()
                            if node >= 0:
                                newChain.append(node)
                        childChains[base] = newChain
                if len(notused) > 0:
                    bases = [*range(self.drone_number, self.node_number())]
                    bases.remove(masterBase)
                    for node in notused:
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

    def printNodeSolution(self, solution):
        for nodeIndex in solution:
            print(self.labels[nodeIndex], end=" ")

        print()

        self.processNodeSolution(solution)

        self.printSolution()

    def printSolution(self):
        for baseIndex in range(self.drone_number, self.node_number()):
            print(self.longlabels[baseIndex], end="")

            solbasePrevNode = baseIndex
            for solbaseIndex, solbaseNode in enumerate(self.chainsOrder[baseIndex]):
                print(" <--" + str(self.chainsFlows[baseIndex][solbaseIndex])
                      + "/" + str(self.net[solbasePrevNode][solbaseNode])
                      + "-- " + str(self.longlabels[solbaseNode]), end="")
                solbasePrevNode = solbaseNode
            print()

    def print(self, mode = 'full'):
        net_bandwidth = []
        for i in range(0, self.node_number()):
            for j in range(i+1, self.node_number()):
                if (self.net[i][j] > 0):
                    net_bandwidth.append(self.net[i][j])
        node_distance = []
        node_range    = []
        for i in range(0, self.drone_number):
            for j in range(i+1, self.node_number()):
                distance = round(math.dist(self.pos[i], self.pos[j]))
                node_distance.append(distance)
                if (self.net[i][j] > 0):
                    node_range.append(distance)
        if mode == 'full':
            print('==========================================')
            print('number of base links       = ', self.base_number)
            print('drone number               = ', self.drone_number)
            print('node bandwidth min/avg/max = ', numpy.min(self.bandwidth), '/', round(numpy.average(self.bandwidth)), '/', numpy.max(self.bandwidth))
            print('net  bandwidth min/avg/max = ', numpy.min(net_bandwidth ), '/', round(numpy.average(net_bandwidth )), '/', numpy.max(net_bandwidth ))
            print('net  distance  min/avg/max = ', numpy.min(node_distance ), '/', round(numpy.average(node_distance )), '/', numpy.max(node_distance ))
            print('net  range     min/avg/max = ', numpy.min(node_range    ), '/', round(numpy.average(node_range    )), '/', numpy.max(node_range    ))
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
            
    def draw(self, showSolution = False, fileName = None):
        # G = networkx.from_numpy_matrix(self.net)
        G = networkx.Graph(self.net)

        edgelist = list(G.edges())

        edge_labels = {}
        edge_width = list()
        edge_color = list()
        edge_to_index = {}

        maxEdgeValue = 1;
        for edgeIndex, edge in enumerate(edgelist):
            maxEdgeValue = max(maxEdgeValue, self.net[edge[1]][edge[0]])

        for edgeIndex, edge in enumerate(edgelist):
            edge_to_index[(edge[0], edge[1])] = edgeIndex
            if (not showSolution):
                edge_labels[(edge[0], edge[1])] = self.net[edge[1]][edge[0]]
            edge_width.append(1 if showSolution else 10 * self.net[edge[1]][edge[0]] / maxEdgeValue)
            edge_color.append("#696969")  # DimGray

        node_size = numpy.empty(self.node_number())
        node_color = numpy.array(["#000000"] * self.node_number())
        dark_color = numpy.array(["#000000"] * self.node_number())
        edgecolors = numpy.array(["#000000"] * self.node_number())

        maxNodeValue = 1;
        for i in range(self.drone_number):
            maxNodeValue = max(maxNodeValue, self.bandwidth[i])

        for i in range(self.drone_number):
            node_size[i] = self.bandwidth[i] * 10000 / maxNodeValue
            node_color[i] = "#D2B48C"  # Tan
            edgecolors[i] = "#888888"
        for i in range(self.base_number):
            node_size[self.drone_number + i] = 2500
            if i == 0:
                node_color[self.drone_number + i] = "#87CEEB"  # SkyBlue
                dark_color[self.drone_number + i] = "#67AECB"
            elif i == 1:
                node_color[self.drone_number + i] = "#FF7F50"  # Coral
                dark_color[self.drone_number + i] = "#DF5F30"
            elif i == 2:
                node_color[self.drone_number + i] = "#90EE90"  # LightGreen
                dark_color[self.drone_number + i] = "#70CE70"
            elif i == 3:
                node_color[self.drone_number + i] = "#DDA0DD"  # Plum
                dark_color[self.drone_number + i] = "#BD80BD"
            else:
                node_color[self.drone_number + i] = "#DAA520"  # GoldenRod
                dark_color[self.drone_number + i] = "#BA8500"

        if (showSolution):
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

        fig, ax = matplotlib.pyplot.subplots(figsize=(20, 20))  # @UnusedVariable
        networkx.draw_networkx_nodes(
            G,
            self.pos,
            ax=ax,
            node_size=node_size,
            node_color=node_color,
            edgecolors=edgecolors)
        networkx.draw_networkx_labels(
            G,
            self.pos,
            ax=ax,
            labels=self.longlabels)
        networkx.draw_networkx_edges(
            G,
            self.pos,
            ax=ax,
            edgelist=edgelist,
            edge_color=edge_color,
            width=edge_width)
        networkx.draw_networkx_edge_labels(
            G,
            self.pos,
            ax=ax,
            edge_labels=edge_labels)

        if fileName is not None:
            matplotlib.pyplot.savefig(fileName, dpi=300, bbox_inches='tight')

class ExactSolution:
    def __init__(self, droneNet: DroneNet, found, firstTime, fullTime, results):
        self.droneNet = droneNet
        self.found = found
        self.firstTime = firstTime
        self.fullTime = fullTime
        self.results = results

        self.bestResults = []
        self.bestScore = -1
        self.scoreHisto = []

        if self.found:
            for result in self.results:
                score = result[1]
                if score >= self.bestScore:
                    if score > self.bestScore:
                        self.bestScore = score
                        self.bestResults = []
                    self.bestResults.append(result)

            prevHistoValue = -1
            for histoRange in range(10):
                histoValue = (histoRange + 1) * self.bestScore / 10
                histoCount = 0
                for result in self.results:
                    score = result[1]
                    if (score > prevHistoValue) and (score <= histoValue):
                        histoCount += 1
                if (prevHistoValue == -1):
                    prevHistoValue = 0
                self.scoreHisto.append((histoRange, prevHistoValue, histoValue, histoCount))
                prevHistoValue = histoValue

    def print(self, mode = 'full', showAll = False, draw = False) :
        if mode == 'full':
            print('==========================================')
            if self.found:
                print('time F : ', int(1000 * self.firstTime), 'ms')
                print('time T : ', int(1000 * self.fullTime ), 'ms')
                print('total  : ', len(self.results))
                print('best   : ', len(self.bestResults))
                sample = random.choice(self.bestResults)
                print('score  : ', sample[1])
                print('histo  : ', self.scoreHisto)
                print('result : ', sample[0])
                
                self.droneNet.processChainSolution(sample[0])
                
                self.droneNet.printSolution()
                
                if draw:
                    self.droneNet.draw(True)
                
                if showAll:
                    print('------------------------------------------')
                    for result in self.results:
                        print(result)
            else:
                print('Fail')
            
            print('==========================================')
        elif mode == 'csv':
            if self.found:
                print('T', end=';')
                print(int(1000 * self.firstTime), end=';')
                print(int(1000 * self.fullTime ), end=';')
                print(len(self.results), end=';')
                print(len(self.bestResults), end=';')
                sample = random.choice(self.bestResults)
                print(sample[1], end=';')
            else:
                print('F', end=';')
                print('' , end=';')
                print('' , end=';')
                print('' , end=';')
                print('' , end=';')
                print('' , end=';')
    
    @staticmethod
    def printEmpty(mode):
        if mode == 'csv':
            print('-', end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
            print('' , end=';')
    
    
    
class GASolution:
    def __init__(self, droneNet: DroneNet, ga: GA, found, time, score, result):
        self.droneNet = droneNet
        self.ga       = ga
        self.found    = found
        self.time     = time
        self.score    = score
        self.result   = result
        
        self.exact  = False
        self.eaBest = 0
        self.gaTop  = 0
        self.eaAll  = 0
        self.gaScoreFactor     = 0
        self.gaPositionFactor  = 0
    
    def applyExact(self, exact : ExactSolution = None):
        if exact is not None:
            self.exact  = True
            
            self.eaBest = exact.bestScore
            self.gaScoreFactor = 1 if exact.bestScore == 0 else self.score / exact.bestScore
            
            self.gaTop = 0
            for result in exact.results:
                score = result[1]
                if score > self.score:
                    self.gaTop += 1
            self.eaAll = len(exact.results)
            self.gaPositionFactor = self.gaTop / self.eaAll
        
        
    def print(self, mode = 'full', draw = False, drawFileName = None, plotFitness = False):
        if mode == 'full':
            print('==========================================')
            print("Time : {executionTime} ms".format(executionTime=int(1000*self.time)))
            if self.found:
                print("Solution : {result}".format(result=self.result))
                print("Score = {score}".format(score=self.score))
                
                if self.exact:
                    print('GA versus Exact')
                    print('  val {score} / {best} = {gaSol}%'.format(score = self.score, best = self.eaBest, gaSol = int(10000*self.gaScoreFactor   )/100))
                    print('  top {gaTop} / {all} = {gaPos}%' .format(gaTop = self.gaTop, all  = self.eaAll , gaPos = int(10000*self.gaPositionFactor)/100))
                    
                self.droneNet.processChainSolution(self.result)
                self.droneNet.printSolution()
                    
                if draw:
                    self.droneNet.draw(showSolution = True, fileName = drawFileName)
                
                if plotFitness:
                    self.ga.plot_fitness(show=False)
            else:
                print('Genetic algorithm fail')
            
            print('==========================================')
        elif mode == 'csv':
            print(int(1000*self.time), end=';')
            if self.found:
                print('T', end=';')
                print(self.score, end=';')
                
                if self.exact:
                    gaTop = 0
                    
                    print('T', end=';')
                    print(self.eaBest, end=';')
                    print(gaTop, end=';')
                    print(self.eaAll, end=';')
                else:
                    print('F', end=';')
                    print('' , end=';')
                    print('' , end=';')
                    print('' , end=';')
            else:
                print('F', end=';')
                print('' , end=';')
                print('F', end=';')
                print('' , end=';')
                print('' , end=';')
                print('' , end=';')
        
    @staticmethod
    def printEmpty(mode):
        if mode == 'csv':
                print('-', end=';')
                print('' , end=';')
                print('' , end=';')
                print('' , end=';')
                print('' , end=';')
                print('' , end=';')
        
class GenAlg:
    def __init__(
            self,
            timeLimit = 120, # in seconds, for 0 one run with other conditions, for < 0 run without time limit
            anySolution = True,
            
            num_generations    = 1000,
            num_parents_mating =  200,
            sol_per_pop        = 1000,
            keep_elitism       =  100,
            
            fitnessChainMode      = 'bestNodes', # any : For all correct return 1; bestEdges : For correct return sum of surplus on edges; bestNodes : For correct return sum of surplus on nodes
            parent_selection_type = 'sss', # sss : steady_state_selection, rws : roulette_wheel_selection, sus : stochastic_universal_selection, random, tournament, rank
            crossoverFix          = 'best', # best random
            
            mutation_type             = 'random', # random swap inversion scramble adaptive
            mutation_probability_norm = 0.2,
            mutation_num_genes        = 2,
            
            saturateStop = 20):
        
        self.timeLimit                 = timeLimit
        self.anySolution               = anySolution
        self.num_generations           = num_generations
        self.num_parents_mating        = num_parents_mating
        self.sol_per_pop               = sol_per_pop
        self.keep_elitism              = keep_elitism
        self.fitnessChainMode          = fitnessChainMode
        self.parent_selection_type     = parent_selection_type
        self.crossoverFix              = crossoverFix
        self.mutation_type             = mutation_type
        self.mutation_probability_norm = mutation_probability_norm
        self.mutation_num_genes        = mutation_num_genes
        
        self.stopCriteria = [];
        self.stopCriteria.append('saturate_' + str(saturateStop))
        if (anySolution):
            self.stopCriteria.append('reach_0')
        
        
    def run(self, dn: DroneNet) -> GASolution:
        start = perf_counter_ns()
        
        solution = []
        solutionFitness = -1
        executionTime = -1
        
        
        while ((self.timeLimit < 0) or (self.timeLimit > executionTime)) and ((not self.anySolution) or (solutionFitness < 0)):
            ga = GA(
                num_generations         = self.num_generations,
                num_parents_mating      = self.num_parents_mating,
                sol_per_pop             = self.sol_per_pop,
                keep_elitism            = self.keep_elitism,
                
                num_genes               = dn.node_number()-1,
                gene_type               = int,
                gene_space              = range(dn.node_number()-1),
                allow_duplicate_genes   = False,
                
                fitness_func            = lambda solution, solution_idx: dn.fitnessChain(solution, solution_idx, self.fitnessChainMode),
                
                parent_selection_type   = self.parent_selection_type,
                crossover_type          = lambda parents, offspring_size, ga: dn.singleChainCrossover(parents, offspring_size, ga, self.crossoverFix), # single_point two_points uniform scattered
                
                mutation_type           = self.mutation_type, # random swap inversion scramble adaptive
                mutation_probability    = self.mutation_probability_norm / dn.drone_number,
                mutation_num_genes      = self.mutation_num_genes, # or mutation_percent_genes
                mutation_by_replacement = True,
                
                #on_crossover = on_crossover,
                #on_mutation  = on_mutation ,
                
                stop_criteria           = self.stopCriteria)
                
            #print(ga_instance.initial_population)
            
            ga.run()
            
            executionTime = perf_counter_ns() - start
            
            currentSolution, currentSolutionFitness, currentSolutionIdx = ga.best_solution()  # @UnusedVariable
            
            if (currentSolutionFitness > solutionFitness):
                solutionFitness = currentSolutionFitness
                solution        = currentSolution
        
        return GASolution(dn, ga, solutionFitness >= 0, executionTime, solutionFitness, solution)