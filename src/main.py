#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.
from src.main_controller import SimulationController
from src.parameters import NUM_MBS
from src.mhp.MHP import DroneNet
from src.mhp.MHP import GenAlg
from src.parameters import CALCULATE_EXACT_FSO_NET_SOLUTION, FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT
import numpy as np

if __name__ == '__main__':

    #Initialize simulation controller with UE distributed in clusters,
    # and UAVs distributed randomly, and Macro base stations located according
    # to settings in parameters.py (MBS_LOCATIONS)
    sim_ctrl = SimulationController()

    #Generate plot and show UEs, DBSs, and MBSs
    fig, _ = sim_ctrl.generate_plot()
    fig.show()

    #Locate DBSs according to EM algorithm which optimizes channel quality
    sim_ctrl.perform_sinr_en()

    #If we plot again we can see new DBSs locations
    fig, _ = sim_ctrl.generate_plot()
    fig.show()

    #Calculate FSO link capacities for current locations
    sim_ctrl.get_fso_capacities()

    # Plot FSO capacities graph with capacity requirement of each cluster
    sim_ctrl.generate_plot_capacs()

    #Call this to generate new UEs distribution
    sim_ctrl.reset_users_model()

    #The controller has the MBSs and DBSs in self.base_stations
    #The MBSs are in the slice [:NUM_MBS] and DBSs are in [NUM_MBS:]
    mbs_list = sim_ctrl.base_stations[:NUM_MBS]
    dbs_list = sim_ctrl.base_stations[NUM_MBS:]

    #Each DBS has coords attribute (which can also behave like numpy array)
    print(dbs_list[0].coords.x, dbs_list[0].coords.y, dbs_list[0].coords.z)
    np.sum(dbs_list[0].coords)

    #Alternatively we can get locations of all DBSs as arrays
    print(sim_ctrl.get_uavs_locs())
    print(np.sum(sim_ctrl.get_uavs_locs(), 1)) # just to show it behaves like array

    #We can also get the FSO capacities as 2D array where the indexes are indexes of DBSs.
    # I.e., element i,j s.t. i=j is redundant. Otherwise the array show the capacity between the ith DBS and jth DBS
    print(sim_ctrl.fso_links_capacs)

    #To get required capacity from UEs load per DBS
    print(sim_ctrl.get_required_capacity_per_dbs())
    
    # Construct drone net
    dn = DroneNet.createArea(mbs_list, dbs_list, sim_ctrl.get_required_capacity_per_dbs(), sim_ctrl.fso_links_capacs)
    dn.print()
    
    #Exact solution, long time
    if CALCULATE_EXACT_FSO_NET_SOLUTION:
        exactSolution = dn.lookForChainSolution(first=False, mode='bestNodes')
        exactSolution.print()
        if exactSolution.found:
            ga = GenAlg(timeLimit=FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT)
            gaSolution = ga.run(dn)
            gaSolution.print(exactSolution)
    else:
        ga = GenAlg(timeLimit=FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT)
        gaSolution = ga.run(dn)
        #gaSolution.print(draw = True, drawFileName = 'FSO_net_ga_solution')
        gaSolution.print()
