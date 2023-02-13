#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.
from src.main_controller import SimulationController

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

