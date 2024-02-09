#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from src.math_tools import db2lin, wh_to_joules, lin2db
from src.types_constants import LinkType
from src.data_structures import Coords3d

from numpy import exp
from numpy import log



######################################Important paramters to modify
#Boundaries of the area in meters
X_BOUNDARY = [0, 10000]
Y_BOUNDARY = [0, 10000]

SIGMA_UE_PER_CLUSTER = 200 # Variance of UEs in each cluster (larger value -> more dispersion)
CLUSTERS_DENSITY = 0.000002 #Cluster density in area
N_CLUSTERS = X_BOUNDARY[1]*Y_BOUNDARY[1]*CLUSTERS_DENSITY #Resulting number of clusters
MEAN_UES_PER_CLUSTER = 2 #Mean number of UEs in cluster

#Add MBS here below as needed
MBS_HEIGHT = 25  # m
MBS_LOCATIONS = [Coords3d(X_BOUNDARY[0], Y_BOUNDARY[0], MBS_HEIGHT), Coords3d(X_BOUNDARY[1], Y_BOUNDARY[1], MBS_HEIGHT)]

NUM_UAVS = 500 #Number of DBSs. We can increase it if FSO capacities are insufficient, etc.

REQUIRED_UE_RATE = 40e6 #Required Mbps per UE. It can be increased to make the problem harder,
                        # or decrease to relax constraint

MAX_FSO_DISTANCE = 3000 #FSO links between nodes that have distance higher than this are assumed to have capacity 0

TX_POWER_FSO_DRONE = 0.05  # W

CLUSTERING_METHOD = 2 #0 for SINR-EM, and 1 for Kmeans, and 2 for hierarchical

MIN_N_DEGREES = 1

RANDOMIZE_MBS_LOCS = True

COVERAGE_RADIUS = 2000000
###########################################

SKIP_UE_RF_UPDATE = True
#Obstacles
EXTEND_TIMES_FOUR = True

# Users Mobility Model
NUM_OF_USERS = 140
USER_SPEED = [0.5, 0.8]
PAUSE_INTERVAL = [0, 60]
TIME_STEP = 2.5  # Between subsequent users mobility model updates
TIME_SLEEP = 2  # Sleep between updates to allow plotting to keep up
BUILDINGS_AREA_MARGIN = 50
SIMULATION_TIME = 60 * 60 * 2
USER_SPEED_DIVISOR = 1

# SOLAR PANELS
PANEL_EFFICIENCY_FACTOR = 0.2
SOLAR_PANEL_AREA = 1
ABSORPTION_COEFFICIENT_CLOUD = 0.01

# SUN ENVIRONMENT
STARTING_DAY = 1
STARTING_MONTH = 7
STARTING_HOUR = 12  # 24 format
STARTING_MINUTE = 0
MAX_HOUR_DAY = 23
CLOUD_SPEED = 16 * 1
TIME_ZONE = 'Europe/Madrid'
SUN_SEARCH_STEP = 7  # m
SUN_SEARCH_COUNT = 5
MAX_SUN_SEARCH_STEPS = 10
BUILDING_EDGE_MARGIN = 1  # m across each axis
SHADOWED_EDGE_PENALTY = 100 / 3

# Channel model PLOS
PLOS_AVG_LOS_LOSS = 1
PLOS_AVG_NLOS_LOSS = 20
PLOS_A_PARAM = 9.61
PLOS_B_PARAM = 0.16
# PLOS_A_PARAM = 4.9 #Obtained using the method
# PLOS_B_PARAM = 0.4
# PLOS_A_PARAM = 5.05 #Obtained using the method
# PLOS_B_PARAM = 0.38

# Channel model RF
DRONE_TX_POWER_RF = 0.2  # W
USER_BANDWIDTH = 500e3 #*2  # Hz
DRONE_BANDWIDTH = 20e6  #+5e6 # Hz
MBS_BANDWIDTH = 20e6  # Hz
DEFAULT_CARRIER_FREQ_MBS = 2e9  # Hz
DEFAULT_CARRIER_FREQ_DRONE = 2e9 + MBS_BANDWIDTH  # Hz
NOISE_SPECTRAL_DENSITY = -174  # dBm/Hz
NOISE_POWER_RF = db2lin(NOISE_SPECTRAL_DENSITY - 30 + lin2db(USER_BANDWIDTH))  # dBm input -> linear in W
DEFAULT_SNR_THRESHOLD = db2lin(25)  # linear
MBS_TX_POWER_RF = 0.5  # W
SINR_THRESHOLD = db2lin(10)
ASSOCIATION_SCHEME = 'SINR'
USER_SINR_COVERAGE_HIST = 1 #before it was 100

# Channel model FSO
RX_DIAMETER = 0.2  # m
DIVERGENCE_ANGLE = 0.06  # rads
RX_RESPONSIVITY = 0.5
AVG_GML = 3
WEATHER_COEFF = 4.3 * 10 ** -4  # /m
POWER_SPLIT_RATIO = 0.005
FSO_ENERGY_HARVESTING_EFF = 0.2

TX_POWER_FSO_MBS = 380  # W
BANDWIDTH_FSO = 1e9  # Hz
NOISE_VARIANCE_FSO = 0.8 * 1e-9
NOISE_POWER_FSO = 1e-6
EMPIRICAL_SNR_LOSSES = db2lin(15)  # Linear
BEAMWAIST_RADIUS = 0.25e-3 * 10
WAVELENGTH = 1550e-9
AvgGmlLoss = {LinkType.A2G: 3, LinkType.A2A: 3 / 1.5, LinkType.G2G: 5}  # TODO: Get refs


# # UAV (Energy, Speed, etc.)
# class UavParams:
UAV_MASS = 4  # kg
UAV_PROPELLER_RADIUS = 0.25  # [m]
NUMBER_OF_UAV_PROPELLERS = 4
AIR_DENSITY = 1.225 # kg/m2
GRAVITATION_ACCELERATION = 9.80665
PROFILE_DRAG_COEFFICIENT = 0.08
UAV_STARTING_ENERGY = wh_to_joules(222)  # wh
UAV_MAX_ENERGY = wh_to_joules(222)
UAV_MIN_ENERGY = 0
UAV_TRAVEL_SPEED = 13  # m/s
SKIP_ENERGY_UPDATE = False
SKIP_ENERGY_CHARGE = True



UAVS_HEIGHTS = [60]#, 80]#, 100]
# BEAMWAIST_RADII = [0.0045, 0.015, 0.0045, 0.015, 0.0045, 0.015, 0.0045, 0.015]
BEAMWAIST_RADII = [0.01, 0.01, 0.02, 0.02]
NUM_MBS = len(MBS_LOCATIONS)

DRONE_HEIGHT = UAVS_HEIGHTS[0]  # m

UE_HEIGHT = 1.5  # To conform with channel models
MAX_USERS_PER_DRONE = DRONE_BANDWIDTH/ USER_BANDWIDTH
# USER_MOBILITY_SAVE_NAME = 'users_200_truncated'
USER_MOBILITY_SAVE_NAME = 'extended_4_madrids_500_users'

# Calculate solution for drone FSO connection in exact way, exponential time, only for small number of drones
CALCULATE_EXACT_FSO_NET_SOLUTION = True
# Limit calculation of exact solution to the first one, still exponential time
CALCULATE_EXACT_FSO_NET_SOLUTION_FIRST_ONLY = False
# Time limit in seconds for genetic algorithm searching solution for FSO net
FSO_NET_GENECTIC_ALGORITHM_TIME_LIMIT = 10
# For true store extra information about MHP runs
SAVE_MHP_DATA = False
