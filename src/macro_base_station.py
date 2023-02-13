#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from itertools import count
from src.data_structures import Coords3d
from src.apparatus.rf_transciever import RfTransceiver
from src.parameters import  MBS_BANDWIDTH, MBS_TX_POWER_RF, DEFAULT_CARRIER_FREQ_MBS, MBS_LOCATIONS
from src.types_constants import StationType


class MacroBaseStation:
    _ids = count(0, -1)

    def __init__(self, mbs_id: int = None, coords: Coords3d = MBS_LOCATIONS):
        self.id = next(self._ids) if mbs_id is None else mbs_id
        self.coords = coords.copy()
        self.rf_transceiver = RfTransceiver(coords=self.coords, t_power=MBS_TX_POWER_RF, bandwidth=MBS_BANDWIDTH,
                                            bs_id=self.id, carrier_frequency=DEFAULT_CARRIER_FREQ_MBS,
                                            station_type=StationType.UMa)
        self.fso_transceivers = []
        self.station_type = StationType.UMa
