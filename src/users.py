#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from src.environment.user_modeling import UserWalker
from itertools import count
from src.apparatus.rf_transciever import RfTransceiver
from src.types_constants import StationType
from src.parameters import *


class User:
    _ids = count(0)

    def __init__(self, user_walker: UserWalker):
        self.id = next(self._ids)
        self.user_walker = user_walker
        self.coords = self.user_walker.current_coords
        self.rf_transceiver = RfTransceiver(coords=self.coords, user_id=self.id, bandwidth=USER_BANDWIDTH,
                                            station_type=StationType.UE)
