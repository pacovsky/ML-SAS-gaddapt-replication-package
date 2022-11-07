from dataclasses import dataclass
from enum import Enum
import numpy as np


class DroneMode(Enum):
    CHARGING = 0
    MOVING = 1
    RESTING = 2
    DEAD = 3
    IDLE = 4


class Singleton(object):
    _instances = {}
    _count = 0

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._count += 1
            print(cls, f"initialize itself with new keyword for the {cls._count} time")
            cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
        # else:
        #     # print(cls, "not initializing")
        #     try:
        #         print(cls._instances[cls].drone_count)
        #     except:
        #         pass
        return cls._instances[cls]


@dataclass
class SimulationSetup(Singleton):
    """dataclass holding the simulation values"""

    def change_simulation_setup(self, drones, flocks, charger_capacity, charger_centers, field_corners, field_centers,
                                drone_speed, drone_consumption, hysteresis, charger_option):
        self._drone_count = drones
        self._flock_count = flocks
        self.charger_capacity = charger_capacity
        self._charger_centers = charger_centers
        self._field_corners = field_corners
        self._field_centers = field_centers
        self._infinity = self.compute_distance_infinity(drone_speed, drone_consumption)
        self._drone_speed = drone_speed
        self._drone_consumption = drone_consumption
        print(f"self._infinity is: {self._infinity}")
        from drone_charging_example.red_flags.redflags import Hysteresis
        self.Hysteresis = Hysteresis()
        self.Hysteresis.reset(hysteresis)
        self.charger_option = charger_option


    @property
    def distance_infinity(self):
        return self._infinity


    def compute_distance_infinity(self, max_drone_distance_travelled_between_ticks, max_energy_consumption):
        # max_fly_distance = (1 / max_energy_consumption) * max_drone_distance_travelled_between_ticks
        max_fly_distance = max_drone_distance_travelled_between_ticks / max_energy_consumption

        # intuition behind the division factor - we are trying to pin point reachable places for the drone
        # that are the places where the drone can stay for a time corresponding to the travel time
        # imagine following scenario: charger and field are apart from one another by the infinity distance
        # we are setting up this distance by setting up the division factor in:
        #                                                                       `max_fly_distance / division`
        # assuming that the move and protect energy consumption are comparable.

        # division by 2 would mean: fly there with full battery,
        #                           reach the place with half turn immediately back,
        #                           reach the charger with 0 battery

        # division by 3 would mean: that the drone would be able to stay at the field for a maximum duration of 1/3
        #                           in practice much less because we want to reach the charger before reaching the reserve battery level

        # division by 3 would mean: that the drone would be able to stay at the field for a maximum duration of 1/2
        #                           which seems like a reasonable compromise

        return max_fly_distance / 4
        # return 160 # originally which corresponds to: max_fly_distance / 4  # 166

    # TODO should hold also min. drone counts for fields
    batch_size = 600
    input_size = 29
    output_size = 84

    charger_capacity = 1

    _drone_count = 4
    _flock_count = 5
    _infinity = 166

    @property
    def charger_count(self):
        return len(self._charger_centers)

    @property
    def drone_count(self):
        return self._drone_count

    @property
    def flock_count(self):
        return self._flock_count

    @property
    def field_count(self):
        return len(self._field_corners)

    _charger_centers = [
        # (x,y)
        (112.64623, 88.650085),
        (107.35458, 99.290443),
        (118.70316, 99.290443)
    ]

    @property
    def charger_centers(self):
        return self._charger_centers

    @property
    def get_charger_centers_array(self):
        return np.array(self._charger_centers).reshape((1, -1, 2))

    _field_centers = [
        # (x, y)
        (46.0689685, 28.117543),
        (40.2676728, 77.8592145),
        (113.13744, 53.381195),
        (186.28848, 26.867491),
        (184.692248, 136.90757)
    ]

    @property
    def field_centers(self):
        return self._field_centers

    _field_corners = np.array([
        # (x min, y min, x max, y max)
        [17.160196, 23.090456, 75.075447, 33.14463],
        [17.160196, 72.83213, 63.433769, 82.886299],
        [101.59348, 48.354103, 124.64072, 58.408287],
        [174.74452, 21.840401, 197.79176, 31.894581],
        [161.58475, 131.88049, 207.85835, 141.9346],
    ], dtype=np.float16)

    @property
    def field_corners(self):
        return self._field_corners

    def is_in_fields(self, XYs, summary="sum"):
        """For each field computes if XYs are within within the field
            :param XYs: np.array of x0,y0,x1,y1, ...
            :param summary: options are: "sum" / "any" / None
                - sum returns number of birds on the field x
                - any returns boolean on the field x are birds
                - anything else will return raw result
            returns statistics summary
        """
        # append 1 to shape description
        shape = (*XYs[:, :, 0].shape, 1)

        # compute for each field and each xy if the xy is within field
        # multiplication "*" of truth values has the same meaning as "logical and"
        result = \
            (self.field_corners[:, 0] <= XYs[:, :, 0].reshape(shape)) * \
            (self.field_corners[:, 1] <= XYs[:, :, 1].reshape(shape)) * \
            (self.field_corners[:, 2] >= XYs[:, :, 0].reshape(shape)) * \
            (self.field_corners[:, 3] >= XYs[:, :, 1].reshape(shape))

        # result axis are: x - fields; axis y - drones/flocks

        if summary == "sum":
            # returns number of birds on the field x
            return np.sum(result, axis=1)
        elif summary == "any":
            # returns if field x has any birds
            return np.any(result, axis=1)
        else:
            return result
