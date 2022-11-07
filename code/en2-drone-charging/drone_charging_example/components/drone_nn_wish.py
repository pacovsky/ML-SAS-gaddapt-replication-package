from enum import IntEnum


class DroneNNWish(IntEnum):
    """
    Drone wish:
        NONE: a default state for drones.
        FIELD_x: wants to go to the field x.
        CHARGE: wants to be charged.
    """
    NONE = 0
    FIELD_1 = 1
    FIELD_2 = 2
    FIELD_3 = 3
    FIELD_4 = 4
    FIELD_5 = 5
    CHARGE = 6

class ExtendedWish(IntEnum):
    none = 0
    a0 = 1
    p0 = 2
    a1 = 3
    p1 = 4
    a2 = 5
    p2 = 6
    a3 = 7
    p3 = 8
    a4 = 9
    p4 = 10
    a_chargers = 11
    u_chargers = 12
    dead = 13

class GroupedWishState(IntEnum):
    NONE = 0
    FIELD_1 = 1
    FIELD_2 = 2
    FIELD_3 = 3
    FIELD_4 = 4
    FIELD_5 = 5
    CHARGER = 6
    DEAD_KO = 7

    INVALID = -1
    # def __str__(self):
    #     return IntEnum.__str__(self).split('.')[1]

    @classmethod
    def is_field(cls, value):
        return cls.NONE < value < cls.CHARGER


class RedFlagsEnum(IntEnum):
    NOTHING = 0
    LOCAL_ONLY = 1
    EVERYTHING = 2