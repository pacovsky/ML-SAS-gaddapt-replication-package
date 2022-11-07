import builtins
import random
import math
from typing import Optional, TYPE_CHECKING

import numpy as np
from components.drone_state import DroneState
from components.drone_nn_wish import DroneNNWish
from world import ENVIRONMENT, WORLD
from ml_deeco.simulation import Agent
if TYPE_CHECKING:
    from components.charger import Charger

# from red_flags.preprocess_data import distance_normalization, battery_distance_normalization
from red_flags.simulation_setup import SimulationSetup
from ml_deeco.simulation import SimulationGlobals
PRINT = False


def print(*args):
    if PRINT:
        builtins.print(args)

class Drone(Agent):
    """
    The drones protect the fields from birds by moving to the field and scaring the flocks of birds away.
    In programming perspective, drone components have access to shared `WORLD` and they can find the position to protect.
    In a real-life scenario, it is assumed that additional sensors will perform the detection of birds, and it can be read from them.

    Attributes
    ----------
    droneRadius: int
        Protecting radius of drone.
    droneSpeed: float
        The moving speed of the drone.
    droneMovingEnergyConsumption: float
        The energy consumption per time-step for drone moving.
    droneProtectingEnergyConsumption: float
        The energy consumption per time-step for protecting/standing drone.
    battery: float
        The battery percentage of the drone.
    target: Point
        The target location of the drone.
    targetField: Field
        The target field to protect.
    targetCharger: Charger
        The selected charger.
    closestCharger: Charger
        the closest charger which is picked by pre-assignment ensemble.
    alert: float
        If computed battery is below this value, it is assumed a critical battery level.
    state: DroneState
        IDLE: a default state for drones.
        PROTECTING: when the drones are protecting the fields.
        MOVING_TO_CHARGER: when the drones are moving/queuing for a charger.
        CHARGING: when the drones are being charged.
        TERMINATED: when the drones' battery is 0, and they do not operate anymore.
    nn_wish : DroneNNWish

    """
    def __init__(self, location):
        """
        Creates a drone object with the given location and ENVIRONMENT settings.

        Parameters
        ----------
        location : Point
            Starting point for the drone.
        """
        self.droneRadius = ENVIRONMENT.droneRadius
        self.droneSpeed = ENVIRONMENT.droneSpeed
        self.droneMovingEnergyConsumption = ENVIRONMENT.droneMovingEnergyConsumption
        self.droneProtectingEnergyConsumption = ENVIRONMENT.droneProtectingEnergyConsumption
        self._state = DroneState.IDLE
        self.target = None
        self.targetField = None
        self.targetCharger: Optional[Charger] = None
        self.closestCharger: Optional[Charger] = None
        self.alert = 0.15
        self.battery = 1 - (ENVIRONMENT.droneBatteryRandomize * random.random() * (1 - self.alert) + self.alert)
        self.nn_wish = DroneNNWish.NONE
        self.DEBUG_nn_wish = DroneNNWish.NONE
        Agent.__init__(self, location, self.droneSpeed)

    @property
    def state(self):
        """

        Returns
        -------
        DroneState
            IDLE: a default state for drones.
            PROTECTING: when the drones are protecting the fields.
            MOVING_TO_CHARGING: when the drones are moving/queuing for a charger.
            CHARGING: when the drones are being charged.
            TERMINATED: when the drones' battery is 0, and they do not operate anymore.
        """
        return self._state

    @property
    def get_the_ensml_project_compatible_enum_numbering(self) -> int:
        """ return the numbers according to the following key:
            CHARGING = 0
            MOVING = 1
            RESTING = 2 # was not used in simulation
            DEAD = 3
            IDLE = 4
        """
        state = self._state
        if state == DroneState.CHARGING:
            return 0
        if state == DroneState.MOVING_TO_FIELD:
            return 1
        if state == DroneState.MOVING_TO_CHARGER:
            return 1
        if state == DroneState.TERMINATED:
            return 3
        if state == DroneState.IDLE:
            return 4

        if state == DroneState.PROTECTING:
            # the data shows following pattern: at first 1 then 4
            # probably corresponds to the situation that switches the ensemble from approaching field to scaring upon
            # arrival to the field and reaching the desired position where they stay until birds are gone
            return 4  # alternatively 1 can be used as well

    @property
    def get_the_ensml_compatible_row(self):
        # energy, mode, x, y
        state = self.get_the_ensml_project_compatible_enum_numbering
        loc = self.location

        # battery_distance, non_linear_battery = battery_distance_normalization(self.battery)
        return self.battery, state, loc.x, loc.y
        # return battery_distance, state, loc.x, loc.y, non_linear_battery

    @state.setter
    def state(self, value):
        """
        Sets the drone state.

        Parameters
        ----------
        value : DroneState
        """
        self._state = value

    def timeToEnergy(self, time, consumptionRate=None):
        """
        Computes the amount of energy which is consumed in the given time.

        Parameters
        ----------
        time : int
            The time (in time steps).
        consumptionRate : float, optional
            The battery consumption per time step, defaults to `self.droneMovingEnergyConsumption`.

        Returns
        -------
        float
            The amount of energy consumed in given time.
        """
        if consumptionRate is None:
            consumptionRate = self.droneMovingEnergyConsumption
        return time * consumptionRate

    def findClosestCharger(self):
        """
        Finds the closest charger comparing the drone distance and WORLD chargers.

        Returns
        -------
        charger
            The closest charger to the drone.
        """
        return min(WORLD.chargers, key=lambda charger: self.location.distance(charger.location))

    def timeToFlyToCharger(self, charger=None):
        """
        Computes the time needed to get to the charger or the closest charger.

        Parameters
        ----------
        charger : Charger, optional
            Specify the charger for measuring the distance, defaults to `self.closestCharger`.

        Returns
        -------
        float
            The time steps needed to fly to the (given) charger.
        """
        if charger is None:
            charger = self.closestCharger
            if self.closestCharger is None:  # Added to be more resilient
                # print("reached unintended pathway when ensembles are used")
                charger = self.findClosestCharger()

        return self.location.distance(charger.location) / self.speed

    def energyToFlyToCharger(self, charger=None):
        """

        Computes the energy needed to fly to the specified charger.

        Parameters
        ----------
        charger : Charger, optional
             Specify the charger for measuring the distance.

        Returns
        -------
        float
            The energy required to fly to the closest or given charger.
        """
        return self.timeToEnergy(self.timeToFlyToCharger(charger))

    def timeToFlyToFieldAndThenToClosestCharger(self, field):
        distance_to_field = field.closestDistanceToDrone(self)
        field_to_charger_distance = min(field.location.distance(charger.location) for charger in WORLD.chargers)
        total_distance = distance_to_field + field_to_charger_distance
        total_time = total_distance / self.speed
        return total_time
        # return self.timeToEnergy(total_time)

    def computeBatteryAfterTime(self, time: int):
        """
        Computes the battery after given time (assuming the `self.droneMovingEnergyConsumption` energy consumption per time step).

        Parameters
        ----------
        time : int
            Time steps.

        Returns
        -------
        float
            Battery after spending energy in given time steps.
        """
        return self.battery - self.timeToEnergy(time)

    def timeToDoneCharging(self):
        """
        Computes how long it will take to fly to the closest charger and get fully charged, assuming the charger is free.

        Returns
        -------
        int
            The time which the drone will be done charging.
        """
        batteryWhenGetToCharger = self.battery - self.energyToFlyToCharger()

        closestCharger = self.closestCharger

        if closestCharger is None:
            closestCharger = self.findClosestCharger()
            # print("closestCharger was None", self.id)

        timeToCharge = (1 - batteryWhenGetToCharger) * closestCharger.chargingRate
        return self.timeToFlyToCharger() + timeToCharge

    def needsCharging(self, timeToStartCharging: int):
        """
        Checks if the drone needs charging assuming it will take `timeToStartCharging` time steps to get to the charger and start charging.

        In ML-Based model, the waiting time is predicted and is part of the `timeToStartCharging`.
        If computed battery is below threshold, the function returns true.

        Parameters
        ----------
        timeToStartCharging : int
            The time the drone needs to get to the charger (and possibly wait) and start charging.

        Returns
        -------
        bool
            Whether the drone needs charging or does not.
        """
        if self.state == DroneState.TERMINATED:
            return False
        futureBattery = self.computeBatteryAfterTime(timeToStartCharging)
        if futureBattery < 0:
            return False
        return futureBattery < self.alert

    def checkBattery(self):
        """
        It checks the battery if is below or equal to 0, it is assumed the drone is dead and it will get removed from the given tasks.
        """
        if self.battery <= 0 and not self.state == DroneState.TERMINATED:
            self.battery = 0
            self.state = DroneState.TERMINATED
            if self.targetField is not None:
                self.targetField.unassign(self)
            self.targetCharger = None
            self.closestCharger = None
                
    def move(self):
        """
        If the drone has not reached target,
        it moves the drone by calling the (super) Agent moving function,
        with addition of decreasing the battery in moving consumption rate.

        Otherwise only decreases the battery in protecting consumption rate.
        """
        if self.location == self.target:
            self.battery = self.battery - self.droneProtectingEnergyConsumption
            return

        self.battery = self.battery - self.droneMovingEnergyConsumption
        super().move(self.target)

    def set_state(self, nn_wish):
        def s(wish):
            return str(DroneNNWish(wish))

        """
        Called from simulation
        """
        self.DEBUG_nn_wish = nn_wish  # todo test         if nn_wish == -1

        if self.battery == 0:
            self.state = DroneState.TERMINATED
            self.target = None
            self._set_target_field(None)
            self.targetCharger = None
            return
        elif self.state == DroneState.TERMINATED:
            print(f"REDFLAG: drone {self.id} has {self.battery} battery but is deemed TERMINATED ")
            self.id = f"resurected {self.id}"
            return

        # force fly to charger when the way to charger will hit the reserve level
        if nn_wish != DroneNNWish.CHARGE and self.needsCharging(self.timeToFlyToCharger()):
            #  TODO when was the closest charger computed?
            print(f"RED FLAG - CHANGING WISH- low battery situation ({self.id} wanted {s(nn_wish)}) time {self.timeToFlyToCharger()}")
            nn_wish = DroneNNWish.CHARGE

        # force charging to finish
        previous_wish = self.nn_wish
        if previous_wish == DroneNNWish.CHARGE and \
                nn_wish != DroneNNWish.CHARGE and \
                self.battery < 0.9 and \
                self.targetCharger is not None and \
                self in self.targetCharger.chargingDrones:  # already in charger
            print(f"{self.id} wanted to leave previous wish {s(previous_wish)} for {s(nn_wish)} but did not have full battery {self.battery}")
            nn_wish = previous_wish
        else:
            self.nn_wish = nn_wish

        # remove everything
        # drone.target = None
        # drone._set_target_field(None)
        # drone.targetCharger = None
        self.closestCharger = None

        if nn_wish == DroneNNWish.NONE:
            self.state = DroneState.IDLE
            self._set_target_field(None)
            self.targetCharger = None #TODO - check charger assign

            # RED FLAG idle on low energy and in reach of a charger nearby? # when not second part returns False
            if self.needsCharging(0) and self.needsCharging(self.timeToFlyToCharger()):
                print(f"REDFLAG: unreachable when the force above??? drone {self.id} was IDLE and already at alert battery level => sending to charger")

        elif nn_wish == DroneNNWish.CHARGE:
            # RED FLAG
            if self.battery >= 1:
                print("REDFLAG: overchared", self.id)

                # is the drone already disconnected?
                if self.targetCharger is None:
                    print("REDFLAG2: overchared", self, self.DEBUG_nn_wish)
                    self.state = DroneState.IDLE  # this gets triggered in actuate and not in drone_charging pre ens

                    # next line causes loop if the charger is far away -
                    # self.set_state(DroneNNWish.FIELD_5) #send the drone away (here I would like random direction but this way is deterministic)

                    # so this is ~safer
                    self.state = DroneState.MOVING_TO_FIELD
                    self._set_target_field(WORLD.fields[4])

            # if drone.state != DroneState.CHARGING:
            #     drone.state = DroneState.MOVING_TO_CHARGER
            #     drone.closestCharger = drone.findClosestCharger()
            #     drone.targetCharger = drone.closestCharger
            if self.state != DroneState.CHARGING and self.state != DroneState.MOVING_TO_CHARGER:
                # setup to trigger the correct behavior in actuate and not in drone_charging pre ens
                # TODO chceck how the drones are waiting for charger - doesn't this cause the battery bug?
                self.state = DroneState.IDLE
                self.closestCharger = self.findClosestCharger()
                self.targetCharger = self.closestCharger  # triggers move to charger # TODO this makes drones loose energy when already waiting nearby charger

        # drone wants to go to fields FIELD
        elif DroneNNWish.NONE < nn_wish < DroneNNWish.CHARGE:
            fly_to_field = WORLD.fields[nn_wish - 1]

            # RED FLAG distance works?
            time = self.timeToFlyToFieldAndThenToClosestCharger(fly_to_field)
            energy_after_time = self.computeBatteryAfterTime(time)
            if energy_after_time < self.alert:  # alert is battery reserve level
                # print(f"REDFLAG: don't fly to field - on low energy {self.id} (energy would be: {energy_after_time})")
                print(f"REDFLAG: flying to a field - on low energy {self.id} (energy now: {self.battery} and would be: {energy_after_time})")
                # TODO do something?
                # todo does distance to the field make a difference - I am far away, or I am standing there
            if self.state != DroneState.PROTECTING:
                self.state = DroneState.MOVING_TO_FIELD

                self._set_target_field(fly_to_field)
        else:
            assert False, "Unreachable"

    def _set_target_field(self, value):
        if self.targetField is not None:
            self.targetField.unassign(self)
        self.targetField = value

    def set_state_minimal(self, nn_wish):
        if self.battery <= 0:
            if self.state != DroneState.TERMINATED:
                pass

            self.state = DroneState.TERMINATED
            return

        if nn_wish == DroneNNWish.NONE:
            self.state = DroneState.IDLE
            return

        if nn_wish == DroneNNWish.CHARGE:
            if self.state != DroneState.CHARGING and self.state != DroneState.MOVING_TO_CHARGER:
                self.closestCharger = self.findClosestCharger()
                self.targetCharger = self.closestCharger
                self.target = self.targetCharger.provideLocation(self)
                self.state = DroneState.MOVING_TO_CHARGER
            return

        # in fields
        if DroneNNWish.NONE < nn_wish < DroneNNWish.CHARGE:
            fly_to_field = WORLD.fields[nn_wish - 1]
            self._set_target_field(fly_to_field)
            self.target = self.targetField.location
            if self.state != DroneState.PROTECTING:
                self.state = DroneState.MOVING_TO_FIELD
            return

        # changes from moving to protecting or charging is done in materialize

        assert False, "Unreachable"

    # def set_state_minimal(self, nn_wish):
    #     """sets state, does checking, informs, but does not apply the fixes"""
    #     def s(wish):
    #         return str(DroneNNWish(wish))
    #     """
    #     Called from simulation
    #     """
    #     self.DEBUG_nn_wish = nn_wish  # todo test         if nn_wish == -1
    #
    #     if self.battery == 0:
    #         self.state = DroneState.TERMINATED
    #         self.target = None
    #         self._set_target_field(None)
    #         self.targetCharger = None
    #         return
    #     elif self.state == DroneState.TERMINATED:
    #         print(f"REDFLAG: drone {self.id} has {self.battery} battery but is deemed TERMINATED ")
    #         self.id = f"resurected {self.id}"
    #         return
    #
    #     # force fly to charger when the way to charger will hit the reserve level
    #     if nn_wish != DroneNNWish.CHARGE and self.needsCharging(self.timeToFlyToCharger()):
    #         #  TODO when was the closest charger computed?
    #         print(f"RED FLAG - CHANGING WISH- low battery situation ({self.id} wanted {s(nn_wish)}) time {self.timeToFlyToCharger()}")
    #
    #     # force charging to finish
    #     previous_wish = self.nn_wish
    #     if previous_wish == DroneNNWish.CHARGE and \
    #             nn_wish != DroneNNWish.CHARGE and \
    #             self.battery < 0.9 and \
    #             self.targetCharger is not None and \
    #             self in self.targetCharger.chargingDrones:  # already in charger
    #         print(f"{self.id} wanted to leave previous wish {s(previous_wish)} for {s(nn_wish)} but did not have full battery {self.battery}")
    #
    #     self.nn_wish = nn_wish
    #
    #     if nn_wish == DroneNNWish.NONE:
    #         self.state = DroneState.IDLE
    #         self._set_target_field(None)
    #         self.targetCharger = None
    #
    #         # RED FLAG idle on low energy and in reach of a charger nearby? # when not second part returns False
    #         if self.needsCharging(0) and self.needsCharging(self.timeToFlyToCharger()):
    #             print(f"REDFLAG: unreachable when the force above??? drone {self.id} was IDLE and already at alert battery level => sending to charger")
    #
    #     elif nn_wish == DroneNNWish.CHARGE:
    #         # RED FLAG
    #         if self.battery >= 1:
    #             print("REDFLAG: overchared", self.id)
    #
    #             # is the drone already disconnected?
    #             if self.targetCharger is None:
    #                 print("REDFLAG2: overchared", self, self.DEBUG_nn_wish)
    #
    #     # drone wants to go to fields FIELD
    #     elif DroneNNWish.NONE < nn_wish < DroneNNWish.CHARGE:
    #         fly_to_field = WORLD.fields[nn_wish - 1]
    #
    #         # RED FLAG distance works?
    #         time = self.timeToFlyToFieldAndThenToClosestCharger(fly_to_field)
    #         energy_after_time = self.computeBatteryAfterTime(time)
    #         if energy_after_time < self.alert:  # alert is battery reserve level
    #             print(f"REDFLAG: flying to a field - on low energy {self.id} (energy now: {self.battery} and would be: {energy_after_time})")
    #         if self.state != DroneState.PROTECTING:
    #             self.state = DroneState.MOVING_TO_FIELD
    #             self._set_target_field(fly_to_field)
    #     else:
    #         assert False, "Unreachable"

    def actuate(self):
        """
        It performs the actions of the drone in one time-step.
        For each state it performs differently:
        TERMINATED:
            Returns, no actions.
        IDLE or PROTECTING:
            checks the targetCharger, because if the charger is set it means the drone needs charging; the state wil be changed to MOVING_TO_CHARGER.
            If not, then it checks the targetField, if it is set, it means that the drone is required on a filed; the state will be changed to MOVING_TO_FIELD.
            If the drone targets are not set, it remains IDLE, and will not consume battery. It is assumed, that it landed or returned to hanger in a real-life scenario.
        MOVING_TO_CHARGER:
            It removes the drone from the field, and start moving toward the charger, with moving energy consumption rate. When it reached to the charger, the drone will consume standing energy until it is landed on the charger. We are not alway certian that by the time drone gets to the charger it will be free. This is due to the fact, the charging rate is changed regarding how many other drones are being charged somewhere else.
        MOVING_TO_FIELD:
            Starts moving toward the field and when reached, it will change the state to PROTECTING.
        CHARGING:
            The drone starts being charged by the charger until the battery level gets to 1. When the battery is ful, the state will be changed to IDLE.  
        
        In each timestep it is checking the battery, to see if the drone is still alive.
        """

        def do():
            if self.state == DroneState.TERMINATED:
                # if self.battery > 0:
                #     assert self.battery == 0
                return
            if self.battery <= 0:
                self.battery = 0
                self.state = DroneState.TERMINATED
                return

            if self.state < DroneState.MOVING_TO_CHARGER:  # IDLE, PROTECTING or MOVING TO FIELD
                if self.targetCharger is not None:
                    self.state = DroneState.MOVING_TO_CHARGER
                else:
                    if self.targetField is None:
                        self.state = DroneState.IDLE
                        # edited: added battery drain
                        self.battery = self.battery - self.droneProtectingEnergyConsumption
                        return
                    self.target = self.targetField.assignPlace(self)
                    self.state = DroneState.MOVING_TO_FIELD

            if self.state == DroneState.MOVING_TO_CHARGER:
                if self.targetField is not None:
                    self.targetField.unassign(self)
                if self.targetCharger is None:
                    if self.battery == 1: # HOTFIX
                        print(f"WARNING not an error - with unplugging the charger: ({self})")
                        self.state = DroneState.IDLE
                        self.battery = self.battery - self.droneProtectingEnergyConsumption
                        return
                    else:
                        print(f"ERROR - lost target charger: ({self}), using the closest")
                        self.targetCharger = self.findClosestCharger()
                self.target = self.targetCharger.provideLocation(self)
                # if self.location != self.targetCharger.location: #
                if self.location != self.target:
                    self.move()
                else:
                    self.battery = self.battery - self.droneProtectingEnergyConsumption

            if self.state == DroneState.MOVING_TO_FIELD:
                if self.location == self.target:
                    self.state = DroneState.PROTECTING
                    self.battery = self.battery - self.droneProtectingEnergyConsumption
                else:
                    self.move()
            if self.state == DroneState.CHARGING:
                if self.battery >= 1:
                    self.targetCharger = None
                    self.state = DroneState.IDLE
            self.checkBattery()

        do()
        if self.targetField is not None and self.state not in [DroneState.PROTECTING, DroneState.MOVING_TO_FIELD]:
            print(f"targetField not None OCCURED {str(DroneState(self.state))}")
            self.targetField.unassign(self)

    def isProtecting(self, point):
        """
        Checks if the given point is protected by the drone.

        Parameters
        ----------
        point : Point
            A given point on the field.

        Returns
        -------
        bool
            True if the given point is within the radius of the drone and drone's state is protecting.
        """
        return (self.state == DroneState.PROTECTING or self.state == DroneState.MOVING_TO_FIELD) and self.location.distance(
            point) <= self.droneRadius

    def protectRadius(self):
        """
        Gives the radius of the drone as form of rectangle to be presented in visualization.

        Returns
        -------
        Point, Point, Point, Point
            Top X, Top Y, Bottom X and Bottom Y
        """
        startX = self.location.x - self.droneRadius
        endX = self.location.x + self.droneRadius
        startY = self.location.y - self.droneRadius
        endY = self.location.y + self.droneRadius
        startX = 0 if startX < 0 else startX
        startY = 0 if startY < 0 else startY
        return startX, startY, endX, endY

    @property
    def int_id(self):
        return self._id - 1

    def __repr__(self):
        """
        Returns
        -------
        string
            Represent the drone instance in one line.
        """
        return f"{self.id}: {str(self.state)}, {str(self.nn_wish)} battery={self.battery}"
