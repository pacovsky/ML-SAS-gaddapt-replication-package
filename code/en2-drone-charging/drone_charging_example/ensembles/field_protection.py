import random
from typing import TYPE_CHECKING
from world import WORLD
from components.drone_state import DroneState
from components.drone_nn_wish import DroneNNWish
from components.drone import Drone
from ml_deeco.simulation import Ensemble, oneOf, someOf
from ml_deeco.utils import verbosePrint
if TYPE_CHECKING:
    from components.field import Field


class FieldProtection(Ensemble):
    """

    The field protection is materialized with one field and one drone.
    The drone is selected if it is idle, to protect the field.

    Parameters
    ----------
    field : Field
        The field component.

    Properties:
    ---------
    drone: Drone (oneOf) Drones
    """

    field: 'Field'

    def __init__(self, field: 'Field'):
        """
        Initate the ensemble.

        Parameters
        ----------
        field : Field
            The field which needs protection.
        """
        self.field = field

    drones: Drone = someOf(Drone)

    def priority_turned_on(self):
        food_left = self.field.allCrops - self.field.damage
        remaining_food_per_drone = food_left / len(self.field.places)
        free_spaces = len(self.field.places) - len(self.field.protectingDrones)
        unprotected_crops = remaining_food_per_drone * free_spaces

        # how many birds on the field
        birds_rate = sum(1 for b in WORLD.birds if b.field == self.field) / len(WORLD.birds)

        try:
            self.field_ratio
            self.total_world_food
        except AttributeError:
            self.field_ratio = len(self.field.places) / sum(len(f.places) for f in WORLD.fields)
            self.total_world_food = sum((f.allCrops) for f in WORLD.fields)

        # print(" " * 10, "computing priority", self.field.id, original(), f'{remaining_food_per_drone=}', f'{unprotected_crops=}')
        self.Priority = birds_rate * unprotected_crops / self.total_world_food + self.field_ratio
        # small value to push forward the larger field - the fluctuations there will be more common
        return self.Priority

    def priority(self):
        """

        The ensembles are sorted to the priority. 
        If a field has no protectors, it will come as a negative int < 0.
        Otherwise they are sorted to the fact which has less protectors.

        Returns
        -------
        float
            The importance of the field.
        """
        return 0.1  # arbitrary number - we use ens just to trigger the state

        if len(self.field.protectingDrones) == 0:
            return -len(self.field.places)
        # if there is no drone assigned, it tries to assign at least one
        return len(self.field.protectingDrones) / len(self.field.places)

    @drones.cardinality
    def drones(self):
        return self.get_cardinality()

    def get_cardinality(self):
        return 0, len(self.field.places)

    @drones.select
    def drones(self, drone, otherEnsembles):
        """
    
        Selects the drone to be used for protecting.

        Parameters
        ----------
        drone : Drone
            The drone to be selected.
        otherEnsembles : List
            unused in this concept, but defined by Ensemble definition.

        Returns
        -------
        bool
            if the drone is selected or not.
        """

        def not_in_other_ensemble(self):
            # method to do the print
            res = not any(ens for ens in otherEnsembles if isinstance(ens, FieldProtection) and drone in ens.drones)
            print("called not_in_other_ensemble", res, self.field, drone)
            return res

        # drone_idle = drone.state == DroneState.IDLE
        # field_has_spaces = len(self.field.places) > len(self.field.protectingDrones)
        # drone_in_other_field = any(ens for ens in otherEnsembles if isinstance(ens, FieldProtection) and drone in ens.drones)
        # assign = not drone_in_other_field and drone_idle and field_has_spaces
        # return assign
        # if random.random() > 0.99:
        #     print("without other ens instances")
        # return drone.state == DroneState.IDLE and \
        #     len(self.field.places) > len(self.field.protectingDrones) and \
        #     not any(ens for ens in otherEnsembles if isinstance(ens, FieldProtection) and drone in ens.drones)

        # return drone.state == DroneState.IDLE and \
        #         len(self.field.places) > len(self.field.protectingDrones) and \
        #         drone.nn_wish == DroneNNWish(self.field._id)

        # takes idle OR just charged drones that are not already assigned to any other filed
        # if the current field has free spaces and still some undamaged crops
        # and the nn_wish is in [fly_to_this_field or do_nothing]

        # TODO this might overfill the fields
        # len(self.field.places) > len(self.field.protectingDrones)
        # or better test the drones - are they on the field (fly 2 and on the field difference)

        return drone.state != DroneState.TERMINATED and (
            drone.nn_wish == DroneNNWish(self.field._id)
           # ) and (
           #  not_in_other_ensemble(self)
        )


        # return (
        #     # (drone.state == DroneState.IDLE or drone.battery == 1 and drone.state != DroneState.MOVING_TO_FIELD) and
        #        #todo test without the above condition
        #     len(self.field.places) > len(self.field.protectingDrones) and
        #     self.field.allCrops != self.field.damage
        #    ) and (
        #     drone.nn_wish == DroneNNWish(self.field._id) or (
        #     # drone.nn_wish in [DroneNNWish.NONE, DroneNNWish.CHARGE]) and
        #     drone.nn_wish in [DroneNNWish.NONE]) and
        #     # contrary to the name it computes energy to field as well
        #     (drone.battery - drone.energyToFlyToCharger(self.field)) > drone.alert
        #    ) and (
        #     not_in_other_ensemble(self)
        # )

    @drones.utility
    def drones(self, drone):
        """

        Utilize the drone list to select the most suitable drone.
        In this case the closest drone will work better.

        Parameters
        ----------
        drone : Drone
            The drone to be selected.

        Returns
        -------
        int
            The distance of the drone to the closest field.
        """

        #TODO utility function should follow the nn_wish first
        # drone.nn_wish == DroneNNWish(self.field._id)
        return - self.field.closestDistanceToDrone(drone)  # == quickest arrival

    def actuate(self):
        """
        Assing selected drone to the field, indirectly.
        Basically, this ensemble tells the drone which field it must protect.
        """
        for d in self.drones:
            d.targetField = self.field
            d.state = DroneState.MOVING_TO_FIELD
            # verbosePrint(f"Protecting Ensemble: assigning {d.id} to {self.field.id}", 0)

def getEnsembles(args):
    """

    Creates a list of ensembles per each field.

    Returns
    -------
    list
        List of field protection ensembles
    """
    ensembles = [FieldProtection(field) for field in WORLD.fields]
    return ensembles
