import math
from typing import List, TYPE_CHECKING
from world import ENVIRONMENT, WORLD
from components.drone_state import DroneState
from components.drone_nn_wish import DroneNNWish
from components.drone import Drone
from ml_deeco.estimators import NumericFeature, CategoricalFeature
from ml_deeco.simulation import Ensemble, someOf
from ml_deeco.utils import verbosePrint
if TYPE_CHECKING:
    from components.charger import Charger

# The order of the ensembles is:
#  1. DroneChargingPreassignment
#  2. DroneChargingAssignment
#  3. AcceptedDronesAssignment

# def verbosePrint(message, minVerbosity):
#     print("---=" * (minVerbosity - 1), end="")
#     print(message)

class DroneChargingPreAssignment(Ensemble):
    """

    The Pre-assignment tells the drones that where is the potential charger and vise-versa.
    It has to come first; therefore, it has priority of 3.

    Parameters
    ----------
    charger : Charger
        The charger component.

    Properties:
    ---------
    drones: List (someOf) Drones
    """
    charger: 'Charger'

    def __init__(self, charger: 'Charger'):
        """

        Initiate the pre-assignment charger ensemble.

        Parameters
        ----------
        charger : Charger
            the targer charger.
        """
        self.charger = charger

    def priority(self):
        """

        Arbitrary as 3, to make sure the ensemble works before others.

        Returns
        -------
        int
            3
        """
        self.Priority = 3
        return self.Priority

    drones: List[Drone] = someOf(Drone)

    @drones.cardinality
    def drones(self):
        """

        The length of drones list is defined.

        Returns
        -------
        tuple
            [0, all drones in the environment]
        """

        self.Cardinality = 0, ENVIRONMENT.droneCount
        return self.Cardinality

    @drones.select
    def drones(self, drone, otherEnsembles):
        """

        Defines which drones are the potential ones to the charger.

        Parameters
        ----------
        drone : Drone
            The query drone.
        otherEnsembles : list
            unused in this concept, followed the definition of ensemble.

        Returns
        -------
        bool
            If the drone is selected.
        """
        return drone.state not in (DroneState.TERMINATED,  DroneState.CHARGING) and \
               (drone.nn_wish == DroneNNWish.CHARGE) and \
            drone.findClosestCharger() == self.charger
        # drone.nn_wish == DroneNNWish.CHARGE and \
        # drone.state == DroneState.MOVING_TO_CHARGER or

        # or rather better - which chargers are for a given drone accesible
        # drone.computeBatteryAfterTime(drone.timeToFlyToCharger(self.charger)) > 0

    def actuate(self):
        """

        For all selected drones, the potential charger is set.
        For the charger, the list of potential drones is set.
        """
        self.charger.waitingDrones = self.drones
        # self.charger.potentialDrones = self.drones
        for drone in self.drones:
            drone.closestCharger = self.charger
        verbosePrint(f"DroneChargingPreassignment: assigned {len(self.drones)} to {self.charger.id}", 4)


class AcceptedDronesAssignment(Ensemble):
    """

    This ensemble only ensure that the drones are being charged in with the charger.

    Parameters
    ----------
    charger : Charger
        The charger component.

    Properties:
    ---------
    drones: List (someOf) Drones 
    """
    charger: 'Charger'

    def __init__(self, charger: 'Charger'):
        """

        Initiate the ensemble.

        Parameters
        ----------
        charger : Charger
            The targeted charger.
        """
        self.charger = charger

    def priority(self):
        """

        Arbitrary set as 1, ensuring it will come after Pre-Assignment ensemble and the accepting ensemble.

        Returns
        -------
        int
            1
        """
        self.Priority = 1  # The order of AcceptedDronesAssignment ensembles can be arbitrary as they don't influence each other.
        return self.Priority

    drones: List[Drone] = someOf(Drone)

    @drones.cardinality
    def drones(self):
        """

        The length of drones.

        Returns
        -------
        tuple
            [0, charger capacity]
        """
        self.Cardinality = 0, self.charger.acceptedCapacity
        return self.Cardinality

    @drones.select
    def drones(self, drone, otherEnsembles):
        """

        Selects the drone to be charrged :
        1- the drone is not terminated
        2- drone is accepted by the charger or the drone is in waiting queue
        3- the charger will be free before/close to the time drone flies there

        Parameters
        ----------
        drone : Drone
            The query drone.
        otherEnsembles : list
            unused in this concept, following the definition of ensemble.

        Returns
        -------
        bool
            If True, the drone is accepted.
        """
        if drone.state == DroneState.TERMINATED:
            return False
        # was accepted before or needs charging (is waiting) and the charger will be free
        return drone in self.charger.acceptedDrones or \
            drone in self.charger.waitingDrones and \
            self.charger.timeToDoneCharging(len(self.drones)) <= drone.timeToFlyToCharger()

    @drones.utility
    def drones(self, drone):
        """

        sorts the drone toward their time to done charging.

        Parameters
        ----------
        drone : Drone
            The candidate drone.

        Returns
        -------
        int
            time to done charing.
        """
        if drone in self.charger.acceptedDrones:
            return 1  # keep the accepted drones from previous time steps
        return -drone.timeToDoneCharging()

    def actuate(self):
        """

        Updates the accepted drone list.
        """
        verbosePrint(f"AcceptedDronesAssignment: assigned {len(self.drones)} to {self.charger.id}", 4)
        self.charger.acceptedDrones = self.drones


ensembles: List[Ensemble]


def getEnsembles(args):
    """

    creates a list of ensembles for all types of charging assignments.

    Returns
    -------
    list
        List of ensembles.
    """
    global ensembles

    ensembles = [
        DroneChargingPreAssignment(charger) for charger in WORLD.chargers
    ] + [
        AcceptedDronesAssignment(charger) for charger in WORLD.chargers
    ]

    return ensembles
