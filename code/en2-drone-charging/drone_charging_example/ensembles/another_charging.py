import math
from typing import List, TYPE_CHECKING
from world import ENVIRONMENT, WORLD
from components.drone_state import DroneState
from components.drone_nn_wish import DroneNNWish
from components.drone import Drone
from components.charger import Charger
from ml_deeco.estimators import NumericFeature, CategoricalFeature
from ml_deeco.simulation import Ensemble, someOf
from ml_deeco.utils import verbosePrint
MAXINT = 999
FILE_VERBOSITY = 0

# The order of the ensembles is:
# 1. ReachableDrones
# 2. LowBatteryDrones
# 3. EnoughBatteryDrones
# 4. ReachableChargers
# 5.


def reset_drone_variables(drone):
    # charger.forced.append(self.drone)  # TODO null in top most ensemble
    # self.drone.targetCharger = charger #resets
    # self.drone.target = charger.location #resets
    # self.drone.reachableChargers = self.chargers #resets

    # ch.join_queue(d)
    pass


class WaitingDronesInCharger(Ensemble):
    """
    Allows one drone to wait in queue.

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
        self.charger = charger

    def priority(self):
        self.Priority = MAXINT + 1 #really first
        return self.Priority

    drones: List[Drone] = someOf(Drone)

    @drones.cardinality
    def drones(self):
        return self.get_cardinality()

    def get_cardinality(self):
        return 0, 1

    @drones.select
    def drones(self, drone, otherEnsembles):
        charger = self.charger
        charger.lastWaitingDrones = charger.waitingDrones
        charger.waitingDrones = []

        return False

    def actuate(self):
        pass


class ReachableDrones(Ensemble):
    """
    The Pre-assignment tells the charger and vise-versa which are the potential drones.
    It has to come first; therefore, it has the biggest priority.

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

        Arbitrary high, to make sure the ensemble works before others.

        Returns
        -------
        int
            maxit
        """
        self.Priority = MAXINT
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
        return self.get_cardinality()

    def get_cardinality(self):
        return 0, ENVIRONMENT.droneCount

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

        # which chargers are for a given drone accessible
        # return drone.state not in (DroneState.TERMINATED, DroneState.MOVING_TO_CHARGER, DroneState.CHARGING) and \

        #TODO possible to add also the ForcedCharging - this flag would be removed by start charging/termination only
        return drone.state not in (DroneState.TERMINATED, DroneState.CHARGING) and \
           (drone.battery - drone.energyToFlyToCharger(self.charger)) > 0

    def actuate(self):
        """
        For all selected drones, the potential charger is set.
        For the charger, the list of potential drones is set.
        """
        self.charger.reachableDrones = self.drones

        # TODO remove
        # for drone in self.drones:
        #     drone.closestCharger = self.charger
        # verbosePrint(f"ReachableDrones: assigned {len(self.drones)} to {self.charger.id}", 4)


class LowBatteryDrones(Ensemble):
    charger: 'Charger'

    def __init__(self, charger: 'Charger'):
        self.charger = charger

    def priority(self):
        self.Priority = MAXINT - 1
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

        return self.get_cardinality()

    def get_cardinality(self):
        return 0, ENVIRONMENT.droneCount

    @drones.select
    def drones(self, drone, otherEnsembles):
        """

        Defines which drones are the potential ones to the charger.

        Parameters
        ----------
        drone : Drone
            The query drone.
        otherEnsembles : list
            at this moment just ReachableDrones.

        Returns
        -------
        bool
            If the drone is selected.
        """

        # which chargers are for a given drone accessible
        # return drone.state not in (DroneState.TERMINATED, DroneState.MOVING_TO_CHARGER, DroneState.CHARGING) and \

        # ens is just ReachableDrones
        ens = [ens for ens in otherEnsembles if ens.charger == self.charger]
        assert len(ens) == 1
        reachableDrones = ens[0]
        # assert type(reachableDrones) == ReachableDrones
        return drone in reachableDrones.drones and \
               (drone.battery - drone.energyToFlyToCharger(self.charger)) <= drone.alert


    def actuate(self):
        """
        For all selected drones, the potential charger is set.
        For the charger, the list of potential drones is set.
        """
        self.charger.lowBatteryDrones = self.drones


class AllowedChargingDrones(Ensemble):
    """Allowed to charge on battery level drones"""
    charger: 'Charger'

    def __init__(self, charger: 'Charger'):
        self.charger = charger

    def priority(self):
        self.Priority = MAXINT - 5
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

        return self.get_cardinality()

    def get_cardinality(self):
        return 0, ENVIRONMENT.droneCount

    @drones.select
    def drones(self, drone, otherEnsembles):
        """

        Defines which drones are the potential ones to the charger.

        Parameters
        ----------
        drone : Drone
            The query drone.
        otherEnsembles : list
            at this moment just ReachableDrones.

        Returns
        -------
        bool
            If the drone is selected.
        """

        # which chargers are for a given drone accessible
        # return drone.state not in (DroneState.TERMINATED, DroneState.MOVING_TO_CHARGER, DroneState.CHARGING) and \

        # ens is just ReachableDrones
        ens = [ens for ens in otherEnsembles if ens.charger == self.charger]
        assert len(ens) == 1
        reachableDrones = ens[0]
        # assert type(reachableDrones) == ReachableDrones
        return drone in reachableDrones.drones and \
               (drone.battery - drone.energyToFlyToCharger(self.charger)) <= (drone.alert * 2)


    def actuate(self):
        """
        For all selected drones, the potential charger is set.
        For the charger, the list of potential drones is set.
        """
        self.charger.lowBatteryDrones = self.drones


class EnoughBatteryDrones(Ensemble):
    """
    The EnoughBatteryDrones answers question if the drone can reach any charger without triggering alert level.
    #
    # Parameters
    # ----------
    # charger : Charger
    #     The charger component.
    #
    # Properties:
    # ---------
    # drones: List (someOf) Drones
    """
    chargers: List['Charger']

    def __init__(self, chargers: List['Charger']):
        self.chargers = chargers

    def priority(self):
        self.Priority = MAXINT - 2
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

        return self.get_cardinality()

    def get_cardinality(self):
        return 0, ENVIRONMENT.droneCount

    @drones.select
    def drones(self, drone, otherEnsembles):
        """

        Defines which drones are the potential ones to the charger.

        Parameters
        ----------
        drone : Drone
            The query drone.
        otherEnsembles : list
            at this moment ReachableDrones and LowBatteryDrones

        Returns
        -------
        bool
            If the drone is selected.
        """
        drone.can_reach_charger_before_alert = False #althoug there is another charger that triggers the alert

        for charger in self.chargers:
            ens = [ens for ens in otherEnsembles if ens.charger == charger]
            assert 1 <= len(ens) <= 2, "always reachable sometimes low battery"

            reachableDrones = ens[0]  # can we trust the order of ensembles??
            assert type(reachableDrones) == ReachableDrones
            assert len(ens) == 1 or type(ens[1]) == LowBatteryDrones
            if drone in reachableDrones.drones and len(ens) == 1:
                return True
        return False

    def actuate(self):
        """
        For all selected drones, the has enough energy.

        """
        for d in self.drones:
            d.can_reach_charger_before_alert = True


class ReachableChargers(Ensemble):
    """
    The Pre-assignment tells the charger and vise-versa which are the potential drones.
    It has to come first; therefore, it has the biggest priority.

    Parameters
    ----------
    charger : Charger
        The charger component.

    Properties:
    ---------
    drones: List (someOf) Drones
    """
    drone: 'Drone'

    def __init__(self, drone: 'Drone'):
        """

        Initiate the pre-assignment charger ensemble.

        Parameters
        ----------
        charger : Charger
            the targer charger.
        """
        self.drone = drone
        # drone. = self.drone.findClosestCharger()

    def priority(self):
        """

        Arbitrary high, to make sure the ensemble works before others.

        Returns
        -------
        int
            maxit
        """
        self.Priority = MAXINT - 3
        return self.Priority

    chargers: List[Charger] = someOf(Charger)

    @chargers.cardinality
    def chargers(self):
        """

        The length of chargers list is defined.

        Returns
        -------
        tuple
            [0, all chargers in the environment]
        """

        return self.get_cardinality()

    def get_cardinality(self):
        return 0, ENVIRONMENT.chargerCount

    @chargers.select
    def chargers(self, charger, otherEnsembles):
        """

        Defines which chargers are the potential ones to the charger.

        Parameters
        ----------
        charger : Charger
            The query drone.
        otherEnsembles : list
            unused in this concept, followed the definition of ensemble.

        Returns
        -------
        bool
            If the drone is selected.
        """

        # which chargers are for a given drone accessible
        # return drone.state not in (DroneState.TERMINATED, DroneState.MOVING_TO_CHARGER, DroneState.CHARGING) and \

        return (self.drone.battery - self.drone.energyToFlyToCharger(charger)) > 0

    @chargers.utility
    def chargers(self, charger):
        # sort by the increasing distance to a given charger
        return - self.drone.location.sq_distance(charger.location)

    def actuate(self):
        """
        For all selected chargers, the potential charger is set.
        For the charger, the list of potential chargers is set.
        """
        self.drone.reachableChargers = self.chargers


class ForcedCharger(Ensemble):
    """Revels which drones have only one charger to choose from

    Parameters
    ----------
    charger : Charger
        The charger component.

    Properties:
    ---------
    drones: List (someOf) Drones
    """
    drone: 'Drone'

    def __init__(self, drone: 'Drone'):
        """

        Initiate the pre-assignment charger ensemble.

        Parameters
        ----------
        charger : Charger
            the targer charger.
        """
        self.drone = drone

    def priority(self):
        """

        Arbitrary high, to make sure the ensemble works before others.

        Returns
        -------
        int
            maxit
        """
        self.Priority = MAXINT - 4
        return self.Priority

    chargers: List[Charger] = someOf(Charger)

    @chargers.cardinality
    def chargers(self):
        """

        The length of chargers list is defined.

        Returns
        -------
        tuple
            [0, all chargers in the environment]
        """

        return self.get_cardinality()

    def get_cardinality(self):
        return 1, 1

    @chargers.select
    def chargers(self, charger, otherEnsembles):
        """

        Defines which chargers are the potential ones to the charger.

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
        return len(self.drone.reachableChargers) == 1 and \
               charger == self.drone.reachableChargers[0] and \
                self.drone.battery - self.drone.energyToFlyToCharger(charger) < self.drone.alert
        #TODO? not in waitingDrones and cca
        # # which chargers are for a given drone accessible
        # # return drone.state not in (DroneState.TERMINATED, DroneState.MOVING_TO_CHARGER, DroneState.CHARGING) and \
        #
        # #otherEnsembles[:-4]
        # ens = [ens for ens in otherEnsembles if hasattr(ens, 'drone') and ens.drone == self.drone]
        #
        # assert len(ens) == 1, len(ens)
        # reachableChargers = ens[0]
        # # assert type(reachableChargers) == ReachableChargers
        # return len(reachableChargers.chargers) == 1

    def actuate(self):
        """
        For all selected drones, the potential charger is set.
        For the charger, the list of potential drones is set.
        """
        if self.chargers:
            assert len(self.chargers) == 1
            charger = self.chargers[0]

            if self.drone not in charger.waitingDrones:
                charger.waitingDrones.append(self.drone)

            self.drone.targetCharger = charger
            self.drone.target = charger.location
            self.drone.state = DroneState.MOVING_TO_CHARGER
            verbosePrint(f"Forced: assigned {(self.drone)} to {charger.id}", FILE_VERBOSITY)


# class EnoughBatteryDrones2(Ensemble):
#     charger: 'Charger'
#
#     def __init__(self, charger: 'Charger'):
#         self.charger = charger
#
#     def priority(self):
#       self.Priority = MAXINT - 2
#       return self.Priority
#
#     drones: List[Drone] = someOf(Drone)
#
#     @drones.cardinality
#     def drones(self):
#         """
#
#         The length of drones list is defined.
#
#         Returns
#         -------
#         tuple
#             [0, all drones in the environment]
#         """
#         return self.get_cardinality()
#     
#     def get_cardinality():
#         return 0, ENVIRONMENT.droneCount
#
#
#     @drones.select
#     def drones(self, drone, otherEnsembles):
#         """
#
#         Defines which drones are the potential ones to the charger.
#
#         Parameters
#         ----------
#         drone : Drone
#             The query drone.
#         otherEnsembles : list
#             at this moment ReachableDrones and LowBatteryDrones
#
#         Returns
#         -------
#         bool
#             If the drone is selected.
#         """
#
#         ens = [ens for ens in otherEnsembles if ens.charger == self.charger]
#         assert 1 <= len(ens) <= 2, "always reachable sometimes low battery"
#         reachableDrones = ens[0] # can we trust the order of ensembles??
#         assert type(reachableDrones) == ReachableDrones
#         assert len(ens) == 1 or  type(ens[1]) == LowBatteryDrones
#         return drone in reachableDrones.drones and len(ens) == 1
#
#     def actuate(self):
#         """
#         For all selected drones, the potential charger is set.
#         For the charger, the list of potential drones is set.
#         """
#         self.charger.enoughBatteryDrones = self.drones

class GlobalQueue(Ensemble):
    # QUEUE:= Reachable - Forced & NN || MAX energy by the eta (on different chs) <= drone.alert
    # sort by the MAX energy (form smallest)
    # foreach drone in QUEUE up to number of drones
    # try to assign drone to the charger, until no charger in drones reachables or successful assignment
    #todo check the prev ensembles
    # todo use somehow the result
    """
    The Pre-assignment tells the charger and vise-versa which are the potential drones.
    It has to come first; therefore, it has the biggest priority.

    Parameters
    ----------
    charger : Charger
        The charger component.

    Properties:
    ---------
    drones: List (someOf) Drones
    """

    def __init__(self):
        """

        Initiate the pre-assignment charger ensemble.

        Parameters
        ----------
        charger : Charger
            the targer charger.
        """

    def priority(self):
        """

        Arbitrary high, to make sure the ensemble works before others.

        Returns
        -------
        int
            maxit
        """
        self.Priority = MAXINT - 6
        return self.Priority

    drones: List[Drone] = someOf(Drone)
    otherEnsembles: List[Ensemble] = None

    @drones.cardinality
    def drones(self):
        """

        The length of drones list is defined.

        Returns
        -------
        tuple
            [0, all drones in the environment]
        """
        return self.get_cardinality()

    def get_cardinality(self):
        return 0, ENVIRONMENT.droneCount


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
        # # ReachableChargers = [ens for ens in otherEnsembles if hasattr(ens, 'drone') and ens.drone == drone][0].chargers
        # # [ens for ens in otherEnsembles if type(ens) == ReachableChargers]
        self.otherEnsembles = otherEnsembles

        def remaining_energy_level_upon_arrival_to_nearest_charger(drone):
            # nearest charger
            if drone.reachableChargers:
                return drone.battery - max(drone.energyToFlyToCharger(ch) for ch in drone.reachableChargers)
            return False  # will run out of battery

        # ens = [ens for ens in otherEnsembles if hasattr(ens, 'drone') and ens.drone == drone]
        return drone.state not in (DroneState.TERMINATED, DroneState.CHARGING) and \
            (drone not in [ens for ens in otherEnsembles if type(ens) == ForcedCharger]) and \
            (drone.nn_wish == DroneNNWish.CHARGE or
                remaining_energy_level_upon_arrival_to_nearest_charger(drone) <= drone.alert)

    @drones.utility
    def drones(self, drone):
        # sort from smallest energy buffer to highest
        nearest_charger = drone.findClosestCharger()
        return -(drone.battery - drone.energyToFlyToCharger(nearest_charger))

    def actuate(self):
        """
        For all selected drones, the potential charger is set.
        For the charger, the list of potential drones is set.
        """
        # reachableChargers = [ens for ens in self.otherEnsembles if hasattr(ens, 'drone') and type(ens) == ReachableChargers]
        # try to assign drone to the charger, until no charger in drones reachable's or successful assignment
        for d in self.drones:
            # for ch in [e for e in reachableChargers if e.drone == d][0].chargers:
            found = False
            for ch in d.reachableChargers:
                # if ch.join_queue(d): # TODO
                if d in ch.acceptedDrones:
                    found = True
                    break

            if found:
                # drone already accepted
                continue

            for ch in d.reachableChargers:
                # TODO what about the ch.chargingDrones ????
                if ch.acceptedCapacity > len(ch.acceptedDrones) and \
                        ch.timeToDoneCharging(len(ch.acceptedDrones)) <= d.timeToFlyToCharger():
                    ch.acceptedDrones.append(d)
                    d.targetCharger = ch
                    d.target = d.targetCharger.location
                    d.state = DroneState.MOVING_TO_CHARGER

                    # remove from other ch, and from other list in this ch
                    for other_ch in WORLD.chargers:
                        if ch != other_ch:
                            if d in other_ch.acceptedDrones:
                                print("THIS HAPPEND") #TODO remove
                                other_ch.acceptedDrones.remove(d)
                        if d in other_ch.waitingDrones:
                            other_ch.waitingDrones.remove(d)
                            print("THIS HAPPPEND") #TODO remove

                    break

        # verbosePrint(f"ReachableDrones: assigned {len(self.drones)} to {self.charger.id}", 4)



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
        return 1  # The order of AcceptedDronesAssignment ensembles can be arbitrary as they don't influence each other.

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
        return self.get_cardinality()

    def get_cardinality(self):
        return 0, self.charger.acceptedCapacity

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
        if len(self.drones) != len(set(self.drones)): # TODO remove
            print("\n" *20, "MULTIPLE ASSIGNMENT!")
            breakpoint()
            raise 32
        self.charger.acceptedDrones = self.drones

#TODO histeresis - add to drone last predictions
# TODO Global Ensemble
#             self.register_global_red_flag(TooMuchDronesAssignToTheCharger())
#             self.register_global_red_flag(WrongDroneCountField())

# class DroneChargingAssignment(Ensemble):
#     """
#
#     The drone charging assignment checks if any potential drones requires charging.
#     In this ensemble, the data for ML-Based model is collected.
#     The priority of this ensemble is 2, ensuring that it will run before accepting.
#     The drones that are selected will be added to the waiting queue.
#
#     Parameters
#     ----------
#     charger : Charger
#         The charger component.
#
#     Properties:
#     ---------
#     drones: List (someOf) Drones With Time Estimator
#     """
#     charger: 'Charger'
#
#     def __init__(self, charger: 'Charger'):
#         """
#
#         initiate the charging ensemble.
#
#         Parameters
#         ----------
#         charger : Charger
#             The targetted charger.
#         """
#         self.charger = charger
#
#     def priority(self):
#         """
#
#         Arbitrary set as 2, ensuring it will come after Pre-Assignment ensemble and before the accepting ensemble.
#
#         Returns
#         -------
#         int
#             2
#         """
#         return 2
#
#     drones: List[Drone] = someOf(Drone).withTimeEstimate().using(WORLD.waitingTimeEstimator)
#
#     @drones.cardinality
#     def drones(self):
#         """
#
#         The length of drones.
#
#         Returns
#         -------
#         tuple
#             [0, all drones in the environment]
#         """
#         return self.get_cardinality()
#
#     def get_cardinality(self):
#         return 0, ENVIRONMENT.droneCount
#
#     @drones.select
#     def drones(self, drone, otherEnsembles):
#         """
#
#         Select the drone to be in the waiting queue, which:
#         1-  Not Terminated
#         2-  Drone in potentail drones.
#         3-  Selected by the NN as needing charging
#         # 3-  Needs charging: in ML-based, the waiting time is estimated
#
#
#         Parameters
#         ----------
#         drone : Drone
#             The query drone.
#         otherEnsembles : list
#             unused here, following the definition of the ensemble.
#
#         Returns
#         -------
#         bool
#             if True, the drone is selected to be in waiting queue.
#         """
#         if drone.state == DroneState.TERMINATED:
#             return False
#
#         return drone in self.charger.potentialDrones
#
#         waitingTimeEstimate = self.chargers.estimate(drone)
#         timeToFlyToCharger = drone.timeToFlyToCharger()
#         # needs charging
#         return drone in self.charger.potentialDrones and \
#                drone.nn_wish == DroneNNWish.CHARGE
#         # drone.needsCharging(waitingTimeEstimate + timeToFlyToCharger)
#         # it would be possible to do some kind of load balancing with other ensembles
#
#     def time_to_empty_space(self):
#         """just an estimate for the first change in the charger
#         returns 0 if there are still free places in the charger
#         or returns time needed for the first drone to leave the charger
#          """
#         if len(self.charger.chargingDrones) <= self.charger.acceptedCapacity:
#             return 0
#         minimum_to_charge = min([1 - drone.battery for drone in self.charger.chargingDrones])
#         return minimum_to_charge / self.charger.chargingRate  # todo in this simulation it is not a constant
#
#     def time_to_empty_space_precise(self):
#         """what do I need to precisely compute:
#                 queue of waiting chargers -
#                     for the chargers that are not yet charging:
#                         batttery level when they get to be charged
#                         compleation time from that battery level up to 100%
#                     number of spaces in the charger
#                     charging speed - !!! in this simulation it is not a constant!!!
#
#         """
#         pass
#
#     def poznamky(self):
#         """
#             potential chargers - the ones that can reach at this moment this charger
#             want to be charged - the ones that the ML chose to put in the charger
#             waiting queue chargers assigned only to this charger
#             waiting queue - chargers assigned to multiple chargers
#                 x -------------- x
#                            ooo
#                  \--------/ \\//
#                  GOAL: the one dr[o]ne should go left, the rest right
#                         the two chargers [x] may have capacity 1 or 2 to make sense of it
#
#             if the drone is in multiple ens chargers the ens should decide to take the drone themselves
#             utility functions - closeness - if there would be another chargers on the left one they would get there first...
#                             - so we need extra - if there is a coming drone with higher priority don't let other drone come
#                             - battery level at arrival
#
#             how to force first fill the closest charger and then the rest???
#         """
#         pass
#
#
#     def is_preassigned(self, drone):
#
#         # return drone in self.chargers
#         return drone in self.charger.potentialDrones
#
#     def is_accepted(self, drone):
#         return drone in self.charger.acceptedDrones
#
#     def actuate(self):
#         """
#
#         The waiting queue will be updated to the list of current chargers.
#         """
#         verbosePrint(f"DroneChargingAssignment: assigned {len(self.drones)} to {self.charger.id}", 4)
#         # if self.chargers:
#         #     print(f"DEBUG: self.charger.waitingDrones = self.chargers {self.chargers}")
#         self.charger.waitingDrones = self.drones

# ensembles: List[Ensemble]

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
        # WaitingDronesInCharger(charger) for charger in WORLD.chargers] + [
        ReachableChargers(drone) for drone in WORLD.drones] + [
        ForcedCharger(drone) for drone in WORLD.drones] + [
        GlobalQueue()
    ]
        # [ReachableDrones(charger) for charger in WORLD.chargers] + \
        # [LowBatteryDrones(charger) for charger in WORLD.chargers] + \
        # [EnoughBatteryDrones(WORLD.chargers)] + \
    return ensembles
