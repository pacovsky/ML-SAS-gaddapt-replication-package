import itertools
from datetime import datetime
from typing import Optional, Callable, List, Tuple, TYPE_CHECKING
from drone_charging_example.components.drone_nn_wish import DroneNNWish, RedFlagsEnum
from drone_charging_example.components.drone_state import DroneState
from drone_charging_example.red_flags.redflags import RedFlags, Hysteresis
from drone_charging_example.red_flags.preprocess_data import process_data, xy_to_distance
from drone_charging_example.red_flags.simulation_setup import SimulationSetup, Singleton
from drone_charging_example.components.option_enums import SetState, SimulationType, RedflagsEnum
from ensemble_resolution.fall_to_ensemble import FALL_TO_ENSEMBLE, CHARGING_ENSEMBLE


import numpy as np
import tensorflow as tf
import sys
import wandb
import os

from collections import namedtuple

from ml_deeco.utils import verbosePrint

if TYPE_CHECKING:
    from ml_deeco.simulation import Ensemble, Component


class SimulationGlobals:
    """
    Storage for the global simulation data such as a list of all estimators and the current time step.

    This simplifies the implementation as it allows for instance the TimeEstimate to access the current time step of the simulation.
    """

    def __init__(self):
        if 'SIMULATION_GLOBALS' in locals():
            raise RuntimeError("Do not create a new instance of the SimulationGlobals. Use the SIMULATION_GLOBALS global variable instead.")
        self.estimators = []
        self.currentTimeStep = 0


    def initEstimators(self):
        """Initialize the estimators. This has to be called after the components and ensembles are imported and before the simulation is run."""
        # for est in self.estimators:
        #     est.init()
        pass


SIMULATION_GLOBALS = SimulationGlobals()


def materialize_ensembles(components, ensembles):
    """
    Performs the materialization of all ensembles. That includes actuating the materialized ensembles and collecting data for the estimates.

    Parameters
    ----------
    components : List['Component']
        All components in the system.
    ensembles : List['Ensemble']
        All potential ensembles in the system.

    Returns
    -------
    List['Ensemble']
        The materialized ensembles.
    """
    materializedEnsembles = []

    potentialEnsembles = sorted(ensembles)
    for ens in potentialEnsembles:
        if ens.materialize(components, materializedEnsembles):
            materializedEnsembles.append(ens)
            ens.actuate()
    # for ens in materializedEnsembles:
    #     ens.collectEstimatesData(components)

    return materializedEnsembles


def actuate_components(components):
    """
    Performs component actuation. Runs the actuate function on all components and collects the data for the estimates.

    Parameters
    ----------
    components : List['Component']
        All components in the system.
    """
    for component in components:
        component.actuate()
        verbosePrint(f"{component}", 4)
    # for component in components:
    #     component.collectEstimatesData()


def run_simulation(
    args: object,
    components: List['Component'],
    ensembles: List['Ensemble'],
    steps: int,
    stepCallback: Optional[Callable[[List['Component'], List['Ensemble'], int, float, float, float, float], None]] = None,
):
    """
    Runs the simulation with `components` and `ensembles` for `steps` steps.

    Parameters
    ----------
    components
        All components in the system.
    ensembles
        All potential ensembles in the system.
    steps
        Number of steps to run.
    stepCallback
        This function is called after each simulation step. It can be used for example to log data from the simulation. The parameters are:
            - list of all components in the system,
            - list of materialized ensembles (in this time step),
            - current time step (int).
    """

    def update_setup(world, env, args):
        birds = world.birds
        drones = world.drones
        fields = world.fields
        chargers = world.chargers

        drone_count = len(drones)
        flock_count = len(birds)
        charger_capacity = env.chargerCapacity  # all chargers have the same value

        charger_centers = [(c.location.x, c.location.y) for c in chargers]

        fields_position = [(f.topLeft.x, f.topLeft.y, f.bottomRight.x, f.bottomRight.y) for f in fields]
        field_corners = np.array(fields_position, dtype=np.float16)
        field_centers = [((f[0]+f[2])/2, (f[1]+f[3])/2) for f in fields_position]

        drone_speed = drones[0].speed
        drone_consumption = max(drones[0].droneProtectingEnergyConsumption,
                                drones[0].droneMovingEnergyConsumption)

        s = SimulationSetup()
        s.change_simulation_setup(drone_count, flock_count, charger_capacity, charger_centers, field_corners,
                                  field_centers, drone_speed, drone_consumption, args.hysteresis, args.charger_option)

    def get_data_from_components():
        zero_label = np.zeros((1, len(WORLD.drones), 16), np.int8)

        chargers = np.array([x.get_the_ensml_compatible_row for x in WORLD.chargers])[np.newaxis, ...]
        drones = np.array([x.get_the_ensml_compatible_row for x in WORLD.drones])[np.newaxis, ...]
        flocks = np.array([x.get_the_ensml_compatible_row for x in WORLD.birds])[np.newaxis, ...]
        Components = namedtuple('Components', ['chargers', 'drones', 'flocks'])
        return Components(chargers, drones, flocks), zero_label

    def print_damage_info(do_print=True):
        totalDamage = sum([field.damage for field in WORLD.fields])
        totalCorp = sum([field.allCrops for field in WORLD.fields])
        deadDrones = len([drone for drone in WORLD.drones if drone.state == DroneState.TERMINATED])
        if do_print:
            print("-" * 20 + f"step: {step + 1}" +
              f" - Damage: {totalDamage}/{totalCorp} - Dead drones: {deadDrones}/{len(WORLD.drones)}")
        return totalDamage, totalCorp, deadDrones, len(WORLD.drones)

    def update_drones_by_the_nn(model, args):
        assert args.which_simulation in {
            SimulationType.random_behavior,
            SimulationType.predict_on_not_normalized,
            SimulationType.baseline,
            SimulationType.bettersimplebaseline,
            SimulationType.betterbaseline,
            SimulationType.nn,
            SimulationType.red_flagsF,
        }
        # needed at for the baseline
        for drone in WORLD.drones:
            drone.closestCharger = drone.findClosestCharger()

        # get "the other format" from this simulation components
        components_data = get_data_from_components()

        # TODO try with different NN
        # TODO try removing the caching behavior of the field position (at least make None, always after some time)

        # y values are empty
        X, _, common_dict, specific_dict = process_data(components_data, return_dict_as_well=True, normalize=False)

        if args.which_simulation == SimulationType.random_behavior:
            prediction_shape = (len(WORLD.drones), 7)  # number of outputs
            seed_state = np.random.get_state()
            random_prediction = np.random.random(prediction_shape)
            np.random.set_state(seed_state)
            random_prediction = tf.math.softmax(random_prediction).numpy()
            prediction = random_prediction
        elif args.which_simulation == SimulationType.predict_on_not_normalized:  # todo try also
            # another almost random assignment, just for sanity reasons
            # this should yield worse result
            prediction = model.predict(X)

        elif args.which_simulation == SimulationType.baseline:
            prediction = manual_rules(common_dict, specific_dict)

        elif args.which_simulation == SimulationType.red_flagsF:
            prediction = manual_rules(common_dict, specific_dict)
            rf = RedFlags(X, prediction, common_dict, specific_dict, components_data, WORLD, args)  # , sum_up_prediction=True)
            rf.process_red_flags(RedFlagsEnum.LOCAL_ONLY)

            prediction = rf.updated_prediction
            answer = np.argmax(prediction, axis=1)
            # answer2 = rf.get_answer()
            # assert all([DroneNNWish(a) == DroneNNWish(b) for a, b in zip(answer, answer2)])
            return answer


        elif args.which_simulation == SimulationType.nn:
            X, _ = process_data(components_data, return_dict_as_well=False)
            prediction = model.predict(X)
        else:
            raise NotImplementedError()


        # prediction hysteresis - this would probably be better to implement within the decision, if last step...
        if args.hysteresis:
            # h = SimulationSetup().Hysteresis#TODO hyster
            # prediction = h.compute_prediction(prediction)
            raise NotImplementedError() # missing added chosen state

        if args.redflags != RedflagsEnum.none:
            rf = RedFlags(X, prediction, common_dict, specific_dict, components_data, WORLD, args)  # , sum_up_prediction=True)
            rf.process_red_flags(RedFlagsEnum.LOCAL_ONLY)

            if args.redflags == RedflagsEnum.answer:
                answer = rf.get_answer()
            elif args.redflags == RedflagsEnum.argmax:
                prediction = rf.updated_prediction
                answer = np.argmax(prediction, axis=1)
        else:
            answer = np.argmax(prediction, axis=1)

            # prediction = rf.updated_prediction  # == hopefully equivalent to rf.get_answer()
            # if not np.all(np.argmax(prediction, axis=1) == np.array([int(i) for i in rf.get_answer()])):
            #     import sys
            #     print(np.argmax(prediction, axis=1), file=sys.stderr)
            #     print(np.array([int(i) for i in rf.get_answer()]), file=sys.stderr)
            #     raise Exception("FOUND OUT NOT MATCH")

        # for network of a type 110 the categories are: + charge, dead
        #   + plus each field has 2 values - protect and approach
        # none, a0, p0, a1, p1, a2, p2, a3, p3, a4, p4, a_chargers, u_chargers, dead

        # answer now == none, f0, f1, f2, f3, f4, chargers, dead

        # local red flags algorithm
        # check the argmax values for each drone separately
        # - # - # - # - # - # - # - # - # - # - # - # - # - # -
        # argmax NNc candidates, NO_solution, Optional_candidates, Selected_candidates
        # - # - # - # - # - # - # - # - # - # - # - # - # - # -
        #    argmax NNc candidates, rejected_states (falsifiable only), selected state(s) per drone
        # - # - # - # - # - # - # - # - # - # - # - # - # - # -
        #
        ## take top x argmax-es for a drone
        # for each drone (at once):
        #    take the argmax
        #    check viability of that solution by local red flags
        #    remove the solution if wrong (optionally add another possible candidate for a move)
        #
        # for drones with removed solutions:
        #   check if there is another NNc candidate (probability of action that is > 0)
        #   -> repeat until ^previous^condition^ is false or until there is something that was not removed
        #
        #   o> when there are no NNc candidates left, check the optional candidate list
        #       - intersect with NO_solution
        #       - check with previous method (watch for cycles) - when adding a candidate - check the NO_solution list
        #
        #   o> if this still yields nothing
        #       there is IDLE option (that itself might already be on NO_solution)
        #       check all the options not yet in NO_solution
        #      x> is there a way that everything might be wrong?
        #       I don't think so (with current red flags)
        #       # go to field if ENOUGH battery, idle when MEDIUM and go to charger when LOW should be
        #
        # this can be done for all of the options -> will produce (multiple) selected_candidates
        #
        # return list of selected_candidates for each drone - (in order of preferences)

        # global red flags algorithm
        # - # - # - # - # - # - # - # - # - # - # - # - # - # -
        # after the redflags locals are computed - chces the 3 matrixes / who is now the argmax - rejected solutions + sudgested - how to deal with suggestions? as a extra point for prropability  TODO

        #   start with all drones with respective selected_candidates (including numeric value of a preference=0 is (acceptable for added values))
        #
        #   on the red flag - not enough -
        #   on the red flag - too much -
        #
        #   local RED flags - list
        #   [x] no_go_to_field_OH_OH_go_to_charger (prepare ahead of time)
        #   [x] ChargeOnVeryLowEnergy
        #   [x] has-done-nothing-wants-to-charge
        #   [x] high-energy-want-to-charge
        #   [x] terminated with battery > 0
        #
        #   local BROWN flags
        #   [x] drone on empty field
        #   [x] high energy idle drones very far from birds
        #   alternatively:
        #       drones are on field that is far from birds2
        #   [] IDLE_drones_nearby_empty_charger_with_less_battery_then_they_would_have_if_they_came_straight_from_charger
        #       - this is not the same as  "has-done-nothing-wants-to-charge"
        #           - it is more like has done something, is idle, should be charging

        #
        #   global RED flag - list
        #   [] fields with birds without drones -- this should suggest not order to take care of that...

        #   [] too_much_drones_in_the_field_ens  # it is not really a problem if not counting the overbooking (IDLE drones...)
        #   [x] too_much_drones_in_the_charger
        #   [] drone_in_reach_of_occupied_field_but_does_not_go_to_the_field #global because someone else could (put it to the locals at first as local suggestion?)

        #   CONSIDERATION: getting rid of the BROWN flags
        #   fields have to say max number of available drones
        #   BROWN FLAGS - allow exceed this number of drones? (it is not strictly error)
        #   - e.g. drones are protecting, we know that the energy level of protecting drone is going to be low,
        #           so we prepare another to be near the field ahead of time
        #   but if all the drones with similar battery level are flying to the same field it is an error
        #   -> solution better tuned RED flags more fain grain / or stricter disallowing valid positions BROWN flags
        #
        #   BROWN flags
        #   drones_are_on_field_that_is_far_from_birds
        #   too_much_drones_in_charger
        #   too_much_drones_fly_to_the_field

        # CONTINUITY PROBLEMS
        #  ???how to tell??? with previous state?
        #   changing intent before reaching goal
        #       charging - battery is not yet 100%
        #       flying to a field - when the birds are still there
        # continuity problems? - history descions - is it ok to switch FIELD1, FIELD2, F1, F2? or should we stip?

        # prediction categories are: do nothing ; go to field nr#, go to charger and charge itself

        # in charger assign = go to charger
        # none, f0, f1, f2, f3, f4, chargers


        # if args.hysteresis: #TODO
        #     h = SimulationSetup().Hysteresis
        #     # if this is always added to 0 the magnitude does not mater
        #     h.save_chosen_states(
        #         (np.eye(7))[answer??? check if correct
        #     )


        for a, drone in zip(answer, WORLD.drones):
            if a > 6:
                # print(drone.id, "TERMINATED")
                a = 0
            if a == -1:
                print("""
                      raise RuntimeError("this is wrong state")
                      """)

            if args.set_state == SetState.use_set_state_minimal:
                drone.set_state_minimal(DroneNNWish(a))
            elif args.set_state == SetState.use_set_state:
                drone.set_state(DroneNNWish(a))
            elif args.set_state == SetState.set_wish:
                drone.nn_wish = DroneNNWish(a)
            elif args.set_state == SetState.do_nothing:
                pass
            else:
                raise NotImplementedError()

            #24 - 830, 942, 60, 60
            #24 -12, 14, 3, 3

            #8 - 740, 809, 598, 613 | 740 <-uplne bez
            # 5, 5, 0, 1, 5

    def manual_rules(common_dict, specific_dict):
        """baseline"""
        answers = []
        # for field in WORLD.fields: sorted by closeness, if the field is bird_only_fields
        birds_on_fields = common_dict['counts_0'].reshape(-1)
        drones_on_fields = common_dict['counts_1'].reshape(-1)
        birds_only_fields = common_dict['counts_2'].reshape(-1)
        for drone in WORLD.drones:
            # terminated drones
            if drone.state == DroneState.TERMINATED:
                answers.append(DroneNNWish.NONE)
                continue

            drone.closestCharger = drone.findClosestCharger()
            # if charging or in need of
            if drone.state not in [DroneState.MOVING_TO_CHARGER, DroneState.CHARGING] and \
                    drone.battery - drone.energyToFlyToCharger(drone.closestCharger) <= drone.alert:
                # fly to charger
                drone.target = drone.closestCharger.location
                answers.append(DroneNNWish.CHARGE)
                continue

            # continue_to_prev_field_target:
            if drone.targetField:
                if drone.target != drone.location:
                    answers.append(DroneNNWish(drone.targetField._id - 1))
                    continue

            # if birds on reachable field
            #   fly there
            # else: wait

            ascending_fields_idx = specific_dict['distances_drone_fields'][drone.int_id].argsort()
            ascending_fields_idx = np.squeeze(ascending_fields_idx)
            field_found = False

            # priority to go to field with birds without drones
            for i in ascending_fields_idx:
                if birds_only_fields[i]:
                    # energyToFlyToCharger computes the energy correctly to anything with .location local variable
                    birds_field = WORLD.fields[i]
                    if drone.battery - drone.energyToFlyToCharger(birds_field) >= drone.alert:
                        answers.append(DroneNNWish(i + 1))
                        field_found = True
                        break

                    else:
                        # ascending order causes always to hit this branch when there is no option
                        break

            if field_found:
                continue

            # all fields with bird already have drones, join the closest
            for i in ascending_fields_idx:
                if birds_on_fields[i]:
                    # energyToFlyToCharger computes the energy correctly to anything with .location local variable
                    birds_field = WORLD.fields[i]
                    if drone.battery - drone.energyToFlyToCharger(birds_field) >= drone.alert:
                        answers.append(DroneNNWish(i + 1))
                        field_found = True
                        break

                    else:
                        # ascending order causes always to hit this branch when there is no option
                        break

            if field_found:
                continue
            answers.append(DroneNNWish.NONE)

        return np.eye(7)[answers]

    def manual_rules2(common_dict, specific_dict):
        answers = []
        # for field in WORLD.fields: sorted by closeness, if the field is bird_only_fields
        birds_on_fields = common_dict['counts_0'].reshape(-1)
        drones_on_fields = common_dict['counts_1'].reshape(-1)
        birds_only_fields = common_dict['counts_2'].reshape(-1)
        for drone in WORLD.drones:
            # terminated drones
            if drone.state == DroneState.TERMINATED:
                answers.append(DroneNNWish.NONE)
                continue

            drone.closestCharger = drone.findClosestCharger()
            # if charging or in need of
            if drone.state not in [DroneState.MOVING_TO_CHARGER, DroneState.CHARGING] and \
                    drone.battery - drone.energyToFlyToCharger(drone.closestCharger) <= drone.alert:
                # fly to charger
                drone.target = drone.closestCharger.location
                answers.append(DroneNNWish.CHARGE)
                continue

            # continue_to_prev_field_target:
            if drone.targetField:
                if drone.target != drone.location:
                    answers.append(DroneNNWish(drone.targetField._id - 1))
                    continue


            ascending_fields_idx = specific_dict['distances_drone_fields'][drone.int_id].argsort()
            ascending_fields_idx = np.squeeze(ascending_fields_idx)
            field_found = False


            # priority to go to field with birds without drones
            for i in ascending_fields_idx:
                if birds_only_fields[i]:
                    # energyToFlyToCharger computes the energy correctly to anything with .location local variable
                    birds_field = WORLD.fields[i]
                    if drone.battery - drone.energyToFlyToCharger(birds_field) >= drone.alert:
                        answers.append(DroneNNWish(i + 1))
                        field_found = True
                        break

                    else:
                        # ascending order causes always to hit this branch when there is no option
                        break

            if field_found:
                continue

            # all fields with bird already have drones, join the closest
            for i in ascending_fields_idx:
                if birds_on_fields[i]:
                    # energyToFlyToCharger computes the energy correctly to anything with .location local variable
                    birds_field = WORLD.fields[i]
                    if drone.battery - drone.energyToFlyToCharger(birds_field) >= drone.alert:
                        answers.append(DroneNNWish(i + 1))
                        field_found = True
                        break

                    else:
                        # ascending order causes always to hit this branch when there is no option
                        break

            if field_found:
                continue
            answers.append(DroneNNWish.NONE)

        return np.eye(7)[answers]

    def manual_rules3(common_dict, specific_dict):
        """bettersimplebaseline"""
        answers = []
        # for field in WORLD.fields: sorted by closeness, if the field is bird_only_fields
        birds_on_fields = common_dict['counts_0'].reshape(-1)
        drones_on_fields = common_dict['counts_1'].reshape(-1)
        birds_only_fields = common_dict['counts_2'].reshape(-1)
        for drone in WORLD.drones:
            # terminated drones
            if drone.state == DroneState.TERMINATED:
                answers.append(DroneNNWish.NONE)
                continue

            drone.closestCharger = drone.findClosestCharger()
            # if charging or in need of
            if drone.state not in [DroneState.MOVING_TO_CHARGER, DroneState.CHARGING] and \
                    drone.battery - drone.energyToFlyToCharger(drone.closestCharger) <= drone.alert:
                # fly to charger
                drone.target = drone.closestCharger.location
                answers.append(DroneNNWish.CHARGE)
                continue

            # continue_to_prev_field_target:
            if drone.targetField:
                if drone.target != drone.location:
                    answers.append(DroneNNWish(drone.targetField._id - 1))
                    continue

            # if birds on reachable field
            #   fly there
            # else: wait

            ascending_fields_idx = specific_dict['distances_drone_fields'][drone.int_id].argsort()
            ascending_fields_idx = np.squeeze(ascending_fields_idx)
            field_found = False

            #  join the closest occupied field
            for i in ascending_fields_idx:
                if birds_on_fields[i]:
                    # energyToFlyToCharger computes the energy correctly to anything with .location local variable
                    birds_field = WORLD.fields[i]
                    if drone.battery - drone.energyToFlyToCharger(birds_field) >= drone.alert:
                        answers.append(DroneNNWish(i + 1))
                        field_found = True
                        break

                    else:
                        # ascending order causes always to hit this branch when there is no option
                        break

            if field_found:
                continue
            answers.append(DroneNNWish.NONE)

        return np.eye(7)[answers]

    def manual_rules2F(common_dict, specific_dict):
        """betterbaselineF"""
        answers = []
        # for field in WORLD.fields: sorted by closeness, if the field is bird_only_fields
        birds_on_fields = common_dict['counts_0'].reshape(-1)
        drones_on_fields = common_dict['counts_1'].reshape(-1)
        birds_only_fields = common_dict['counts_2'].reshape(-1)
        for drone in WORLD.drones:
            # terminated drones
            if drone.state == DroneState.TERMINATED:
                answers.append(DroneNNWish.NONE)
                continue

            drone.closestCharger = drone.findClosestCharger()
            # if charging or in need of
            if drone.state not in [DroneState.MOVING_TO_CHARGER, DroneState.CHARGING] and \
                    drone.battery - drone.energyToFlyToCharger(drone.closestCharger) <= drone.alert:
                # fly to charger
                drone.target = drone.closestCharger.location
                answers.append(DroneNNWish.CHARGE)
                continue

            # continue_to_prev_field_target:
            if drone.targetField:
                # if drone.target != drone.location:
                    answers.append(DroneNNWish(drone.targetField._id - 1))
                    continue

            # if birds on reachable field
            #   fly there
            # else: wait

            ascending_fields_idx = specific_dict['distances_drone_fields'][drone.int_id].argsort()
            ascending_fields_idx = np.squeeze(ascending_fields_idx)
            field_found = False

            # priority to go to field with birds without drones
            for i in ascending_fields_idx:
                if birds_only_fields[i]:
                    # energyToFlyToCharger computes the energy correctly to anything with .location local variable
                    birds_field = WORLD.fields[i]
                    if drone.battery - drone.energyToFlyToCharger(birds_field) >= drone.alert:
                        answers.append(DroneNNWish(i + 1))
                        field_found = True
                        break

                    else:
                        # ascending order causes always to hit this branch when there is no option
                        break

            if field_found:
                continue

            # all fields with bird already have drones, join the closest
            for i in ascending_fields_idx:
                if birds_on_fields[i]:
                    # energyToFlyToCharger computes the energy correctly to anything with .location local variable
                    birds_field = WORLD.fields[i]
                    if drone.battery - drone.energyToFlyToCharger(birds_field) >= drone.alert:
                        answers.append(DroneNNWish(i + 1))
                        field_found = True
                        break

                    else:
                        # ascending order causes always to hit this branch when there is no option
                        break

            if field_found:
                continue
            answers.append(DroneNNWish.NONE)

        return np.eye(7)[answers]

    def manual_rules3F(common_dict, specific_dict):
        """bettersimplebaselineF"""
        answers = []
        # for field in WORLD.fields: sorted by closeness, if the field is bird_only_fields
        birds_on_fields = common_dict['counts_0'].reshape(-1)
        drones_on_fields = common_dict['counts_1'].reshape(-1)
        birds_only_fields = common_dict['counts_2'].reshape(-1)
        for drone in WORLD.drones:
            # terminated drones
            if drone.state == DroneState.TERMINATED:
                answers.append(DroneNNWish.NONE)
                continue

            drone.closestCharger = drone.findClosestCharger()
            # if charging or in need of
            if drone.state not in [DroneState.MOVING_TO_CHARGER, DroneState.CHARGING] and \
                    drone.battery - drone.energyToFlyToCharger(drone.closestCharger) <= drone.alert:
                # fly to charger
                drone.target = drone.closestCharger.location
                answers.append(DroneNNWish.CHARGE)
                continue

            # continue_to_prev_field_target:
            if drone.targetField:
                # if drone.target != drone.location:
                    answers.append(DroneNNWish(drone.targetField._id - 1))
                    continue

            # if birds on reachable field
            #   fly there
            # else: wait

            ascending_fields_idx = specific_dict['distances_drone_fields'][drone.int_id].argsort()
            ascending_fields_idx = np.squeeze(ascending_fields_idx)
            field_found = False

            #  join the closest occupied field
            for i in ascending_fields_idx:
                if birds_on_fields[i]:
                    # energyToFlyToCharger computes the energy correctly to anything with .location local variable
                    birds_field = WORLD.fields[i]
                    if drone.battery - drone.energyToFlyToCharger(birds_field) >= drone.alert:
                        answers.append(DroneNNWish(i + 1))
                        field_found = True
                        break

                    else:
                        # ascending order causes always to hit this branch when there is no option
                        break

            if field_found:
                continue
            answers.append(DroneNNWish.NONE)

        return np.eye(7)[answers]

    def save_debug_state(prediction, red_flags=False):
        argm = prediction.argmax(1)  # ARGMAX

        if red_flags:
            for a, drone in zip(argm, WORLD.drones):
                if drone.DEBUG_nn_wish != DroneNNWish(a):
                    drone.DEBUG_nn_wish = f".{str(drone.DEBUG_nn_wish)[12:]} -> {str(DroneNNWish(a))[12:]}"
        else:
            for a, drone in zip(argm, WORLD.drones):
                drone.DEBUG_nn_wish = DroneNNWish(a)

    def assign_drones(results, hysteresis):
        for a, dr in zip(results, WORLD.drones):
            dr.set_state_minimal(DroneNNWish(a))
            dr.nn_wish = DroneNNWish(a)
        if hysteresis:
            hysteresis.add_chosen(
                (np.eye(7))[list(results)]
            )

    def composition_algorithm():
        # get nn result
        components_data = get_data_from_components()
        X, _ = process_data(components_data, return_dict_as_well=False)

        if args.redflags != RedflagsEnum.none:# or 'baseline':
            _, _, common_dict, specific_dict = process_data(components_data, return_dict_as_well=True, normalize=False)
        else:
            common_dict, specific_dict = None, None

        if args.which_simulation in [#SimulationType.baseline_with_rf, HAS A COUNER PAR IN in_composiion_wihtout_nn
                                     SimulationType.baseline_with_rf_argmax,
                                     SimulationType.baseline_with_composiion]:
            prediction = manual_rules2(common_dict, specific_dict)
        else:
            if not model:
                print("ERROR tries to use non existing model:", args.which_simulation, file=sys.stderr)
                exit(0)
            prediction = model.predict(X)

        save_debug_state(prediction)

        if args.hysteresis:
            hyst = hysteresis.compute_hysteresis()
            if hyst is not None:
                if hysteresis.clip:
                    hyst = hyst.clip(0, hysteresis.clip)
                    # if hysteresis.type == 'a':
                    #     hyst = hyst / 3
                prediction += hyst
                prediction = prediction.clip(0, 1)


        if args.redflags != RedflagsEnum.none:
            rf = RedFlags(X, prediction, common_dict, specific_dict, components_data, WORLD, args)  # , sum_up_prediction=True)

            rf.process_red_flags()  # so that 1111 would work
            prediction = rf.updated_prediction
            save_debug_state(prediction, True)

        # get the unlimited None ensebmle, FieldProtection and charging ensembles cardinalities
        # from drone_charging_example.ensembles.field_protection import FieldProtection
        from drone_charging_example.ensembles.another_charging import GlobalQueue

        # from ensembles.another_charging import GlobalQueue

        # if this fails -> following will work, but is ugly
        # ensembles[0].__class__.__name__ == 'FieldProtection'
        from ensembles.field_protection import FieldProtection  # don't change to drone_charging_example.ensembles.field_protection
        fields = [e for e in ensembles if isinstance(e, FieldProtection)]
        assert len(fields) > 0
        # charger = [e for e in ensembles if isinstance(e, GlobalQueue)] # the cardinality is len drones -> not good

        charging_drones = set()
        if args.chargers_fixed:
            for c in WORLD.chargers:
                for e in c.chargingDrones:
                    charging_drones.add(e.int_id)

        # including some waiting queue
        total_chargers_capacity = sum(ENVIRONMENT.chargerCapacity for ch in WORLD.chargers) + len(charging_drones)

        # FALL_TO_ENSEMBLE, FIELDS, CHARGING_ENSEMBLE
        cardinalities = [(0, prediction.shape[1])] + [e.get_cardinality() for e in fields] + [(0, total_chargers_capacity)]

        mins = [m[0] for m in cardinalities]
        maxs = [m[1] for m in cardinalities]

        if args.which_simulation in [
            # greedy_ensemble_partial_sets,
            # greedy_ensemble_partial_sets_priority
            SimulationType.greedy_ensemble_priority,
            SimulationType.greedy_ensemble1_priority,
            SimulationType.greedy_ensemble_obey_partial,
            SimulationType.greedy_ensemble_obey,
        ]:
            priors = [-1] + [e.priority_turned_on() for e in fields] + [1]
        else:
            # priors = [-1] + [e.priority() for e in fields] + [1]
            # none, fields, charger
            priors = [-1, 0.1, 0.1, 0.1, 0.1, 0.1, 1]

        def get_better_mins(mins, maxs, drone_count):
            # scale if needed
            total_drones_ens = sum(maxs[1:])
            scale_up_factor = 1 + drone_count // total_drones_ens
            mins = [(m*scale_up_factor*0.5)//1 for m in mins]
            maxs = [(m*scale_up_factor)//1 for m in maxs]

            # alive_drone_count = sum(map(lambda d: 1 if d.battery > 0 else 0, WORLD.drones))
            _mins = [0]
            _maxs = [maxs[0]]
            priorities = []
            for f in WORLD.fields:
                food_left = f.allCrops - f.damage
                places = len(f.places)
                remaining_food_per_drone = food_left / places
                free_spaces = places - len(f.protectingDrones)
                unprotected_crops = remaining_food_per_drone * free_spaces

                # how many birds on the field
                birds = sum(1 for b in WORLD.birds if b.field == f)
                birds_rate = birds / len(WORLD.birds)
                _min = 0
                if birds > 0:
                    _min = 1
                if birds > places and places > 1:
                     _min = 2
                if remaining_food_per_drone == 0:
                    _min = 0
                    _max = 0
                else:
                    _max = 9999
                _mins.append(_min)
                _maxs.append(_max)
                priorities.append(birds_rate * unprotected_crops)

            _mins.append(mins[-1])
            _maxs.append(maxs[-1])
            return [max(x, y) for x, y in zip(_mins, mins)],\
                   [min(x, y) for x, y in zip(_maxs, maxs)],
        mins, maxs = get_better_mins(mins, maxs, X.shape[0])


        if args.which_simulation in [SimulationType.argmax,  SimulationType.baseline_with_rf_argmax]:
            argm = prediction.argmax(1)  # ARGMAX
            assign_drones(argm, hysteresis)

        elif args.which_simulation in greedy_to_alg_dict:
            # do not use dead drones in the ensembles
            dead_drones = [i for i, d in enumerate(WORLD.drones) if d.battery <= 0]

            greedy_algorithm = None
            greedy_algorithm = greedy_to_alg_dict[args.which_simulation]

            assignment, assigned_drones = greedy_algorithm(prediction, dead_drones, charging_drones, priors, mins, maxs)

            # add back dead drones
            for d in dead_drones:
                assignment[d] = DroneNNWish.NONE

            if len(assignment) != len(WORLD.drones):
                missing = set(range(len(WORLD.drones))) - assignment.keys()
                print(f"WRONG values - Some [{len(WORLD.drones) - len(assignment)}]: missing: {missing} drone(s) did not get value: ",
                      assignment.keys(), args.which_simulation, file=sys.stderr)
                for d in missing:
                    assignment[d] = DroneNNWish.NONE
            assignment = dict(sorted(assignment.items()))

            assert len(assignment) == len(WORLD.drones), f'{args.seed}, {len(assignment)}, {len(WORLD.drones)} {prediction.shape = }'

            assign_drones(assignment.values(), hysteresis)
        elif args.which_simulation in slow_algs_set:  # not a good place to put to TODO refactor
            # do not use dead drones in the ensembles
            dead_drones = [i for i, d in enumerate(WORLD.drones) if d.battery <= 0]
            from ensemble_resolution.gready_resolution import SLOW, transform_slow_result
            slow = SLOW(prediction, dead_drones, priors, mins, maxs, args.which_simulation)
            if slow is None:
                print(f'resolution failed in step: {step} -> fallback to argmax', file=sys.stderr)
                argm = prediction.argmax(1)  # ARGMAX
                assign_drones(argm, hysteresis)
            else:
                assignments = transform_slow_result(slow, dead_drones, args.which_simulation)
                path = args.save_files_stump.rsplit('/', 1)[0]
                print(path)
                print(args.save_files_stump.rsplit('/', 1)[1])
                os.makedirs(path, exist_ok=True)

                pickle_path = f"{args.save_files_stump}_step_{args.current_step}.npz"

                # assignments, prediction, dead_drones, priors, mins, maxs, args.which_simulation
                np.savez_compressed(pickle_path, prediction=prediction, dead_drones=dead_drones, prior=priors, mins=mins, maxs=maxs,
                                    transf_result=assignments)
                assign_drones(assignments, hysteresis)
        else:
            raise Exception(f"WRONG VALUE {args.which_simulation = }")

        # handles only state changes upon reaching the fields / chargers
        materializedEnsembles = materialize_ensembles(components, ensembles)

        # print(f"using new execution path instead of old \"compatibility\", {args.which_ensembles = }")
        # print(f'{len(materializedEnsembles) = }')

    from world import WORLD, ENVIRONMENT
    update_setup(WORLD, ENVIRONMENT, args)

    from ensemble_resolution.gready_resolution import GreedyAlgorithms
    # greedy_algs_set = {
    #     SimulationType.greedy_ensemble,
    #     SimulationType.greedy_ensemble2,
    #     SimulationType.greedy_ensemble_priority,
    #
    #     SimulationType.greedy_drone,
    #     SimulationType.greedy_drone2,
    #     SimulationType.greedy_drone_ens
    # }
    greedy_to_alg_dict = {
        SimulationType.greedy_ensemble: GreedyAlgorithms.first_satisfy_ensembles, #greedy_ensemble
        SimulationType.greedy_ensemble2: GreedyAlgorithms.first_satisfy_ensembles_partial_sets, #greedy_ensemble2 is parial ses
        SimulationType.greedy_ensemble_priority: GreedyAlgorithms.first_satisfy_ensembles_partial_sets_priorities, # field priories are used
        SimulationType.greedy_ensemble1_priority: GreedyAlgorithms.first_satisfy_ensembles_no_sets_priorities,
        SimulationType.greedy_ensemble_obey_partial: GreedyAlgorithms.first_satisfy_ensembles_partial_sets_priorities_check_minima,
        SimulationType.greedy_ensemble_obey: GreedyAlgorithms.first_satisfy_ensembles_no_sets_priorities_check_minima,

        SimulationType.greedy_drone: GreedyAlgorithms.DroneFirst.satisfy_drone,
        SimulationType.greedy_drone2: GreedyAlgorithms.DroneFirst.satisfy_drone_inn,
        SimulationType.greedy_drone_ens: GreedyAlgorithms.DroneFirst.satisfy_ensemble,

        # SimulationType.baseline_with_rf: GreedyAlgorithms.DroneFirst.satisfy_drone,
        SimulationType.baseline_with_rf_argmax: None,
        SimulationType.baseline_with_composiion: GreedyAlgorithms.DroneFirst.satisfy_drone,

    }
    greedy_algs_set = set(greedy_to_alg_dict.keys())
    slow_algs_set = {SimulationType.slowProb, SimulationType.slowPoints, SimulationType.slowBoth}
    accepted_sims_set = {SimulationType.argmax} | slow_algs_set | greedy_algs_set
    in_composiion_wihtout_nn = {   # SimulationType.baseline_with_rf,
        SimulationType.baseline_with_rf_argmax,
        SimulationType.baseline_with_composiion}

     # allow no model
    if args.which_simulation in (greedy_algs_set | {SimulationType.argmax}) and \
            args.which_simulation not in in_composiion_wihtout_nn:
        model_path = get_model_path(args)
        try:
            model = tf.keras.models.load_model(model_path)
        except ValueError:
            import tensorflow_addons as tfa
            model = tf.keras.models.load_model(model_path, custom_objects={'StochasticDepth': tfa.layers.StochasticDepth})
    else:
        model = None
    hysteresis = SimulationSetup().Hysteresis if args.hysteresis else None

    for step in range(steps+1):  # + 1 just to yield the last statistics
        args.current_step = step
        damage, crop, dead_drones, total_drones = print_damage_info()
        if stepCallback:
            stepCallback(components, [], step, damage, crop, dead_drones, total_drones)

        if dead_drones == total_drones:  # pre optimization - all drones dead
            print("LAST_STEP_IS", step)
            print(f"which sim {str(args.which_simulation)}\n" * 2)

            # crop instead of damage because birds are going to eat everything
            if stepCallback:
                stepCallback(components, [], step + 1, crop, crop, dead_drones, total_drones)
            return crop, crop, dead_drones, total_drones
        if step >= steps - 1:
            print(f"which sim {str(args.which_simulation)}\n" * 2)
            return damage, crop, dead_drones, total_drones

        verbosePrint(f"Step {step + 1}:", 3)
        SIMULATION_GLOBALS.currentTimeStep = step

        # TODO test argmax in both versions
        # if args.which_simulation in greedy_algs_set or args.which_simulation == SimulationType.argmax:
        # if args.which_simulation in greedy_algs_set | slow_algs_set:

        if args.which_simulation in accepted_sims_set:
            composition_algorithm()
        elif args.which_simulation in [SimulationType.red_flagsF,
            SimulationType.baseline, SimulationType.betterbaseline, SimulationType.bettersimplebaseline, SimulationType.betterbaselineF, SimulationType.bettersimplebaselineF]:
            components_data = get_data_from_components()
            _, _, common_dict, specific_dict = process_data(components_data, return_dict_as_well=True, normalize=False)

            if args.which_simulation == SimulationType.betterbaseline:
                prediction = manual_rules2(common_dict, specific_dict)
            if args.which_simulation == SimulationType.red_flagsF:
                import copy
                # args = parser.parse_args()
                args_copy = copy.deepcopy(args)
                args_copy.redflags = RedflagsEnum.answer

                ans = update_drones_by_the_nn(model, args_copy)
                assign_drones(ans, hysteresis=False)

                materializedEnsembles = materialize_ensembles(components, ensembles)
                actuate_components(components)

                if stepCallback:
                    # stepCallback(components, materializedEnsembles, step)
                    stepCallback(components, [], step, damage, crop, dead_drones, total_drones)
                continue
            answer = np.argmax(prediction, axis=1)
            assign_drones(answer, hysteresis=False)
            materialize_ensembles(components, ensembles)

        else: #FIXME keeping somewhat - backwards compatibility
            if args.set_state != SetState.do_not_enter_the_processing:
                update_drones_by_the_nn(model, args)
            if args.materialize_ensembles:
                #the results of the simulation only -> wish only will not work, must use set state, or directly set state
                materializedEnsembles = materialize_ensembles(components, ensembles)
            else:
                #support to switch the states of variables
                for drone in WORLD.drones:
                    if drone.state in [DroneState.MOVING_TO_FIELD, DroneState.MOVING_TO_CHARGER]:
                        if drone.target == drone.location:
                            if drone.state == DroneState.MOVING_TO_FIELD:
                                drone.state = DroneState.PROTECTING
                            else:
                                drone.state = DroneState.CHARGING


        actuate_components(components)

        if stepCallback:
            # stepCallback(components, materializedEnsembles, step)
            stepCallback(components, [], step, damage, crop, dead_drones, total_drones)


def fff(prediction, dead_drones, priors, mins, maxs):
    assignment_e, assigned_drones_e = GreedyAlgorithms.first_satisfy_ensembles(prediction, dead_drones, priors, mins, maxs)
    assignment_fd, assigned_drones_fd = GreedyAlgorithms.DroneFirst.satisfy_drone(prediction, dead_drones, priors, mins, maxs)
    assignment_fdi, assigned_drones_fdi = GreedyAlgorithms.DroneFirst.satisfy_drone_inn(prediction, dead_drones, priors, mins, maxs)
    assignment_fe, assigned_drones_fe = GreedyAlgorithms.DroneFirst.satisfy_ensemble(prediction, dead_drones, priors, mins, maxs)

    assignment_string = ''
    if assignment_e == assignment_fd:
        assignment_string += '@'
    if assignment_e == assignment_fdi:
        assignment_string += '€'
    if assignment_e == assignment_fe:
        assignment_string += '$'
    if assignment_fd == assignment_fe:
        assignment_string += '%'
    if assignment_fd == assignment_fdi:
        assignment_string += '='
    if assignment_fdi == assignment_fe:
        assignment_string += '+'

    print("assignment_string", assignment_string, file=sys.stderr)
    print('', assigned_drones_e, '\n', assigned_drones_fd, '\n', assigned_drones_fdi, '\n', assigned_drones_fe)

    raise 123

def get_model_path(args):
    def select_which_model(model):
        # todo is this true?
        #  models are normaly saved into: something like: /dev/shm/nn-logs/2022_04_01_06-18-37-0.9772498607635498.h5
        #  we have to manualy do:  cp /dev/shm/nn-logs/2022_04_01_06-18-37-0.9772498607635498.h5 /root/redflags-honza/external_utils/models/

        # # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\108\happy-plasma-5029.h5'
        # # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\108\legendary-armadillo-5025.h5'
        # # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\108\confused-shape-5023.h5'
        # # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\108\robust-resonance-4462.h5'
        # # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\108\lambent-festival-10.h5'
        # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\108\vibrant-dragon-1.h5'
        '''
            ###############################################################################################################
            # Name                     Runtime  epochs  final_learning_rate  smoothing  train_files  evalualtion_accuracy #
            ###############################################################################################################
            # happy-plasma-5029         1190     1       0.00001              0.1487     166667       0.97511             #
            # legendary-armadillo-5025  1171     1       0.00001              0.0377     166667       0.97508             #
            # confused-shape-5023       2747     1       0.00001              0.2354     166667       0.97460             #
            # robust-resonance-4462     460      5       0.00001              0.0081     16667        0.97516             #
            # lambent-festival-10       35983    500     0                    0.01       16667        0.97643             #
            # vibrant-dragon-1          650      5       0                    0.004      16667        0.97489             #
            ###############################################################################################################
            '''

        # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\110\comic-gorge-5045.h5' #deep
        # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\110\desert-gorge-5075.h5' #bn_l2
        # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\110\dashing-vortex-5020.h5' #deep
        # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\110\lemon-smoke-5059.h5' #bn_l2
        # model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\110\upbeat-snow-4694.h5' #bn_l2 "/dev/shm/nn-logs/2022_02_07_09-34-14-0.9702981114387512.h5"

        # best deep
        model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\109\2022_04_01_19-15-21-0.9782962203025818.h5' # amber-yogurt-217

        # best 256
        model_path = '/root/redflags-honza/external_utils/models/2022_04_01_06-18-37-0.9772498607635498.h5' # driven-snowball-188
        model_path = '/root/redflags-honza/external_utils/models/2022_03_31_21-25-43-0.9772105813026428.h5' # lemon-donkey-149

        # model combinations
        model_path = '/root/redflags-honza/external_utils/models/(9, 8, 7)'  # 0.9786-869287490845 # C1 combination
        # model_path = '/root/redflags-honza/external_utils/models/x9842' # 0.9785-636067390442 # C2

        # model_path = '/root/redflags-honza/external_utils/models/2022_03_30_03-33-46-0.9779280424118042.h5' #9 # decent-sea-34
        # model_path = '/root/redflags-honza/external_utils/models/2022_03_30_03-11-01-0.9774134755134583.h5' #8 # robust-vortex-33
        # model_path = '/root/redflags-honza/external_utils/models/2022_03_30_00-24-19-0.9771632552146912.h5' #7 # glamorous-pond-6
        # model_path = '/root/redflags-honza/external_utils/models/2022_03_29_23-23-11-0.9767979383468628.h5' #6
        # model_path = '/root/redflags-honza/external_utils/models/2022_03_30_13-54-40-0.9765704870223999.h5' #5
        # model_path = '/root/redflags-honza/external_utils/models/2022_03_30_00-50-41-0.9765072464942932.h5' #4
        # model_path = '/root/redflags-honza/external_utils/models/2022_03_30_01-20-38-0.9763468503952026.h5' #3
        # model_path = '/root/redflags-honza/external_utils/models/2022_03_30_12-53-09-0.9763191938400269.h5' #2
        # model_path = '/root/redflags-honza/external_utils/models/2022_03_31_12-36-53-0.9762064814567566.h5' #1
        # model_path = '/root/redflags-honza/external_utils/models/2022_03_31_09-48-32-0.9761964678764343.h5' #0

        """"""
        # TESTING RESULTS:                               VALIDATION RESULTS:
        # C1 [0.053262561559677124, 0.9786356091499329], [([0.05313698574900627, 0.9786862134933472], '/root/redflags-honza/external_utils/models/(9, 8, 7)'),
        # C2 [0.11663825809955597, 0.9785196185112],  ([0.11651721596717834, 0.9785635471343994], '/root/redflags-honza/external_utils/models/x9842'),
        # #9 [0.0522942841053009, 0.977927565574646],  ([0.05217164754867554, 0.9779478311538696], '/root/redflags-honza/external_utils/models/2022_03_30_03-33-46-0.9779280424118042.h5'),
        # #8 [0.05308815836906433, 0.9774132370948792],  ([0.05294010415673256, 0.9774811267852783], '/root/redflags-honza/external_utils/models/2022_03_30_03-11-01-0.9774134755134583.h5'),
        # #7 [0.11836966127157211, 0.9771579504013062],  ([0.11821414530277252, 0.9772240519523621], '/root/redflags-honza/external_utils/models/2022_03_30_00-24-19-0.9771632552146912.h5'),
        # #6 [0.11916141957044601, 0.9767947793006897],  ([0.11897893249988556, 0.976844847202301], '/root/redflags-honza/external_utils/models/2022_03_29_23-23-11-0.9767979383468628.h5'),
        # #5 [0.4900893568992615, 0.9765711426734924],  ([0.4899742007255554, 0.9766247868537903], '/root/redflags-honza/external_utils/models/2022_03_30_13-54-40-0.9765704870223999.h5'),
        # #4 [0.4917091131210327, 0.9765066504478455],  ([0.4915824234485626, 0.9765464663505554], '/root/redflags-honza/external_utils/models/2022_03_30_00-50-41-0.9765072464942932.h5'),
        # #3 [0.05682304501533508, 0.9763514995574951],  ([0.05670536682009697, 0.9763811826705933], '/root/redflags-honza/external_utils/models/2022_03_30_01-20-38-0.9763468503952026.h5'),
        # #2 [0.8007562160491943, 0.9763232469558716],  ([0.8006662130355835, 0.9763506650924683], '/root/redflags-honza/external_utils/models/2022_03_30_12-53-09-0.9763191938400269.h5'),
        # #1 [0.12107113748788834, 0.976207971572876],  ([0.12090647965669632, 0.9762551784515381], '/root/redflags-honza/external_utils/models/2022_03_31_12-36-53-0.9762064814567566.h5'),
        # #0 [0.05704856663942337, 0.9761985540390015],  ([0.0568840354681015, 0.9762539863586426], '/root/redflags-honza/external_utils/models/2022_03_31_09-48-32-0.9761964678764343.h5')]

        if model in ['C1', '', None]:
            model_path = '/root/redflags-honza/external_utils/models/(9, 8, 7)'  # 0.9786-869287490845 # C1 combination
        elif model == 'C2':
            model_path = '/root/redflags-honza/external_utils/models/x9842' # 0.9785-636067390442 # C2
        elif model == 'C1_1':
            model_path = '/root/redflags-honza/external_utils/models/2022_03_30_03-33-46-0.9779280424118042.h5' #9 # decent-sea-34
        elif model == 'C1_2':
            model_path = '/root/redflags-honza/external_utils/models/2022_03_30_03-11-01-0.9774134755134583.h5' #8 # robust-vortex-33
        elif model == 'C1_3':
            model_path = '/root/redflags-honza/external_utils/models/2022_03_30_00-24-19-0.9771632552146912.h5' #7 # glamorous-pond-6
        elif model == '256':
            model_path = '/root/redflags-honza/external_utils/models/2022_04_01_06-18-37-0.9772498607635498.h5'  # driven-snowball-188
        elif model == '256B':
            model_path = '/root/redflags-honza/external_utils/models/2022_03_31_21-25-43-0.9772105813026428.h5'  # lemon-donkey-149
        elif model == 'deep':
            model_path = r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\109\2022_04_01_19-15-21-0.9782962203025818.h5'  # amber-yogurt-217
            model_path = '/root/redflags-honza/external_utils/models/2022_04_01_19-15-21-0.9782962203025818.h5'  # amber-yogurt-217
        else:
            raise ModuleNotFoundError("this model is non existent")
        return model_path

    model_path = select_which_model(args.load_model)
    manual_debug = False
    if not tf.io.gfile.exists(model_path) or manual_debug:

        if model_path == r'C:\Users\X\Pycharms\ensml\drones_from_scratch\models\110\upbeat-snow-4694.h5 ':
            model_path = "/dev/shm/nn-logs/2022_02_07_09-34-14-0.9702981114387512.h5"
        else:
            print(f"file {model_path} was not found - reaching to wandb to get the filename to search the right file")
            def get_filename_from_wandb_name(name):
                api = wandb.Api()
                my_entity = "deepcharles"
                my_project = "basic_network_compact_110"

                # my_projects = [p.name for p in api.projects(my_entity)]
                my_projects = ['basic_network_compact_110',
                               'basic_network_compact_108',
                               'basic_network_compact2',
                               'basic_network_compact',
                               'basic_network',
                               'trans_net_109',
                               ]
                for my_project in my_projects:
                    runs = api.runs(path=f"{my_entity}/{my_project}", filters={"display_name": name})
                    lruns = list(runs)
                    # lruns = [r.name for r in lruns]
                    # print(lruns)
                    if lruns:
                        run = lruns[0]
                        fs = [f for f in run.files() if 'h5' in f.name]
                        if fs:
                            return (my_project, fs[0].name)

            n = model_path.split('\\')[-1].split('/')[-1].split('.h5')[0]

            project_filename = get_filename_from_wandb_name(n)
            if project_filename:
                project, filename = project_filename
                model_path = f"/dev/shm/nn-logs/{filename}"
                if not tf.io.gfile.exists(model_path):
                    raise FileNotFoundError(model_path)
                print("found a new model_path")
                print(model_path)
            else:
                print('not found ??')
    return model_path


def run_experiment(
    iterations: int,
    simulations: int,
    steps: int,
    prepareSimulation: Callable[[int, int], Tuple[List['Component'], List['Ensemble']]],
    prepareIteration: Optional[Callable[[int], None]] = None,
    iterationCallback: Optional[Callable[[int], None]] = None,
    simulationCallback: Optional[Callable[[List['Component'], List['Ensemble'], int, int], None]] = None,
    stepCallback: Optional[Callable[[List['Component'], List['Ensemble'], int], None]] = None,
    args: object = None,
):
    """
    Runs `iterations` iteration of the experiment. Each iteration consist of running the simulation `simulations` times (each simulation is run for `steps` steps) and then performing training of the Estimator (ML model).

    Parameters
    ----------
    iterations
        Number of iterations to run.
    simulations
        Number of simulations to run in each iteration.
    steps
        Number of steps to perform in each simulation.
    prepareSimulation
        Prepares the components and ensembles for the simulation.
        Parameters:
            - current iteration,
            - current simulation (in the current iteration).
        Returns:
            - list of components,
            - list of potential ensembles.
    prepareIteration
        Performed at the beginning of each iteration.
        Parameters:
            - current iteration.
    iterationCallback
        Performed at the end of each iteration (after the training of Estimators).
        Parameters:
            - current iteration.
    simulationCallback
        Performed after each simulation.
        Parameters:
            - list of components (returned by `prepareSimulation`),
            - list of potential ensembles (returned by `prepareSimulation`),
            - current iteration,
            - current simulation (in the current iteration).
    stepCallback
        This function is called after each simulation step. It can be used for example to log data from the simulation. The parameters are:
            - list of all components in the system,
            - list of materialized ensembles (in this time step),
            - current time step (int).
    """

    SIMULATION_GLOBALS.initEstimators()
    assert iterations == 1, "deal with the output"
    for iteration in range(iterations):
        verbosePrint(f"Iteration {iteration + 1} started at {datetime.now()}:", 1)
        if prepareIteration:
            prepareIteration(iteration)

        for simulation in range(simulations):
            verbosePrint(f"Simulation {simulation + 1} started at {datetime.now()}:", 2)

            components, ensembles = prepareSimulation(iteration, simulation)

            output = run_simulation(args, components, ensembles, steps, stepCallback)

            if simulationCallback:
                simulationCallback(components, ensembles, iteration, simulation)

        for estimator in SIMULATION_GLOBALS.estimators:
            estimator.endIteration()

        if iterationCallback:
            iterationCallback(iteration)
    return output


if __name__ == '__main__':
    import glob
    from ensemble_resolution.gready_resolution import measure_prediction_quality_sum_probabilities, \
        measure_prediction_quality_sum_probabilities_penalize_rf, measure_prediction_quality_sum_points, \
        measure_prediction_quality_sum_probabilities_OLD, measure_prediction_quality_sum_points_OLD, GreedyAlgorithms

    gas = [
        GreedyAlgorithms.first_satisfy_ensembles,
        GreedyAlgorithms.DroneFirst.satisfy_drone,
        GreedyAlgorithms.DroneFirst.satisfy_drone_inn,
        GreedyAlgorithms.DroneFirst.satisfy_ensemble,
    ]
    # glob_path = '/dev/shm/results/results-999[7-9]/**/slowBoth/**/*.npz'

    import enum
    SlowAlgorithm = enum.Enum('SlowAlgorithm', 'POINT PROB BOTH')

    def get_path_to_npz_DATA_LOST(which_algorithm):
        return {
            SlowAlgorithm.POINT: '/dev/shm/results/results-999[7-9]/**/slowPoints/**/*.npz',
            SlowAlgorithm.PROB: '/dev/shm/results/results-999[7-9]/**/slowProb/**/*.npz',
            SlowAlgorithm.BOTH: '/dev/shm/results/results-999[7-9]/**/*.npz',
        }[which_algorithm]

    def get_path_to_npz(which_algorithm):
        return {
            SlowAlgorithm.POINT: '/dev/shm/results/results-9995/**/slowPoints/**/*.npz',
            SlowAlgorithm.PROB: '/dev/shm/results/results-9995/**/slowProb/**/*.npz',
            SlowAlgorithm.BOTH: '/dev/shm/results/results-9995/**/*.npz',
        }[which_algorithm]


    which_algorithm = SlowAlgorithm.POINT
    which_algorithm = SlowAlgorithm.PROB
    glob_path = get_path_to_npz(which_algorithm)
    # glob_path = '/dev/shm/results/results-None/local_only/SetState.use_set_state_minimal/slowProb/4_4drones-lower-resolution/*_82*npz' # UNKNOWN SEED!
    # glob_path = '/dev/shm/results/results-42/local_only/SetState.use_set_state_minimal/slowProb/4_4drones-lower-resolution/*.npz'
    # glob_path = '/dev/shm/results/results-42/local_only/SetState.use_set_state_minimal/slowProb/4_4drones-lower-resolution/*_63*.npz'

    # #TODO remove
    # gas = gas[:1]    # 7
    # gas = gas[1:2]   # 8
    # gas = gas[2:3]   # 9
    # gas = gas[:3]    # 10

    # problematic values are the one that are negative :- best result - other = rest of the best and since best is max...
    Counts = namedtuple('Counts', ['negative', 'zero', 'positive', 'percent_reached'], defaults=(0, 0, 0, 0))

    equal_result = {}
    count_diff = {}

    metrics = {}
    diffs_count = 5

    for g in gas:
        # same_output, (negative, zero, positive)
        metrics[g] = np.zeros(diffs_count * 2)

        equal_result[g] = [0]
        count_diff[g] = []
        for _ in range(diffs_count):
            count_diff[g].append(Counts())
    file_counter = 0


    def update_vals(gas, list_diffs):
        for i, (c, d) in enumerate(zip(count_diff[gas], list_diffs)):
            if d < 1:
                assert d >= -0.05, f"{list_diffs} {d=}, {d<0 = }"
                updated = count_diff[gas][i]._replace(percent_reached=c[3] + d)
                count_diff[gas][i] = updated


    def update_counts(gas, list_diffs):
        for i, (c, d) in enumerate(zip(count_diff[gas], list_diffs)):
            if d == 0:
                updated = count_diff[gas][i]._replace(zero=c[1] + 1)
            elif d > 0:
                updated = count_diff[gas][i]._replace(positive=c[2] + 1)
            else:
                updated = count_diff[gas][i]._replace(negative=c[0] + 1)

            count_diff[gas][i] = updated

    def to_list_sets(assign):
        assignment = [set() for _ in f['mins']]
        # for d_idx in dead_drones:
        #     assignment[0].add(d_idx)
        for a, i, in enumerate(assign):
            assignment[i].add(a)
        return assignment

    # for file, _ in zip(glob.iglob(glob_path, recursive=True), range(200)):
    for file in glob.iglob(glob_path, recursive=True):
        file_counter += 1
        f = np.load(file)
        assert sorted(list(f.keys())) == ['dead_drones', 'maxs', 'mins', 'prediction', 'prior', 'transf_result']

        f_prediction = f['prediction']
        f_dead_drones = f['dead_drones']

        # best results
        br = f['transf_result']
        br_sets = to_list_sets(br)  # expected_assignment_list_set
        br_compatibility = GreedyAlgorithms.compatibility_output(br_sets, f_prediction, f_dead_drones, 'string')

        br_points = measure_prediction_quality_sum_points(f_prediction, br_sets)
        br_prob_rf = measure_prediction_quality_sum_probabilities_penalize_rf(f_prediction, br_sets)
        br_prob = measure_prediction_quality_sum_probabilities(f_prediction, br_sets)

        # expected_quality_prob_OLD = measure_prediction_quality_sum_probabilities_OLD(f['prediction'], assignment, f['dead_drones'])
        # expected_quality_points_OLD = measure_prediction_quality_sum_points_OLD(f['prediction'], assignment, f['dead_drones'])

        br_prob_OLD = measure_prediction_quality_sum_probabilities_OLD(f_prediction, br_compatibility[0], f_dead_drones)
        br_points_OLD = measure_prediction_quality_sum_points_OLD(f_prediction, br_compatibility[0], f_dead_drones)

        for greedy_alg in gas:
            result_compatibility = greedy_alg(f_prediction, f_dead_drones, f['prior'], f['mins'], f['maxs'])
            gas_assign, gas_assign_drones = result_compatibility

            # the algorithm returns the same assigment
            if np.equal(br, list(gas_assign.values())).all():
                equal_result[greedy_alg][0] += 1
                continue

            # check whether the dead drones are added to the right group
            all_dead_drones_are_in_fallback = all(d in gas_assign_drones[FALL_TO_ENSEMBLE] for d in f_dead_drones)
            if not all_dead_drones_are_in_fallback:
                print("all_dead_drones_are_in_fallback failed", f_dead_drones, gas_assign_drones[FALL_TO_ENSEMBLE])
            assert all_dead_drones_are_in_fallback

            # check whether the ensembles have correct sizes -- the mins and maxs
            correct_sizes_check_failed = False
            for i, m, M in zip(range(len(f['mins'])), f['mins'], f['maxs']):
                if m <= len(gas_assign_drones[i]) <= M:
                    # print('ok', end=' ') #TODO remove
                    pass
                else:
                    correct_sizes_check_failed = True
                    break
            # skip wrong assign
            if correct_sizes_check_failed:
                continue

            ga_sets = to_list_sets(list(x[1] for x in sorted(gas_assign.items())))
            # ga_sets = to_list_sets(list(gas_assign.values()))
            ga_points = measure_prediction_quality_sum_points(f_prediction, ga_sets)
            ga_prob_rf = measure_prediction_quality_sum_probabilities_penalize_rf(f_prediction, ga_sets)
            ga_prob = measure_prediction_quality_sum_probabilities(f_prediction, ga_sets)

            # old measurement discard dead drones
            ga_prob_OLD = measure_prediction_quality_sum_probabilities_OLD(f_prediction, gas_assign, f_dead_drones)
            ga_points_OLD = measure_prediction_quality_sum_points_OLD(f_prediction, gas_assign, f_dead_drones)

            #allclose == with tolerance
            if (not np.allclose(ga_prob, ga_prob_OLD, atol=1e-15, rtol=1e-15) or ga_points != ga_points_OLD):
                if len(f_dead_drones) == 0:  # not does not work with np.array
                    print(f"{ga_prob=} != {ga_prob_OLD=} or {ga_points=} != {ga_points_OLD=}):")

            # diff1 = br_prob_OLD - ga_prob_OLD
            # diff2 = br_prob - ga_prob
            # diff3 = br_prob_rf - ga_prob_rf
            # diff4 = br_points - ga_points
            # diff5 = br_points_OLD - ga_points_OLD

            diff_info = ['prob_OLD, prob, prob_rf, points, points_old'.split(', ')]
            diff1 = diff2 = diff3 = diff4 = diff5 = 0
            # best result - greedy algorithm
            if which_algorithm in [SlowAlgorithm.PROB, SlowAlgorithm.BOTH]:
                diff1 = br_prob_OLD - ga_prob_OLD
                diff2 = br_prob - ga_prob
                diff3 = br_prob_rf - ga_prob_rf
            if which_algorithm in [SlowAlgorithm.POINT, SlowAlgorithm.BOTH]:
                diff4 = br_points - ga_points
                diff5 = br_points_OLD - ga_points_OLD

            if any(map(lambda x: x < 0, [diff1, diff2, diff3, diff4, diff5])):
                pass
                # print(f"---|- {ga_sets=}, {br_sets=}, {f_dead_drones=} ", ([diff1, diff2, diff3, diff4, diff5]), f'{file=}, {greedy_alg=}, ', (
                # f"{br_prob_OLD=}, {br_prob=}, {br_prob_rf=}, {br_points=}, {br_points_OLD=}, {f_dead_drones=}"))

            update_counts(greedy_alg, [diff1, diff2, diff3, diff4, diff5])

            metrics[greedy_alg] = metrics[greedy_alg] + [diff1, diff2, diff3, diff4, diff5,
                                 diff1**2, diff2**2, diff3**2, diff4**2, diff5**2]

            if which_algorithm in [SlowAlgorithm.PROB, SlowAlgorithm.BOTH]:
                diff1 = ga_prob_OLD / br_prob_OLD
                if ga_prob_OLD < 0 or br_prob_OLD < 0:
                    diff1 = 0
                    print(f"df1 {ga_prob_OLD} {br_prob_OLD}")
                diff2 = ga_prob / br_prob
                if ga_prob < 0 or br_prob < 0:
                    diff2 = 0
                    print(f"df2 {ga_prob} {br_prob}")

                diff3 = ga_prob_rf / br_prob_rf
                if ga_prob_rf < 0 or br_prob_rf < 0:
                    diff3 = 0

            if which_algorithm in [SlowAlgorithm.POINT, SlowAlgorithm.BOTH]:
                diff4 = ga_points / br_points
                if ga_points < 0 or br_points < 0:
                    diff4 = 0


                diff5 = ga_points_OLD / br_points_OLD
                if ga_points_OLD < 0 or br_points_OLD < 0:
                    diff5 = 0

            update_vals(greedy_alg, [diff1, diff2, diff3, diff4, diff5])

    # else:
    #     raise Exception("no files found")

    # for g in gas:
    #     print('\n')
    #     print('--- ' * 4)
    #     print(g.__name__)
    #     print("counts")
    #     print(equal_result[g])
    #     for s in equal_result[g]:
    #         print(s / file_counter)
    #     print("diffs")
    #     print(count_diff[g])
    #     for s in count_diff[g]:
    #         print(np.array(s) / file_counter)
    #     m = metrics[g]
    #     print(m)
    #     print(m[:diffs_count] / file_counter, np.sqrt(m[diffs_count:]) / file_counter)
    #
    # print('---' * 15 + '\n'*20)
    # this is what I need for the plot -> ['Same solution', 'Same score', 'Worse']
    diff_info = 'prob_OLD, prob, prob_rf, points, points_old'.split(', ')
    for g in gas:
        print('\n')
        print(which_algorithm)
        print('--- ' * 4)
        print(g.__name__)
        print(f"equal_result: {equal_result[g]} rate: {equal_result[g][0]/file_counter}")
        print("diffs", count_diff[g])

        if which_algorithm in [SlowAlgorithm.PROB, SlowAlgorithm.BOTH]:
            for s, info in list(zip(count_diff[g], diff_info))[:3]:
                # print(info, np.array(s) / file_counter)
                # print(f'[{equal_result[g][0]}, {s.zero}, {s.positive}]')
                print(f'[{equal_result[g][0]}, {s.zero}, {s.positive}]  # {info} # {g.__name__}"'
                      f' {s.percent_reached=} {0 if s.positive == 0 else s.percent_reached/s.positive}')

        if which_algorithm in [SlowAlgorithm.POINT, SlowAlgorithm.BOTH]:
            for s, info in list(zip(count_diff[g], diff_info))[3:]:
                # print(info, np.array(s))
                # print(np.array(s) / file_counter)

                print(f'[{equal_result[g][0]}, {s.zero}, {s.positive}]  # {info} # {g.__name__}"'
                      f' {s.percent_reached=} {1.0 if s.positive == 0 else s.percent_reached/s.positive}')

        m = metrics[g]
        print(f"metrics: {m}")
        print(m[:diffs_count] / file_counter, np.sqrt(m[diffs_count:]) / file_counter)




    # d1p = 3592
    # d1n = 1616
    # d10 = 12752
    #
    # d5p = 1799
    # d5n = 1767
    # d50 = 14394

greedy_result_comp = [1729, 1476, 409, 0]  # prob_OLD # first_satisfy_ensembles
greedy_result_comp = [1729, 1476, 409, 0]  # prob # first_satisfy_ensembles
greedy_result_comp = [1729, 1476, 409, 0]  # prob_rf # first_satisfy_ensembles

greedy_result_comp = [3337, 59, 462, 0]  # prob_OLD # satisfy_drone
greedy_result_comp = [3337, 59, 462, 0]  # prob # satisfy_drone
greedy_result_comp = [3337, 59, 462, 0]  # prob_rf # satisfy_drone

greedy_result_comp = [3337, 59, 462, 0]  # prob_OLD # satisfy_drone_inn
greedy_result_comp = [3337, 59, 462, 0]  # prob # satisfy_drone_inn
greedy_result_comp = [3337, 59, 461, 0]  # prob_rf # satisfy_drone_inn

greedy_result_comp = [3337, 59, 462, 0]  # prob_OLD # satisfy_ensemble
greedy_result_comp = [3337, 59, 462, 0]  # prob # satisfy_ensemble
greedy_result_comp = [3337, 59, 462, 0]  # prob_rf # satisfy_ensemble

################

greedy_result_comp = [3029, 150, 222, 0]  # points # satisfy_drone_inn
greedy_result_comp = [3029, 150, 222, 0]  # points_old # satisfy_drone_inn

greedy_result_comp = [3029, 150, 222, 0]  # points # satisfy_drone
greedy_result_comp = [3029, 150, 222, 0]  # points_old # satisfy_drone

greedy_result_comp = [1646, 1346, 316, 0]  # points # first_satisfy_ensembles
greedy_result_comp = [1646, 1346, 316, 0]  # points_old # first_satisfy_ensembles

############ 250 NEGATIVE!!!! how ???
greedy_result_comp = [3029, 122, 0, 250]  # points # satisfy_ensemble
greedy_result_comp = [3029, 122, 0, 250]  # points_old # satisfy_ensemble
