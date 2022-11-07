import sys

import numpy as np
from drone_charging_example.components.drone_nn_wish import DroneNNWish, ExtendedWish, GroupedWishState, RedFlagsEnum
from drone_charging_example.components.drone_state import DroneState
import typing
from typing import List
from collections import Counter, namedtuple, defaultdict
from drone_charging_example.red_flags.simulation_setup import Singleton

#meaning?
VERBOSITY = 0 # levels 0-5, none, errors, critical, warnings, info
              # levels 0-5, everything, info, warnings, critical, errors


class Visitor:
    def __init__(self):
        self.verbosity = VERBOSITY
        return

    def __str__(self):
        return self.__class__.__name__

    # def visit(self):
    #     context = self.context #cycle
    #     pass


def argmax_priorities(prediction):
    """
        position 0 has the index of the maximum
        position 1 has the index of 2nd biggest
        last position has the index of minimum
    """

    return np.argsort(-prediction, axis=1)


def argmax_top_k_priorities(prediction, top_x):
    """test speed compared to the argmax_priorities[:top_k]"""
    p = prediction.copy()
    argmaxs = []

    for i in range(min(prediction.shape[1], top_x)):
        argmax = np.argmax(p, axis=1)
        arange = np.arange(len(p))

        p[arange, argmax] = 0
        argmaxs.append(argmax)

    return argmaxs


class Hysteresis(Singleton):

    def __init__(self):
        self.states = []

        self.use_chosen_states = False
        self.size = 5
        self.rate = None  # only applicable by the side importance
        self.eval_function = self._compute_average_prediction
        self.type = ''
        self.clip = None

    def reset(self, string):
        """takes string as a setup: [a/f/b]-rate-size
            A-0.4-9
            f-0.3-4
            param size:
                max number of remembered previous states
        """

        if not string:
            return

        # assert string[0] != 'a' or '-' in string, "a should not have rate"
        eval_dict = dict()
        eval_dict['a'] = self._compute_average_prediction
        eval_dict['f'] = self._compute_floating_prediction_front_dominant
        eval_dict['b'] = self._compute_floating_prediction  # back
        eval_dict['m'] = self._compute_average_prediction_manual
        s = string.split('-')

        if s[0][0].lower() in ['a', 'f', 'b']:
            self.clip = None

        if len(s) == 2:# and s[0][0].lower() in ['a', 'f', 'b', 'c','d','h']:  # compatibility with a0.5-5
            s = [s[0][0], s[0][1:], s[1]]

        assert len(s) == 3
        # assert s[0] != 'A', f'capital A not allowed only F, B allowed, got {s[0]}'
        # assert s[0] != 'C', f'capital A not allowed only F, B allowed, got {s[0]}'
        if s[0].isupper() and s[0].lower() in ['a', 'c', 'e', 'j', 'm', 'p', 's', 'v']:
            print(f'capital A not allowed only F, B groups allowed, got {s[0]}', file=sys.stderr)
            exit(0)

        self.type = s[0]
        self.normalize_each_step = self.type[0].isupper()

        if self.type[0].lower() not in ['a', 'f', 'b']:
            # XXX = set([chr(ord('a') + x) for x in range(29 - 3)]) - set(['c', 'h', 'd', 'a', 'f', 'b'])
            # a, b, c = ['a', 'c'], ['f', 'h'], ['b', 'd']
            # while len(XXX) > 2:
            #     D, E, F = sorted(XXX)[:3]
            #     a.append(D)
            #     XXX.remove(D)
            #     b.append(E)
            #     XXX.remove(E)
            #     c.append(F)
            #     XXX.remove(F)
            a,b,c = (['a', 'c', 'e', 'j', 'm', 'p', 's', 'v'],
                 ['f', 'h', 'g', 'k', 'n', 'q', 't', 'w'],
                 ['b', 'd', 'i', 'l', 'o', 'r', 'u', 'x'])

            _char = char = self.type[0].lower()
            X = None
            if char in a:
                char = 'a'
                X = a
            elif char in b:
                char = 'f'
                X = b
            elif char in c:
                char = 'b'
                X = c

            assert X is not None

            self.type = char + self.type[1:]
            i = X.index(_char)
            self.clip = [None, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2][i]

        # print(self.type.lower(), file=sys.stderr, )
        self.eval_function = eval_dict[self.type.lower()]
        self.rate = float(s[1]) if len(s) > 1 and len(s[1]) > 0 else None
        self.size = int(s[2])


    def add_chosen(self, state): #TODO from collections import deque
        if self.size == len(self.states):
            self.states = self.states[1:]
        self.states.append(state)

    def _compute_floating_prediction_front_dominant(self):
        return self._compute_floating_prediction(last_dominant=False)

    def _compute_floating_prediction(self, last_dominant=True):
        assert len(self.states) > 0, "has to be called on elements`"
        rate = self.rate
        rate1 = 1 - rate
        # init step

        if last_dominant:
            l = 1
            result = np.array(self.states[0])
        else:
            l = -1
            result = np.array(self.states[-1])
        output = 0

        for pred in self.states[::l]:
            result = result * rate + pred * rate1
            output = output * rate + pred
            # output = (output + pred) * rate

        # output = output / output.sum(1)
        # output = output / self.size
        # result /= self.size



        if self.normalize_each_step:
            return result
        return output

    def _compute_average_prediction(self):
        return np.mean(self.states, axis=0)

    def _compute_average_prediction_manual(self):
        # np.
        #  np.mean(self.states, axis=0)
        pass

    def compute_hysteresis(self):
        if len(self.states) == 0:
            return None
        res = self.eval_function()
        m = np.max(res, axis=1)
        m = m.clip(1, np.inf)
        # print("hysteresis: ",res, m,res / m[:, np.newaxis])
        return res / m[:, np.newaxis]



class RedFlags:
    def __init__(self, X, prediction, common_dict, specific_dict, components_data, WORLD, args):
        self.X = X
        self.common_dict = common_dict
        self.specific_dict = specific_dict
        self.components_data = components_data

        self._locals: LocalRedFlags = []
        self._globals: GlobalRedFlags = []
        self.verbosity = VERBOSITY

        self.combined_prediction = None
        # self.combined_prediction = self.process_extended_prediction(prediction, sum_up_prediction=True)
        self.combined_prediction = self.extend_compact_prediction(prediction)

        self.original_argmax = None
        self.local_red_flags_argmax = None

        self.argmax_candidates = None  # = argmax_priorities(self.combined_prediction)
        self.WORLD = WORLD

        wishes_len = len(GroupedWishState) - 1
        #todo when this would be a int instead of bool - then I could distinguish the amount how strong is the rejection
        # allowing red flags and brown flags at once
        self._rejected_states = np.full((wishes_len, len(X)), False)  # state x drones - False == not rejected
        self._optional_candidates = np.full((wishes_len, len(X)), False)  # True - some red flag thought this would be the only option
        self._change_probabilities = np.full((wishes_len, len(X)), 1.0, dtype=np.float16) # multiplies the values at the end
        self._selected_candidates = []

        self._drone_consumption = np.array([d.droneMovingEnergyConsumption for d in WORLD.drones])
        self._drone_speed = np.array([d.speed for d in WORLD.drones])
        self._drone_alert = np.array([d.alert for d in WORLD.drones])
        self._drone_battery = np.array([d.battery for d in WORLD.drones])

        self._field_places = np.array([len(f.places) for f in self.WORLD.fields])
        if self.verbosity >= 4:
            print("FIELD places:", self._field_places)

        self.drone_count = len(WORLD.drones)
        self.field_count = len(WORLD.fields)

        self.drones: List[DroneStateSelector] = []

        self.subfolder = args.subfolder

    def register_local_red_flag(self, visitor: Visitor):
        self._locals.append(visitor)
        visitor.context = self

    def register_global_red_flag(self, visitor: Visitor):
        self._globals.append(visitor)
        visitor.context = self

    def energy_from_distance(self, distance):
        drone_speed = self._drone_speed
        drone_consumption = self._drone_consumption
        if len(distance.shape) > 1:
            reshape = []
            for _ in distance.shape:
                reshape += [1]
            reshape[0] = -1
            drone_speed = drone_speed.reshape(reshape)
            drone_consumption = drone_consumption.reshape(reshape)

        energy_consumed = drone_consumption * (distance / drone_speed)
        return energy_consumed

    def __str__(self):
        return self.__class__.__name__

    def execute_local_red_flags(self):
        for visitor in self._locals:
            visitor.context = self
            visitor.visit()

        rejected_states = self._rejected_states.transpose()
        optional_candidates = self._optional_candidates.transpose()
        # selected_candidates = self._selected_candidates.transpose()

        # clipping
        np.clip(self._change_probabilities, 0, 10, out=self._change_probabilities)
        # np.clip(self._change_probabilities, 0, np.inf, out=self._change_probabilities)

        # equivalent to:
        # self._change_probabilities[self._change_probabilities < 0] = 0
        # self._change_probabilities[self._change_probabilities > 10] = 10

        self.combined_prediction *= self._change_probabilities.transpose()

        optional_candidates = optional_candidates * 1.1
        rejected_states = rejected_states * -20
        candidates = self.combined_prediction + optional_candidates + rejected_states  # this would have to change in case of field ID != ID column
        self.original_argmax = argmax_priorities(self.combined_prediction)

        self.updated_prediction = candidates.copy()
        # this removes smoothing # we are doing the inverse, insead of adding so that it wont be 0, we remove
        # self.updated_prediction[candidates < 0] = 0
        self.updated_prediction[candidates < 0] = -0.01  # forcing those forbidden options to appear after all 0 values from nn

        self.updated_prediction = self.updated_prediction[:, :-1]  # removing dead column

        argmax_candidates = argmax_priorities(candidates)
        candidates = -np.sort(-candidates)

        argmax_candidates[candidates <= 0] = -1

        self.argmax_candidates = argmax_candidates

        drones = []
        for i, (a, c, o) in enumerate(zip(argmax_candidates, candidates, self.original_argmax), start=1):
            drones.append(DroneStateSelector(i, a, c, o))
        self.drones: List[DroneStateSelector] = drones

    def execute_global_red_flags(self):
        for visitor in self._globals:
            visitor.context = self
            visitor.visit()

    def process_extended_prediction(self, prediction, sum_up):
        """sum up or use max """
        assert prediction.shape[-1] in [14], """
            ERROR: expecting extended prediction  
               for network of a type 110 the categories are: + charge, dead 
                 + plus each field has 2 values - protect and approach 
               none, a0, p0, a1, p1, a2, p2, a3, p3, a4, p4, a_chargers, u_chargers, dead
                (that is: approach field, and protect fields, approach and use charger)"""

        names = ["none", "a0", "p0", "a1", "p1", "a2", "p2", "a3", "p3", "a4", "p4", "a_chargers", "u_chargers", "dead"]
        pre2 = np.array_split(prediction, [1, 3, 5, 7, 9, 11, 12, 13], axis=1)
        none = pre2[0]

        if sum_up:
            fields = [p.sum(axis=1) for p in pre2[1:-3]]
            chargers = pre2[-3] + pre2[-2]
        else:
            fields = [p.max(axis=1) for p in pre2[1:-3]]
            chargers = np.max(pre2[-3:-1], axis=0)
        dead = pre2[-1]

        # reshape because some are (x,1,), some only (x,)
        combined_prediction = np.column_stack([x.reshape(-1, ) for x in [none, *fields, chargers, dead]])
        return combined_prediction

    def extend_compact_prediction(self, prediction):
        assert prediction.shape[-1] in [7], "meaning of the columns [none, *fields, chargers]"
        # battery = self.components_data[0].drones[:, :, 0]
        dead = self.components_data[0].drones[:, :, 0] <= 0
        dead = dead.transpose()

        # zero the rest options
        predictions = prediction * (1 - dead)

        combined_prediction = np.column_stack([predictions, dead])
        return combined_prediction

    def process_red_flags(self, postprocessing=RedFlagsEnum.EVERYTHING):
        if postprocessing == RedFlagsEnum.NOTHING:
            if self.combined_prediction is not None:
                return np.argmax(self.combined_prediction, axis=1)  #TODO this would have to change in case of field ID != ID column
            else:
                raise Exception(f"combined_prediction is not set")
        elif postprocessing == RedFlagsEnum.LOCAL_ONLY:
            self._helper_fill_flags()
            self.execute_local_red_flags()

        elif postprocessing == RedFlagsEnum.EVERYTHING:
            if not self._locals and not self._globals:
                self._helper_fill_flags()
                pass

            # local is first stage
            self.execute_local_red_flags()

            # means it uses predicted values
            self.execute_global_red_flags()

        if self.verbosity >= 3:
            print("\t drones ID \t battery \t\t Wish State")
            for drone in self.drones:
                if drone.selected_state != drone.original_wish:
                    print(f"\t{drone.id}: {self._drone_battery[drone.id -1]:.4f} {str(GroupedWishState(drone.original_wish))} -> {str(GroupedWishState(drone.selected_state))}")
                else:
                    if self.verbosity < 4:
                        continue
                    print(f"\t{drone.id}: {self._drone_battery[drone.id -1]:.4f} {str(GroupedWishState(drone.original_wish))}")

    def get_answer(self):
        # returning array of the len drones and states to do
        return [d.selected_state for d in self.drones]

    def _helper_fill_flags(self, subfolder=None):
        if subfolder is None:
            subfolder = self.subfolder
        # print("REGISTER_FLAGS: ", self.subfolder)

        # # # # # self.register_global_red_flag(BirdOnlyFields())
        # # # # # self.register_global_red_flag(TooMuchDronesAssignToTheField())
        # # # # # self.register_global_red_flag(TooMuchDronesWantToFlyToTheFieldWithSimilarBatteryAndETA())
        # # # # # self.register_global_red_flag(NotEnoughDronesProtectTheField())

        if subfolder is None:  # == just local
            subfolder = "11111100"
        elif subfolder == "local_only":
            subfolder = "11111100"
        elif subfolder == "global_only":
            subfolder = "00000011"
        elif subfolder == "local_and_too_much":
            subfolder = "11111101"
        elif subfolder == "local_and_wrong_count":
            subfolder = "11111110"

        number_of_ifs = 8
        if len(subfolder) >= number_of_ifs:
            binary = subfolder

            if len(subfolder) > number_of_ifs:
                # remove prefix (that is for saving file)
                binary = binary.split('_')[1]

                # string = r'C:\Users\X\Pycharms\milad\en2-drone-charging\VENV_DIR\Scripts\python.exe C:/Users/X/Pycharms/milad/en2-drone-charging/drone_charging_example/run.py experiments/8drones.yaml -a -r '
                # t = 2
                # pprint([string + ''.join(x) for x in set(it.permutations('0' * t + '1' * (9-t)))])
        else:
            raise NotImplementedError("temporarily turned off")
            # convert binary number writen as an int
            integer = int(subfolder)
            binary = bin(integer)[2:]  # 'get rid of 0b'
            fill = number_of_ifs - len(binary)
            binary = binary + '0' * fill

        assert len(binary) == number_of_ifs

        for i, rf in zip(binary[:-2], (
            HighEnergyLevelChargeStopper,
            HighEnergyWantsToCharge,
            ChargeOnVeryLowEnergy,
            LowEnergyFlyToField,
            DroneOnEmptyField,
            HighEnergyIdleDronesVeryFarFromBirds
        )):
            if int(i) and rf is not None:
                self.register_local_red_flag(rf())

        for i, rf in zip(binary[-2:], (
            WrongDroneCountField,
            TooMuchDronesAssignToTheCharger
        )):
            if int(i) and rf is not None:
                self.register_global_red_flag(rf())


class LocalRedFlags(Visitor):
    context = None
    pass


class UnderAttackFieldWithoutDrones(LocalRedFlags):
    def visit(self):
        context = self.context

        # # TODO copy paste  only of manual rules
        # context.specific_dict['energies'].reshape(-1) > 0.000  # False mean dead
        #
        # for i in ascending_fields_idx:
        #     if birds_only_fields[i]:
        #         # energyToFlyToCharger computes the energy correctly to anything with .location local variable
        #         birds_field = WORLD.fields[i]
        #         if drone.battery - drone.energyToFlyToCharger(birds_field) <= drone.alert:
        #             answers.append(DroneNNWish(i + 1))
        #             field_found = True
        #             break
        #
        #         else:
        #             # ascending order causes always to hit this branch when there is no option
        #             break
        #
        # if field_found:
        #     continue
        #
        # # all fields with bird


class FalseDetectionOfTerminated(LocalRedFlags):
    def visit(self):
        raise NotImplementedError("this is useless")
        context = self.context
        """this sets up the rejected state (instead of updating)"""
        flags = context.specific_dict['energies'].reshape(-1) > 0.000  # False mean dead

        # #this should be argmax comparison not _rejected_states
        # knowledge = context._rejected_states[GroupedWishState.DEAD_KO]
        # if self.verbosity >= 4:
        #     assert knowledge.shape == flags.shape, (knowledge, flags)
        #     if np.all(knowledge != flags):
        #         for i, (k, f) in enumerate(zip(knowledge, flags)):
        #             if k != f:
        #                 print(f"drone {i} TERMINATION changed to {f}")
        context._rejected_states[GroupedWishState.DEAD_KO] = flags


class HighEnergyWantsToCharge(LocalRedFlags):
    def visit(self):
        context = self.context
        flags = context.specific_dict['energies'].reshape(-1) > 0.75
        context._rejected_states[GroupedWishState.CHARGER] += flags


        x = context.specific_dict['energies'].reshape(-1)
        y = -4 * (x - 0.55) ** 3  # +- 0.55 on the resulting value
        # https://www.wolframalpha.com/input?i=-4+*+%28x+-+0.55%29+**+3%2C++x+from+0+to+1
        # https://www.wolframalpha.com/input?i=-4%28x+-+0.55%29%5E3%2C++x+from+0+to+1
        context._change_probabilities[GroupedWishState.CHARGER] += y


class HighEnergyLevelChargeStopper(LocalRedFlags):
    """this brown flag triggers when the drone wants to charge although energy level > energy needed to fly to the nearest charger + 2* alert"""
    def visit(self):
        context = self.context
        energy = context.specific_dict['energies'].reshape(-1)
        nearest_charger_distance = context.specific_dict['distances_drone_chargers'].reshape(context.drone_count, -1).min(axis=1)
        energy_consumed = context.energy_from_distance(nearest_charger_distance)
        forbid_charging = energy > (energy_consumed + 1.5 * context._drone_alert)
        context._rejected_states[GroupedWishState.CHARGER] += forbid_charging


class ChargeOnVeryLowEnergy(LocalRedFlags):  # HighEnergyWantsToCharge has also part of this smooth operation
    def visit(self):
        context = self.context
        flags = context.specific_dict['energies'].reshape(-1) <= context._drone_alert
        context._optional_candidates[GroupedWishState.CHARGER] += flags


class DroneOnEmptyField(LocalRedFlags):
    def visit(self): #  doesn't forbid, just lower prob. todo + smooth on distance to of the birds to field
        context = self.context
        # birds_on_fields = context.common_dict['counts_0'].reshape(-1)
        # drones_on_fields = context.common_dict['counts_1'].reshape(-1)
        # bird_only_fields = context.common_dict['counts_2'].reshape(-1)
        drone_only_fields = context.common_dict['counts_3'].reshape(-1)

        # get indexes of the fields
        field_enum_index = (~drone_only_fields).argsort()[:np.count_nonzero(drone_only_fields)] + 1  # +1 is to skip none

        # old code that removed the option
        # context._rejected_states[field_enum_index] += np.full(context.drone_count, True) # TODO this disables the removal
        context._change_probabilities[field_enum_index] -= np.full(context.drone_count, 0.3)  # TODO smooth by the distance from the birds
        # context.specific_dict['distances_flock_fields']
        # context.specific_dict['distances_drone_fields']

        #closest drone to field distance np.min(context.specific_dict['distances_flock_fields'], axis=0)
        # context.common_dict['closest_2_flocks_distance_0']
        #context.specific_dict['distances_drone_fields'].min(1)
        # context.specific_dict['distances_drone_fields'] / context.common_dict['closest_2_flocks_distance_0'] ## this is rate between drones


class HighEnergyIdleDronesVeryFarFromBirds(LocalRedFlags):
    def visit(self):
        context = self.context
        #this method require computation of matrix (all drones x all birds)

        energy = context.specific_dict['energies'].reshape(-1)

        drones = np.array([(d.location.x, d.location.y) for d in context.WORLD.drones])
        flocks = np.array([(d.location.x, d.location.y) for d in context.WORLD.birds])
        drones = drones.reshape(-1, 1, 2)

        top_k = 1
        distances = np.linalg.norm(drones - flocks, axis=2)
        smallest_k_distances = np.partition(distances, top_k - 1)[:, :top_k]
        smallest_k_distances = smallest_k_distances.reshape(-1)

        very_far_threshold = 40
        energy_threshold = 0.5
        # idle_drones = context.argmax_candidates[:, 0] == GroupedWishState.NONE
        # far_idles = idle_drones * (energy > energy_threshold) * (smallest_k_distances > very_far_threshold)
        far_idles = (energy > energy_threshold) * (smallest_k_distances > very_far_threshold)
        context._rejected_states[GroupedWishState.NONE] += far_idles


class LowEnergyFlyToField(LocalRedFlags): #TODO make a RED FLAG - check just trip there
    def visit(self):
        context = self.context
        energy = context.specific_dict['energies'].reshape(-1, 1)

        distances_drone_fields = context.specific_dict['distances_drone_fields'].reshape(context.drone_count, -1)
        field_s_nearest_charger = context.specific_dict['distances_chargers_fields'].reshape(-1, context.field_count).min(axis=0)
        distance_to_field_and_back_to_charger = distances_drone_fields + field_s_nearest_charger
        energy_consumed = context.energy_from_distance(distance_to_field_and_back_to_charger)

        # find out if it will drain at least so much battery to trigger battery.alert level
        not_enough_battery = energy_consumed + context._drone_alert.reshape(-1, 1) >= energy
        context._rejected_states[GroupedWishState.FIELD_1:GroupedWishState.FIELD_5 + 1] += not_enough_battery.transpose()


class GlobalRedFlags(Visitor):
    """this is allowed to touch the argmax values"""
    pass

    """
        dont bother with vectors instrucitons
        ?
        From the drones that are tought of remove the once with task of probablity > 1
        
        list of drones still to assign, 
        
        BirdOnlyFields -- needs IDLE drones
        TooMuchDronesInTheCharger -- throw away the biggest battery -> make the 2nd in order
        TooMuchDronesAssignToTheField -> make the 2nd in order with 2 other versions 1. specification again generalization
        
        
        
        
        list of drones - each drone knows its number, list of states he wants (in the order)
        
        DO ONCE / iterate until END ?list of red flags is empty?
        *   process red flag
        *   if flag kicks the drone away - remove the listing he wanted, and all in the list of already triggered
            * flag that red flag has triggered
                * per field if it is field
                * remove red flag if it has triggered
        
       *THE RED FLAGS could HAVE INFO WHAT THEY DO - kick out of charger...., IDLE.... but there is difference and hence
                    the max number in field / charger ... vs... fly to a field it is better
       
       
        * what to do with none state left drones?
            * let them do what their original argmax said
            
        
        TooMuchDronesInTheCharger *cieling 1.5 capacity  is the hard threshold (for alfa version) - kick the one that is with lowest wish...
        TooMuchDronesAssignToTheField *cieling 1.5, capacity is the hard threshold (for alfa version)
        BirdOnlyFields 
    """


class BirdOnlyFields(GlobalRedFlags):
    def visit(self):
        context = self.context
        """how many to send is 1 enough? or full field capacity?
        what if it I have more need than drones,
        am I allowed to steal other needed drones

            look first on the IDLE drones,
        """
        bird_only_fields = context.common_dict['counts_2'].reshape(-1)
        field_enum_index = (~bird_only_fields).argsort()[:np.count_nonzero(bird_only_fields)] + 1  # +1 is to skip none

        # the drones that have the strongest wish to be IDLE
        idle_drones = context.argmax_candidates[:, 0] == GroupedWishState.NONE

        possible_assign = []
        for field in field_enum_index:
            context.argmax_candidates[idle_drones] == field
            # numbers: drone - field
        #TODO
        # assaign the drones


        """
        situation:
                multiple fields, mutliple drones
                drone closeness, drone battery, drone wish
            
        
        """


class TooMuchDronesAssignToTheCharger(GlobalRedFlags):
    def visit(self):
        context = self.context
        """ DEFINITIONSNs: "combine state vs 2 states":
                        means whether we distinguish approaching and charging states for charger.

            IMPORTANT NOTE: in all following options !!!! 1 charger may have multiple charging places !!!!

            Depending on map we can encounter following cases (with increasing complexity):
                1 charger with 2 states
                1 charger with combine state
                multiple chargers with 2 states
                multiple chargers with combined states

            The last case is the general one, so it make sense only to deal with that one.
                (although the solution to previous cases could be straight forward)
                    e.g. for the first case:
                        reasonable solution would be to disallow charging for:
                        approaching drone that would reach the charger with most battery left

            Let's discuss the possible solution for the last case: 'multiple chargers with combined states'
            Since there are multiple ways how to deal with it.
                0. it is possible to reconstruct the 2 states to be able to deal with this situation

                1. easy option)
                    let a random subset pass
                        - this would not be deterministic

                2. complicated option)
                    obtain the data which drone is flying to which charger
                    for drones with distance to charger > epsilon
                        use leftover battery as the desciding factor

                3. overly complicated option)
                    on top of previous option do the assignment to the chargers manually
                        (time to charge, detection which charger is used, compute waiting times, ...)

                4. less complicated option) look who was already charging - let them finish,
                    fill the rest taking the drones by the: (lower is better)
                        a) ~~distance to closest charger~~ [does not allow to take the incoming time to a consideration]
                        b) ~~battery level~~ [does not allow - decision now is good time to charge]
                        c) ~~look at the drone preferences + take argmax~~ BROWN flag only

                        d) allow overbooking when: (again assuming 1 charger):
                                1) the drone is not in ensemble from previous steps but does not have other option
                                            (e.g. all possible states for this drone were eliminated by other red flags)
                                2) just the pseudo-random deletions (so that it would be deterministic)
                                    A solution that would not systematicaly get rid of valid solution
                                        but I assume that the drones would display a lagy behavior



            this method implements the overbooking less complicated option (4.d.1)
            """

        to_be_charged = [d for d in context.drones if d.selected_state == GroupedWishState.CHARGER]
        if not to_be_charged:
            return

        capacities = [ch.acceptedCapacity for ch in context.WORLD.chargers]
        total_capacity = sum(capacities)

        # the naïve approach of one charger only
        if total_capacity >= len(to_be_charged):
            return

        count_to_charge = len(to_be_charged)
        overbooked_by = count_to_charge - total_capacity

        charger_drones_ids = [d.id for d in to_be_charged]
        prev_state = [d.state for d in context.WORLD.drones if d._id in charger_drones_ids]

        # keep the previous assignment
        not_yet_charging = list(
            map(lambda x: x[1],
                filter(lambda x:
                             # x[0] not in [DroneState.MOVING_TO_CHARGER, DroneState.CHARGING],
                             x[0] != DroneState.CHARGING,
                             zip(prev_state, to_be_charged))))

        # remaining_places = total_capacity - (count_to_charge - len(not_yet_charging))
        not_yet_charging.sort(key=lambda x: -x.freedom)
        for d in not_yet_charging:
            if overbooked_by <= 0:
                break

            # if the drone has still another viable option
            if d.freedom > 2 or d.freedom == 2 and not\
                    (d.has_possible_state(GroupedWishState.NONE) or
                     d.has_possible_state(GroupedWishState.DEAD_KO)):

                d.remove_state(GroupedWishState.CHARGER)
                overbooked_by -= 1


# class TooMuchDronesAssignToTheField(GlobalRedFlags): # this is nor redflag
# class TooMuchDronesWantToFlyFromTheChargerAtOnceToTheField # this is subset of following
class TooMuchDronesWantToFlyToTheFieldWithSimilarBatteryAndETA(GlobalRedFlags): #TODO

    def visit(self):
        context = self.context
        # """idle is not a solution - the drone stays on the field - so it will trigger the things again?"""
        """solutions:
                select drone with a) lowest wish until the correct amount is reached
                    remove that field form options for a given drone
                                  b) highest battery - has option to do something else (or wait until the protector gets tired)

             A) allow overbooking
                similar position as with the `TooMuchDronesInTheCharger` - we need 2 states, but here we know that the given field is just one

             B)
                1)
                    - compute who is on the field, they get the allowance (sorted by lowest battery level)
                    - fill the rest by the closeness of the drones
                        or their wish sizes

                2) allow the same number of incomings as well rest not

                    although there might be an optimal solution that would suffer
                    closer drones have low battery so they would protect the field,
                        until the more charged would reach the field - this can have arbitrary many iterations

        """
        raise NotImplementedError()
        context.argmax_candidates

        times_ids = [timeToField(drone) for drone in dronesToSameField]
        times_ids.sort(key=lambda x: x[0])


class NotEnoughDronesProtectTheField(GlobalRedFlags):
    """or fly to...
        * order fields by the size of attackers
            * start with the most vulnerable
        * options)
            ignoring what the drones wants to do:
            * take by the priority of argmax in the drones (on position 3 vs position 4)
            * take by the priority of arg value in the drones (value 0.31 vs 0.14)

            looking into what they do or what they have:
            * take idle drones first
                * then the charging that are not in the ensembles or charging
                * then from too much fields
                ? finally from other fields ?? -> would propagate wrongly

            * take battery into consideration

    """
    pass


class WrongDroneCountField(GlobalRedFlags):
    """A rough try of dealing with wrong counts on the fields at once"""

    def __init__(self):
        super().__init__()
        self.context = None

    @staticmethod
    def logging_info(balance, too_much, not_enough):
        if balance == 0:
            print("reassigning within fields should covers the needs")
        elif balance < 0:
            print("in the sum fields need more drones that were assigned")
        else:
            print("in the sum more than enough drones have been assigned")

        if len(too_much) > 0:
            print("too much drones are assigned to (a) filed(s)")
            # this is not a Red flag in some cases

        if len(not_enough) > 0:
            print("extra drones are needed on the field(s)")


    def visit(self):
        context = self.context
        assigned_states_count = Counter(d.selected_state for d in context.drones)

        too_much = {}
        not_enough = {}
        has_noone = set()
        balance = 0

        birds_on_fields = context.common_dict['counts_0'].reshape(-1)
        # drones_on_fields = context.common_dict['counts_1'].reshape(-1)

        drones_needed = context._field_places * (birds_on_fields > 0)  # this shows the the current need
        # check fields
        for i, (need, places) in enumerate(zip(drones_needed, context._field_places), start=1):
            diff = 0
            if assigned_states_count[i] > places:
                # overbooked
                diff = assigned_states_count[i] - places
                too_much[GroupedWishState(i)] = diff
            if assigned_states_count[i] < need:
                # need help
                diff = assigned_states_count[i] - need
                not_enough[GroupedWishState(i)] = - diff
                if not assigned_states_count[i]:
                    has_noone.add(GroupedWishState(i))
            balance += diff

        if self.verbosity >= 4:
            self.logging_info(balance, too_much, not_enough)

        # is change needed?
        if len(not_enough) + len(too_much) == 0:
            return

        if len(not_enough) > 0:
            # order fields by the size of attackers groups
            priority = np.argsort(-birds_on_fields)
            priority += 1  # make priority indexable for the next line
            priority = [GroupedWishState(p) for p in priority if p in not_enough]

            self.reassign_drones(context.drones, priority, not_enough, has_noone, too_much)

    def reassign_drones(self, drones, priority, not_enough, has_noone, too_much):
        """
            param: priority: List[GroupedWishState], list of fields in order of importance
            param: drones: List[DroneStateSelector] ,
            param: not_enough list of field indices with fields that have assigned not enough drones,
                    with the number of drones that should be assigned
            param: too_much dict of field indices with fields that have assigned too much drones,
                    with the number of drones which might be removed
            param: has_noone: set of field indices where none of the drone is assigned to

        """
        drone_with_states = defaultdict(list)
        for d in drones:
            drone_with_states[d.selected_state].append(d)

        order_in_which_to_try_to_find_drones = [
            GroupedWishState.NONE,
            GroupedWishState.CHARGER,
        ]

        # we might have the fields that have more than enough drones assigned in that case add those fields
        if too_much:
            order_in_which_to_try_to_find_drones.extend(too_much)

        # create dict of fields with list of drones that may be assigned to this field
        field_drones_candidates = defaultdict(list)
        for field in priority:
            for states in order_in_which_to_try_to_find_drones:
                if states not in drone_with_states:
                    continue
                for drone in drone_with_states[states]:
                    if drone.has_possible_state(field):
                        field_drones_candidates[field].append(drone)

        used_drones = set()
        if not field_drones_candidates:
            return

        # todo alternative assignment, for each field with need - check that at least one drone is assigned at first, then continue
        for i in range(self.context.field_count):
            if not has_noone:
                break

            solved = set()
            for field in has_noone:
                if field in solved:
                    continue

                for drone in field_drones_candidates[field]:
                    # take the most restricted drones first 
                    # this might be wrong - the restricted drones are the once that does not have much energy
                    if drone.freedom == i + 1:
                        # don't use too much of too_much drones (drones from another field)
                        another_field = drone.selected_state
                        if too_much and GroupedWishState.is_field(another_field):
                            if too_much[another_field] > 0:
                                too_much[another_field] -= 1
                            else:
                                continue

                        if self.verbosity >= 4:
                            print(f"drone ({drone}) reassigned to the field {field}")
                        drone.select_state(field)
                        solved.add(field)
                        used_drones.add(drone)

                        if not_enough[field] == 1:
                            not_enough.pop(field)
                            priority.remove(field)
                        else:
                            not_enough[field] -= 1
                        break
            has_noone -= solved

        for field in priority:
            list_of_assignable_drones = field_drones_candidates[field]
            need = not_enough[field]

            # greedy algorithm # TODO here might be something clever(er) # at least sort the drones increasengly by the freedom
            assigned = 0
            for drone in list_of_assignable_drones:
                if assigned > need:
                    break
                if drone in used_drones:
                    continue

                # don't use too much of too_much drones (drones from another field)
                another_field = drone.selected_state
                if too_much and GroupedWishState.is_field(another_field):
                    if too_much[another_field] > 0:
                        too_much[another_field] -= 1
                    else:
                        continue

                used_drones.add(drone)
                drone.select_state(field)
                assigned += 1
                if self.verbosity > 4:
                    print(f"drone reassigned {drone}")


class DroneStateSelector:
    def __init__(self, idd, argmax, prob, original):
        self._allowed_states = []
        self._probabilities = {}
        self._chosen = None
        self.id = idd
        self.original_wish: GroupedWishState = original[0]

        for a, p in zip(argmax, prob):
            if a == -1:
                break

            state = GroupedWishState(a)
            self._allowed_states.append(state)
            self._probabilities[state] = p

    def __str__(self):
        return f"Drone: {self.id}, {str(self.selected_state)}, freedom: {self.freedom}"

    @property
    # degrees of freedom
    def freedom(self):
        return len(self._allowed_states)

    @property
    def selected_state(self) -> GroupedWishState:
        if self._chosen:
            return self._chosen
        if self._allowed_states:
            return self._allowed_states[0]
        return self.original_wish

    def remove_state(self, state: GroupedWishState):
        if state == self._chosen:
            self._chosen = None

        if state in self._allowed_states:
            self._allowed_states.remove(state)
            # self._probabilities.pop(state)
        else:
            raise Exception("should never happen")

    def select_state(self, value: GroupedWishState, freeze=True):
        if value in self._allowed_states:
            self._chosen = value
            if freeze:
                self._allowed_states = [value]
                # p = self._probabilities.pop(value)
                # self._probabilities.clear()
                # self._probabilities[value] = p
        else:
            raise ReferenceError(f"trying to set value that is not allowed {value}")

    def has_possible_state(self, state: GroupedWishState):
        return state in self._allowed_states


#TODO remove chargers block to leave ensemble

#%%
if __name__ == '__main__':
    h = Hysteresis()

    rate = 0.8
    length = 9
    h.reset(f"f{rate}-{length}")


    def get_new_result(a=None):
        # print('-------------')
        if a is not None:
            print(a, end='\t'*2)
            h.add_chosen(a)
        np.set_printoptions(precision=3)

        def print_hys(string):
            h.reset(f"{string}{rate}-{length}")
            res = h.compute_hysteresis()

            np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
            print(f"{string}: {res}", end='\t\t')
            return res

        for letter in 'fbaFB':
            res = print_hys(letter)
        print()
        return res

    def get_state_for_debug(seed=None, drones=12, categories=7):
        if seed:
            np.random.seed(seed)
        return np.eye(drones)[np.random.random((categories, categories)).argmax(1)][:, :categories]

#%%
    def real_sign():
        state = get_state_for_debug(111)
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
        state = get_state_for_debug(222)
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
        state = get_state_for_debug(333)
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
        state = get_state_for_debug(111)

        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
#%%
    #
    #
    # for _ in range(10):
    #     state = get_state_for_debug()
    #     get_new_result(state)
    #
    # state = get_state_for_debug()
    # result = get_new_result(state)
#%%
    def easy_sign():
        get_new_result()
        state = np.array([[1, 0, 0, 0]])
        get_new_result(state)
        state = np.array([[1, 0, 0, 0]])
        get_new_result(state)
        state = np.array([[1, 0, 0, 0]])
        get_new_result(state)
        state = np.array([[0, 0, 1, 0]])
        get_new_result(state)
        state = np.array([[1, 0, 0, 0]])
        get_new_result(state)
        state = np.array([[0, 0, 1, 0]])
        get_new_result(state)

        state = np.array([[0, 0, 0, 1]])
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)
        get_new_result(state)

# %%

        # state = np.array([[1, 0, 0, 0]])


    real_sign()
    # easy_sign()

