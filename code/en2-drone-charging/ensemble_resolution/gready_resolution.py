#%%
import sys

import numpy as np
import tensorflow as tf
from itertools import combinations, accumulate, chain
from drone_charging_example.components.option_enums import SimulationType
from ensemble_resolution.fall_to_ensemble import FALL_TO_ENSEMBLE, CHARGING_ENSEMBLE
from drone_charging_example.components.drone_nn_wish import DroneNNWish
from sys import maxsize


class GreedyAlgorithms:
    @staticmethod
    def first_satisfy_ensembles_no_sets_priorities_check_minima(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
        return GreedyAlgorithms.first_satisfy_ensembles(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs, check_minima=True)

    @staticmethod
    def first_satisfy_ensembles_partial_sets_priorities_check_minima(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
        return GreedyAlgorithms.first_satisfy_ensembles(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs, better=True, check_minima=True)

    @staticmethod
    def first_satisfy_ensembles_no_sets_priorities(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
        #seems the same but gets different ens priorities when
        return GreedyAlgorithms.first_satisfy_ensembles(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs)

    @staticmethod
    def first_satisfy_ensembles_partial_sets_priorities(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
        return GreedyAlgorithms.first_satisfy_ensembles(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs, better=True)

    @staticmethod
    def first_satisfy_ensembles_partial_sets(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
        return GreedyAlgorithms.first_satisfy_ensembles(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs, better=True)
    @staticmethod

    def first_satisfy_ensembles(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs, better=False, check_minima=False):
        r, t = greedy_algorithm_ens_first(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs, better, check_minima)

        def ddict2dict(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = ddict2dict(v)
            return dict(d)

        def dd2listSet(d, x):
            y = []
            for _ in range(x.shape[1]):
                y.append(set())
            for xx, yy in d.items():
                y[xx] = yy
            return y

        y = dd2listSet(t, x)
        res = GreedyAlgorithms.compatibility_output(y, x, dead_drones, 'first_satisfy_ensembles')

        return res

    """ Greedy naïve algorithms:

        Definition: drone with highest preference := drone that has highest preference across all drones and ensembles

    * ens first tries to satisfy the ensembles with the drones

    * drone first try to satisfy the most decided drone first

    option 1: implemented in greedy_algorithm_drone_first_satisfy_drone
        takes the drone that has the highest preference, and satisfy this drone assigning to the ensemble that has still free spaces (according to the drone preferences)

    option 1b: not implemented
            minor change:
                if drone preference at the current level is already equal to 0, add a drone to list - fill up later
                and after all drones with preferences are dealt with assign the remaining according to their remaining preference
                e.i. which states are forbidden and which ens are not yet full


    option 2: implemented in greedy_algorithm_drone_first_satisfy_preference
        takes the drone that has the highest preference, tries to assign to selected ensemble,

    option 1,2 differences for loop trough the the options in a drone vs loop through the drones
    """
    class DroneFirst:
        @staticmethod
        def satisfy_drone(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
            res = greedy_algorithm_drone_first_satisfy_drone(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs)
            return GreedyAlgorithms.compatibility_output(res, x, dead_drones, 'satisfy_drone')

        @staticmethod
        def satisfy_drone_inn(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
            res = greedy_algorithm_drone_first_satisfy_drone_inn(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs)
            return GreedyAlgorithms.compatibility_output(res, x, dead_drones, 'satisfy_drone_inn')


        @staticmethod
        def satisfy_ensemble(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
            res = greedy_algorithm_drone_first_satisfy_preference(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs)
            return GreedyAlgorithms.compatibility_output(res, x, dead_drones, 'satisfy_ensemble')

    def compatibility_output(list_of_sets, x, dead_drones, string):
        # makes GreedyAlgorithms to have the same output
        from collections import defaultdict
        assign_drones = defaultdict(list)
        assign = {}
        for i, los in enumerate(list_of_sets):
            for d in sorted(los):
                assign_drones[i].append(d)
                assign[d] = i

        assign = dict(sorted(assign.items()))

        # check if all drones are assigned
        drone_count = x.shape[0]
        if drone_count != len(assign):
            if drone_count != len(assign) + len(dead_drones):
                print(f"wrong +=-*()^%$#,  {drone_count=}, assign count={len(assign)}")
            else:
                # add dead drones
                for d in dead_drones:
                    assign[d] = DroneNNWish.NONE
                    assign_drones[DroneNNWish.NONE].append(d)

        assert drone_count == len(assign), f"{drone_count=} == {len(assign)=} error"

        return assign, assign_drones

def testing_validity_ensembles(x, forbidden_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
    minimum_count = sum(ens_mins)
    maximum_count = sum(ens_maxs)
    allowed_drone_count = x.shape[0] - len(forbidden_drones)
    assert allowed_drone_count >= 0

    # print(f"Are drones: {allowed_drone_count} within borders of [{minimum_count}-{maximum_count}]?"
    #       f" {minimum_count <= allowed_drone_count <= maximum_count}")
    return minimum_count <= allowed_drone_count <= maximum_count


def greedy_algorithm_drone_first_satisfy_drone(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
    """ Greedy naïve algorithm drone first try to satisfy the most decided drone first

    option 1: takes the drone that has the highest preference,
    and satisfy this drone assigning to the ensemble that has still free spaces (according to the drone preferences)

    Definition: drone with highest preference := drone that has highest preference across all drones and ensembles
    """
    # todo possible enhancmend - if the drone has preference reached 0 - let first satisfy other drones

    valid = testing_validity_ensembles(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs)
    init_forced = (False, True) if sum(ens_mins) == 0 or not valid else (None, False)

    assignment = [set() for _ in ens_mins]
    assert len(ens_mins) == 7, "this count might change 0 but it is better to be explicit for now"

    # sort drones according to their priority to any ensemble
    z = x.copy()
    z.sort(1)
    drones_priorities = z[:, -1]  # order of drone resolution
    drones_priorities_idx = (-drones_priorities).argsort()  # index of drones in order

    used_drones = set(dead_drones)
    for d_idx in dead_drones:
        assignment[FALL_TO_ENSEMBLE].add(d_idx)

    if charging_drones:
        used_drones.update(set(charging_drones))
        for d_idx in charging_drones:
            assignment[CHARGING_ENSEMBLE].add(d_idx)

    for d_idx in drones_priorities_idx:
        if d_idx in used_drones:
            continue
        order_of_ens = (-x[d_idx]).argsort()
        forced, forced_evaluated = init_forced
        for ens_idx in order_of_ens:
            # try to assign drone as early as possible
            if len(assignment[ens_idx]) < ens_maxs[ens_idx]:
                if len(assignment[ens_idx]) > ens_mins[ens_idx]:
                    #do we sill have freedom to choose
                    if not forced_evaluated:
                        forced_evaluated = True
                        # do we need only to fill forced ensembles?
                        remaining_drone_count_forced_ens = sum(m - len(a) for a, m in zip(assignment, ens_mins) if len(a) < m)

                        remaining_unassigned_drones = x.shape[0] - len(used_drones)
                        forced = remaining_unassigned_drones <= remaining_drone_count_forced_ens
                    # already over the limit, go to an ensemble where there is still missing drone
                    if forced:
                        continue
                assignment[ens_idx].add(d_idx)
                used_drones.add(d_idx)
                break

        if d_idx not in used_drones:
            # add to None ensemble
            assignment[FALL_TO_ENSEMBLE].add(d_idx)
            used_drones.add(d_idx)

    return assignment


def greedy_algorithm_drone_first_satisfy_drone_inn(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
    """ Greedy naïve algorithm drone first try to satisfy the most decided drone first if not null

    option 1b: takes the drone that has the highest preference, and satisfy this drone assigning to the ensemble that has still free spaces (according to the drone preferences)
        but if his preference is <= 0 skip the drone in first iteration

    Definition: drone with highest preference := drone that has highest preference across all drones and ensembles
    """

    valid = testing_validity_ensembles(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs)
    init_forced = (False, True) if sum(ens_mins) == 0 or not valid else (None, False)

    assignment = [set() for _ in ens_mins]

    # sort drones according to their priority to any ensemble
    z = x.copy()
    z.sort(1)
    drones_priorities = z[:, -1]  # order of drone resolution
    drones_priorities_idx = (-drones_priorities).argsort()  # index of drones in order

    used_drones = set(dead_drones)
    for d_idx in dead_drones:
        assignment[FALL_TO_ENSEMBLE].add(d_idx)

    if charging_drones:
        used_drones.update(set(charging_drones))
        for d_idx in charging_drones:
            assignment[CHARGING_ENSEMBLE].add(d_idx)

    todo_drones = set()
    for assign_todo_drones in range(2):
        if assign_todo_drones:
            if not todo_drones:
                break
            # we have finished an iteration & now only drones left to examine are the one in todo_drones
            drones_priorities_idx = todo_drones
        for d_idx in drones_priorities_idx:
            if d_idx in used_drones:
                continue
            order_of_ens = (-x[d_idx]).argsort()
            forced, forced_evaluated = init_forced
            for ens_idx in order_of_ens:
                # try to assign drone as early as possible
                if len(assignment[ens_idx]) < ens_maxs[ens_idx]:
                    if len(assignment[ens_idx]) > ens_mins[ens_idx]:
                        if not forced_evaluated:
                            forced_evaluated = True
                            # do we need only to fill forced ensembles?
                            remaining_drone_count_forced_ens = sum(m - len(a) for a, m in zip(assignment, ens_mins) if len(a) < m)
                            remaining_unassigned_drones = x.shape[0] - len(used_drones)
                            forced = remaining_unassigned_drones <= remaining_drone_count_forced_ens
                        if forced:
                            continue

                    # If the drone has preference reached 0 - let first satisfy other drones
                    if x[d_idx, ens_idx] <= 0 and not assign_todo_drones:  # not the second iteration
                        # since drone values are sorted the rest options in the current are not greater
                        todo_drones.add(d_idx)
                        break
                    assignment[ens_idx].add(d_idx)
                    used_drones.add(d_idx)  # probably not needed...
                    break

            if d_idx not in used_drones and d_idx not in todo_drones:
                # add to None ensemble
                assignment[FALL_TO_ENSEMBLE].add(d_idx)
                used_drones.add(d_idx)

    return assignment


def greedy_algorithm_drone_first_satisfy_preference(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
    testing_validity_ensembles(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs)
    """ Greedy naïve algorithm drone first 
    option 2: takes the drone that has the highest preference, tries to assign to selected ensemble, 
        if it fails the option is removed and drone is put again in choose pool
    Definition: drone with highest preference := drone that has highest preference across all drones and ensembles
    """
    assignment = [set() for _ in ens_mins]

    # sort drones according to their priority to any ensemble
    z = x.copy()
    z.sort(1)
    drones_priorities = z[:, -1]  # order of drone resolution
    drones_priorities_idx = (-drones_priorities).argsort()  # index of drones in order

    used_drones = set(dead_drones)
    for d_idx in dead_drones:
        assignment[FALL_TO_ENSEMBLE].add(d_idx)

    if charging_drones:
        used_drones.update(set(charging_drones))
        for d_idx in charging_drones:
            assignment[CHARGING_ENSEMBLE].add(d_idx)

    skip_forcing = sum(ens_mins) == 0

    forced = False

    sanity = 0  # TODO fix this seems to be wrong we should change the drones_priorities_idx as well...
    while len(used_drones) < x.shape[0]:
        sanity += 1
        if sanity > x.shape[0] * 20:
            raise Exception("sanity error")
        changed_pool = False
        for d_idx in drones_priorities_idx:
            if d_idx in used_drones:
                continue

            order_of_ens = (-x[d_idx]).argsort()

            ens_idx = order_of_ens[0]
            # try to assign drone as early as possible
            if len(assignment[ens_idx]) < ens_maxs[ens_idx]:
                if not skip_forcing and not forced and len(assignment[ens_idx]) >= ens_mins[ens_idx]:
                    # compute if we need only to fill forced ensembles?
                    remaining_drone_count_forced_ens = sum(m - len(a) for a, m in zip(assignment, ens_mins) if len(a) < m)
                    remaining_unassigned_drones = x.shape[0] - len(used_drones)
                    forced = remaining_unassigned_drones <= remaining_drone_count_forced_ens
                if not forced or len(assignment[ens_idx]) < ens_mins[ens_idx]:
                    assignment[ens_idx].add(d_idx)
                    used_drones.add(d_idx)
                    continue

            # disable the state for unsuccessful assignment
            if d_idx not in used_drones:
                x[d_idx, ens_idx] = -100
                changed_pool = True
                break
    return assignment


def greedy_algorithm_ens_first(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs, use_parial_sets, check_minima):
    """ Greedy naïve algorithm
    Fill the top ensemble with the highest probability option.
    (for the first loop it is the ensemble with highest priority and the drones that have the first option this ensemble)
        > Max: take subset of the ones that have none other option then the ones with the highest values.
        >= min && <= max: take all
        < min: repeat selection with lower preferences (e.g. for second loop of the first ensemble it will look at the drones 2nd preference)
            :repeat until ensemble >= min or until there are no more not seen drones
                if the minimum is not reached - this ensemble will not materialize

        param x: drones * ensembles
    """

    used_drones = set()
    assignment = {}  # drone_idx = ens_idx
    assigned_drones = {}  # ens_idx = drone_idx

    def assign_drone(d, ens_idx):
        assignment[d] = ens_idx
        used_drones.add(d)
        if ens_idx not in assigned_drones:
            assigned_drones[ens_idx] = []
        assigned_drones[ens_idx].append(d)

    for d in dead_drones:
        assign_drone(d, FALL_TO_ENSEMBLE)
    for d in charging_drones:
        assign_drone(d, CHARGING_ENSEMBLE)

    order = np.argsort(-x)
    # values = -np.sort(-x)
    values = np.take_along_axis(x, order, axis=1)  # todo test speed
    priorities_order = np.argsort(ens_priorities)[::-1]
    drone_count, ens_count = x.shape

    # ens_maxs = ens_maxs.copy() # don't influence outside world
    # ens_mins = ens_mins.copy() # don't influence outside world
    remaining_mins = np.cumsum([ens_mins[x] for x in priorities_order[::-1]])[::-1]
    remaining_mins = list(remaining_mins)[1:] + [0]
    # # remaining_maxs = np.cumsum([ens_maxs[x] for x in priorities_order[::-1]])[::-1]
    # if check_minima:
    #     raise NotImplementedError()
    # len(used_drones)

    def get_maximum(rmin, ens_idx):
        # rmin needed in the next steps
        if check_minima:
            available_maximum = drone_count - len(used_drones) - rmin
            if available_maximum < ens_maxs[ens_idx]:
                if available_maximum < ens_mins[ens_idx]:
                    ens_maxs[ens_idx] = 0
                else:
                    ens_maxs[ens_idx] = available_maximum
        return int(ens_maxs[ens_idx])

    # assuming different priorities for all ensembles
    for ens_idx, rmin in zip(priorities_order, remaining_mins):
        if len(used_drones) == drone_count:
            # assigned all drones
            break
        ens_maxs_ens_idx = get_maximum(rmin, ens_idx)
        # search for drones according to their priorities in stages
        found_before = []
        for drone_priority in range(ens_count):
            found = []
            for drone in range(drone_count):
                if drone not in used_drones and order[drone, drone_priority] == ens_idx:
                    found.append(drone)
            if len(found_before) + len(found) >= ens_mins[ens_idx]:
                # use all from found_before
                if len(found_before) + len(found) <= ens_maxs_ens_idx:
                    found_before.extend(found)
                else:
                    remove = len(found_before) + len(found) - ens_maxs_ens_idx
                    # select best subset from found according to the value
                    try:
                        found = [f[0] for f in sorted(list(zip(found, *values[[found], ens_idx])), key=lambda l: -l[1])[:-remove]]
                    except Exception as E:
                        print("ERROR", found, found_before, ens_maxs_ens_idx, ens_mins[ens_idx],  file=sys.stderr)
                        raise E
                    found_before.extend(found)
                for d in found_before:
                    assign_drone(d, ens_idx)
                found_before = []
                break
            else:
                found_before.extend(found)
        if found_before:
            # ensemble could not be satisfied try to use the drones elsewhere
            if use_parial_sets:
                for d in found_before:
                    assign_drone(d, ens_idx)
            else:
                pass

    if len(used_drones) != drone_count:
        # not all drones found assignment
        for drones in sorted(set(range(x.shape[0])) - used_drones):
            assign_drone(drones, FALL_TO_ENSEMBLE)

    assignment = dict(sorted(assignment.items()))
    assigned_drones = dict(sorted(assigned_drones.items()))
    return assignment, assigned_drones


def measure_prediction_quality_sum_probabilities_OLD(x, ass, dead: set):
    return sum(x[d, e] for d, e in ass.items() if d not in dead)


def measure_prediction_quality_sum_probabilities(x, assignment):
    suma = 0
    for ens_idx, ens_elements in enumerate(assignment):
        for d in ens_elements:
            suma += x[d, ens_idx]
    return suma


def measure_prediction_quality_sum_probabilities_penalize_rf(x, assignment):
    suma = 0
    for ens_idx, ens_elements in enumerate(assignment):
        for d in ens_elements:
            val = x[d, ens_idx]
            if val > 0:
                pass
            elif val == 0:
                val = -1
            else:
                val = val - 10  # penalization
            suma += val
    return suma


def measure_prediction_quality_sum_points(x, assignment):
    order = np.argsort(-x)
    size = x.shape[1]
    suma = 0
    for ens_idx, ens_elements in enumerate(assignment):
        for d in ens_elements:
            s = size - list(order[d]).index(ens_idx)
            suma += s
    return suma


def measure_prediction_quality_sum_points_OLD(x, ass, dead: set):
    order = np.argsort(-x)
    size = x.shape[1]
    suma = 0
    for d in ass:
        if d in dead:
            continue
        # print(order[d, ass[d]])
        s = size - list(order[d]).index(ass[d])
        # d = 7 - np.where(order[d] == ass[d])[0][0]
        # print(s==d)
        suma += s
    return suma


def recursive_asasign(min_maxs, free_drones: set, must_use, selected, depth=0, maxdepth=6):  # 6 because the 7th will take the rest
    if depth >= maxdepth:
        yield selected + [free_drones]
        # print("in assign returning", [free_drones] + selected)
        # yield [free_drones] + selected  # none category is the first
        return

    min_, max_ = min_maxs[depth]
    for i in range(min_, max_ + 1):
        if len(free_drones) - (must_use[depth] + i - min_) < 0:
            # not enough drones to cover the rest
            return

        for combination in combinations(free_drones, i):
            c = set(combination)
            remaining = free_drones - c
            # print("in assign adding c", c)
            for assignment in recursive_asasign(min_maxs, remaining, must_use, selected + [c], depth + 1, maxdepth):
                # print("in assign", selected + [c])
                yield assignment


def transform_slow_result(result, dead_drones, which_simulation):
    best_points, best_prob, (best_boths_points, best_boths_prob), \
    best_points_assignment, best_prob_assignment, best_both_assignment, \
    same_result_for_different_metrics = result

    if which_simulation == SimulationType.slowProb:
        set_assign = best_prob_assignment
    elif which_simulation == SimulationType.slowPoints:
        set_assign = best_points_assignment
    elif which_simulation == SimulationType.slowBoth:
        set_assign = best_both_assignment
    else:
        raise NotImplementedError()

    listup = []
    for ens_idx, ens_elements in enumerate(set_assign):
        for d in ens_elements:
            listup += [(d, ens_idx)]

    # add back dead drones
    for dd in dead_drones:
        listup += [(dd, 0)]

    listup = sorted(listup)
    # print(sorted(listup))
    assignments = [x[1] for x in listup]
    return assignments


def SLOW(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs, which_simulation=SimulationType.slowBoth):
    if testing_validity_ensembles(x, dead_drones, charging_drones, ens_priorities, ens_mins, ens_maxs):
        remaining_must_use = list(accumulate(ens_mins[::-1]))[::-1] # the sum of mis per remaining ensemble
        min_max_list = list(zip(ens_mins, ens_maxs))
        free_drones_set = set(range(x.shape[0])) - set(dead_drones)

        best_prob, best_points = 0, 0
        best_both_prob, best_both_points = 0, 0
        best_prob_assignment, best_points_assignment, best_both_assignment = None, None, None

        # [a for a in recursive_asasign(min_max_list, free_drones_set, remaining_must_use, []) if a[6] == {2}] #DELETE
        # [a for a in recursive_asasign(min_max_list, free_drones_set, remaining_must_use, [])]

        #[(measure_prediction_quality_sum_probabilities(x, b), b) for b in [a for a in recursive_asasign(min_max_list, free_drones_set, remaining_must_use, [])]]
        # alls = [(measure_prediction_quality_sum_probabilities(x, b), b) for b in [a for a in recursive_asasign(min_max_list, free_drones_set, remaining_must_use, [])]]  # if a[6] == {2}]

        #max(map(lambda x:x[0],[(measure_prediction_quality_sum_probabilities(x, b), b) for b in [a for a in recursive_asasign(min_max_list, free_drones_set, remaining_must_use, [])]]))
        # print("starting assign")
        # countER = 10
        for assignment in recursive_asasign(min_max_list, free_drones_set, remaining_must_use, []):
            # if countER < 0:
            #     exit(1)
            # else:
            #     countER -=1
            prob, points, changed = 0, 0, ''

            if which_simulation == SimulationType.slowProb:
                prob = measure_prediction_quality_sum_probabilities(x, assignment)
                if prob > best_prob:
                    best_prob = prob
                    best_prob_assignment = assignment
                    changed += '-'
            elif which_simulation == SimulationType.slowPoints:
                points = measure_prediction_quality_sum_points(x, assignment)
                if points > best_points:
                    best_points = points
                    best_points_assignment = assignment
                    changed += '*'
            elif which_simulation == SimulationType.slowBoth:
                prob = measure_prediction_quality_sum_probabilities(x, assignment)
                if prob > best_prob:
                    best_prob = prob
                    best_prob_assignment = assignment
                    changed += '-'
                # skip the extra slow evaluation if we know it would not be better
                if prob >= best_both_prob:
                    points = measure_prediction_quality_sum_points(x, assignment)
                    # since we skip this * is not always reliable (may miss some best point assignments)
                    if points > best_points:
                        best_points = points
                        best_points_assignment = assignment
                        changed += '*'
                    if points >= best_both_points:
                        # both at least match the best
                        if (prob > best_both_prob) or (points > best_both_points):
                            best_both_prob = prob
                            best_both_points = points
                            best_both_assignment = assignment
                            changed += '+'
            else:
                raise NotImplementedError()

            if changed:
                # this is for the "hack" - I want to have this debug output when running > /dev/null
                print(changed, points, f"{prob:.5f}", assignment, file=sys.stderr)

        return best_points, best_prob, (best_both_points, best_both_prob), \
               best_points_assignment, best_prob_assignment, best_both_assignment, \
               best_points == best_both_points and best_prob == best_both_prob

    else:
        # this situation in unsatisfiable
        return None


    min_maxs = list(zip(ens_mins, ens_maxs))

#%%


def main():
    def main_create_test_input():
        np.random.seed(42)
        # a = np.random.random((16, 7))  # 16 drones; 1 charger, 5 fields, 1 none ensemble
        a = np.random.random((11, 7))  # 12 drones; 1 charger, 5 fields, 1 none ensemble
        a = tf.nn.softmax(a, axis=1).numpy()
        a[5] = np.zeros(7)
        a[5, 3] = 1
        a[8] = np.zeros(7)
        a[8, 3] = 0.1
        a[8, 5] = 0.1
        a[8, 6] = 0.8
        return a
    main_a = main_create_test_input()
    # priorities_order = np.arange(7)
    main_priorities_order = [0, 1, 2, 3, 4, 5, 6]
    main_mins = [0, 1, 1, 2, 1, 1, 3]
    main_maxs = [16, 5, 2, 3, 4, 3, 3]
    dead_drones = {2, 3}

    c = False
    if not c:
        # main_ass, main_ass2\
        ass = greedy_algorithm_drone_first_satisfy_drone(main_a, dead_drones, main_priorities_order, main_mins, main_maxs)
        ass2 = greedy_algorithm_ens_first(main_a, dead_drones, main_priorities_order, main_mins, main_maxs)
        ass == ass2
    else:
        # slow = SLOW(main_a, dead_drones, main_priorities_order, main_mins, main_maxs)
        # print(slow)
        min_max_main = list(zip(main_mins, main_maxs))
        remaining_must_use = list(accumulate(main_mins[::-1]))[::-1]
        free_drones_set = set(range(main_a.shape[0])) - set(dead_drones)

        min_max_main= [(0, 7), (0, 6), (0, 6), (0, 8), (0, 2), (0, 4), (0, 4)]
        free_drones_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        remaining_must_use = [0, 0, 0, 0, 0, 0, 0]
        # asss = [assignment for assignment in recursive_asasign(min_max_main, free_drones_set, remaining_must_use, [])]
        asss = []
        import time
        start = time.perf_counter()

        for i, assignment in enumerate(recursive_asasign(min_max_main, free_drones_set, remaining_must_use, [])):
            # if i % 1_000_000 == 0:
            #     end = time.perf_counter()
            print(i, assignment)
                # start = end
            # asss.append(assignment)

        print("final length", i)

        print(len(asss))
        print(set(asss))

        assignment = asss[333]
        # main_ass, main_ass2 = greedy_algorithm_drone_first(main_a, dead_drones, main_priorities_order, main_mins, main_maxs)
        main_ass, main_ass2 = greedy_algorithm_ens_first(main_a, dead_drones, main_priorities_order, main_mins, main_maxs)
        # import collections
        # c = collections.Counter(ass.values())
        # print(c)
        print(f'{main_ass=}')
        print(f'{main_ass2=}')
        prob = measure_prediction_quality_sum_probabilities_OLD(main_a, main_ass, dead_drones)
        print(prob)

        points = measure_prediction_quality_sum_points_OLD(main_a, main_ass, dead_drones)
        print(points)

        print(main_a.shape)
        # assignment = [set([]), set([1]), set([3]), set([2,4]), set([]), set([6,7]), set([8])]
        import time
        start = time.perf_counter()
        alive = sorted(list(free_drones_set))
        print(alive)
        for x in range(100_000):
            y = measure_prediction_quality_sum_points(main_a, assignment)
            # y = measure_prediction_quality_sum_probabilities_OLD(main_a, main_ass, dead_drones)
        end = time.perf_counter()
        print(end - start, y)


if __name__ == '__main__':
    main()