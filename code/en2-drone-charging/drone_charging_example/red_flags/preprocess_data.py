import logging
import time

import numpy as np
from drone_charging_example.red_flags.simulation_setup import SimulationSetup, DroneMode

def print_logging_info(verbose, prev, message):
    if verbose:
        cur = time.perf_counter()
        sec = cur - prev
        logging.info(f'{message} time: {sec}s ({sec / 60}min)')
        return cur
    return prev


#COMMON INTERFACE "i"
#def interface(processed, raw, normalize):
    #"""if the element should not be part of the results add to blacklist"""
    # distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    # chargers, droness, flocks = raw

def i_get_mode_moving(processed, raw, _):
    # approaching charger :=  moving - {approach ensml} - {scaring}
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    chargers, droness, flocks = raw

    modes = (droness[:, :, 1] == 1).transpose().reshape(drone_count, -1, 1)
    # timeit.timeit(lambda: (droness[:, :, 1]).transpose().reshape(4, -1, 1) == 1, number=100)
    return modes


def i_get_mode_charging(processed, raw, _):
    # approaching charger :=  moving - {approach ensml} - {scaring}
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    chargers, droness, flocks = raw

    modes = (droness[:, :, 1] == 0).transpose().reshape(drone_count, -1, 1)
    # timeit.timeit(lambda: (droness[:, :, 1]).transpose().reshape(4, -1, 1) == 1, number=100)
    return modes


def i_get_mode_dead(processed, raw, _):
    # approaching charger :=  moving - {approach ensml} - {scaring}
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    chargers, droness, flocks = raw

    modes = (droness[:, :, 1] == 3).transpose().reshape(drone_count, -1, 1)
    # timeit.timeit(lambda: (droness[:, :, 1]).transpose().reshape(4, -1, 1) == 1, number=100)
    return modes


def i_get_energy(processed, raw, _):
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    chargers, droness, flocks = raw
    energies = droness[:, :, 0].transpose().reshape(drone_count, -1, 1)
    return energies

def i_get_energy2(processed, raw, _):
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    chargers, droness, flocks = raw
    energies = droness[:, :, 0].transpose().reshape(drone_count, -1, 1)

    return [*battery_distance_normalization(energies)] # to trigger addition of lists


def i_process_other_drones(_, raw, __):
    # distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    chargers, droness, flocks = raw
    other_drones_columns = process_other_droness(droness)
    return other_drones_columns


def i_normalized_summary_distances(_, raw, normalize):
    # distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    chargers, droness, flocks = raw
    flocks_new_columns = get_closest_flocks_for_drones(droness, flocks, normalize=normalize)
    return flocks_new_columns


def i_chargers_occupancy(_, raw, __):
    # distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    chargers, droness, flocks = raw
    # charges -> as a occupancy rate (1 means all used)
    capacity = SimulationSetup().charger_count * SimulationSetup().charger_capacity
    chargers_occupancy = np.sum(chargers, axis=1) / capacity
    chargers_occupancy = chargers_occupancy.reshape(-1, 1)
    return chargers_occupancy


def i_onions(processed, _, normalize):
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    # chargers, droness, flocks = raw
    onions = get_onion_distances(distances_drone_fields, distances_flock_fields, drone_count, flock_count, normalize)
    return onions


def i_avg_field_distance(processed, _, __):
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    # chargers, droness, flocks = raw
    field_distance_for_all_drones = compute_average_field_distance(distances_drone_fields, drone_count)
    return field_distance_for_all_drones


def i_closest_2_drones_distance(processed, _, __):
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    # chargers, droness, flocks = raw
    closest_2_drones_distance = get_x_closest_distances(distances_drone_fields)
    return closest_2_drones_distance


def i_closest_2_flocks_distance(processed, _, __):
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    # chargers, droness, flocks = raw
    closest_2_flocks_distance = get_x_closest_distances(distances_flock_fields)
    return closest_2_flocks_distance


def i_drones_field_closeness(processed, _, __):
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    # chargers, droness, flocks = raw
    # number of drones that are closest to a field x
    drones_field_closeness = within_fields_distance_comparison(distances_drone_fields, drone_count)
    return drones_field_closeness


def i_flocks_field_closeness(processed, _, __):
    distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    # chargers, droness, flocks = raw
    flocks_field_closeness = within_fields_distance_comparison(distances_flock_fields, flock_count)
    return flocks_field_closeness


def i_counts_on_field(_, raw, __):
    # distances_drone_fields, distances_flock_fields, drone_count, flock_count = processed
    chargers, droness, flocks = raw
    counts = counts_on_field(SimulationSetup(), droness, flocks)
    return counts


def add_to_dict(dictionary, name, list_or_item):
    if type(list_or_item) == list:
        for i, item in enumerate(list_or_item):
            new_name = f'{name}_{i}'
            add_to_dict(dictionary, new_name, item)
    else:
        dictionary[name] = list_or_item


def get_transform_data_dicts(data, first=None, verbose=True, normalize=True):
    """
        Converts the loaded data from DataTransformation.SplitDataByType
        to the style where the new columns don't have exact information about other drones, but only summaries, and
        the x, y coordinates are transformed to the distances from fixed points e.i. chargers, fields

        :param data has the same format as the result from method load_all_at_once
        :param first contains the result of the time.perf_counter() before we started loading data
        :param verbose

        :returns 2 dicts - common and specific
            some columns are shared across drones - common but some of them are only drone specific
            expected shape for specific: (4, 100200, -1)
                           for common:   (100200, -1)
    """
    (chargers, droness, flocks), labelss = data  # same format as in load_all_at_once

    # some columns are shared across drones - common but some of them are only drone specific
    # expected shape for specific: (4, 100200, -1)
    #                for common:   (100200, -1)

    common_dict = {}  # shared across drones
    specific_dict = {}  # not shared across drones

    prev = first if first else time.perf_counter()
    prev = print_logging_info(verbose, prev, 'loaded')

    setup = SimulationSetup()
    drone_count = setup.drone_count
    flock_count = setup.flock_count

    field_centers = np.array(setup.field_centers)
    distances_flock_fields = xy_to_distance(flocks, field_centers, normalize)
    prev = print_logging_info(verbose, prev, 'flocks_xy_to_distance')

    distances = drone_xy_to_distances(droness, drone_count, normalize=normalize)
    prev = print_logging_info(verbose, prev, 'drones_xy_to_distance')
    distances_drone_fields = distances[:, :, setup.charger_count:]
    distances_drone_chargers = distances[:, :, :setup.charger_count]

    # following lines makes variable chargers count fixed in output
    # deals with both overflow and underflow
    limited_number_of_chargers = 3
    distances_drone_chargers.sort(axis=2)
    distances_drone_chargers = distances_drone_chargers[:, :, :limited_number_of_chargers]
    if distances_drone_chargers.shape[2] < limited_number_of_chargers:
        extra_columns_needed = limited_number_of_chargers - distances_drone_chargers.shape[2]
        last = distances_drone_chargers[:, :, -1:]
        #fill with last
        filling = [last for x in range(extra_columns_needed)]
        distances_drone_chargers = np.concatenate([distances_drone_chargers, *filling], axis=2)


    prev = print_logging_info(verbose, prev, 'splitting the data')

    distances_chargers_fields = xy_to_distance(setup.get_charger_centers_array, field_centers, normalize)
    prev = print_logging_info(verbose, prev, 'distances_chargers_fields')

    add_to_dict(specific_dict, 'distances_drone_fields', distances_drone_fields.transpose([1, 0, 2]))
    add_to_dict(specific_dict, 'distances_drone_chargers', distances_drone_chargers.transpose([1, 0, 2])) # can take just one as a minimum? or top 3...
    add_to_dict(specific_dict, 'distances_flock_fields', distances_flock_fields.transpose([1, 0, 2]))
    add_to_dict(specific_dict, 'distances_chargers_fields', distances_chargers_fields.transpose([1, 0, 2]))
    prev = print_logging_info(verbose, prev, 'distances transposes')

    processed = (distances_drone_fields, distances_flock_fields, drone_count, flock_count)
    raw = (chargers, droness, flocks)

    for dic, fce, name, msg in [
        (specific_dict, i_get_energy, 'energies', 'the real energy value'),
        (specific_dict, i_get_energy2, 'energies', 'energies transformation'),
        (specific_dict, i_get_mode_moving, 'i_get_mode_moving', 'i_get_mode_moving'),
        (specific_dict, i_get_mode_charging, 'i_get_mode_charging', 'i_get_mode_charging'),
        (specific_dict, i_get_mode_dead, 'i_get_mode_dead', 'i_get_mode_dead'),
        (specific_dict, i_process_other_drones, 'other_drones_summary', 'other drones energy summary per group (charging and not)'),

        # The following method was removed due to conceptual and actual problems
        #   - computes 2D matrix of distances between flocks and drones
        #   - then takes the top k elements in axis of flocks - or drones - which fixes the amount of that element
        #  so if we want to have both variable it follows to ignore such a method
        #   - the implementation however does not produce flocks2drones but drones2flocks
        # (specific_dict, i_normalized_summary_distances, 'distances_flocks_to_2closest_drones_median_furthers', 'get_closest_flocks_for_drones'),  # FIXME ? historical string - function had wrong name...

        (common_dict, i_chargers_occupancy, 'chargers_occupancy', 'chargers_occupancy'),
        (common_dict, i_onions, 'onions', 'onions'),
        (common_dict, i_closest_2_drones_distance, 'closest_2_drones_distance', 'closest_2_drones_distance'),
        (common_dict, i_closest_2_flocks_distance, 'closest_2_flocks_distance', 'closest_2_flocks_distance'),
        (common_dict, i_drones_field_closeness, 'drones_field_closeness', 'number of drones closest to the field'),
        (common_dict, i_flocks_field_closeness, 'flocks_field_closeness', 'number of flocks closest to the field'),
        (common_dict, i_counts_on_field, 'counts', 'counts_on_field'), #TODO this has the fields in order
        # quick test shows that compute_average_field_distance does not greatly improve results
        #(specific_dict, i2_avg_field_distance, 'field_distance_for_all_drones', 'average_field_distance')
    ]:
        res = fce(processed, raw, normalize)
        add_to_dict(dic, name, res)
        prev = print_logging_info(verbose, prev, msg)

    return common_dict, specific_dict


def merge_data(common_dict, specific_dict, labelss, verbose=2):
    prev = time.perf_counter()

    blacklist = ['energies', 'i_get_mode_moving', 'i_get_mode_charging', 'i_get_mode_dead', 'distances_flock_fields', 'distances_chargers_fields']  # TODO whitelist instead?
    # "blacklist items are for red_flags use only - it should not appear in normal results"
    cd = [i for d, i in common_dict.items() if d not in blacklist]
    sd = [i for d, i in specific_dict.items() if d not in blacklist]

    # concatenate all in one
    X = np.concatenate([np.concatenate([*x, *cd], axis=1) for x in zip(*sd)])
    prev = print_logging_info(verbose, prev, 'final X concatenation')

    additional_labels = [specific_dict['i_get_mode_charging'], specific_dict['i_get_mode_dead'],] # is used to create y
    labels, label_weights = make_labels_categorical(labelss, additional_labels)
    y = np.concatenate([x for x in labels.transpose([1, 0, 2])])
    print_logging_info(verbose, prev, 'final labelss transpose')

    return X, y

def process_data(data, first=None, verbose=True, return_dict_as_well=False, normalize=True):
    """Converts the loaded data from DataTransformation.SplitDataByType
        to the style where the new columns don't have exact information about other drones, but only summaries, and
        the x, y coordinates are transformed to the distances from fixed points e.i. chargers, fields
    """
    labels = data[1]
    common_dict, specific_dict = get_transform_data_dicts(data, first, verbose, normalize)

    X, y = merge_data(common_dict, specific_dict, labels)

    if return_dict_as_well:
        return X, y, common_dict, specific_dict

    return X, y


def make_labels_categorical(labels, extended_labels=None, sample_weights=False):
    """    :param labels:
    :param modes_as_input: drone state when it is extended_labels - charging, dead

    :param sample_weights: - if true replace with default measured value if array replace with the given value

    :returns 0 - none
        17 -extended_labels - charging
        18 -dead
    """
    if extended_labels is not None:
        # used in enhanced version with extra columns
        # adds the column with extended_labels
        extra = [c.transpose((1, 0, 2)) for c in extended_labels]
        columns = [labels] + extra
        labels = np.concatenate(columns, axis=2)


    # extend labels by the first column - no ensemble AKA one hot encoding
    label_no_ensemble = 1 - np.max(labels, axis=2)
    new_shape = (*label_no_ensemble.shape, 1)
    label_no_ensemble = label_no_ensemble.reshape(new_shape)
    labels = np.concatenate([label_no_ensemble, labels], axis=2)

    # weights 0 is_ensemble_nothing
    class_weights = np.array(
        [0.01265993, 0.04169432, 0.05427943, 0.16586222, 0.1316951, 0.05075509, 0.3279661, 0.3279661, 0.67032915,
         0.67032915, 0.92158854, 0.906047, 1., 1., 0.12028203, 0.1892202, 0.21455792], dtype=np.float32)

    if not sample_weights:
        return labels, class_weights

        # class_weights = {a: b for (a, b) in enumerate(class_weights[1:])}  # first arg is none column
        # class_weights is not supported for 3+ dimensions

        weights_ind = np.argmax(labels, axis=2)
        sample_weights = class_weights[weights_ind]
        # dw = tf.data.Dataset.from_tensor_slices(sample_weights)
    return labels, sample_weights


def enumerate_index_array_skipping_one_element(size=4):
    """ returns a tuple of number and a rest
    [(0, array([1, 2, 3])),
     (1, array([0, 2, 3])),
     (2, array([0, 1, 3])),
     (3, array([0, 1, 2]))]
    """
    for i in range(size):
        x = np.arange(0, i)
        y = np.arange(i + 1, size)
        z = np.hstack([x, y])
        yield i, z


def process_other_droness(droness):
    """computes statistics for #drones groups - energy levels and counts for: charging_drones, alive_not_charging"""
    x = [_process_other_drones(droness[:, rest]) for _, rest in
         enumerate_index_array_skipping_one_element(SimulationSetup().drone_count)]
    x = np.array(x)
    return x


def _process_other_drones(drones): #TODO battery might be normalized
    """computes statistics - energy levels and counts for: charging_drones, alive_not_charging"""
    number_of_drones = drones.shape[1]
    energy = drones[:, :, 0]
    mode = drones[:, :, 1]

    charging_drones = mode == DroneMode.CHARGING.value
    alive_not_charging = (mode != DroneMode.DEAD.value) * (mode != DroneMode.CHARGING.value)  # "*" is "and"

    nans = np.full(number_of_drones, np.nan)
    results = []

    for selected_drones in [charging_drones, alive_not_charging]:
        selected_drones_count = np.sum(selected_drones, axis=1)
        # print(f"_process_other_drones - {selected_drones_count = } / {number_of_drones = } - {drones.shape = }")
        normalized_count = selected_drones_count / number_of_drones

        drones_energy = np.where(selected_drones, energy, nans)

        # nan results are in this case ok
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'(All-NaN slice encountered|Mean of empty slice)')

            energy_min = np.nanmin(drones_energy, axis=1)
            energy_mean = np.nanmean(drones_energy, axis=1)
            energy_max = np.nanmax(drones_energy, axis=1)

        np.nan_to_num(energy_min, copy=False, nan=1.0) #TODO this could be separated to 0 and 1 for charging_drones, alive_not_charging
        np.nan_to_num(energy_mean, copy=False, nan=0.5)
        np.nan_to_num(energy_max, copy=False, nan=0.0)

        # invalid_value = [1.0, 0.5, 0.0]  # how to denote wrong output ???
        # invalid_value = [0.0, 0.0, 0.0]
        # invalid_value = [1.0, 1.0, 1.0]

        results.extend([normalized_count, energy_min, energy_mean, energy_max])

    results = np.array(results)
    results = results.transpose()
    return results


def get_closest_flocks_for_drones(droness, flockss, normalize=False):
    """computes distance for each drone for the 2 closest flocks and median"""
    #each line
    top_k = 2
    output = []
    droness = droness[:, :, 2:4].reshape(-1, 4, 1, 2)
    for drones, flocks in zip(droness, flockss):
        # computes distances between all drones and all flocks for one frame
        distances = np.linalg.norm(drones - flocks, axis=2)
        smallest_k_distances = np.partition(distances, top_k - 1)[:, :top_k]

        row_drone_new_data = np.column_stack([
            smallest_k_distances,
            # distances.max(axis=1), # this is irrelevant for a big number of drones
            distances.mean(axis=1),  # this might be as well
        ])
        output.append(row_drone_new_data)
    output = np.stack(output, axis=1)
    if normalize:
        output = distance_normalization(output)
    return output


def drone_xy_to_distances(droness, drone_count=4, normalize=False):
    """Converts drone's x,y to the distances to the chargers and fields"""
    # four drones each has fields of (energy, mode, x,y) fields
    # either working on one example or on batch
    assert droness.shape[2] == 4 \
        and droness.shape[1] == drone_count \
        and ((droness.shape[0] == 1) or (droness.shape[0] % 600 == 0 and droness.shape[0] >= 600)), \
            f"{droness.shape[2]}==4,  {droness.shape[1]}==drones=={drone_count},  {droness.shape[0]} == lines"

    setup = SimulationSetup()
    centers = []
    centers.extend(setup.charger_centers)
    centers.extend(setup.field_centers)
    centers = np.array(centers)

    points = droness[:, :, 2:4]
    new_columns = xy_to_distance(points, centers, normalize)

    return new_columns


def distance_normalization(distances):
    """ to trasform x, y columns use xy_to_distance

        normalization of distances to [0-1]; note that this differs from the normalize_coordinates_xy
        any distance bigger then x will be normalized to 1
    """
    x = distances / SimulationSetup().distance_infinity
    #todo try to make distances not linear - sigmoid / log
    # non_linear_distance = 1 / (1 + math.exp(-2 - 4.4 * (self.battery - n) / n))

    x = np.minimum(x, 1)  # cutoff at one (INFINITY - everything further than ... is 1)
    return x

def battery_distance_normalization(battery):
    # ccaX ~ 1000,
    # 0.ccax ~ 0.005
    # y = (1 + math.exp(- math.log(ccaX) * (x - n) / n + math.log(0.ccaX))) ** -1
    n = 0.6
    non_linear_battery = 1 / (1 + np.exp(-2 - 4.4 * (battery - n) / n))

    ss = SimulationSetup()
    speed = ss._drone_speed
    consumption = ss._drone_consumption
    # battery - normalized as a a distance_normalization (without the division cutoff so can be 0-4 instead 0-1)
    # max distance that the drone can fly normalized to the same scale as the other distances
    battery_as_norm_distance = (np.array(battery / consumption, np.int) * speed) / ss.distance_infinity
    return battery_as_norm_distance, non_linear_battery


def xy_to_distance(xy, xy2, normalize=False):
    """computes the distances between two xy (len 2) columns
    to get results faster use the following condition should hold:    len(xy) > len(xy2)"""

    distances = []
    for point in xy2:
        x = np.linalg.norm(xy - point, axis=2)
        x = np.array(x, dtype=np.float16)  # back to float16

        if normalize:
            x = distance_normalization(x)
        distances.append(x)

    new_columns = np.stack(distances, axis=2)
    return new_columns


def get_onion_distances(drone_distances, flock_distances, drone_count, flock_count, normalize):
    ONIONS = []
    onion = onion_distance(drone_distances, drone_count, normalized_data=normalize)
    ONIONS.extend(onion)
    onion = onion_distance(flock_distances, flock_count, normalized_data=normalize)
    ONIONS.extend(onion)
    # ONIONS = np.column_stack(ONIONS)
    return ONIONS


def onion_distance(distances, count, normalized_data=False, onion_distances=None):
    """for every field computes how many drones or flocks are within each onion layer
        :param distances: columns x,y only
        :param count: number of drones/flocks
        :param normalized_data: is the data normalized to [0-1]
        :param onion_distances: the onion distances
    """

    if not onion_distances:
        onion_distances = np.array([25, 50, 75, 100])
        if normalized_data and max(onion_distances) > 1:
            onion_distances = distance_normalization(onion_distances)

    onion = []
    for d in onion_distances:
        counts_of_entities_under_distance = np.count_nonzero(distances < d, axis=1) / count
        onion.append(counts_of_entities_under_distance)

    return onion


def get_x_closest_distances(distances, closest_count=2):
    """for each fields returns distances to closest elements e.g. field 1: has closest drone in distance 14"""
    assert closest_count > 0
    # closest_2_drones_distance = np.partition(distances, closest_x - 1, axis=1)[:, :closest_x, :]
    close = np.partition(distances, closest_count - 1, axis=1)
    closest_distances = [close[:, i, :] for i in range(closest_count)]
    return closest_distances


def counts_on_field(simulation_setup, droness, flocks, birds_cutoff=3):
    bird_count_in_fields = simulation_setup.is_in_fields(flocks)
    drone_count_in_fields = simulation_setup.is_in_fields(droness[:, :, 2:4])
    fields_with_birds_without_drones = ((bird_count_in_fields > 0) * (drone_count_in_fields == 0))
    fields_with_drones_without_birds = ((drone_count_in_fields > 0) * (bird_count_in_fields == 0))
    bird_count_in_fields = np.minimum(bird_count_in_fields / birds_cutoff, 1)  # cutoff on the 3
    drone_count_in_fields = drone_count_in_fields / simulation_setup.drone_count
    return [bird_count_in_fields, drone_count_in_fields, fields_with_birds_without_drones, fields_with_drones_without_birds]


def within_fields_distance_comparison(drone_distances, element_count, field_count=5):
    """number of elements that are closest to a field x"""
    closest = drone_distances.argpartition(0)[:, :, 0]
    counts = [np.count_nonzero(closest == x, axis=1) for x in range(field_count)]
    stack = np.stack(counts, axis=1) / element_count
    return stack


def compute_average_field_distance(drone_distances, drone_count):
    """weighted distance drone - fields, by the avg distance"""
    # avg drone distance to each field
    avg = drone_distances[:, :, :].mean(axis=1)
    # avg = avg / DISTANCE_OF_INFINITY
    # avg = np.where(avg > 1, 1, avg)
    field_distance_for_all_drones = []
    for i in range(drone_count):
        # field_distance_for_current_drone
        fdcd = drone_distances[:, i, :] / avg
        fdcd2 = fdcd / fdcd.sum(axis=1).reshape((-1, 1))
        field_distance_for_all_drones.append(fdcd2)
    return np.array(field_distance_for_all_drones)


def normalize_coordinates_xy(xy):
    """normalize x,y columns to [0,1]; note that this differs from the distance_normalization"""

    # the numbers are extreme values in each dimension that appeared in the simulation runs
    min_n = np.array([16, 16])
    max_n = np.array([207.875, 146.375])
    max_n = max_n - min_n
    return (xy - min_n) / max_n

