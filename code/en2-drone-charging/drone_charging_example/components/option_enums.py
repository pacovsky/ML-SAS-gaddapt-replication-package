from enum import IntEnum, Enum, auto

class SetState(Enum):
    """

    """
    do_nothing = auto()
    set_wish = auto()  # this is probably the right one
    use_set_state = auto()
    use_set_state_minimal = auto()

    do_not_enter_the_processing = auto() # do_nothing == do_not_enter_the_processing (&& if baseline == nn_wish)
    #  = auto()


class ChargerOption(IntEnum):
    """

    """
    percentage = auto()
    in_use = auto()
    full = auto()
    random = auto()


class SimulationType(IntEnum):
    """

    """
    random_behavior = auto()
    predict_on_not_normalized = auto()
    baseline = auto()
    betterbaseline = auto()
    bettersimplebaseline = auto()
    betterbaselineF = auto()
    bettersimplebaselineF = auto()

    # 'betterbaseline' what is EQUAL to baseline_with_rf_argmax with no hys and no rf
    #  '0baseline_with_rf_argmax',      what has influence:rf,hys + 01 ch_fix DOES NOT
    #  '0baseline_with_composiion_rf2', what has influence:rf,hys + 01 ch_fix # TODO the 0 rf and 0 hys a B hys?

    # baseline_with_rf = auto() #error - was same as baseline_with_composiion
    baseline_with_rf_argmax = auto()
    baseline_with_composiion = auto()

    red_flagsF = auto()
    nn = auto()

    greedy_ensemble = auto()  # does not allow sets
    greedy_ensemble2 = auto() #_partial_sets
    greedy_ensemble_priority = auto() # priority is the 2 version
    greedy_ensemble1_priority = auto() # priority
    greedy_ensemble_obey = auto()  # creates ens only if allowed by the rest + priority & no partial sets
    greedy_ensemble_obey_partial = auto()  # creates ens only if allowed by the rest + priority & no partial sets

    greedy_drone = auto() #OK
    greedy_drone2 = auto()#ValueError: too many values to unpack (expected 2) #gready_resolution.py", line 288 # remaining_drone_count_forced_ens
    greedy_drone_ens = auto() #ValueError: too many values to unpack (expected 2) #gready_resolution.py", line 288 # remaining_drone_count_forced_ens

    argmax = auto()
    slowPoints = auto()
    slowProb = auto()
    slowBoth = auto()


class RedflagsEnum(IntEnum):
    answer = auto()
    argmax = auto()
    none = auto()


