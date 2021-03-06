from vivarium_gates_iv_iron.constants import data_keys


class TransitionString(str):
    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split("_TO_")
        return obj

#############################
# Pregnancy Model Variables #
#############################
PREGNANCY_MODEL_NAME = data_keys.PREGNANCY.name
NOT_PREGNANT_STATE = "not_pregnant"
PREGNANT_STATE = "pregnant"
POSTPARTUM_STATE = "postpartum"
MATERNAL_DISORDER_STATE = "maternal_disorder"
NO_MATERNAL_DISORDER_STATE = "no_maternal_disorder"
PREGNANCY_MODEL_STATES = (
    NOT_PREGNANT_STATE,
    PREGNANT_STATE,
    MATERNAL_DISORDER_STATE,
    NO_MATERNAL_DISORDER_STATE,
    POSTPARTUM_STATE
)
PREGNANCY_MODEL_TRANSITIONS = (
    TransitionString(f"{NOT_PREGNANT_STATE}_TO_{PREGNANT_STATE}"),
    TransitionString(f"{PREGNANT_STATE}_TO_{MATERNAL_DISORDER_STATE}"),
    TransitionString(f"{PREGNANT_STATE}_TO_{NO_MATERNAL_DISORDER_STATE}"),
    TransitionString(f"{MATERNAL_DISORDER_STATE}_TO_{POSTPARTUM_STATE}"),
    TransitionString(f"{NO_MATERNAL_DISORDER_STATE}_TO_{POSTPARTUM_STATE}"),
    TransitionString(f"{POSTPARTUM_STATE}_TO_{NOT_PREGNANT_STATE}"),
)
LIVE_BIRTH_OUTCOME = "live_birth"
STILLBIRTH_OUTCOME = "stillbirth"
OTHER_OUTCOME = "other"
INVALID_OUTCOME = "invalid"  # Also used as invalid sex of child
PREGNANCY_OUTCOMES = (LIVE_BIRTH_OUTCOME, STILLBIRTH_OUTCOME, OTHER_OUTCOME, INVALID_OUTCOME)

STATE_MACHINE_MAP = {
    PREGNANCY_MODEL_NAME: {
        "states": PREGNANCY_MODEL_STATES,
        "transitions": PREGNANCY_MODEL_TRANSITIONS,
    },
}

MODERATE_MATERNAL_HEMORRHAGE_STATE = "moderate_maternal_hemorrhage"
SEVERE_MATERNAL_HEMORRHAGE_STATE = "severe_maternal_hemorrhage"
NOT_MATERNAL_HEMORRHAGE_STATE = "not_maternal_hemorrhage"
MATERNAL_HEMORRHAGE_STATES = (
    MODERATE_MATERNAL_HEMORRHAGE_STATE,
    SEVERE_MATERNAL_HEMORRHAGE_STATE,
    NOT_MATERNAL_HEMORRHAGE_STATE,
)

STATES = tuple(
    state for model in STATE_MACHINE_MAP.values() for state in model["states"]
)
TRANSITIONS = tuple(
    state for model in STATE_MACHINE_MAP.values() for state in model["transitions"]
)
