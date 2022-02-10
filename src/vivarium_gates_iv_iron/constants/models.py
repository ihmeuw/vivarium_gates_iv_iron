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
PREGNANCY_MODEL_STATES = (
    NOT_PREGNANT_STATE,
    PREGNANT_STATE,
    POSTPARTUM_STATE,
)
PREGNANCY_MODEL_TRANSITIONS = (
    TransitionString(f"{NOT_PREGNANT_STATE}_TO_{PREGNANT_STATE}"),
    TransitionString(f"{PREGNANT_STATE}_TO_{POSTPARTUM_STATE}"),
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


STATES = tuple(
    state for model in STATE_MACHINE_MAP.values() for state in model["states"]
)
TRANSITIONS = tuple(
    state for model in STATE_MACHINE_MAP.values() for state in model["transitions"]
)
