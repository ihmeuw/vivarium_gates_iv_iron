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

# TODO input details of model states and transitions

PREGNANCY_MODEL_NAME = data_keys.PREGNANCY.name
SUSCEPTIBLE_STATE = f"susceptible_to_{PREGNANCY_MODEL_NAME}"
PREGNANT_STATE = "pregnant"
POSTPARTUM_STATE = "postpartum"
PREGNANCY_MODEL_STATES = (
    SUSCEPTIBLE_STATE,
    PREGNANT_STATE,
    POSTPARTUM_STATE,
)
PREGNANCY_MODEL_TRANSITIONS = (
    TransitionString(f"{SUSCEPTIBLE_STATE}_TO_{PREGNANT_STATE}"),
    TransitionString(f"{PREGNANT_STATE}_TO_{POSTPARTUM_STATE}"),
    TransitionString(f"{POSTPARTUM_STATE}_TO_{SUSCEPTIBLE_STATE}"),
)

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
