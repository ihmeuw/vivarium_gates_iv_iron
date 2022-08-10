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
PREPOSTPARTUM_STATES = (MATERNAL_DISORDER_STATE, NO_MATERNAL_DISORDER_STATE)
PREGNANCY_MODEL_TRANSITIONS = (
    TransitionString(f"{NOT_PREGNANT_STATE}_TO_{PREGNANT_STATE}"),
    TransitionString(f"{PREGNANT_STATE}_TO_{MATERNAL_DISORDER_STATE}"),
    TransitionString(f"{PREGNANT_STATE}_TO_{NO_MATERNAL_DISORDER_STATE}"),
    TransitionString(f"{MATERNAL_DISORDER_STATE}_TO_{POSTPARTUM_STATE}"),
    TransitionString(f"{NO_MATERNAL_DISORDER_STATE}_TO_{POSTPARTUM_STATE}"),
    TransitionString(f"{POSTPARTUM_STATE}_TO_{NOT_PREGNANT_STATE}"),
)
FULL_TERM_OUTCOME = "full_term"
NOT_FULL_TERM_OUTCOME = "not_full_term"
LIVE_BIRTH_OUTCOME = "live_birth"
STILLBIRTH_OUTCOME = "stillbirth"
OTHER_OUTCOME = "other"
INVALID_OUTCOME = "invalid"  # Also used as invalid sex of child
PREGNANCY_OUTCOMES = (LIVE_BIRTH_OUTCOME, STILLBIRTH_OUTCOME, OTHER_OUTCOME, INVALID_OUTCOME)

ANEMIA_LEVELS = ('no', 'mild', 'moderate', 'severe')
INVALID_BMI_ANEMIA = "invalid"
LOW_BMI_ANEMIC = "low_bmi_anemic"
LOW_BMI_NON_ANEMIC = "low_bmi_non_anemic"
NORMAL_BMI_ANEMIC = "normal_bmi_anemic"
NORMAL_BMI_NON_ANEMIC = "normal_bmi_non_anemic"
BMI_ANEMIA_CATEGORIES = (
    INVALID_BMI_ANEMIA,
    LOW_BMI_ANEMIC,
    LOW_BMI_NON_ANEMIC,
    NORMAL_BMI_ANEMIC,
    NORMAL_BMI_NON_ANEMIC,
)

INVALID_TREATMENT = "invalid"
NO_TREATMENT = "uncovered"
TREATMENT = "covered"
IV_IRON_TREATMENT_STATUSES = (
    INVALID_TREATMENT,
    NO_TREATMENT,
    TREATMENT,
)

IFA_SUPPLEMENTATION = "ifa"
MMS_SUPPLEMENTATION = "mms"
BEP_SUPPLEMENTATION = "bep"
SUPPLEMENTATION_CATEGORIES = (
    INVALID_TREATMENT,
    NO_TREATMENT,
    IFA_SUPPLEMENTATION,
    MMS_SUPPLEMENTATION,
    BEP_SUPPLEMENTATION,
)


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
