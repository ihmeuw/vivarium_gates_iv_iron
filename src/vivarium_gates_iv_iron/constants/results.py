import itertools

import pandas as pd

from vivarium_gates_iv_iron.constants import models

#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = "total_population"
TOTAL_YLDS_COLUMN = "years_lived_with_disability"
TOTAL_YLLS_COLUMN = "years_of_life_lost"

# Columns from parallel runs
INPUT_DRAW_COLUMN = "input_draw"
RANDOM_SEED_COLUMN = "random_seed"
OUTPUT_SCENARIO_COLUMN = "placeholder_branch_name.scenario"

# Add due to make_results bug
OUTPUT_INPUT_DRAW_COLUMN = 'input_data.input_draw_number'
OUTPUT_RANDOM_SEED_COLUMN = 'randomness.random_seed'

STANDARD_COLUMNS = {
    "total_population": TOTAL_POPULATION_COLUMN,
    "total_ylls": TOTAL_YLLS_COLUMN,
    "total_ylds": TOTAL_YLDS_COLUMN,
}

THROWAWAY_COLUMNS = [f"{state}_event_count" for state in models.STATES]

TOTAL_POPULATION_COLUMN_TEMPLATE = "total_population_{POP_STATE}"
PERSON_TIME_COLUMN_TEMPLATE = (
    "person_time_in_{YEAR}_in_age_group_{AGE_GROUP}"
)
DEATH_COLUMN_TEMPLATE = (
    "death_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_in_age_group_{AGE_GROUP}"
)
YLLS_COLUMN_TEMPLATE = (
    "ylls_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_in_age_group_{AGE_GROUP}"
)
YLDS_COLUMN_TEMPLATE = (
    "ylds_due_to_{CAUSE_OF_DISABILITY}_in_{YEAR}_in_age_group_{AGE_GROUP}"
)
STATE_PERSON_TIME_COLUMN_TEMPLATE = (
    "{STATE}_person_time_in_{YEAR}_in_age_group_{AGE_GROUP}"
)
TRANSITION_COUNT_COLUMN_TEMPLATE = (
    "{TRANSITION}_event_count_in_{YEAR}_in_age_group_{AGE_GROUP}"
)

PREGNANCY_OUTCOME_COUNT_COLUMN_TEMPLATE = (
    "{PREGNANCY_OUTCOME}_count_in_{YEAR}_in_age_group_{AGE_GROUP}"
)

PREGNANCY_STATE_PERSON_TIME_COLUMN_TEMPLATE = (
    "{PREGNANCY_STATE}_with_{PREGNANCY_OUTCOME}_person_time_in_{YEAR}_in_age_group_{AGE_GROUP}"
)

PREGNANCY_TRANSITION_COUNT_COLUMN_TEMPLATE = (
    "{PREGNANCY_TRANSITION}_count_in_{YEAR}_in_age_group_{AGE_GROUP}"
)

MATERNAL_DISORDER_INCIDENT_COUNT_COLUMN_TEMPLATE = (
    "incident_cases_of_maternal_disorders_in_{YEAR}_in_age_group_{AGE_GROUP}"
)
MATERNAL_HEMORRHAGE_INCIDENT_COUNT_COLUMN_TEMPLATE = (
    "incident_cases_of_maternal_hemorrhage_in_{YEAR}_in_age_group_{AGE_GROUP}"
)
MATERNAL_HEMORRHAGE_PERSON_TIME_COLUMN_TEMPLATE = (
    "maternal_hemorrhage_person_time_in_{YEAR}_in_age_group_{AGE_GROUP}"
)
HEMOGLOBIN_EXPOSURE_SUM_COLUMN_TEMPLATE = (
    "hemoglobin_exposure_sum_{YEAR}_in_age_group_{AGE_GROUP}_among_{PREGNANCY_STATE}_with_{MATERNAL_HEMORRHAGE_STATE}"
)
COLUMN_TEMPLATES = {
    "population": TOTAL_POPULATION_COLUMN_TEMPLATE,
    "person_time": PERSON_TIME_COLUMN_TEMPLATE,
    "deaths": DEATH_COLUMN_TEMPLATE,
    "ylls": YLLS_COLUMN_TEMPLATE,
    "ylds": YLDS_COLUMN_TEMPLATE,
    "state_person_time": STATE_PERSON_TIME_COLUMN_TEMPLATE,
    "transition_count": TRANSITION_COUNT_COLUMN_TEMPLATE,
    "pregnancy_outcome_counts": PREGNANCY_OUTCOME_COUNT_COLUMN_TEMPLATE,
    "pregnancy_state_person_time": PREGNANCY_STATE_PERSON_TIME_COLUMN_TEMPLATE,
    "pregnancy_transition_counts": PREGNANCY_TRANSITION_COUNT_COLUMN_TEMPLATE,
    "maternal_disorder_incident_counts": MATERNAL_DISORDER_INCIDENT_COUNT_COLUMN_TEMPLATE,
    "maternal_hemorrhage_incident_counts": MATERNAL_HEMORRHAGE_INCIDENT_COUNT_COLUMN_TEMPLATE,
    "hemoglobin_exposure_sum": HEMOGLOBIN_EXPOSURE_SUM_COLUMN_TEMPLATE,
}

NON_COUNT_TEMPLATES = []

POP_STATES = ("living", "dead", "tracked", "untracked")
SEXES = ("female",)
YEARS = tuple(range(2022, 2025))
# TODO - add literals for ages in the model
AGE_GROUPS = (
    '5_to_9',
    '10_to_14',
    '15_to_19',
    '20_to_24',
    '25_to_29',
    '30_to_34',
    '35_to_39',
    '40_to_44',
    '45_to_49',
    '50_to_54',
    '55_to_59',
)
# TODO - add causes of death
CAUSES_OF_DEATH = (
    "other_causes",
    "maternal_disorders",
)
# TODO - add causes of disability
CAUSES_OF_DISABILITY = (
    "maternal_disorders",
)

TEMPLATE_FIELD_MAP = {
    "POP_STATE": POP_STATES,
    "YEAR": YEARS,
    "SEX": SEXES,
    "AGE_GROUP": AGE_GROUPS,
    "CAUSE_OF_DEATH": CAUSES_OF_DEATH,
    "CAUSE_OF_DISABILITY": CAUSES_OF_DISABILITY,
    "PREGNANCY_STATE": models.PREGNANCY_MODEL_STATES,
    "PREGNANCY_OUTCOME": models.PREGNANCY_OUTCOMES,
    "PREGNANCY_TRANSITION": models.PREGNANCY_MODEL_TRANSITIONS,
    "MATERNAL_HEMORRHAGE_STATE": models.MATERNAL_HEMORRHAGE_STATES
}


def RESULT_COLUMNS(kind="all"):
    if kind not in COLUMN_TEMPLATES and kind != "all":
        raise ValueError(f"Unknown result column type {kind}")
    columns = []
    if kind == "all":
        for k in COLUMN_TEMPLATES:
            columns += RESULT_COLUMNS(k)
        columns = list(STANDARD_COLUMNS.values()) + columns
    else:
        template = COLUMN_TEMPLATES[kind]
        filtered_field_map = {
            field: values
            for field, values in TEMPLATE_FIELD_MAP.items()
            if f"{{{field}}}" in template
        }
        fields, value_groups = filtered_field_map.keys(), itertools.product(
            *filtered_field_map.values()
        )
        for value_group in value_groups:
            columns.append(
                template.format(
                    **{field: value for field, value in zip(fields, value_group)}
                )
            )
    return columns


def RESULTS_MAP(kind):
    if kind not in COLUMN_TEMPLATES:
        raise ValueError(f"Unknown result column type {kind}")
    columns = []
    template = COLUMN_TEMPLATES[kind]
    filtered_field_map = {
        field: values
        for field, values in TEMPLATE_FIELD_MAP.items()
        if f"{{{field}}}" in template
    }
    fields, value_groups = list(filtered_field_map.keys()), list(
        itertools.product(*filtered_field_map.values())
    )
    for value_group in value_groups:
        columns.append(
            template.format(
                **{field: value for field, value in zip(fields, value_group)}
            )
        )
    df = pd.DataFrame(value_groups, columns=map(lambda x: x.lower(), fields))
    df["key"] = columns
    df["measure"] = kind
    return df.set_index("key").sort_index()
