import itertools

import pandas as pd

from vivarium_gates_iv_iron.constants import models, data_values

#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = "total_population"
TOTAL_YLDS_COLUMN = "years_lived_with_disability"
TOTAL_YLLS_COLUMN = "years_of_life_lost"

# Columns from parallel runs
INPUT_DRAW_COLUMN = "input_draw"
RANDOM_SEED_COLUMN = "random_seed"
OUTPUT_SCENARIO_COLUMN = "intervention.scenario"

# Add due to make_results bug
OUTPUT_INPUT_DRAW_COLUMN = "input_data.input_draw_number"
OUTPUT_RANDOM_SEED_COLUMN = "randomness.random_seed"

STANDARD_COLUMNS = {
    "total_population": TOTAL_POPULATION_COLUMN,
    "total_ylls": TOTAL_YLLS_COLUMN,
    "total_ylds": TOTAL_YLDS_COLUMN,
}

THROWAWAY_COLUMNS = [f"{state}_event_count" for state in models.STATES]

TOTAL_POPULATION_COLUMN_TEMPLATE = "total_population_{POP_STATE}"

DEATH_COLUMN_TEMPLATE = "death_due_to_{CAUSE_OF_DEATH}_year_{YEAR}_age_{AGE_GROUP}_pregnancy_status_{PREGNANCY_STATE}_maternal_supplementation_{SUPPLEMENTATION}_antenatal_iv_iron_{ANTENATAL_IV_IRON}_postpartum_iv_iron_{POSTPARTUM_IV_IRON}"
YLLS_COLUMN_TEMPLATE = "ylls_due_to_{CAUSE_OF_DEATH}_year_{YEAR}_age_{AGE_GROUP}_pregnancy_status_{PREGNANCY_STATE}_maternal_supplementation_{SUPPLEMENTATION}_antenatal_iv_iron_{ANTENATAL_IV_IRON}_postpartum_iv_iron_{POSTPARTUM_IV_IRON}"
YLDS_COLUMN_TEMPLATE = "ylds_due_to_{CAUSE_OF_DISABILITY}_year_{YEAR}_age_{AGE_GROUP}_pregnancy_status_{PREGNANCY_STATE}_maternal_supplementation_{SUPPLEMENTATION}_antenatal_iv_iron_{ANTENATAL_IV_IRON}_postpartum_iv_iron_{POSTPARTUM_IV_IRON}"
PREGNANCY_OUTCOME_COUNT_COLUMN_TEMPLATE = "{PREGNANCY_OUTCOME}_count_year_{YEAR}_age_{AGE_GROUP}"
PREGNANCY_STATE_PERSON_TIME_COLUMN_TEMPLATE = (
    "{PREGNANCY_STATE}_with_{PREGNANCY_OUTCOME}_with_{MATERNAL_HEMORRHAGE_STATE}_person_time_year_{YEAR}_age_{AGE_GROUP}"
)
PREGNANCY_TRANSITION_COUNT_COLUMN_TEMPLATE = (
    "{PREGNANCY_TRANSITION}_count_year_{YEAR}_age_{AGE_GROUP}"
)
MATERNAL_DISORDER_INCIDENT_COUNT_COLUMN_TEMPLATE = (
    "incident_cases_of_maternal_disorders_year_{YEAR}_age_{AGE_GROUP}"
)
MATERNAL_HEMORRHAGE_INCIDENT_COUNT_COLUMN_TEMPLATE = (
    "incident_cases_of_{WITH_MATERNAL_HEMORRHAGE_STATE}_year_{YEAR}_age_{AGE_GROUP}"
)
HEMOGLOBIN_EXPOSURE_SUM_COLUMN_TEMPLATE = (
    "hemoglobin_exposure_sum_among_{PREGNANCY_STATE}_with_{MATERNAL_HEMORRHAGE_STATE}_year_{YEAR}_age_{AGE_GROUP}_maternal_supplementation_{SUPPLEMENTATION}_antenatal_iv_iron_{ANTENATAL_IV_IRON}_postpartum_iv_iron_{POSTPARTUM_IV_IRON}"
)
ANEMIA_LEVEL_PERSON_TIME_COLUMN_TEMPLATE = (
    "{ANEMIA_LEVEL}_anemia_person_time_among_{PREGNANCY_STATE}_with_{MATERNAL_HEMORRHAGE_STATE}_year_{YEAR}_age_{AGE_GROUP}_maternal_supplementation_{SUPPLEMENTATION}_antenatal_iv_iron_{ANTENATAL_IV_IRON}_postpartum_iv_iron_{POSTPARTUM_IV_IRON}"
)
MATERNAL_BMI_PERSON_TIME_COLUMN_TEMPLATE = (
    "bmi_person_time_{BMI_CATEGORY}_year_{YEAR}_age_{AGE_GROUP}_pregnancy_status_{PREGNANCY_STATE}"
)
INTERVENTION_PERSON_TIME_COLUMN_TEMPLATE = (
    "person_time_{INTERVENTION_CATEGORY}_bmi_{BMI_CATEGORY}_year_{YEAR}_age_{AGE_GROUP}_pregnancy_status_{PREGNANCY_STATE}"
)
INTERVENTION_COUNT_COLUMN_TEMPLATE = (
    "count_of_{INTERVENTION_CATEGORY}_bmi_{BMI_CATEGORY}_year_{YEAR}_pregnancy_status_{PREGNANCY_STATE}"
)

COLUMN_TEMPLATES = {
    "population": TOTAL_POPULATION_COLUMN_TEMPLATE,
    "deaths": DEATH_COLUMN_TEMPLATE,
    "ylls": YLLS_COLUMN_TEMPLATE,
    "ylds": YLDS_COLUMN_TEMPLATE,
    "pregnancy_outcome_counts": PREGNANCY_OUTCOME_COUNT_COLUMN_TEMPLATE,
    "pregnancy_state_person_time": PREGNANCY_STATE_PERSON_TIME_COLUMN_TEMPLATE,
    "pregnancy_transition_counts": PREGNANCY_TRANSITION_COUNT_COLUMN_TEMPLATE,
    "maternal_disorder_incident_counts": MATERNAL_DISORDER_INCIDENT_COUNT_COLUMN_TEMPLATE,
    "maternal_hemorrhage_incident_counts": MATERNAL_HEMORRHAGE_INCIDENT_COUNT_COLUMN_TEMPLATE,
    "hemoglobin_exposure_sum": HEMOGLOBIN_EXPOSURE_SUM_COLUMN_TEMPLATE,
    "anemia_state_person_time": ANEMIA_LEVEL_PERSON_TIME_COLUMN_TEMPLATE,
    "maternal_bmi_person_time": MATERNAL_BMI_PERSON_TIME_COLUMN_TEMPLATE,
    "intervention_person_time": INTERVENTION_PERSON_TIME_COLUMN_TEMPLATE,
    "intervention_counts": INTERVENTION_COUNT_COLUMN_TEMPLATE,
}

NON_COUNT_TEMPLATES = []

POP_STATES = ("living", "dead", "tracked", "untracked")
SEXES = ("female",)
YEARS = tuple(range(2020, 2041))
AGE_GROUPS = (
    "5_to_9",
    "10_to_14",
    "15_to_19",
    "20_to_24",
    "25_to_29",
    "30_to_34",
    "35_to_39",
    "40_to_44",
    "45_to_49",
    "50_to_54",
    "55_to_59",
)
CAUSES_OF_DEATH = (
    "other_causes",
    "maternal_disorders",
)
CAUSES_OF_DISABILITY = ("maternal_disorders", "anemia")
INTERVENTION_CATEGORIES = tuple([
    *[f'antenatal_iv_iron_{s}' for s in models.IV_IRON_TREATMENT_STATUSES],
    *[f'postpartum_iv_iron_{s}' for s in models.IV_IRON_TREATMENT_STATUSES],
    *[f'maternal_supplementation_{s}' for s in models.SUPPLEMENTATION_CATEGORIES],
])


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
    "MATERNAL_HEMORRHAGE_STATE": models.MATERNAL_HEMORRHAGE_STATES,
    "WITH_MATERNAL_HEMORRHAGE_STATE": models.MATERNAL_HEMORRHAGE_STATES[:-1],
    "ANEMIA_LEVEL": data_values.ANEMIA_DISABILITY_WEIGHTS.keys(),
    "SUPPLEMENTATION": models.SUPPLEMENTATION_CATEGORIES,
    "ANTENATAL_IV_IRON": models.IV_IRON_TREATMENT_STATUSES,
    "POSTPARTUM_IV_IRON": models.IV_IRON_TREATMENT_STATUSES,
    "BMI_CATEGORY": models.BMI_ANEMIA_CATEGORIES,
    "INTERVENTION_CATEGORY": INTERVENTION_CATEGORIES
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
