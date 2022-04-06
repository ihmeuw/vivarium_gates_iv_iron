from typing import NamedTuple

####################
# Project metadata #
####################

ARTIFACT_INDEX_COLUMNS = [
    'location',
    'sex',
    'age_start',
    'age_end',
    'year_start',
    'year_end'
]

PROJECT_NAME = "vivarium_gates_iv_iron"
CLUSTER_PROJECT = "proj_cost_effect"

CLUSTER_QUEUE = "all.q"
MAKE_ARTIFACT_MEM = "10G"
MAKE_ARTIFACT_CPU = "1"
MAKE_ARTIFACT_RUNTIME = "3:00:00"
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = [
    "Sub-Saharan Africa",
    "South Asia",
    "LMICs"
]


class __Scenarios(NamedTuple):
    baseline: str = "baseline"
    # TODO - add scenarios here


SCENARIOS = __Scenarios()

GBD_2019_ROUND_ID = 6
GBD_2020_ROUND_ID = 7


class __AgeGroup(NamedTuple):
    BIRTH_ID = 164
    EARLY_NEONATAL_ID = 2
    LATE_NEONATAL_ID = 3
    MONTHS_1_TO_5 = 388
    MONTHS_6_TO_11 = 389
    MONTHS_12_TO_23 = 238
    YEARS_2_TO_4 = 34

    GBD_2019_LBWSG_EXPOSURE = {BIRTH_ID, EARLY_NEONATAL_ID, LATE_NEONATAL_ID}
    GBD_2019_LBWSG_RELATIVE_RISK = {EARLY_NEONATAL_ID, LATE_NEONATAL_ID}
    GBD_2019_SIDS = {LATE_NEONATAL_ID}
    GBD_2020 = {
        EARLY_NEONATAL_ID,
        LATE_NEONATAL_ID,
        MONTHS_1_TO_5,
        MONTHS_6_TO_11,
        MONTHS_12_TO_23,
        YEARS_2_TO_4,
    }


AGE_GROUP = __AgeGroup()

USE_PLW_LOCATION_WEIGHTS = False
