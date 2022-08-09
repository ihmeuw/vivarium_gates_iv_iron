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
CLUSTER_PROJECT = "proj_simscience"

CLUSTER_QUEUE = "all.q"
MAKE_ARTIFACT_MEM = "10G"
MAKE_ARTIFACT_CPU = "1"
MAKE_ARTIFACT_RUNTIME = "3:00:00"
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = [
    "Sub-Saharan Africa",
    "South Asia",
    "LMICs",
    "Ethiopia",
    "Nigeria",
    "India",
]


class __Scenarios(NamedTuple):
    baseline: str
    oral_iron: str
    antenatal_iv_iron: str
    postpartum_iv_iron: str
    antenatal_and_postpartum_iv_iron: str


SCENARIOS = __Scenarios(*__Scenarios._fields)
