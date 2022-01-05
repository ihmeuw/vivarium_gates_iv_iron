from typing import NamedTuple

####################
# Project metadata #
####################

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
    # TODO: add "LMICs"
]


class __Scenarios(NamedTuple):
    baseline: str = "baseline"
    # TODO - add scenarios here


SCENARIOS = __Scenarios()
