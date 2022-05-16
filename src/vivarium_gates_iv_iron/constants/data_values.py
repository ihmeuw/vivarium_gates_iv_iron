import pandas as pd
import numpy as np
from typing import NamedTuple

from vivarium_gates_iv_iron.constants import models


class _Durations(NamedTuple):
    DETECTION: float = 6 / 52
    PARTIAL_TERM: float = 24 / 52
    FULL_TERM: float = 40 / 52
    PREPOSTPARTUM: float = 1 / 52
    POSTPARTUM: float = 5 / 52


DURATIONS = _Durations()

MATERNAL_HEMORRHAGE_HEMOGLOBIN_POSTPARTUM_SHIFT = 6.8  # g/L
PROBABILITY_MODERATE_MATERNAL_HEMORRHAGE = (0.85, 0.81, 0.89)

# state: (mean_params, sd_params)
HEMOGLOBIN_CORRECTION_FACTORS = {
    models.NOT_PREGNANT_STATE: (
        (1., 1., 1.), (1., 1., 1.)
    ),
    models.PREGNANT_STATE: (
        (0.919325, 0.86, 0.98), (1.032920188, 1.032920188, 1.032920188)
    ),
}


class _HemoglobinDistributionParameters(NamedTuple):
    XMAX: int = 220
    EULERS_CONSTANT: float = np.euler_gamma
    GAMMA_DISTRIBUTION_WEIGHT: float = 0.4
    MIRROR_GUMBEL_DISTRIBUTION_WEIGHT: float = 0.6


HEMOGLOBIN_DISTRIBUTION_PARAMETERS = _HemoglobinDistributionParameters()


ANEMIA_DISABILITY_WEIGHTS = {"none": 0., "mild": 0.004, "moderate": 0.052, "severe": 0.149}


# tuples are: (age_start, age_end, severe_upper, moderate_upper, mild_upper)
_hemoglobin_threshold_data = {"pregnant": [(5, 15, 80, 110, 115),
                                           (15, 57, 70, 100, 110)],
                              "not_pregnant": [(5, 15, 80, 110, 115),
                                               (15, 57, 80, 110, 120)]}
_hemoglobin_state_map = {"pregnant": models.PREGNANCY_MODEL_STATES[1:],
                         "not_pregnant": [models.NOT_PREGNANT_STATE]}
_htd = []
for key, states in _hemoglobin_state_map.items():
    for state in states:
        for row in _hemoglobin_threshold_data[key]:
            _htd.append((state, "Female", *row))

HEMOGLOBIN_THRESHOLD_DATA = pd.DataFrame(_htd,
                                         columns=["pregnancy_status", "sex", "age_start", "age_end", "severe",
                                                  "moderate", "mild"])
