import pandas as pd
import numpy as np
from typing import NamedTuple

from vivarium_gates_iv_iron.constants import models

TOTAL_POSTPARTUM_DAYS = 42
PREPOSTPARTUM_DURATION_DAYS = 7
POSTPARTUM_DURATION_DAYS = TOTAL_POSTPARTUM_DAYS - PREPOSTPARTUM_DURATION_DAYS
PREPOSTPARTUM_DURATION_RATIO = PREPOSTPARTUM_DURATION_DAYS / TOTAL_POSTPARTUM_DAYS
POSTPARTUM_DURATION_RATIO = POSTPARTUM_DURATION_DAYS / TOTAL_POSTPARTUM_DAYS

MATERNAL_HEMMORHAGE_HEMOGLOBIN_POSTPARTUM_SHIFT = 6.8  # g/L


class _HemoglobinDistributionParameters(NamedTuple):
    XMAX: int = 220
    EULERS_CONSTANT: float = np.euler_gamma
    GAMMA_DISTRIBUTION_WEIGHT: float = 0.4
    MIRROR_GUMBEL_DISTRIBUTION_WEIGHT: float = 0.6
    PREGNANCY_MEAN_ADJUSTMENT_FACTOR: tuple = (0.919325, 0.86, 0.98)  # 95% confidence interval
    PREGNANCY_STANDARD_DEVIATION_ADJUSTMENT_FACTOR: tuple = (1.032920188, 1.032920188, 1.032920188)
    NO_PREGNANCY_MEAN_ADJUSTMENT_FACTOR: tuple = (1., 1., 1.)
    NO_PREGNANCY_STANDARD_DEVIATION_ADJUSTMENT_FACTOR: tuple = (1., 1., 1.)


HEMOGLOBIN_DISTRIBUTION_PARAMETERS = _HemoglobinDistributionParameters()

# ADJUSTMENT_FACTORS_BY_PREGNANCY_STATE = {
#     models.NOT_PREGNANT_STATE: (HEMOGLOBIN_DISTRIBUTION_PARAMETERS.NO_PREGNANCY_MEAN_ADJUSTMENT_FACTOR,
#                                 HEMOGLOBIN_DISTRIBUTION_PARAMETERS.NO_PREGNANCY_STANDARD_DEVIATION_ADJUSTMENT_FACTOR),
#     models.PREGNANT_STATE: (HEMOGLOBIN_DISTRIBUTION_PARAMETERS.PREGNANCY_MEAN_ADJUSTMENT_FACTOR,
#                             HEMOGLOBIN_DISTRIBUTION_PARAMETERS.PREGNANCY_STANDARD_DEVIATION_ADJUSTMENT_FACTOR),
#     models.MATERNAL_DISORDER_STATE: (HEMOGLOBIN_DISTRIBUTION_PARAMETERS.PREGNANCY_MEAN_ADJUSTMENT_FACTOR,
#                             HEMOGLOBIN_DISTRIBUTION_PARAMETERS.PREGNANCY_STANDARD_DEVIATION_ADJUSTMENT_FACTOR),
#     models.NO_MATERNAL_DISORDER_STATE: (HEMOGLOBIN_DISTRIBUTION_PARAMETERS.PREGNANCY_MEAN_ADJUSTMENT_FACTOR,
#                             HEMOGLOBIN_DISTRIBUTION_PARAMETERS.PREGNANCY_STANDARD_DEVIATION_ADJUSTMENT_FACTOR),
#     models.POSTPARTUM_STATE: (HEMOGLOBIN_DISTRIBUTION_PARAMETERS.PREGNANCY_MEAN_ADJUSTMENT_FACTOR,
#                             HEMOGLOBIN_DISTRIBUTION_PARAMETERS.PREGNANCY_STANDARD_DEVIATION_ADJUSTMENT_FACTOR),
# }


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
