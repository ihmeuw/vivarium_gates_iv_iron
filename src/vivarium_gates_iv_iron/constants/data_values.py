import pandas as pd
from typing import NamedTuple

TOTAL_POSTPARTUM_DAYS = 42
PREPOSTPARTUM_DURATION_DAYS = 7
POSTPARTUM_DURATION_DAYS = TOTAL_POSTPARTUM_DAYS - PREPOSTPARTUM_DURATION_DAYS
PREPOSTPARTUM_DURATION_RATIO = PREPOSTPARTUM_DURATION_DAYS / TOTAL_POSTPARTUM_DAYS
POSTPARTUM_DURATION_RATIO = POSTPARTUM_DURATION_DAYS / TOTAL_POSTPARTUM_DAYS


class _HemoglobinDistributionParameters(NamedTuple):
    XMAX: int = 220
    EULERS_CONSTANT: float = 0.57721566490153286060651209008240243104215933593992
    GAMMA_DISTRIBUTION_WEIGHT: float = 0.4
    MIRROR_GUMBEL_DISTRIBUTION_WEIGHT: float = 0.6
    PREGNANCY_MEAN_ADJUSTMENT_FACTOR: tuple = (0.919325, 0.86, 0.98)  # 95% confidence interval
    PREGNANCY_STANDARD_DEVIATION_ADJUSTMENT_FACTOR: float = 1.032920188


HEMOGLOBIN_DISTRIBUTION_PARAMETERS = _HemoglobinDistributionParameters()


class _AnemiaDisabilityWeights(NamedTuple):
    MILD: float = 0.004
    MODERATE: float = 0.052
    SEVERE: float = 0.149


ANEMIA_DISABILITY_WEIGHTS = _AnemiaDisabilityWeights()

hemoglobin_threshold_data = {
    'sex': 'Female',
    'pregnancy_status': ['pregnant', 'not_pregnant'] * 2,
    'age_start': [5, 5, 15, 15],
    'age_end': [14, 14, 57, 57],
    'mild_anemia': [(110, 114), (110, 114), (100, 109), (110, 119)],
    'moderate_anemia': [(80, 110), (80, 110), (70, 100), (80, 110)],
    'severe_anemia': [(0, 80), (0, 80), (0, 70), (0, 80)]
}
HemoglobinThresholds = pd.DataFrame(hemoglobin_threshold_data).set_index(keys=['sex', 'age_start', 'age_end'])
